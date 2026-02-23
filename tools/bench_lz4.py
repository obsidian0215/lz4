#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hw_telemetry import TelemetryProbe, apply_freq_percent

# Paths
LZ4_BIN = os.environ.get("LZ4_CPU_BIN", "/root/lz4/programs/lz4")
LZ4_GPU_BIN = "/root/lz4/lz4_gpu/lz4_gpu"
SAMPLES_DIR = "/root/samples"
RESULTS_DIR = "/root/lz4/exp_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "lz4_param_sweep.csv")
LZ4_DAEMON_SOCKET_PATH = "/tmp/lz4_gpu_daemon.sock"
LZ4_DAEMON_PID_PATH = "/tmp/lz4_gpu_daemon.pid"

# Configuration Space
CPU_BLOCK_SIZES = ["64K", "256K", "1M"]
GPU_BLOCK_SIZES = ["16K", "32K", "64K", "128K"]
CPU_THREADS = [1]
HASH_LOGS = [14, 15, 16]
LOCAL_SIZES = [1]
GPU_ACCELS = [1, 2, 4]

SCOPE_COMPRESS = "compress"
SCOPE_DECOMPRESS = "decompress"
SCOPE_COMMON = "common"

SCOPE_METRICS = {
    SCOPE_COMPRESS: ["ratio", "comp_kernel_reported_mbs"],
    SCOPE_DECOMPRESS: ["dec_kernel_reported_mbs"],
    SCOPE_COMMON: ["ratio", "comp_kernel_reported_mbs", "dec_kernel_reported_mbs"],
}


def _is_executable_file(path):
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def _try_build_lz4_cpu_binary():
    build_plans = [
        (["make", "-C", "/root/lz4/programs", "lz4", "-j8"], "/root/lz4/programs/lz4"),
        (["make", "-C", "/root/lz4", "lz4", "-j8"], "/root/lz4/lz4"),
    ]
    for cmd, out_bin in build_plans:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0 and _is_executable_file(out_bin):
                return out_bin
        except Exception:
            continue
    return None


def resolve_lz4_cpu_binary():
    candidates = []
    env_bin = os.environ.get("LZ4_CPU_BIN")
    if env_bin:
        candidates.append(env_bin)
    candidates.extend([
        "/root/lz4/programs/lz4",
        "/root/lz4/lz4",
    ])

    for c in candidates:
        if _is_executable_file(c):
            return c

    built = _try_build_lz4_cpu_binary()
    if built:
        return built

    raise FileNotFoundError(
        "Cannot find/build LZ4 CPU binary. Tried env LZ4_CPU_BIN, /root/lz4/programs/lz4, /root/lz4/lz4 and PATH."
    )


def parse_int_list(value, default_list):
    if value is None:
        return list(default_list)
    s = str(value).strip()
    if not s:
        return list(default_list)
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out if out else list(default_list)


def parse_optional_int_list(value):
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def parse_str_list(value, default_list):
    if value is None:
        return list(default_list)
    s = str(value).strip()
    if not s:
        return list(default_list)
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        out.append(tok)
    return out if out else list(default_list)


def safe_median(vals):
    cleaned = [float(v) for v in vals if v is not None]
    if not cleaned:
        return 0.0
    return float(statistics.median(cleaned))


def safe_mad(vals, center=None):
    cleaned = [float(v) for v in vals if v is not None]
    if not cleaned:
        return 0.0
    c = safe_median(cleaned) if center is None else float(center)
    dev = [abs(v - c) for v in cleaned]
    return float(statistics.median(dev)) if dev else 0.0


def lz4_infer_change_scope(paths):
    if not paths:
        return SCOPE_COMMON

    comp_hit = False
    decomp_hit = False
    common_hit = False

    common_markers = [
        "lz4_gpu/lz4_gpu.cl",
        "lz4_gpu/lz4_gpu_core",
        "lz4_gpu/lz4_gpu.c",
        "lz4_gpu/lz4_gpu_daemon.c",
        "lz4_gpu/lz4_gpu_utils",
        "lz4_gpu/protocol",
    ]
    comp_markers = ["compress", "encoder", "lz4hc", "hash", "accel", "ratio"]
    decomp_markers = ["decompress", "decomp", "decoder", "decode", "unlz4", "lz4cat"]

    for p in paths:
        pl = p.lower()
        if any(m in pl for m in common_markers):
            common_hit = True
            continue
        de_hit = any(m in pl for m in decomp_markers)
        cp_hit = any(m in pl for m in comp_markers)
        decomp_hit = decomp_hit or de_hit
        comp_hit = comp_hit or cp_hit

    if common_hit or (comp_hit and decomp_hit):
        return SCOPE_COMMON
    if comp_hit:
        return SCOPE_COMPRESS
    if decomp_hit:
        return SCOPE_DECOMPRESS
    return SCOPE_COMMON


def detect_scope_from_git(repo_root, base_ref):
    changed = []
    cmds = [
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        ["git", "diff", "--name-only", "HEAD"],
    ]
    for cmd in cmds:
        try:
            res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                changed = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
                break
        except Exception:
            continue
    return lz4_infer_change_scope(changed), changed


def aggregate_runs(run_stats):
    if not run_stats:
        return {}

    numeric_keys = [
        "ratio",
        "comp_mbs",
        "dec_mbs",
        "comp_time_s",
        "dec_time_s",
        "comp_kernel_reported_mbs",
        "dec_kernel_reported_mbs",
        "cpu_freq_start_mhz",
        "cpu_freq_end_mhz",
        "gpu_freq_start_mhz",
        "gpu_freq_end_mhz",
        "cpu_energy_j",
        "gpu_energy_j",
    ]

    out = {}
    for k in numeric_keys:
        vals = [float(s.get(k, 0.0) or 0.0) for s in run_stats]
        med = safe_median(vals)
        out[k] = med
        out[f"mad_{k}"] = safe_mad(vals, center=med)

    out["throughput_semantics"] = run_stats[0].get("throughput_semantics", "op_time_bench")
    out["energy_source"] = run_stats[0].get("energy_source", "none")
    out["roundtrip_verified"] = all(bool(s.get("roundtrip_verified", False)) for s in run_stats)
    out["repeat_count"] = len(run_stats)
    out["repeat_failures"] = sum(0 if bool(s.get("roundtrip_verified", False)) else 1 for s in run_stats)
    return out


def fmtf(v, digits):
    return f"{float(v):.{digits}f}"


def resolve_single_sample(single_file, samples_root, discovered_samples):
    if not single_file:
        return discovered_samples

    cand = Path(single_file)
    if cand.is_file():
        return [cand]

    root_cand = Path(samples_root) / single_file
    if root_cand.is_file():
        return [root_cand]

    by_name = [p for p in discovered_samples if p.name == single_file]
    if len(by_name) == 1:
        return by_name
    if len(by_name) > 1:
        raise SystemExit(f"--single-file matched multiple files named '{single_file}', please provide full path")

    raise SystemExit(f"--single-file not found: {single_file}")


def build_freq_points(cpu_points, gpu_points, cpu_single, gpu_single):
    cpu_targets = cpu_points if cpu_points else [cpu_single]
    gpu_targets = gpu_points if gpu_points else [gpu_single]
    freq_points = []
    seen = set()
    for c in cpu_targets:
        for g in gpu_targets:
            key = (c, g)
            if key in seen:
                continue
            seen.add(key)
            freq_points.append(key)
    return freq_points

def parse_size_to_bytes(s):
    s = str(s).strip()
    if not s:
        return 0
    unit = s[-1].upper()
    if unit == 'K':
        return int(float(s[:-1]) * 1024)
    if unit == 'M':
        return int(float(s[:-1]) * 1024 * 1024)
    if unit == 'G':
        return int(float(s[:-1]) * 1024 * 1024 * 1024)
    return int(s)


def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def files_match(a, b):
    if not a or not b:
        return False
    if not os.path.exists(a) or not os.path.exists(b):
        return False
    return compute_sha256(a) == compute_sha256(b)


def file_matches_hash(path, expected_hash):
    if not path or not expected_hash:
        return False
    if not os.path.exists(path):
        return False
    return compute_sha256(path) == expected_hash


def run_control_action(control_script_path, action):
    if not os.path.exists(control_script_path):
        return "missing_script"

    cmd = [control_script_path, action]
    if os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return "ok"
        msg = ((res.stderr or "") + "\n" + (res.stdout or "")).strip().splitlines()
        short = msg[0][:80] if msg else ""
        return f"failed:{res.returncode}:{short}"
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def _read_pid(pid_path):
    try:
        with open(pid_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def _pid_alive(pid):
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _socket_accepting(socket_path):
    if not socket_path or not os.path.exists(socket_path):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            s.connect(socket_path)
        return True
    except OSError:
        return False


def is_daemon_running(pid_path, socket_path):
    pid = _read_pid(pid_path)
    pid_alive = _pid_alive(pid)
    sock_ready = _socket_accepting(socket_path)
    if pid_alive and sock_ready:
        return True
    if (not pid_alive) and sock_ready:
        return True
    return False


def start_daemon(bin_path, pid_path, socket_path, timeout_s=8.0):
    if is_daemon_running(pid_path, socket_path):
        return None, "already_running"

    proc = subprocess.Popen([bin_path, "--daemon"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if is_daemon_running(pid_path, socket_path):
            return proc, "started"
        if proc.poll() is not None:
            break
        time.sleep(0.1)

    code = proc.poll()
    raise RuntimeError(f"failed to start daemon (exit={code})")


def stop_daemon(bin_path):
    try:
        res = subprocess.run([bin_path, "--stop-daemon"], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return "ok"
        msg = ((res.stderr or "") + "\n" + (res.stdout or "")).strip().splitlines()
        short = msg[0][:80] if msg else ""
        return f"failed:{res.returncode}:{short}"
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def parse_gpu_output(output):
    stats = {}
    try:
        ratio_match = re.search(r"\((\d+\.\d+)% ratio\)", output)
        if ratio_match:
            stats['ratio'] = float(ratio_match.group(1))

        kernel_tp_match = re.search(r"Kernel Throughput\s*:\s*([0-9]+\.?[0-9]*)\s*MB/s", output)
        if kernel_tp_match:
            stats['kernel_tp'] = float(kernel_tp_match.group(1))

        incl_tp_match = re.search(r"Inclusive Throughput\s*:\s*([0-9]+\.?[0-9]*)\s*MB/s", output)
        if incl_tp_match:
            stats['inclusive_tp'] = float(incl_tp_match.group(1))

    except Exception:
        pass
    return stats


def parse_lz4_cpu_bench_output(output, input_size):
    stats = {
        'ratio': 0.0,
        'comp_mbs': 0.0,
        'dec_mbs': 0.0,
        'comp_time_s': 0.0,
        'dec_time_s': 0.0,
        'parsed': False,
    }

    # Example quiet line:
    # -1       334836 (3.927) 1254.63 MB/s 7746.1 MB/s  _structured_1.txt
    line_re = re.compile(
        r"^\s*-\d+\s+(\d+)\s+\(([0-9]+(?:\.[0-9]+)?)\)\s+([0-9]+(?:\.[0-9]+)?)\s+MB/s\s+([0-9]+(?:\.[0-9]+)?)\s+MB/s",
        re.MULTILINE,
    )
    matches = list(line_re.finditer(output or ""))
    if not matches:
        return stats

    m = matches[-1]
    c_size = int(m.group(1))
    comp_tp = float(m.group(3))
    dec_tp = float(m.group(4))
    ratio_pct = (100.0 * c_size / input_size) if input_size > 0 else 0.0
    comp_time_s = (input_size / (comp_tp * 1024.0 * 1024.0)) if comp_tp > 0 and input_size > 0 else 0.0
    dec_time_s = (input_size / (dec_tp * 1024.0 * 1024.0)) if dec_tp > 0 and input_size > 0 else 0.0

    stats.update({
        'ratio': ratio_pct,
        'comp_mbs': comp_tp,
        'dec_mbs': dec_tp,
        'comp_time_s': comp_time_s,
        'dec_time_s': dec_time_s,
        'parsed': True,
    })
    return stats


def run_lz4_cpu(file_path, bs, threads, orig_hash, telemetry=None):
    print(f"Bench_CPU: {file_path.name} BS={bs} T={threads}")
    stats = {
        'ratio': 0,
        'comp_mbs': 0,
        'dec_mbs': 0,
        'comp_time_s': 0,
        'dec_time_s': 0,
        'comp_kernel_reported_mbs': 0,
        'dec_kernel_reported_mbs': 0,
        'throughput_semantics': 'op_time_bench',
        'roundtrip_verified': False,
        'cpu_freq_start_mhz': 0,
        'cpu_freq_end_mhz': 0,
        'gpu_freq_start_mhz': 0,
        'gpu_freq_end_mhz': 0,
        'cpu_energy_j': 0,
        'gpu_energy_j': 0,
        'energy_source': 'none',
    }
    snap_start = telemetry.snapshot() if telemetry else None
    try:
        in_sz = file_path.stat().st_size
        if in_sz <= 0:
            return stats

        cmd = [LZ4_BIN, "-q", f"-B{str(bs).upper()}", f"-T{threads}", "-b1", str(file_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = (res.stdout or "") + (res.stderr or "")
        parsed = parse_lz4_cpu_bench_output(output, in_sz)

        if parsed.get('parsed'):
            stats['ratio'] = parsed.get('ratio', 0)
            stats['comp_mbs'] = parsed.get('comp_mbs', 0)
            stats['dec_mbs'] = parsed.get('dec_mbs', 0)
            stats['comp_time_s'] = parsed.get('comp_time_s', 0)
            stats['dec_time_s'] = parsed.get('dec_time_s', 0)

        verified = (res.returncode == 0) and parsed.get('parsed', False)
        stats['roundtrip_verified'] = bool(verified)
        if not verified:
            print(f"  [CPU] bench parse failed for {file_path} BS={bs} T={threads}", flush=True)
    except Exception as e:
        print(f"CPU error: {e}")

    snap_end = telemetry.snapshot() if telemetry else None
    if telemetry and snap_start and snap_end:
        stats.update(telemetry.diff(snap_start, snap_end))
        stats['energy_source'] = telemetry.describe_sources()

    return stats


def run_lz4_gpu(file_path, bs, hl, lsz, accel, orig_hash, telemetry=None, use_daemon=False):
    print(f"Bench_GPU: {file_path.name} BS={bs} HL={hl} LSZ={lsz} A={accel}")
    bs_arg = str(bs).lower()
    stats = {
        'ratio': 0,
        'comp_mbs': 0,
        'dec_mbs': 0,
        'comp_time_s': 0,
        'dec_time_s': 0,
        'comp_kernel_reported_mbs': 0,
        'dec_kernel_reported_mbs': 0,
        'throughput_semantics': 'op_time_bench',
        'roundtrip_verified': False,
        'cpu_freq_start_mhz': 0,
        'cpu_freq_end_mhz': 0,
        'gpu_freq_start_mhz': 0,
        'gpu_freq_end_mhz': 0,
        'cpu_energy_j': 0,
        'gpu_energy_j': 0,
        'energy_source': 'none',
    }
    tmp_lz4 = None
    tmp_dec_path = None
    gpu_cmd_prefix = [LZ4_GPU_BIN, "--use-daemon"] if use_daemon else [LZ4_GPU_BIN]
    snap_start = telemetry.snapshot() if telemetry else None
    try:
        with tempfile.NamedTemporaryFile(prefix="bench_lz4_tmp_", suffix=".lz4", delete=False) as tf:
            tmp_lz4 = tf.name
        res_c = subprocess.run(
            gpu_cmd_prefix + ["-v", "-b", bs_arg, "-H", str(hl), "-l", str(lsz), "-a", str(accel), str(file_path), "-o", tmp_lz4],
            capture_output=True, text=True, check=False
        )
        stats_c = parse_gpu_output((res_c.stdout or "") + (res_c.stderr or ""))
        stats['comp_kernel_reported_mbs'] = stats_c.get('kernel_tp', 0)
        stats['comp_mbs'] = stats_c.get('inclusive_tp', 0) or stats_c.get('kernel_tp', 0)

        in_sz = file_path.stat().st_size
        out_sz = os.path.getsize(tmp_lz4) if tmp_lz4 and os.path.exists(tmp_lz4) else 0
        if in_sz > 0 and out_sz > 0:
            stats['ratio'] = (out_sz / in_sz) * 100.0
        if in_sz > 0 and stats['comp_mbs'] > 0:
            stats['comp_time_s'] = in_sz / (stats['comp_mbs'] * 1024.0 * 1024.0)

        tmp_dec = tempfile.NamedTemporaryFile(prefix="bench_lz4_gpu_dec_", suffix="", delete=False)
        tmp_dec.close()
        tmp_dec_path = tmp_dec.name
        cmd_d = gpu_cmd_prefix + ["-v", "-d", tmp_lz4, "-o", tmp_dec.name, "-l", str(lsz), "-H", str(hl)]
        res_d = subprocess.run(cmd_d, capture_output=True, text=True, check=False)
        stats_d = parse_gpu_output((res_d.stdout or "") + (res_d.stderr or ""))
        stats['dec_kernel_reported_mbs'] = stats_d.get('kernel_tp', 0)
        stats['dec_mbs'] = stats_d.get('inclusive_tp', 0) or stats_d.get('kernel_tp', 0)
        if in_sz > 0 and stats['dec_mbs'] > 0:
            stats['dec_time_s'] = in_sz / (stats['dec_mbs'] * 1024.0 * 1024.0)

        verified = file_matches_hash(tmp_dec.name, orig_hash)
        stats['roundtrip_verified'] = verified
        if not verified:
            print(f"  [GPU] Roundtrip mismatch for {file_path} (BS={bs} HL={hl} LSZ={lsz})", flush=True)
    except Exception as exc:
        print(f"GPU error: {exc}")
    finally:
        safe_remove(tmp_dec_path)
        safe_remove(tmp_lz4)

    snap_end = telemetry.snapshot() if telemetry else None
    if telemetry and snap_start and snap_end:
        stats.update(telemetry.diff(snap_start, snap_end))
        stats['energy_source'] = telemetry.describe_sources()

    return stats

def main():
    global LZ4_BIN
    parser = argparse.ArgumentParser(description='Bench LZ4 CPU/GPU sweep (supports --limit for quick runs)')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of samples (0 = all)')
    parser.add_argument('--samples', default=SAMPLES_DIR, help='Samples directory (default: /root/samples)')
    parser.add_argument('--cpu-only', action='store_true', help='Run CPU sweep only (skip GPU)')
    parser.add_argument('--gpu-only', action='store_true', help='Run GPU sweep only (skip CPU)')
    parser.add_argument('--cpu-threads', default='1,2', help='CPU thread list, comma-separated (default: 1,2)')
    parser.add_argument('--cpu-block-sizes', default=','.join(CPU_BLOCK_SIZES), help='CPU block sizes, comma-separated')
    parser.add_argument('--gpu-block-sizes', default=','.join(GPU_BLOCK_SIZES), help='GPU block sizes, comma-separated')
    parser.add_argument('--hash-logs', default='15,16', help='GPU hash logs, comma-separated')
    parser.add_argument('--local-sizes', default='1', help='GPU local sizes, comma-separated')
    parser.add_argument('--gpu-accels', default='1,2,4', help='GPU acceleration factors, comma-separated')
    parser.add_argument('--no-gpu-daemon', action='store_true', help='Run GPU benchmark without daemon mode')
    parser.add_argument('--cpu-freq-percent', type=int, default=None, help='Try setting CPU freq percent before sweep (0-100)')
    parser.add_argument('--gpu-freq-percent', type=int, default=None, help='Try setting GPU freq percent before sweep (0-100)')
    parser.add_argument('--cpu-freq-points', default='', help='CPU freq points for sweep, comma-separated (overrides single percent when set)')
    parser.add_argument('--gpu-freq-points', default='', help='GPU freq points for sweep, comma-separated (overrides single percent when set)')
    parser.add_argument('--single-file', default='', help='Only benchmark one file (path or basename under samples dir)')
    parser.add_argument('--no-telemetry', action='store_true', help='Disable freq/energy telemetry collection')
    parser.add_argument('--repeats', type=int, default=3, help='Repeat each benchmark case N times and aggregate with median/MAD (default: 3)')
    parser.add_argument('--save-repeats', action='store_true', help='Write per-repeat raw rows in addition to aggregated row')
    parser.add_argument('--change-scope', choices=['auto', SCOPE_COMPRESS, SCOPE_DECOMPRESS, SCOPE_COMMON], default='auto', help='Metric scope for comparison: auto|compress|decompress|common')
    parser.add_argument('--base-ref', default='dev', help='Git base ref used when --change-scope=auto (default: dev)')
    parser.add_argument('--repo-root', default='/root/lz4', help='Repo root used for git diff scope detection')
    args = parser.parse_args()

    if args.cpu_only and args.gpu_only:
        raise SystemExit('Cannot use --cpu-only and --gpu-only together')

    repeats = max(1, int(args.repeats))

    if not args.gpu_only:
        LZ4_BIN = resolve_lz4_cpu_binary()
        print(f"[CPU-BIN] using {LZ4_BIN}")

    cpu_threads = parse_int_list(args.cpu_threads, CPU_THREADS)
    cpu_block_sizes = parse_str_list(args.cpu_block_sizes, CPU_BLOCK_SIZES)
    gpu_block_sizes = parse_str_list(args.gpu_block_sizes, GPU_BLOCK_SIZES)
    hash_logs = parse_int_list(args.hash_logs, HASH_LOGS)
    local_sizes = parse_int_list(args.local_sizes, LOCAL_SIZES)
    gpu_accels = parse_int_list(args.gpu_accels, GPU_ACCELS)
    use_gpu_daemon = (not args.no_gpu_daemon) and (not args.cpu_only)
    cpu_freq_points = parse_optional_int_list(args.cpu_freq_points)
    gpu_freq_points = parse_optional_int_list(args.gpu_freq_points)
    freq_points = build_freq_points(cpu_freq_points, gpu_freq_points, args.cpu_freq_percent, args.gpu_freq_percent)

    telemetry = None if args.no_telemetry else TelemetryProbe()
    if telemetry is not None:
        print(f"Telemetry sources: {telemetry.describe_sources()}")

    if args.change_scope == 'auto':
        metric_scope, changed_files = detect_scope_from_git(args.repo_root, args.base_ref)
        scope_source = f"auto:{args.base_ref}"
    else:
        metric_scope, changed_files = args.change_scope, []
        scope_source = "manual"
    relevant_metrics = SCOPE_METRICS.get(metric_scope, SCOPE_METRICS[SCOPE_COMMON])
    print(f"[Scope] source={scope_source} scope={metric_scope} relevant_metrics={','.join(relevant_metrics)}")
    if changed_files:
        print(f"[Scope] changed_files_detected={len(changed_files)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    samples = sorted([p for p in Path(args.samples).glob("*") if p.is_file()])

    if args.single_file:
        samples = resolve_single_sample(args.single_file, args.samples, samples)

    if not samples:
        print(f"No samples found in {args.samples}")
        return

    if args.limit and args.limit > 0 and not args.single_file:
        samples = samples[:args.limit]

    hash_cache = {}

    daemon_proc = None
    daemon_state = "disabled"

    try:
        if use_gpu_daemon:
            daemon_proc, daemon_state = start_daemon(LZ4_GPU_BIN, LZ4_DAEMON_PID_PATH, LZ4_DAEMON_SOCKET_PATH)
            print(f"[Daemon] LZ4 GPU daemon state: {daemon_state}")

        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "File", "FreqPoint", "CPUFreqTargetPct", "GPUFreqTargetPct",
                "Engine", "Threads_LSZ", "BlockSize", "HashLog", "Acceleration", "Ratio%",
                "CompMBs", "DecMBs", "CompTime_s", "DecTime_s",
                "CompKernelReported_MBs", "DecKernelReported_MBs",
                "MAD_Ratio", "MAD_CompKernel", "MAD_DecKernel",
                "CPUFreqStart_MHz", "CPUFreqEnd_MHz", "GPUFreqStart_MHz", "GPUFreqEnd_MHz",
                "CPUEnergy_J", "GPUEnergy_J", "EnergySource",
                "CPUFreqApply", "GPUFreqApply",
                "ThroughputSemantics", "Roundtrip_OK",
                "RepeatIndex", "RepeatCount", "AggMethod",
                "ChangeScope", "ScopeSource", "RelevantMetricKeys"
            ])

            for point_idx, (cpu_freq_target, gpu_freq_target) in enumerate(freq_points, start=1):
                cpu_freq_apply = apply_freq_percent('/root/lz4/tools/cpu_control.sh', cpu_freq_target)
                gpu_freq_apply = apply_freq_percent('/root/lz4/tools/gpu_control.sh', gpu_freq_target)
                print(f"[FreqPoint {point_idx}] CPU={cpu_freq_target} apply={cpu_freq_apply}; GPU={gpu_freq_target} apply={gpu_freq_apply}")

                for sample in samples:
                    sample_key = str(sample)
                    if sample_key not in hash_cache:
                        hash_cache[sample_key] = compute_sha256(sample_key)
                    orig_hash = hash_cache[sample_key]

                    # CPU Sweep: block size isn't always supported by CPU tool; convert and use benchmark flag
                    if not args.gpu_only:
                        for t in cpu_threads:
                            for bs in cpu_block_sizes:
                                runs = [run_lz4_cpu(sample, bs, t, orig_hash, telemetry=telemetry) for _ in range(repeats)]
                                if args.save_repeats:
                                    for i, raw in enumerate(runs, start=1):
                                        writer.writerow([
                                            sample.name,
                                            point_idx,
                                            "" if cpu_freq_target is None else cpu_freq_target,
                                            "" if gpu_freq_target is None else gpu_freq_target,
                                            "CPU", t, bs, "N/A", "N/A", fmtf(raw['ratio'], 2),
                                            fmtf(raw['comp_mbs'], 2), fmtf(raw['dec_mbs'], 2),
                                            fmtf(raw['comp_time_s'], 6), fmtf(raw['dec_time_s'], 6),
                                            fmtf(raw['comp_kernel_reported_mbs'], 2), fmtf(raw['dec_kernel_reported_mbs'], 2),
                                            "", "", "",
                                            fmtf(raw['cpu_freq_start_mhz'], 2),
                                            fmtf(raw['cpu_freq_end_mhz'], 2),
                                            fmtf(raw['gpu_freq_start_mhz'], 2),
                                            fmtf(raw['gpu_freq_end_mhz'], 2),
                                            fmtf(raw['cpu_energy_j'], 6),
                                            fmtf(raw['gpu_energy_j'], 6),
                                            raw.get('energy_source', 'none'),
                                            cpu_freq_apply,
                                            gpu_freq_apply,
                                            raw.get('throughput_semantics', 'cpu_kernel_bench'),
                                            "yes" if raw.get('roundtrip_verified') else "no",
                                            i,
                                            repeats,
                                            "raw",
                                            metric_scope,
                                            scope_source,
                                            ",".join(relevant_metrics),
                                        ])

                                cpu_stats = aggregate_runs(runs)
                                writer.writerow([
                                    sample.name,
                                    point_idx,
                                    "" if cpu_freq_target is None else cpu_freq_target,
                                    "" if gpu_freq_target is None else gpu_freq_target,
                                    "CPU", t, bs, "N/A", "N/A", fmtf(cpu_stats['ratio'], 2),
                                    fmtf(cpu_stats['comp_mbs'], 2), fmtf(cpu_stats['dec_mbs'], 2),
                                    fmtf(cpu_stats['comp_time_s'], 6), fmtf(cpu_stats['dec_time_s'], 6),
                                    fmtf(cpu_stats['comp_kernel_reported_mbs'], 2), fmtf(cpu_stats['dec_kernel_reported_mbs'], 2),
                                    fmtf(cpu_stats['mad_ratio'], 3), fmtf(cpu_stats['mad_comp_kernel_reported_mbs'], 3), fmtf(cpu_stats['mad_dec_kernel_reported_mbs'], 3),
                                    fmtf(cpu_stats['cpu_freq_start_mhz'], 2),
                                    fmtf(cpu_stats['cpu_freq_end_mhz'], 2),
                                    fmtf(cpu_stats['gpu_freq_start_mhz'], 2),
                                    fmtf(cpu_stats['gpu_freq_end_mhz'], 2),
                                    fmtf(cpu_stats['cpu_energy_j'], 6),
                                    fmtf(cpu_stats['gpu_energy_j'], 6),
                                    cpu_stats.get('energy_source', 'none'),
                                    cpu_freq_apply,
                                    gpu_freq_apply,
                                    cpu_stats.get('throughput_semantics', 'cpu_kernel_bench'),
                                    "yes" if cpu_stats.get('roundtrip_verified') else "no",
                                    "median",
                                    repeats,
                                    "median_mad",
                                    metric_scope,
                                    scope_source,
                                    ",".join(relevant_metrics),
                                ])
                                f.flush()

                    # GPU Sweep
                    if not args.cpu_only:
                        for bs in gpu_block_sizes:
                            for hl in hash_logs:
                                for lsz in local_sizes:
                                    for accel in gpu_accels:
                                        runs = [run_lz4_gpu(sample, bs, hl, lsz, accel, orig_hash, telemetry=telemetry, use_daemon=use_gpu_daemon) for _ in range(repeats)]
                                        if args.save_repeats:
                                            for i, raw in enumerate(runs, start=1):
                                                writer.writerow([
                                                    sample.name,
                                                    point_idx,
                                                    "" if cpu_freq_target is None else cpu_freq_target,
                                                    "" if gpu_freq_target is None else gpu_freq_target,
                                                    "GPU", lsz, bs, hl, accel, fmtf(raw['ratio'], 2),
                                                    fmtf(raw['comp_mbs'], 2), fmtf(raw['dec_mbs'], 2),
                                                    fmtf(raw['comp_time_s'], 6), fmtf(raw['dec_time_s'], 6),
                                                    fmtf(raw['comp_kernel_reported_mbs'], 2), fmtf(raw['dec_kernel_reported_mbs'], 2),
                                                    "", "", "",
                                                    fmtf(raw['cpu_freq_start_mhz'], 2),
                                                    fmtf(raw['cpu_freq_end_mhz'], 2),
                                                    fmtf(raw['gpu_freq_start_mhz'], 2),
                                                    fmtf(raw['gpu_freq_end_mhz'], 2),
                                                    fmtf(raw['cpu_energy_j'], 6),
                                                    fmtf(raw['gpu_energy_j'], 6),
                                                    raw.get('energy_source', 'none'),
                                                    cpu_freq_apply,
                                                    gpu_freq_apply,
                                                    raw.get('throughput_semantics', 'gpu_kernel_bench'),
                                                    "yes" if raw.get('roundtrip_verified') else "no",
                                                    i,
                                                    repeats,
                                                    "raw",
                                                    metric_scope,
                                                    scope_source,
                                                    ",".join(relevant_metrics),
                                                ])

                                        gpu_stats = aggregate_runs(runs)
                                        writer.writerow([
                                            sample.name,
                                            point_idx,
                                            "" if cpu_freq_target is None else cpu_freq_target,
                                            "" if gpu_freq_target is None else gpu_freq_target,
                                            "GPU", lsz, bs, hl, accel, fmtf(gpu_stats['ratio'], 2),
                                            fmtf(gpu_stats['comp_mbs'], 2), fmtf(gpu_stats['dec_mbs'], 2),
                                            fmtf(gpu_stats['comp_time_s'], 6), fmtf(gpu_stats['dec_time_s'], 6),
                                            fmtf(gpu_stats['comp_kernel_reported_mbs'], 2), fmtf(gpu_stats['dec_kernel_reported_mbs'], 2),
                                            fmtf(gpu_stats['mad_ratio'], 3), fmtf(gpu_stats['mad_comp_kernel_reported_mbs'], 3), fmtf(gpu_stats['mad_dec_kernel_reported_mbs'], 3),
                                            fmtf(gpu_stats['cpu_freq_start_mhz'], 2),
                                            fmtf(gpu_stats['cpu_freq_end_mhz'], 2),
                                            fmtf(gpu_stats['gpu_freq_start_mhz'], 2),
                                            fmtf(gpu_stats['gpu_freq_end_mhz'], 2),
                                            fmtf(gpu_stats['cpu_energy_j'], 6),
                                            fmtf(gpu_stats['gpu_energy_j'], 6),
                                            gpu_stats.get('energy_source', 'none'),
                                            cpu_freq_apply,
                                            gpu_freq_apply,
                                            gpu_stats.get('throughput_semantics', 'gpu_kernel_bench'),
                                            "yes" if gpu_stats.get('roundtrip_verified') else "no",
                                            "median",
                                            repeats,
                                            "median_mad",
                                            metric_scope,
                                            scope_source,
                                            ",".join(relevant_metrics),
                                        ])
                                        f.flush()
    finally:
        if use_gpu_daemon:
            stop_state = stop_daemon(LZ4_GPU_BIN)
            print(f"[Daemon] LZ4 GPU daemon stop: {stop_state}")
            if daemon_proc is not None and daemon_proc.poll() is None:
                try:
                    daemon_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    daemon_proc.kill()

        cpu_reset = run_control_action('/root/lz4/tools/cpu_control.sh', 'reset')
        gpu_reset = run_control_action('/root/lz4/tools/gpu_control.sh', 'reset')
        print(f"[Cleanup] CPU reset={cpu_reset}; GPU reset={gpu_reset}")

if __name__ == "__main__":
    main()
