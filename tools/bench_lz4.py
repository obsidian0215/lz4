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
LZ4_DAEMON_SOCKET_PATH = "/tmp/lz4_gpu_daemon.sock"
LZ4_DAEMON_PID_PATH = "/tmp/lz4_gpu_daemon.pid"

# Configuration Space
CPU_BLOCK_SIZES = ["64K", "256K"]
GPU_BLOCK_SIZES = ["16K", "32K", "64K"]
CPU_THREADS = [1, 2, 3]
HASH_LOGS = [14, 15]
LOCAL_SIZES = [1]
GPU_ACCELS = [1, 2, 3]



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


def safe_mean(vals):
    cleaned = [float(v) for v in vals if v is not None]
    if not cleaned:
        return 0.0
    return float(sum(cleaned) / len(cleaned))


def percentile(vals, p):
    cleaned = sorted(float(v) for v in vals if v is not None)
    if not cleaned:
        return 0.0
    if p <= 0:
        return cleaned[0]
    if p >= 100:
        return cleaned[-1]
    pos = (len(cleaned) - 1) * (p / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(cleaned) - 1)
    if lo == hi:
        return cleaned[lo]
    frac = pos - lo
    return cleaned[lo] * (1.0 - frac) + cleaned[hi] * frac


def summarize_dist(vals):
    cleaned = [float(v) for v in vals if v is not None]
    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(cleaned),
        "mean": safe_mean(cleaned),
        "median": safe_median(cleaned),
        "p10": percentile(cleaned, 10),
        "p90": percentile(cleaned, 90),
        "min": min(cleaned),
        "max": max(cleaned),
    }


def aggregate_runs(run_stats):
    if not run_stats:
        return {}

    numeric_keys = [
        "ratio",
        "comp_mbs",
        "dec_mbs",
        "comp_time_s",
        "dec_time_s",
        "cpu_freq_avg_mhz",
        "gpu_freq_avg_mhz",
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


def aggregate_runs_mean(run_stats):
    if not run_stats:
        return {}

    numeric_keys = [
        "ratio",
        "comp_mbs",
        "dec_mbs",
        "comp_time_s",
        "dec_time_s",
        "cpu_freq_avg_mhz",
        "gpu_freq_avg_mhz",
        "cpu_energy_j",
        "gpu_energy_j",
    ]

    out = {}
    for k in numeric_keys:
        vals = [float(s.get(k, 0.0) or 0.0) for s in run_stats]
        out[k] = safe_mean(vals)

    out["throughput_semantics"] = run_stats[0].get("throughput_semantics", "op_time_bench")
    out["energy_source"] = run_stats[0].get("energy_source", "none")
    out["roundtrip_verified"] = all(bool(s.get("roundtrip_verified", False)) for s in run_stats)
    out["repeat_count"] = len(run_stats)
    out["repeat_failures"] = sum(0 if bool(s.get("roundtrip_verified", False)) else 1 for s in run_stats)
    return out


def kernel_comp(stats, engine):
    if engine == "CPU":
        v = float(stats.get("comp_mbs", 0.0) or 0.0)
        return v if v > 0 else None
    v = float(stats.get("comp_mbs", 0.0) or 0.0)
    return v if v > 0 else None


def kernel_dec(stats, engine):
    if engine == "CPU":
        v = float(stats.get("dec_mbs", 0.0) or 0.0)
        return v if v > 0 else None
    v = float(stats.get("dec_mbs", 0.0) or 0.0)
    return v if v > 0 else None


def fmt_metric(v, digits):
    if v is None:
        return "N/A"
    return f"{float(v):.{digits}f}"


def fmt_csv_metric(v, digits):
    if v is None:
        return ""
    return f"{float(v):.{digits}f}"


def fmt_mbs_or_na(v, digits):
    if v is None:
        return "N/A"
    return f"{float(v):.{digits}f}MB/s"


def emit_case_average(file_name, engine, config_label, avg_stats):
    comp_kernel = kernel_comp(avg_stats, engine)
    dec_kernel = kernel_dec(avg_stats, engine)
    print(
        "[CaseAvg] "
        f"file={file_name} engine={engine} cfg={config_label} "
        f"ratio_avg={fmtf(avg_stats.get('ratio', 0.0), 2)}% "
        f"comp_mbs_avg={fmt_mbs_or_na(comp_kernel, 2)} "
        f"dec_mbs_avg={fmt_mbs_or_na(dec_kernel, 2)}"
    )


def build_summary_record(engine, config_label, avg_stats):
    return {
        "engine": engine,
        "config": config_label,
        "ratio": float(avg_stats.get("ratio", 0.0) or 0.0),
        "comp_kernel": kernel_comp(avg_stats, engine),
        "dec_kernel": kernel_dec(avg_stats, engine),
    }


def print_and_save_config_summary(summary_records, out_csv):
    if not summary_records:
        print("[ConfigSummary] no summary records")
        return

    grouped = {}
    for rec in summary_records:
        key = (rec["engine"], rec["config"])
        grouped.setdefault(key, []).append(rec)

    print("\n===== Per-Config Summary Stats (mean/median/p10/p90) =====")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Engine", "Config", "Samples",
            "Ratio_mean", "Ratio_median", "Ratio_p10", "Ratio_p90", "Ratio_min", "Ratio_max",
            "CompMBs_mean", "CompMBs_median", "CompMBs_p10", "CompMBs_p90", "CompMBs_min", "CompMBs_max",
            "DecMBs_mean", "DecMBs_median", "DecMBs_p10", "DecMBs_p90", "DecMBs_min", "DecMBs_max",
        ])

        for (engine, config), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
            ratio_s = summarize_dist([r["ratio"] for r in rows])
            comp_kernel_s = summarize_dist([r["comp_kernel"] for r in rows])
            dec_kernel_s = summarize_dist([r["dec_kernel"] for r in rows])

            print(
                f"[ConfigSummary] {engine} {config} n={ratio_s['count']} | "
                f"ratio(mean/med/p10)={fmt_metric(ratio_s['mean'],2)}/{fmt_metric(ratio_s['median'],2)}/{fmt_metric(ratio_s['p10'],2)}% | "
                f"Cmbs={fmt_metric(comp_kernel_s['mean'],2)}/{fmt_metric(comp_kernel_s['median'],2)}/{fmt_metric(comp_kernel_s['p10'],2)} | "
                f"Dmbs={fmt_metric(dec_kernel_s['mean'],2)}/{fmt_metric(dec_kernel_s['median'],2)}/{fmt_metric(dec_kernel_s['p10'],2)}"
            )

            writer.writerow([
                engine, config, ratio_s["count"],
                fmt_csv_metric(ratio_s["mean"], 4), fmt_csv_metric(ratio_s["median"], 4), fmt_csv_metric(ratio_s["p10"], 4), fmt_csv_metric(ratio_s["p90"], 4), fmt_csv_metric(ratio_s["min"], 4), fmt_csv_metric(ratio_s["max"], 4),
                fmt_csv_metric(comp_kernel_s["mean"], 4), fmt_csv_metric(comp_kernel_s["median"], 4), fmt_csv_metric(comp_kernel_s["p10"], 4), fmt_csv_metric(comp_kernel_s["p90"], 4), fmt_csv_metric(comp_kernel_s["min"], 4), fmt_csv_metric(comp_kernel_s["max"], 4),
                fmt_csv_metric(dec_kernel_s["mean"], 4), fmt_csv_metric(dec_kernel_s["median"], 4), fmt_csv_metric(dec_kernel_s["p10"], 4), fmt_csv_metric(dec_kernel_s["p90"], 4), fmt_csv_metric(dec_kernel_s["min"], 4), fmt_csv_metric(dec_kernel_s["max"], 4),
            ])

    print(f"[ConfigSummary] saved: {out_csv}")


def fmtf(v, digits):
    return f"{float(v):.{digits}f}"


def prepare_results_paths(results_root, csv_name, summary_name):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_root) / "runs" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, str(run_dir / csv_name), str(run_dir / summary_name)


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


def build_freq_points(shared_points, shared_single):
    targets = shared_points if shared_points else [shared_single]
    if not targets:
        targets = [None]

    freq_points = []
    seen = set()
    for v in targets:
        if v in seen:
            continue
        seen.add(v)
        freq_points.append(v)
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


def build_gpu_subprocess_env():
    env = os.environ.copy()
    common_dbg = os.environ.get("GPU_DEBUG_COUNTERS", "").strip()
    if common_dbg and common_dbg != "0":
        env["LZ4_GPU_DEBUG_COUNTERS"] = "1"
        env["LZO_GPU_DEBUG_COUNTERS"] = "1"
    return env


def run_command_with_telemetry(cmd, telemetry=None, env=None, sample_interval_s=0.05):
    if telemetry is None:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        return res, {}

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    samples = []
    start_snap = telemetry.snapshot()
    samples.append(start_snap)

    while True:
        try:
            out, err = proc.communicate(timeout=sample_interval_s)
            break
        except subprocess.TimeoutExpired:
            samples.append(telemetry.snapshot())

    end_snap = telemetry.snapshot()
    samples.append(end_snap)
    delta = telemetry.diff(start_snap, end_snap)

    cpu_avg = safe_mean([s.get("cpu_freq_mhz") for s in samples])
    gpu_avg = safe_mean([s.get("gpu_freq_mhz") for s in samples])

    tel = {
        "elapsed_s": float(delta.get("elapsed_s", 0.0) or 0.0),
        "cpu_freq_avg_mhz": float(cpu_avg),
        "gpu_freq_avg_mhz": float(gpu_avg),
        "cpu_energy_j": float(delta.get("cpu_energy_j", 0.0) or 0.0),
        "gpu_energy_j": float(delta.get("gpu_energy_j", 0.0) or 0.0),
    }
    completed = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    return completed, tel


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


def parse_stable_bench_output(output):
    comp = re.search(
        r"Bench\s+Compress\s*:\s*kernel_tp=([0-9]+\.?[0-9]*)\s*MB/s"
        r"(?:\s+total_tp=([0-9]+\.?[0-9]*)\s*MB/s)?"
        r"\s+ratio=([0-9]+\.?[0-9]*)%",
        output or "",
        re.IGNORECASE,
    )
    dec = re.search(
        r"Bench\s+Decompress\s*:\s*kernel_tp=([0-9]+\.?[0-9]*)\s*MB/s"
        r"(?:\s+total_tp=([0-9]+\.?[0-9]*)\s*MB/s)?"
        r"\s+verify=(OK|FAIL)",
        output or "",
        re.IGNORECASE,
    )
    if not comp or not dec:
        return None
    return {
        "ratio": float(comp.group(3)),
        "comp_kernel_tp": float(comp.group(1)),
        "comp_total_tp": float(comp.group(2)) if comp.group(2) else 0.0,
        "dec_kernel_tp": float(dec.group(1)),
        "dec_total_tp": float(dec.group(2)) if dec.group(2) else 0.0,
        "verify_ok": dec.group(3).upper() == "OK",
    }


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
        'throughput_semantics': 'op_time_bench',
        'roundtrip_verified': False,
        'cpu_freq_avg_mhz': 0,
        'gpu_freq_avg_mhz': 0,
        'cpu_energy_j': 0,
        'gpu_energy_j': 0,
        'comp_cpu_power_w': 0,
        'comp_gpu_power_w': 0,
        'energy_source': 'none',
    }
    tel_window = {}
    try:
        in_sz = file_path.stat().st_size
        if in_sz <= 0:
            return stats

        # Use native lz4 bench output only (no temp file workflow)
        cmd_bench = [LZ4_BIN, "-q", f"-B{str(bs).upper()}", f"-T{threads}", "-b1", str(file_path)]
        bench_res, tel_window = run_command_with_telemetry(cmd_bench, telemetry=telemetry)
        bench_output = (bench_res.stdout or "") + (bench_res.stderr or "")
        parsed = parse_lz4_cpu_bench_output(bench_output, in_sz)
        if parsed.get('parsed'):
            stats['ratio'] = parsed.get('ratio', 0)
            stats['comp_mbs'] = parsed.get('comp_mbs', 0)
            stats['dec_mbs'] = parsed.get('dec_mbs', 0)
            stats['comp_time_s'] = parsed.get('comp_time_s', 0)
            stats['dec_time_s'] = parsed.get('dec_time_s', 0)
            stats['throughput_semantics'] = 'stable_cpu_bench'
            stats['roundtrip_verified'] = (bench_res.returncode == 0)
        else:
            stats['throughput_semantics'] = 'stable_cpu_bench_parse_failed'
            stats['roundtrip_verified'] = False
            print(f"  [CPU] stable bench parse failed for {file_path} BS={bs} T={threads}", flush=True)
    except Exception as e:
        print(f"CPU error: {e}")

    if telemetry and tel_window:
        stats['cpu_freq_avg_mhz'] = float(tel_window.get('cpu_freq_avg_mhz', 0.0) or 0.0)
        stats['gpu_freq_avg_mhz'] = float(tel_window.get('gpu_freq_avg_mhz', 0.0) or 0.0)

        comp_s = float(stats.get('comp_time_s', 0.0) or 0.0)
        dec_s = float(stats.get('dec_time_s', 0.0) or 0.0)
        kernel_total_s = comp_s + dec_s
        elapsed_s = float(tel_window.get('elapsed_s', 0.0) or 0.0)
        kernel_scale = min(1.0, (kernel_total_s / elapsed_s)) if (elapsed_s > 0 and kernel_total_s > 0) else 0.0

        cpu_kernel_energy = float(tel_window.get('cpu_energy_j', 0.0) or 0.0) * kernel_scale
        gpu_kernel_energy = float(tel_window.get('gpu_energy_j', 0.0) or 0.0) * kernel_scale
        comp_share = (comp_s / kernel_total_s) if kernel_total_s > 0 else 0.0

        stats['cpu_energy_j'] = cpu_kernel_energy * comp_share
        stats['gpu_energy_j'] = gpu_kernel_energy * comp_share
        stats['comp_cpu_power_w'] = (cpu_kernel_energy / kernel_total_s) if kernel_total_s > 0 else 0.0
        stats['comp_gpu_power_w'] = (gpu_kernel_energy / kernel_total_s) if kernel_total_s > 0 else 0.0
        stats['energy_source'] = telemetry.describe_sources()

    return stats


def run_lz4_gpu(file_path, bs, hl, lsz, accel, orig_hash, telemetry=None, bench_seconds=3.0):
    print(f"Bench_GPU: {file_path.name} BS={bs} HL={hl} LSZ={lsz} A={accel}")
    bs_arg = str(bs).lower()
    stats = {
        'ratio': 0,
        'comp_mbs': 0,
        'dec_mbs': 0,
        'comp_total_mbs': 0,
        'dec_total_mbs': 0,
        'comp_time_s': 0,
        'dec_time_s': 0,
        'throughput_semantics': 'op_time_bench',
        'roundtrip_verified': False,
        'cpu_freq_avg_mhz': 0,
        'gpu_freq_avg_mhz': 0,
        'cpu_energy_j': 0,
        'gpu_energy_j': 0,
        'comp_cpu_power_w': 0,
        'comp_gpu_power_w': 0,
        'energy_source': 'none',
    }
    tel_window = {}
    try:
        in_sz = file_path.stat().st_size

        bench_cmd = [
            LZ4_GPU_BIN,
            "--bench", str(bench_seconds),
            "-b", bs_arg,
            "-H", str(hl),
            "-l", str(lsz),
            "-a", str(accel),
            str(file_path),
        ]
        bench_res, tel_window = run_command_with_telemetry(bench_cmd, telemetry=telemetry, env=build_gpu_subprocess_env())
        bench_output = (bench_res.stdout or "") + (bench_res.stderr or "")
        stable = parse_stable_bench_output(bench_output)

        if stable:
            stats['ratio'] = stable['ratio']
            stats['comp_mbs'] = stable['comp_kernel_tp']
            stats['dec_mbs'] = stable['dec_kernel_tp']
            stats['comp_total_mbs'] = stable.get('comp_total_tp', 0.0)
            stats['dec_total_mbs'] = stable.get('dec_total_tp', 0.0)
            if in_sz > 0 and stats['comp_mbs'] > 0:
                stats['comp_time_s'] = in_sz / (stats['comp_mbs'] * 1024.0 * 1024.0)
            if in_sz > 0 and stats['dec_mbs'] > 0:
                stats['dec_time_s'] = in_sz / (stats['dec_mbs'] * 1024.0 * 1024.0)
            stats['throughput_semantics'] = 'stable_kernel_bench'
            stats['roundtrip_verified'] = bool(stable['verify_ok']) and (bench_res.returncode == 0)
        else:
            stats['throughput_semantics'] = 'stable_kernel_bench_parse_failed'
            stats['roundtrip_verified'] = False
            print(f"  [GPU] stable bench parse failed for {file_path} (BS={bs} HL={hl} LSZ={lsz} A={accel})", flush=True)
    except Exception as exc:
        print(f"GPU error: {exc}")

    if telemetry and tel_window:
        stats['cpu_freq_avg_mhz'] = float(tel_window.get('cpu_freq_avg_mhz', 0.0) or 0.0)
        stats['gpu_freq_avg_mhz'] = float(tel_window.get('gpu_freq_avg_mhz', 0.0) or 0.0)

        comp_s = float(stats.get('comp_time_s', 0.0) or 0.0)
        dec_s = float(stats.get('dec_time_s', 0.0) or 0.0)
        kernel_total_s = comp_s + dec_s
        elapsed_s = float(tel_window.get('elapsed_s', 0.0) or 0.0)
        kernel_scale = min(1.0, (kernel_total_s / elapsed_s)) if (elapsed_s > 0 and kernel_total_s > 0) else 0.0

        cpu_kernel_energy = float(tel_window.get('cpu_energy_j', 0.0) or 0.0) * kernel_scale
        gpu_kernel_energy = float(tel_window.get('gpu_energy_j', 0.0) or 0.0) * kernel_scale
        comp_share = (comp_s / kernel_total_s) if kernel_total_s > 0 else 0.0

        stats['cpu_energy_j'] = cpu_kernel_energy * comp_share
        stats['gpu_energy_j'] = gpu_kernel_energy * comp_share
        stats['comp_cpu_power_w'] = (cpu_kernel_energy / kernel_total_s) if kernel_total_s > 0 else 0.0
        stats['comp_gpu_power_w'] = (gpu_kernel_energy / kernel_total_s) if kernel_total_s > 0 else 0.0
        stats['energy_source'] = telemetry.describe_sources()

    return stats

def main():
    global LZ4_BIN
    parser = argparse.ArgumentParser(description='Bench LZ4 CPU/GPU sweep (supports --limit for quick runs)')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of samples (0 = all)')
    parser.add_argument('--samples', default=SAMPLES_DIR, help='Samples directory (default: /root/samples)')
    parser.add_argument('--cpu-only', action='store_true', help='Run CPU sweep only (skip GPU)')
    parser.add_argument('--gpu-only', action='store_true', help='Run GPU sweep only (skip CPU)')
    parser.add_argument('--cpu-threads', default=','.join(str(x) for x in CPU_THREADS), help='CPU thread list, comma-separated (default: 1,2)')
    parser.add_argument('--cpu-block-sizes', default=','.join(CPU_BLOCK_SIZES), help='CPU block sizes, comma-separated')
    parser.add_argument('--gpu-block-sizes', default=','.join(GPU_BLOCK_SIZES), help='GPU block sizes, comma-separated')
    parser.add_argument('--hash-logs', default=','.join(str(x) for x in HASH_LOGS), help='GPU hash logs, comma-separated')
    parser.add_argument('--local-sizes', default=','.join(str(x) for x in LOCAL_SIZES), help='GPU local sizes, comma-separated')
    parser.add_argument('--gpu-accels', default=','.join(str(x) for x in GPU_ACCELS), help='GPU acceleration factors, comma-separated')
    parser.add_argument('--freq-percent', type=int, default=None, help='Set both CPU and GPU to one shared frequency percent (0-100)')
    parser.add_argument('--freq-points', default='', help='Shared CPU/GPU frequency points, comma-separated (e.g. 40,70,100)')
    parser.add_argument('--single-file', default='', help='Only benchmark one file (path or basename under samples dir)')
    parser.add_argument('--no-telemetry', action='store_true', help='Disable freq/energy telemetry collection')
    parser.add_argument('--bench-seconds', type=float, default=3.0, help='Benchmark duration in seconds for timed bench paths (default: 3.0)')
    args = parser.parse_args()

    if args.cpu_only and args.gpu_only:
        raise SystemExit('Cannot use --cpu-only and --gpu-only together')

    if not args.gpu_only:
        LZ4_BIN = resolve_lz4_cpu_binary()
        print(f"[CPU-BIN] using {LZ4_BIN}")

    cpu_threads = parse_int_list(args.cpu_threads, CPU_THREADS)
    cpu_block_sizes = parse_str_list(args.cpu_block_sizes, CPU_BLOCK_SIZES)
    gpu_block_sizes = parse_str_list(args.gpu_block_sizes, GPU_BLOCK_SIZES)
    hash_logs = parse_int_list(args.hash_logs, HASH_LOGS)
    local_sizes = parse_int_list(args.local_sizes, LOCAL_SIZES)
    gpu_accels = parse_int_list(args.gpu_accels, GPU_ACCELS)
    use_gpu_daemon = (not args.cpu_only)
    freq_points = build_freq_points(parse_optional_int_list(args.freq_points), args.freq_percent)

    telemetry = None if args.no_telemetry else TelemetryProbe()
    if telemetry is not None:
        print(f"Telemetry sources: {telemetry.describe_sources()}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_dir, results_csv, results_summary_csv = prepare_results_paths(
        RESULTS_DIR,
        "lz4_param_sweep.csv",
        "lz4_param_sweep_config_summary.csv",
    )
    print(f"[Results] run_dir={run_dir}")
    with open(run_dir / "run_meta.txt", "w", encoding="utf-8") as mf:
        mf.write(f"argv={' '.join(sys.argv)}\n")
        mf.write(f"cwd={os.getcwd()}\n")
        mf.write(f"bench_seconds={args.bench_seconds}\n")
        mf.write(f"samples={args.samples}\n")

    samples = sorted([p for p in Path(args.samples).glob("*") if p.is_file()])

    if args.single_file:
        samples = resolve_single_sample(args.single_file, args.samples, samples)

    if not samples:
        print(f"No samples found in {args.samples}")
        return

    if args.limit and args.limit > 0 and not args.single_file:
        samples = samples[:args.limit]

    hash_cache = {}
    summary_records = []

    daemon_proc = None
    daemon_state = "disabled"

    try:
        if use_gpu_daemon:
            daemon_proc, daemon_state = start_daemon(LZ4_GPU_BIN, LZ4_DAEMON_PID_PATH, LZ4_DAEMON_SOCKET_PATH)
            print(f"[Daemon] LZ4 GPU daemon state: {daemon_state}")

        with open(results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "File", "FreqPoint", "CPUFreqTargetPct", "GPUFreqTargetPct",
                "Engine", "Threads_LSZ", "BlockSize", "HashLog", "Acceleration", "Ratio%",
                "CompMBs", "DecMBs", "CompTotalMBs", "DecTotalMBs",
                "CompTime_s", "DecTime_s",
                "CPUFreqAvgKernel_MHz", "GPUFreqAvgKernel_MHz",
                "CompCPUEnergy_J", "CompGPUEnergy_J", "CompCPUPower_W", "CompGPUPower_W",
                "Roundtrip_OK"
            ])

            for point_idx, freq_target in enumerate(freq_points, start=1):
                cpu_freq_target = freq_target
                gpu_freq_target = freq_target
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
                                cpu_stats = run_lz4_cpu(sample, bs, t, orig_hash, telemetry=telemetry)
                                writer.writerow([
                                    sample.name,
                                    point_idx,
                                    "" if cpu_freq_target is None else cpu_freq_target,
                                    "" if gpu_freq_target is None else gpu_freq_target,
                                    "CPU", t, bs, "N/A", "N/A", fmtf(cpu_stats['ratio'], 2),
                                    fmtf(cpu_stats['comp_mbs'], 2), fmtf(cpu_stats['dec_mbs'], 2),
                                    "", "",
                                    fmtf(cpu_stats['comp_time_s'], 6), fmtf(cpu_stats['dec_time_s'], 6),
                                    fmtf(cpu_stats['cpu_freq_avg_mhz'], 2),
                                    fmtf(cpu_stats['gpu_freq_avg_mhz'], 2),
                                    fmtf(cpu_stats['cpu_energy_j'], 6),
                                    fmtf(cpu_stats['gpu_energy_j'], 6),
                                    fmtf(cpu_stats['comp_cpu_power_w'], 6),
                                    fmtf(cpu_stats['comp_gpu_power_w'], 6),
                                    "yes" if cpu_stats.get('roundtrip_verified') else "no",
                                ])
                                f.flush()

                                cpu_cfg_label = f"FP={point_idx};BS={bs};T={t}"
                                emit_case_average(sample.name, "CPU", cpu_cfg_label, cpu_stats)
                                if cpu_stats.get("roundtrip_verified", False):
                                    summary_records.append(build_summary_record("CPU", cpu_cfg_label, cpu_stats))

                    # GPU Sweep
                    if not args.cpu_only:
                        for bs in gpu_block_sizes:
                            for hl in hash_logs:
                                for lsz in local_sizes:
                                    for accel in gpu_accels:
                                        gpu_stats = run_lz4_gpu(sample, bs, hl, lsz, accel, orig_hash, telemetry=telemetry, bench_seconds=args.bench_seconds)
                                        writer.writerow([
                                            sample.name,
                                            point_idx,
                                            "" if cpu_freq_target is None else cpu_freq_target,
                                            "" if gpu_freq_target is None else gpu_freq_target,
                                            "GPU", lsz, bs, hl, accel, fmtf(gpu_stats['ratio'], 2),
                                            fmtf(gpu_stats['comp_mbs'], 2), fmtf(gpu_stats['dec_mbs'], 2),
                                            fmtf(gpu_stats.get('comp_total_mbs', 0), 2),
                                            fmtf(gpu_stats.get('dec_total_mbs', 0), 2),
                                            fmtf(gpu_stats['comp_time_s'], 6), fmtf(gpu_stats['dec_time_s'], 6),
                                            fmtf(gpu_stats['cpu_freq_avg_mhz'], 2),
                                            fmtf(gpu_stats['gpu_freq_avg_mhz'], 2),
                                            fmtf(gpu_stats['cpu_energy_j'], 6),
                                            fmtf(gpu_stats['gpu_energy_j'], 6),
                                            fmtf(gpu_stats['comp_cpu_power_w'], 6),
                                            fmtf(gpu_stats['comp_gpu_power_w'], 6),
                                            "yes" if gpu_stats.get('roundtrip_verified') else "no",
                                        ])
                                        f.flush()

                                        gpu_cfg_label = f"FP={point_idx};BS={bs};HL={hl};LSZ={lsz};ACC={accel}"
                                        emit_case_average(sample.name, "GPU", gpu_cfg_label, gpu_stats)
                                        if gpu_stats.get("roundtrip_verified", False):
                                            summary_records.append(build_summary_record("GPU", gpu_cfg_label, gpu_stats))

        print_and_save_config_summary(summary_records, results_summary_csv)
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
