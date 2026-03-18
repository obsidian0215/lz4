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
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tools.hw_telemetry import TelemetryProbe, apply_freq_percent, apply_freq_mhz

# Paths
IS_WINDOWS = os.name == "nt"
EXEEXT = ".exe" if IS_WINDOWS else ""
TEMP_DIR = Path(tempfile.gettempdir())
LZ4_BIN = os.environ.get("LZ4_CPU_BIN", str(REPO_ROOT / "programs" / f"lz4{EXEEXT}"))
LZ4_GPU_BIN = os.environ.get("LZ4_GPU_BIN", str(REPO_ROOT / "lz4_gpu" / f"lz4_gpu{EXEEXT}"))
LZ4_HYBRID_BIN = os.environ.get("LZ4_HYBRID_BIN", str(REPO_ROOT / "lz4_hybrid" / f"lz4_hybrid{EXEEXT}"))
SAMPLES_DIR = os.environ.get("LZ4_SAMPLES_DIR", str(REPO_ROOT / "samples"))
RESULTS_DIR = os.environ.get("LZ4_RESULTS_DIR", str(REPO_ROOT / "exp_results"))
LZ4_DAEMON_SOCKET_PATH = str(TEMP_DIR / "lz4_gpu_daemon.sock")
LZ4_DAEMON_PID_PATH = str(TEMP_DIR / "lz4_gpu_daemon.pid")
CPU_CONTROL_SCRIPT = str(REPO_ROOT / "tools" / "cpu_control.sh")
GPU_CONTROL_SCRIPT = str(REPO_ROOT / "tools" / "gpu_control.sh")

# Configuration Space
CPU_BLOCK_SIZES = ["1M"]
GPU_BLOCK_SIZES = ["64K", "128K", "256K"]
CPU_THREADS = [1, 4, 0]
LOCAL_SIZES = [1]
GPU_ACCELS = [1, 2]
HYBRID_BLOCK_SIZES = ["64K", "128K", "256K"]
HYBRID_GPU_RATIOS = [0.0, 0.3, 0.5, 0.7, 1.0]
HYBRID_CPU_THREADS = [1, 4, 0]
HYBRID_SPLIT_MODES = ["fixed", "adaptive"]
HYBRID_LOCAL_SIZES = [1]
HYBRID_ACCELS = [1, 2]

# Default frequency configs (for intel iGPU)
DEFAULT_CPU_FREQ_MHZ = "800,1900,3000,5000"
DEFAULT_GPU_FREQ_MHZ = "300,1500"



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
        LZ4_BIN,
        str(REPO_ROOT / "programs" / f"lz4{EXEEXT}"),
        str(REPO_ROOT / f"lz4{EXEEXT}"),
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
        f"Cannot find/build LZ4 CPU binary. Tried {candidates} and PATH."
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


def parse_float_list(value, default_list):
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
        out.append(float(tok))
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
        "comp_total": float(avg_stats.get("comp_total_mbs", 0.0) or 0.0) if avg_stats.get("comp_total_mbs") is not None else None,
        "dec_total": float(avg_stats.get("dec_total_mbs", 0.0) or 0.0) if avg_stats.get("dec_total_mbs") is not None else None,
    }


def print_and_save_config_summary(summary_records, out_csv):
    if not summary_records:
        print("[ConfigSummary] no summary records")
        return

    grouped = {}
    for rec in summary_records:
        key = (rec["engine"], rec["config"])
        grouped.setdefault(key, []).append(rec)

    has_total = any(r.get("comp_total") is not None or r.get("dec_total") is not None for r in summary_records)
    print("\n===== Per-Config Summary Stats (mean/median/p10/p90) =====")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "Engine", "Config", "Samples",
            "Ratio_mean", "Ratio_median", "Ratio_p10", "Ratio_p90", "Ratio_min", "Ratio_max",
            "CompMBs_mean", "CompMBs_median", "CompMBs_p10", "CompMBs_p90", "CompMBs_min", "CompMBs_max",
            "DecMBs_mean", "DecMBs_median", "DecMBs_p10", "DecMBs_p90", "DecMBs_min", "DecMBs_max",
        ]
        if has_total:
            header.extend([
                "CompTotalMBs_mean", "CompTotalMBs_median", "CompTotalMBs_p10", "CompTotalMBs_p90", "CompTotalMBs_min", "CompTotalMBs_max",
                "DecTotalMBs_mean", "DecTotalMBs_median", "DecTotalMBs_p10", "DecTotalMBs_p90", "DecTotalMBs_min", "DecTotalMBs_max",
            ])
        writer.writerow(header)

        for (engine, config), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
            ratio_s = summarize_dist([r["ratio"] for r in rows])
            comp_kernel_s = summarize_dist([r["comp_kernel"] for r in rows])
            dec_kernel_s = summarize_dist([r["dec_kernel"] for r in rows])

            line = (
                f"[ConfigSummary] {engine} {config} n={ratio_s['count']} | "
                f"ratio(mean/med/p10)={fmt_metric(ratio_s['mean'],2)}/{fmt_metric(ratio_s['median'],2)}/{fmt_metric(ratio_s['p10'],2)}% | "
                f"Cmbs={fmt_metric(comp_kernel_s['mean'],2)}/{fmt_metric(comp_kernel_s['median'],2)}/{fmt_metric(comp_kernel_s['p10'],2)} | "
                f"Dmbs={fmt_metric(dec_kernel_s['mean'],2)}/{fmt_metric(dec_kernel_s['median'],2)}/{fmt_metric(dec_kernel_s['p10'],2)}"
            )

            csv_row = [
                engine, config, ratio_s["count"],
                fmt_csv_metric(ratio_s["mean"], 4), fmt_csv_metric(ratio_s["median"], 4), fmt_csv_metric(ratio_s["p10"], 4), fmt_csv_metric(ratio_s["p90"], 4), fmt_csv_metric(ratio_s["min"], 4), fmt_csv_metric(ratio_s["max"], 4),
                fmt_csv_metric(comp_kernel_s["mean"], 4), fmt_csv_metric(comp_kernel_s["median"], 4), fmt_csv_metric(comp_kernel_s["p10"], 4), fmt_csv_metric(comp_kernel_s["p90"], 4), fmt_csv_metric(comp_kernel_s["min"], 4), fmt_csv_metric(comp_kernel_s["max"], 4),
                fmt_csv_metric(dec_kernel_s["mean"], 4), fmt_csv_metric(dec_kernel_s["median"], 4), fmt_csv_metric(dec_kernel_s["p10"], 4), fmt_csv_metric(dec_kernel_s["p90"], 4), fmt_csv_metric(dec_kernel_s["min"], 4), fmt_csv_metric(dec_kernel_s["max"], 4),
            ]
            if has_total:
                comp_total_s = summarize_dist([r["comp_total"] for r in rows])
                dec_total_s = summarize_dist([r["dec_total"] for r in rows])
                line += (
                    f" | CTmbs={fmt_metric(comp_total_s['mean'],2)}/{fmt_metric(comp_total_s['median'],2)}/{fmt_metric(comp_total_s['p10'],2)}"
                    f" | DTmbs={fmt_metric(dec_total_s['mean'],2)}/{fmt_metric(dec_total_s['median'],2)}/{fmt_metric(dec_total_s['p10'],2)}"
                )
                csv_row.extend([
                    fmt_csv_metric(comp_total_s["mean"], 4), fmt_csv_metric(comp_total_s["median"], 4), fmt_csv_metric(comp_total_s["p10"], 4), fmt_csv_metric(comp_total_s["p90"], 4), fmt_csv_metric(comp_total_s["min"], 4), fmt_csv_metric(comp_total_s["max"], 4),
                    fmt_csv_metric(dec_total_s["mean"], 4), fmt_csv_metric(dec_total_s["median"], 4), fmt_csv_metric(dec_total_s["p10"], 4), fmt_csv_metric(dec_total_s["p90"], 4), fmt_csv_metric(dec_total_s["min"], 4), fmt_csv_metric(dec_total_s["max"], 4),
                ])
            print(line)
            writer.writerow(csv_row)

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


def resolve_samples_root(samples_arg):
    root = Path(samples_arg)
    if not root.exists():
        raise SystemExit(f"Samples directory not found: {samples_arg}")

    direct_files = sorted([p for p in root.glob("*") if p.is_file()])
    if direct_files:
        return root, direct_files

    child_dirs = sorted([p for p in root.glob("*") if p.is_dir()])
    if len(child_dirs) == 1:
        nested_files = sorted([p for p in child_dirs[0].glob("*") if p.is_file()])
        if nested_files:
            return child_dirs[0], nested_files

    return root, direct_files


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


def build_freq_mhz_points(cpu_mhz_str, gpu_mhz_str):
    def parse_mhz(s):
        if not s or not s.strip():
            return [None]
        return [int(x.strip()) for x in s.split(',') if x.strip()]
    return parse_mhz(cpu_mhz_str), parse_mhz(gpu_mhz_str)

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


def make_temp_file_path(prefix, suffix):
    return str(TEMP_DIR / f"{prefix}_{os.getpid()}_{int(time.time() * 1000)}{suffix}")


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
    summary = telemetry.summarize_samples(samples)

    tel = {
        "elapsed_s": float(summary.get("elapsed_s", 0.0) or 0.0),
        "cpu_freq_avg_mhz": float(summary.get("cpu_freq_avg_mhz", 0.0) or 0.0),
        "gpu_freq_avg_mhz": float(summary.get("gpu_freq_avg_mhz", 0.0) or 0.0),
        "cpu_energy_j": float(summary.get("cpu_energy_j", 0.0) or 0.0),
        "core_energy_j": float(summary.get("core_energy_j", 0.0) or 0.0),
        "gpu_energy_j": float(summary.get("gpu_energy_j", 0.0) or 0.0),
        "cpu_pkg_peak_power_w": float(summary.get("cpu_pkg_peak_power_w", 0.0) or 0.0),
        "cpu_core_peak_power_w": float(summary.get("cpu_core_peak_power_w", 0.0) or 0.0),
        "gpu_peak_power_w": float(summary.get("gpu_peak_power_w", 0.0) or 0.0),
    }
    completed = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    return completed, tel


def run_command_with_telemetry_cwd(cmd, cwd=None, telemetry=None, env=None, sample_interval_s=0.05):
    if telemetry is None:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env, cwd=cwd)
        return res, {}

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=cwd)
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
    summary = telemetry.summarize_samples(samples)

    tel = {
        "elapsed_s": float(summary.get("elapsed_s", 0.0) or 0.0),
        "cpu_freq_avg_mhz": float(summary.get("cpu_freq_avg_mhz", 0.0) or 0.0),
        "gpu_freq_avg_mhz": float(summary.get("gpu_freq_avg_mhz", 0.0) or 0.0),
        "cpu_energy_j": float(summary.get("cpu_energy_j", 0.0) or 0.0),
        "core_energy_j": float(summary.get("core_energy_j", 0.0) or 0.0),
        "gpu_energy_j": float(summary.get("gpu_energy_j", 0.0) or 0.0),
        "cpu_pkg_peak_power_w": float(summary.get("cpu_pkg_peak_power_w", 0.0) or 0.0),
        "cpu_core_peak_power_w": float(summary.get("cpu_core_peak_power_w", 0.0) or 0.0),
        "gpu_peak_power_w": float(summary.get("gpu_peak_power_w", 0.0) or 0.0),
    }
    completed = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    return completed, tel


def _clamp01(v):
    try:
        x = float(v)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def apply_wall_energy(stats, tel_window, comp_elapsed_s, dec_elapsed_s, energy_source,
                      idle_pkg_power_w=0.0, idle_core_power_w=0.0, idle_gpu_power_w=0.0,
                      gpu_share_hint=0.0):
    stats['cpu_freq_avg_mhz'] = float(tel_window.get('cpu_freq_avg_mhz', 0.0) or 0.0)
    stats['gpu_freq_avg_mhz'] = float(tel_window.get('gpu_freq_avg_mhz', 0.0) or 0.0)
    elapsed_s = float(tel_window.get('elapsed_s', 0.0) or 0.0)
    if elapsed_s <= 0.0:
        stats['cpu_energy_j'] = 0.0
        stats['gpu_energy_j'] = 0.0
        stats['comp_cpu_power_w'] = 0.0
        stats['comp_gpu_power_w'] = 0.0
        stats['energy_source'] = energy_source
        return

    pkg_total_energy = float(tel_window.get('cpu_energy_j', 0.0) or 0.0)
    core_total_energy = float(tel_window.get('core_energy_j', 0.0) or 0.0)
    gpu_total_energy = float(tel_window.get('gpu_energy_j', 0.0) or 0.0)

    pkg_peak_power_w = float(tel_window.get('cpu_pkg_peak_power_w', 0.0) or 0.0)
    core_peak_power_w = float(tel_window.get('cpu_core_peak_power_w', 0.0) or 0.0)
    gpu_peak_power_w = float(tel_window.get('gpu_peak_power_w', 0.0) or 0.0)

    if pkg_peak_power_w <= 0.0 and pkg_total_energy > 0.0:
        pkg_peak_power_w = pkg_total_energy / elapsed_s
    if core_peak_power_w <= 0.0 and core_total_energy > 0.0:
        core_peak_power_w = core_total_energy / elapsed_s
    if gpu_peak_power_w <= 0.0 and gpu_total_energy > 0.0:
        gpu_peak_power_w = gpu_total_energy / elapsed_s

    cpu_peak_power_w = core_peak_power_w if core_peak_power_w > 0.0 else pkg_peak_power_w
    cpu_idle_power_w = float(idle_core_power_w or 0.0) if core_peak_power_w > 0.0 else float(idle_pkg_power_w or 0.0)

    cpu_inc_power_w = max(0.0, cpu_peak_power_w - max(0.0, cpu_idle_power_w))

    gpu_inc_power_direct = None
    if gpu_peak_power_w > 0.0:
        gpu_inc_power_direct = max(0.0, gpu_peak_power_w - max(0.0, float(idle_gpu_power_w or 0.0)))

    gpu_hint = _clamp01(gpu_share_hint)
    if gpu_inc_power_direct is not None:
        gpu_inc_power_w = gpu_inc_power_direct
    else:
        gpu_inc_power_w = cpu_inc_power_w * gpu_hint

    comp_phase_s = max(0.0, float(comp_elapsed_s or 0.0))
    dec_phase_s = max(0.0, float(dec_elapsed_s or 0.0))
    if comp_phase_s <= 0.0 and dec_phase_s <= 0.0:
        comp_phase_s = elapsed_s

    stats['cpu_energy_j'] = cpu_inc_power_w * comp_phase_s
    stats['gpu_energy_j'] = gpu_inc_power_w * comp_phase_s
    stats['dec_cpu_energy_j'] = cpu_inc_power_w * dec_phase_s
    stats['dec_gpu_energy_j'] = gpu_inc_power_w * dec_phase_s
    stats['comp_cpu_power_w'] = cpu_inc_power_w
    stats['comp_gpu_power_w'] = gpu_inc_power_w
    stats['cpu_peak_power_w'] = cpu_peak_power_w
    stats['gpu_peak_power_w'] = gpu_peak_power_w
    stats['cpu_idle_power_w'] = cpu_idle_power_w
    stats['gpu_idle_power_w'] = float(idle_gpu_power_w or 0.0)
    stats['energy_source'] = energy_source


def run_control_action(control_script_path, action):
    if not os.path.exists(control_script_path):
        return "missing_script"
    if IS_WINDOWS:
        return "unsupported_on_windows"

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


def warm_lz4_gpu_daemon(file_path, bs_arg, lsz, accel):
    if IS_WINDOWS:
        return

    sample_path = str(Path(file_path).resolve())

    warm_out = make_temp_file_path("lz4_gpu_warm", ".lz4")
    warm_dec = f"{warm_out}.dec"
    try:
        warm_comp_cmd = [
            LZ4_GPU_BIN,
            "--use-daemon",
            "-B", bs_arg,
            "--local", str(lsz),
            "-a", str(accel),
            "-o", warm_out,
            sample_path,
        ]
        subprocess.run(warm_comp_cmd, capture_output=True, text=True, check=False, env=build_gpu_subprocess_env())

        if os.path.exists(warm_out):
            warm_dec_cmd = [
                LZ4_GPU_BIN,
                "--use-daemon",
                "-d",
                "-o", warm_dec,
                warm_out,
            ]
            subprocess.run(warm_dec_cmd, capture_output=True, text=True, check=False, env=build_gpu_subprocess_env())
    finally:
        safe_remove(warm_out)
        safe_remove(warm_dec)


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


def run_lz4_cpu(file_path, bs, threads, orig_hash, telemetry=None, bench_seconds=3.0):
    print(f"Bench_CPU: {file_path.name} BS={bs} T={threads}")
    sample_path = str(file_path.resolve())
    exec_threads = int(threads) if int(threads) > 0 else max(1, int(os.cpu_count() or 1))
    stats = {
        'ratio': 0.0,
        'comp_mbs': 0.0,
        'dec_mbs': 0.0,
        'comp_total_mbs': 0.0,
        'dec_total_mbs': 0.0,
        'comp_time_s': 0.0,
        'dec_time_s': 0.0,
        'throughput_semantics': 'op_time_bench',
        'roundtrip_verified': False,
        'cpu_freq_avg_mhz': 0.0,
        'gpu_freq_avg_mhz': 0.0,
        'cpu_energy_j': 0.0,
        'gpu_energy_j': 0.0,
        'comp_cpu_power_w': 0.0,
        'comp_gpu_power_w': 0.0,
        'energy_source': 'none',
    }
    tel_window = {}
    idle_pkg_power_w = 0.0
    idle_core_power_w = 0.0
    idle_gpu_power_w = 0.0
    try:
        in_sz = file_path.stat().st_size
        if in_sz <= 0:
            return stats

        if telemetry:
            idle_pkg_power_w = telemetry.measure_idle_pkg_power_w(0.2)
            idle_core_power_w = telemetry.measure_idle_core_power_w(0.2)

        # Use native lz4 bench output only (no temp file workflow)
        cmd_bench = [LZ4_BIN, "-q", f"-B{str(bs).upper()}", f"-T{exec_threads}", "-b1", sample_path]
        bench_res, tel_window = run_command_with_telemetry(cmd_bench, telemetry=telemetry)
        bench_output = (bench_res.stdout or "") + (bench_res.stderr or "")
        parsed = parse_lz4_cpu_bench_output(bench_output, in_sz)
        if parsed.get('parsed'):
            stats['ratio'] = parsed.get('ratio', 0)
            stats['comp_mbs'] = parsed.get('comp_mbs', 0)
            stats['dec_mbs'] = parsed.get('dec_mbs', 0)
            stats['comp_time_s'] = parsed.get('comp_time_s', 0)
            stats['dec_time_s'] = parsed.get('dec_time_s', 0)
            stats['comp_total_mbs'] = parsed.get('comp_mbs', 0)
            stats['dec_total_mbs'] = parsed.get('dec_mbs', 0)

            tmp_comp = make_temp_file_path("lz4_cpu_total", ".lz4")
            tmp_dec = f"{tmp_comp}.dec"
            total_ok = False

            try:
                cmd_comp_total = [
                    LZ4_BIN,
                    "-q",
                    "-z",
                    "-f",
                    f"-B{str(bs).upper()}",
                    f"-T{exec_threads}",
                    sample_path,
                    tmp_comp,
                ]
                t0 = time.perf_counter()
                comp_total_res, _ = run_command_with_telemetry(cmd_comp_total, telemetry=None)
                comp_elapsed_s = max(0.0, time.perf_counter() - t0)

                cmd_dec_total = [
                    LZ4_BIN,
                    "-q",
                    "-f",
                    "-d",
                    tmp_comp,
                    tmp_dec,
                ]
                t1 = time.perf_counter()
                dec_total_res, _ = run_command_with_telemetry(cmd_dec_total, telemetry=None)
                dec_elapsed_s = max(0.0, time.perf_counter() - t1)

                expected_hash = orig_hash

                total_ok = (
                    comp_total_res.returncode == 0
                    and dec_total_res.returncode == 0
                    and file_matches_hash(tmp_dec, expected_hash)
                )

            finally:
                safe_remove(tmp_comp)
                safe_remove(tmp_dec)

            stats['throughput_semantics'] = 'stable_cpu_bench_with_op_total'
            stats['roundtrip_verified'] = (bench_res.returncode == 0) and total_ok
        else:
            stats['throughput_semantics'] = 'stable_cpu_bench_parse_failed'
            stats['roundtrip_verified'] = False
            print(f"  [CPU] stable bench parse failed for {file_path} BS={bs} T={threads}", flush=True)
    except Exception as e:
        print(f"CPU error: {e}")

    if telemetry and tel_window:
        apply_wall_energy(
            stats,
            tel_window,
            float(stats.get('comp_time_s', 0.0) or 0.0),
            float(stats.get('dec_time_s', 0.0) or 0.0),
            telemetry.describe_sources(),
            idle_pkg_power_w=idle_pkg_power_w,
            idle_core_power_w=idle_core_power_w,
            idle_gpu_power_w=idle_gpu_power_w,
            gpu_share_hint=0.0,
        )

    return stats


def run_lz4_gpu(file_path, bs, lsz, accel, orig_hash, telemetry=None, bench_seconds=3.0, use_daemon=False):
    print(f"Bench_GPU: {file_path.name} BS={bs} LSZ={lsz} A={accel}")
    sample_path = str(file_path.resolve())
    bs_arg = str(bs).upper()
    stats = {
        'ratio': 0.0,
        'comp_mbs': 0.0,
        'dec_mbs': 0.0,
        'comp_total_mbs': 0.0,
        'dec_total_mbs': 0.0,
        'comp_time_s': 0.0,
        'dec_time_s': 0.0,
        'throughput_semantics': 'op_time_bench',
        'roundtrip_verified': False,
        'cpu_freq_avg_mhz': 0.0,
        'gpu_freq_avg_mhz': 0.0,
        'cpu_energy_j': 0.0,
        'gpu_energy_j': 0.0,
        'comp_cpu_power_w': 0.0,
        'comp_gpu_power_w': 0.0,
        'energy_source': 'none',
    }
    tel_window = {}
    idle_pkg_power_w = 0.0
    idle_core_power_w = 0.0
    idle_gpu_power_w = 0.0
    try:
        in_sz = file_path.stat().st_size
        gpu_dir = str(Path(LZ4_GPU_BIN).resolve().parent)

        if telemetry:
            idle_pkg_power_w = telemetry.measure_idle_pkg_power_w(0.2)
            idle_core_power_w = telemetry.measure_idle_core_power_w(0.2)
            idle_gpu_power_w = telemetry.measure_idle_gpu_power_w(0.2)

        bench_cmd = [
            LZ4_GPU_BIN,
            "--bench", str(bench_seconds),
            "-B", bs_arg,
            "--local", str(lsz),
            "-a", str(accel),
            sample_path,
        ]
        bench_res, tel_window = run_command_with_telemetry_cwd(bench_cmd, cwd=gpu_dir, telemetry=telemetry, env=build_gpu_subprocess_env())
        bench_output = (bench_res.stdout or "") + (bench_res.stderr or "")
        stable = parse_stable_bench_output(bench_output)

        if stable:
            if use_daemon:
                warm_lz4_gpu_daemon(file_path, bs_arg, lsz, accel)
            stats['ratio'] = stable['ratio']
            stats['comp_mbs'] = stable['comp_kernel_tp']
            stats['dec_mbs'] = stable['dec_kernel_tp']
            stats['comp_total_mbs'] = stable.get('comp_total_tp', 0.0)
            stats['dec_total_mbs'] = stable.get('dec_total_tp', 0.0)

            tmp_comp = make_temp_file_path("lz4_gpu_total", ".lz4")
            tmp_dec = f"{tmp_comp}.dec"
            total_tel = {}
            total_ok = False

            try:
                cmd_comp_total = [LZ4_GPU_BIN]
                if use_daemon:
                    cmd_comp_total.append("--use-daemon")
                cmd_comp_total.extend([
                    "-v",
                    "-B", bs_arg,
                    "--local", str(lsz),
                    "-a", str(accel),
                    "-o", tmp_comp,
                    sample_path,
                ])
                comp_total_res, comp_total_tel = run_command_with_telemetry_cwd(cmd_comp_total, cwd=gpu_dir, telemetry=telemetry, env=build_gpu_subprocess_env())
                comp_total_output = (comp_total_res.stdout or "") + (comp_total_res.stderr or "")
                comp_total_parsed = parse_gpu_output(comp_total_output)

                cmd_dec_total = [LZ4_GPU_BIN]
                if use_daemon:
                    cmd_dec_total.append("--use-daemon")
                cmd_dec_total.extend([
                    "-v",
                    "-d",
                    "-o", tmp_dec,
                    tmp_comp,
                ])
                dec_total_res, dec_total_tel = run_command_with_telemetry_cwd(cmd_dec_total, cwd=gpu_dir, telemetry=telemetry, env=build_gpu_subprocess_env())
                dec_total_output = (dec_total_res.stdout or "") + (dec_total_res.stderr or "")
                dec_total_parsed = parse_gpu_output(dec_total_output)

                comp_elapsed_s = float(comp_total_tel.get('elapsed_s', 0.0) or 0.0)
                dec_elapsed_s = float(dec_total_tel.get('elapsed_s', 0.0) or 0.0)
                total_tel = {
                    'elapsed_s': comp_elapsed_s + dec_elapsed_s,
                    'cpu_freq_avg_mhz': float(comp_total_tel.get('cpu_freq_avg_mhz', 0.0) or 0.0),
                    'gpu_freq_avg_mhz': float(comp_total_tel.get('gpu_freq_avg_mhz', 0.0) or 0.0),
                    'cpu_energy_j': float(comp_total_tel.get('cpu_energy_j', 0.0) or 0.0) + float(dec_total_tel.get('cpu_energy_j', 0.0) or 0.0),
                    'core_energy_j': float(comp_total_tel.get('core_energy_j', 0.0) or 0.0) + float(dec_total_tel.get('core_energy_j', 0.0) or 0.0),
                    'gpu_energy_j': float(comp_total_tel.get('gpu_energy_j', 0.0) or 0.0) + float(dec_total_tel.get('gpu_energy_j', 0.0) or 0.0),
                    'cpu_pkg_peak_power_w': max(float(comp_total_tel.get('cpu_pkg_peak_power_w', 0.0) or 0.0), float(dec_total_tel.get('cpu_pkg_peak_power_w', 0.0) or 0.0)),
                    'cpu_core_peak_power_w': max(float(comp_total_tel.get('cpu_core_peak_power_w', 0.0) or 0.0), float(dec_total_tel.get('cpu_core_peak_power_w', 0.0) or 0.0)),
                    'gpu_peak_power_w': max(float(comp_total_tel.get('gpu_peak_power_w', 0.0) or 0.0), float(dec_total_tel.get('gpu_peak_power_w', 0.0) or 0.0)),
                }

                total_ok = (
                    comp_total_res.returncode == 0
                    and dec_total_res.returncode == 0
                    and file_matches_hash(tmp_dec, orig_hash)
                )

                if comp_elapsed_s > 0.0:
                    stats['comp_time_s'] = comp_elapsed_s
                elif comp_total_parsed.get('inclusive_tp'):
                    stats['comp_total_mbs'] = comp_total_parsed['inclusive_tp']
                if dec_elapsed_s > 0.0:
                    stats['dec_time_s'] = dec_elapsed_s
                elif dec_total_parsed.get('inclusive_tp'):
                    stats['dec_total_mbs'] = dec_total_parsed['inclusive_tp']
            finally:
                safe_remove(tmp_comp)
                safe_remove(tmp_dec)

            stats['throughput_semantics'] = 'stable_kernel_bench_with_op_total'
            stats['roundtrip_verified'] = bool(stable['verify_ok']) and (bench_res.returncode == 0) and total_ok

            if telemetry:
                bench_tel = tel_window if tel_window else {}
                io_tel = total_tel if total_tel else {}
                merged_tel = {
                    'elapsed_s': float(bench_tel.get('elapsed_s', 0.0) or io_tel.get('elapsed_s', 0.0) or 0.0),
                    'cpu_freq_avg_mhz': float(bench_tel.get('cpu_freq_avg_mhz', 0.0) or io_tel.get('cpu_freq_avg_mhz', 0.0) or 0.0),
                    'gpu_freq_avg_mhz': float(bench_tel.get('gpu_freq_avg_mhz', 0.0) or io_tel.get('gpu_freq_avg_mhz', 0.0) or 0.0),
                    'cpu_energy_j': float(bench_tel.get('cpu_energy_j', 0.0) or io_tel.get('cpu_energy_j', 0.0) or 0.0),
                    'core_energy_j': float(bench_tel.get('core_energy_j', 0.0) or io_tel.get('core_energy_j', 0.0) or 0.0),
                    'gpu_energy_j': float(bench_tel.get('gpu_energy_j', 0.0) or io_tel.get('gpu_energy_j', 0.0) or 0.0),
                    'cpu_pkg_peak_power_w': max(float(bench_tel.get('cpu_pkg_peak_power_w', 0.0) or 0.0), float(io_tel.get('cpu_pkg_peak_power_w', 0.0) or 0.0)),
                    'cpu_core_peak_power_w': max(float(bench_tel.get('cpu_core_peak_power_w', 0.0) or 0.0), float(io_tel.get('cpu_core_peak_power_w', 0.0) or 0.0)),
                    'gpu_peak_power_w': max(float(bench_tel.get('gpu_peak_power_w', 0.0) or 0.0), float(io_tel.get('gpu_peak_power_w', 0.0) or 0.0)),
                }
                apply_wall_energy(
                    stats,
                    merged_tel,
                    float(stats.get('comp_time_s', 0.0) or 0.0),
                    float(stats.get('dec_time_s', 0.0) or 0.0),
                    telemetry.describe_sources(),
                    idle_pkg_power_w=idle_pkg_power_w,
                    idle_core_power_w=idle_core_power_w,
                    idle_gpu_power_w=idle_gpu_power_w,
                    gpu_share_hint=1.0,
                )
        else:
            stats['throughput_semantics'] = 'stable_kernel_bench_parse_failed'
            stats['roundtrip_verified'] = False
            print(f"  [GPU] stable bench parse failed for {file_path} (BS={bs} LSZ={lsz} A={accel})", flush=True)
    except Exception as exc:
        print(f"GPU error: {exc}")

    return stats


def run_lz4_hybrid(file_path, bs, gpu_ratio, cpu_threads, local_size, accel, orig_hash, telemetry=None, bench_seconds=3.0, split_mode="fixed", sample_blocks=8):
    print(f"Bench_HYBRID: {file_path.name} BS={bs} mode={split_mode} R={gpu_ratio} T={cpu_threads} LSZ={local_size} A={accel}")
    sample_path = str(file_path.resolve())
    bs_arg = str(bs).lower()
    stats = {
        'ratio': 0.0,
        'comp_mbs': 0.0,
        'dec_mbs': 0.0,
        'comp_total_mbs': 0.0,
        'dec_total_mbs': 0.0,
        'comp_time_s': 0.0,
        'dec_time_s': 0.0,
        'throughput_semantics': 'op_time_bench',
        'roundtrip_verified': False,
        'cpu_freq_avg_mhz': 0.0,
        'gpu_freq_avg_mhz': 0.0,
        'cpu_energy_j': 0.0,
        'gpu_energy_j': 0.0,
        'comp_cpu_power_w': 0.0,
        'comp_gpu_power_w': 0.0,
        'energy_source': 'none',
    }
    idle_pkg_power_w = 0.0
    idle_core_power_w = 0.0
    idle_gpu_power_w = 0.0
    try:
        in_sz = file_path.stat().st_size
        hybrid_dir = str(Path(LZ4_HYBRID_BIN).resolve().parent)

        if telemetry:
            idle_pkg_power_w = telemetry.measure_idle_pkg_power_w(0.2)
            idle_core_power_w = telemetry.measure_idle_core_power_w(0.2)
            idle_gpu_power_w = telemetry.measure_idle_gpu_power_w(0.2)

        bench_cmd = [
            LZ4_HYBRID_BIN,
            "--bench", str(bench_seconds),
            "-b", bs_arg,
            "-l", str(local_size),
            "-a", str(accel),
            "-T", str(cpu_threads),
            sample_path,
        ]
        if split_mode == "adaptive":
            bench_cmd[1:1] = ["--adaptive", "--sample-blocks", str(sample_blocks), "--gpu-ratio", str(gpu_ratio)]
        else:
            bench_cmd[1:1] = ["--gpu-ratio", str(gpu_ratio)]

        bench_res, _ = run_command_with_telemetry_cwd(bench_cmd, cwd=hybrid_dir, telemetry=telemetry, env=build_gpu_subprocess_env())
        bench_output = (bench_res.stdout or "") + (bench_res.stderr or "")
        stable = parse_stable_bench_output(bench_output)
        if stable:
            stats['ratio'] = stable['ratio']
            stats['comp_mbs'] = stable['comp_kernel_tp']
            stats['dec_mbs'] = stable['dec_kernel_tp']
            stats['comp_total_mbs'] = stable.get('comp_total_tp', 0.0)
            stats['dec_total_mbs'] = stable.get('dec_total_tp', 0.0)

            total_ok = False
            total_tel = {}
            tmp_comp = make_temp_file_path("lz4_hybrid_verify_tmp", ".lz4")
            tmp_dec = f"{tmp_comp}.dec"
            try:
                cmd_comp_total = [
                    LZ4_HYBRID_BIN,
                    "-b", bs_arg,
                    "-l", str(local_size),
                    "-a", str(accel),
                    "-T", str(cpu_threads),
                    "-o", tmp_comp,
                    sample_path,
                ]
                if split_mode == "adaptive":
                    cmd_comp_total[1:1] = ["--adaptive", "--sample-blocks", str(sample_blocks), "--gpu-ratio", str(gpu_ratio)]
                else:
                    cmd_comp_total[1:1] = ["--gpu-ratio", str(gpu_ratio)]
                comp_total_res, comp_total_tel = run_command_with_telemetry_cwd(cmd_comp_total, cwd=hybrid_dir, telemetry=telemetry, env=build_gpu_subprocess_env())

                cmd_dec_total = [
                    LZ4_HYBRID_BIN,
                    "-d",
                    "-o", tmp_dec,
                    tmp_comp,
                ]
                if split_mode == "adaptive":
                    cmd_dec_total[1:1] = ["--adaptive", "--sample-blocks", str(sample_blocks), "--gpu-ratio", str(gpu_ratio), "-T", str(cpu_threads)]
                else:
                    cmd_dec_total[1:1] = ["--gpu-ratio", str(gpu_ratio), "-T", str(cpu_threads)]
                dec_total_res, dec_total_tel = run_command_with_telemetry_cwd(cmd_dec_total, cwd=hybrid_dir, telemetry=telemetry, env=build_gpu_subprocess_env())

                comp_elapsed_s = float(comp_total_tel.get('elapsed_s', 0.0) or 0.0)
                dec_elapsed_s = float(dec_total_tel.get('elapsed_s', 0.0) or 0.0)
                if comp_elapsed_s > 0.0:
                    stats['comp_time_s'] = comp_elapsed_s
                if dec_elapsed_s > 0.0:
                    stats['dec_time_s'] = dec_elapsed_s

                total_ok = (
                    comp_total_res.returncode == 0
                    and dec_total_res.returncode == 0
                    and file_matches_hash(tmp_dec, orig_hash)
                )
                total_tel = {
                    'elapsed_s': float(comp_total_tel.get('elapsed_s', 0.0) or 0.0) + float(dec_total_tel.get('elapsed_s', 0.0) or 0.0),
                    'cpu_freq_avg_mhz': float(comp_total_tel.get('cpu_freq_avg_mhz', 0.0) or 0.0),
                    'gpu_freq_avg_mhz': float(comp_total_tel.get('gpu_freq_avg_mhz', 0.0) or 0.0),
                    'cpu_energy_j': float(comp_total_tel.get('cpu_energy_j', 0.0) or 0.0) + float(dec_total_tel.get('cpu_energy_j', 0.0) or 0.0),
                    'core_energy_j': float(comp_total_tel.get('core_energy_j', 0.0) or 0.0) + float(dec_total_tel.get('core_energy_j', 0.0) or 0.0),
                    'gpu_energy_j': float(comp_total_tel.get('gpu_energy_j', 0.0) or 0.0) + float(dec_total_tel.get('gpu_energy_j', 0.0) or 0.0),
                    'cpu_pkg_peak_power_w': max(float(comp_total_tel.get('cpu_pkg_peak_power_w', 0.0) or 0.0), float(dec_total_tel.get('cpu_pkg_peak_power_w', 0.0) or 0.0)),
                    'cpu_core_peak_power_w': max(float(comp_total_tel.get('cpu_core_peak_power_w', 0.0) or 0.0), float(dec_total_tel.get('cpu_core_peak_power_w', 0.0) or 0.0)),
                    'gpu_peak_power_w': max(float(comp_total_tel.get('gpu_peak_power_w', 0.0) or 0.0), float(dec_total_tel.get('gpu_peak_power_w', 0.0) or 0.0)),
                }
            finally:
                safe_remove(tmp_comp)
                safe_remove(tmp_dec)

            stats['throughput_semantics'] = 'stable_kernel_bench_inprocess_op_total'
            stats['roundtrip_verified'] = bool(stable['verify_ok']) and (bench_res.returncode == 0)
            if total_ok:
                stats['roundtrip_verified'] = True
            if telemetry and total_tel:
                apply_wall_energy(
                    stats,
                    total_tel,
                    float(stats.get('comp_time_s', 0.0) or 0.0),
                    float(stats.get('dec_time_s', 0.0) or 0.0),
                    telemetry.describe_sources(),
                    idle_pkg_power_w=idle_pkg_power_w,
                    idle_core_power_w=idle_core_power_w,
                    idle_gpu_power_w=idle_gpu_power_w,
                    gpu_share_hint=float(gpu_ratio),
                )
        else:
            stats['throughput_semantics'] = 'stable_kernel_bench_parse_failed'
            stats['roundtrip_verified'] = False
            print(f"  [HYBRID] stable bench parse failed for {file_path} (BS={bs} mode={split_mode} R={gpu_ratio} T={cpu_threads} LSZ={local_size} A={accel})", flush=True)
    except Exception as exc:
        print(f"HYBRID Error: {exc}")

    return stats

def main():
    parser = argparse.ArgumentParser(description='Bench LZ4 CPU/GPU/Hybrid sweep (supports --limit for quick runs)')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of samples (0 = all)')
    parser.add_argument('--samples', default=SAMPLES_DIR, help='Samples directory (default: /root/samples)')
    parser.add_argument('--cpu-only', action='store_true', help='Run CPU sweep only (skip GPU and Hybrid)')
    parser.add_argument('--gpu-only', action='store_true', help='Run GPU sweep only (skip CPU and Hybrid)')
    parser.add_argument('--hybrid-only', action='store_true', help='Run Hybrid sweep only (skip CPU and GPU)')
    parser.add_argument('--cpu-threads', default=','.join(str(x) for x in CPU_THREADS), help='CPU thread list, comma-separated (default: 1,2)')
    parser.add_argument('--cpu-block-sizes', default=','.join(CPU_BLOCK_SIZES), help='CPU block sizes, comma-separated')
    parser.add_argument('--gpu-block-sizes', default=','.join(GPU_BLOCK_SIZES), help='GPU block sizes, comma-separated')
    parser.add_argument('--local-sizes', default=','.join(str(x) for x in LOCAL_SIZES), help='GPU local sizes, comma-separated')
    parser.add_argument('--gpu-accels', default=','.join(str(x) for x in GPU_ACCELS), help='GPU acceleration factors, comma-separated')
    parser.add_argument('--hybrid-block-sizes', default=','.join(HYBRID_BLOCK_SIZES), help='Hybrid block sizes, comma-separated')
    parser.add_argument('--hybrid-gpu-ratios', default=','.join(str(x) for x in HYBRID_GPU_RATIOS), help='Hybrid GPU ratios, comma-separated')
    parser.add_argument('--hybrid-cpu-threads', default=','.join(str(x) for x in HYBRID_CPU_THREADS), help='Hybrid CPU threads, comma-separated')
    parser.add_argument('--hybrid-split-modes', default=','.join(HYBRID_SPLIT_MODES), help='Hybrid split modes, comma-separated (fixed,adaptive)')
    parser.add_argument('--hybrid-local-sizes', default=','.join(str(x) for x in HYBRID_LOCAL_SIZES), help='Hybrid local sizes, comma-separated')
    parser.add_argument('--hybrid-accels', default=','.join(str(x) for x in HYBRID_ACCELS), help='Hybrid acceleration factors, comma-separated')
    parser.add_argument('--freq-percent', type=int, default=None, help='Set both CPU and GPU to one shared frequency percent (0-100)')
    parser.add_argument('--freq-points', default='', help='Shared CPU/GPU frequency points, comma-separated (e.g. 40,70,100)')
    parser.add_argument('--cpu-freq-points', default=DEFAULT_CPU_FREQ_MHZ, help='CPU frequency points in MHz, comma-separated (default: %(default)s)')
    parser.add_argument('--gpu-freq-points', default=DEFAULT_GPU_FREQ_MHZ, help='GPU frequency points in MHz, comma-separated (default: %(default)s)')
    parser.add_argument('--hybrid-freq-pairs', default='', help='Hybrid CPU+GPU freq pairs in MHz, semicolon-separated (e.g. "800,300;800,1500;5000,300;5000,1500")')
    parser.add_argument('--single-file', default='', help='Only benchmark one file (path or basename under samples dir)')
    parser.add_argument('--no-telemetry', action='store_true', help='Disable freq/energy telemetry collection')
    parser.add_argument('--bench-seconds', type=float, default=3.0, help='Benchmark duration in seconds for timed bench paths (default: 3.0)')
    parser.add_argument('--results-dir', default=RESULTS_DIR, help='Directory for benchmark outputs')
    args = parser.parse_args()

    only_flags = sum([args.cpu_only, args.gpu_only, args.hybrid_only])
    if only_flags > 1:
        raise SystemExit('Cannot combine --cpu-only, --gpu-only, --hybrid-only')

    run_cpu = not args.gpu_only and not args.hybrid_only
    run_gpu = args.gpu_only
    run_hybrid = (not args.cpu_only and not args.gpu_only) or args.hybrid_only

    def emit_gpu_row_from_hybrid(sample, point_idx, cpu_freq_target, gpu_freq_target, bs, lsz, accel, hybrid_stats):
        writer.writerow([
            sample.name,
            point_idx,
            "" if cpu_freq_target is None else cpu_freq_target,
            "" if gpu_freq_target is None else gpu_freq_target,
            "GPU", lsz, bs, 14, accel, fmtf(hybrid_stats['ratio'], 2),
            fmtf(hybrid_stats['comp_mbs'], 2), fmtf(hybrid_stats['dec_mbs'], 2),
            fmtf(hybrid_stats.get('comp_total_mbs', 0), 2),
            fmtf(hybrid_stats.get('dec_total_mbs', 0), 2),
            fmtf(hybrid_stats['comp_time_s'], 6), fmtf(hybrid_stats['dec_time_s'], 6),
            fmtf(hybrid_stats['cpu_freq_avg_mhz'], 2),
            fmtf(hybrid_stats['gpu_freq_avg_mhz'], 2),
            fmtf(hybrid_stats['cpu_energy_j'], 6),
            fmtf(hybrid_stats['gpu_energy_j'], 6),
            fmtf(hybrid_stats['comp_cpu_power_w'], 6),
            fmtf(hybrid_stats['comp_gpu_power_w'], 6),
            "yes" if hybrid_stats.get('roundtrip_verified') else "no",
        ])
        f.flush()

        gpu_cfg_label = f"FP={point_idx};BS={bs};LSZ={lsz};ACC={accel}"
        emit_case_average(sample.name, "GPU", gpu_cfg_label, hybrid_stats)
        if hybrid_stats.get("roundtrip_verified", False):
            summary_records.append(build_summary_record("GPU", gpu_cfg_label, hybrid_stats))

    if run_cpu:
        resolved_lz4_bin = resolve_lz4_cpu_binary()
        globals()["LZ4_BIN"] = resolved_lz4_bin
        print(f"[CPU-BIN] using {resolved_lz4_bin}")

    cpu_threads = parse_int_list(args.cpu_threads, CPU_THREADS)
    cpu_block_sizes = parse_str_list(args.cpu_block_sizes, CPU_BLOCK_SIZES)
    gpu_block_sizes = parse_str_list(args.gpu_block_sizes, GPU_BLOCK_SIZES)
    local_sizes = parse_int_list(args.local_sizes, LOCAL_SIZES)
    gpu_accels = parse_int_list(args.gpu_accels, GPU_ACCELS)
    hybrid_block_sizes = parse_str_list(args.hybrid_block_sizes, HYBRID_BLOCK_SIZES)
    hybrid_gpu_ratios = parse_float_list(args.hybrid_gpu_ratios, HYBRID_GPU_RATIOS)
    hybrid_cpu_threads = parse_int_list(args.hybrid_cpu_threads, HYBRID_CPU_THREADS)
    hybrid_split_modes = parse_str_list(args.hybrid_split_modes, HYBRID_SPLIT_MODES)
    hybrid_local_sizes = parse_int_list(args.hybrid_local_sizes, HYBRID_LOCAL_SIZES)
    hybrid_accels = parse_int_list(args.hybrid_accels, HYBRID_ACCELS)
    use_gpu_daemon = run_gpu and (not IS_WINDOWS)
    freq_points = build_freq_points(parse_optional_int_list(args.freq_points), args.freq_percent)
    cpu_freq_mhz_points, gpu_freq_mhz_points = build_freq_mhz_points(args.cpu_freq_points, args.gpu_freq_points)
    hybrid_freq_pairs = []
    if args.hybrid_freq_pairs:
        for pair_str in args.hybrid_freq_pairs.split(';'):
            parts = pair_str.strip().split(',')
            if len(parts) == 2:
                hybrid_freq_pairs.append((int(parts[0].strip()), int(parts[1].strip())))
    use_mhz_mode = (cpu_freq_mhz_points != [None] or gpu_freq_mhz_points != [None] or len(hybrid_freq_pairs) > 0)
    if use_mhz_mode:
        freq_points = [None]

    if IS_WINDOWS:
        if use_mhz_mode or freq_points != [None]:
            print("[FreqControl] Windows detected; disabling frequency scan and forcing a single default point.")
        use_mhz_mode = False
        freq_points = [None]
        cpu_freq_mhz_points = [None]
        gpu_freq_mhz_points = [None]
        hybrid_freq_pairs = []

    telemetry = None if args.no_telemetry else TelemetryProbe()
    if telemetry is not None:
        print(f"Telemetry sources: {telemetry.describe_sources()}")

    os.makedirs(args.results_dir, exist_ok=True)
    run_dir, results_csv, results_summary_csv = prepare_results_paths(
        args.results_dir,
        "lz4_param_sweep.csv",
        "lz4_param_sweep_config_summary.csv",
    )
    print(f"[Results] run_dir={run_dir}")
    samples_root, samples = resolve_samples_root(args.samples)
    with open(run_dir / "run_meta.txt", "w", encoding="utf-8") as mf:
        mf.write(f"argv={' '.join(sys.argv)}\n")
        mf.write(f"cwd={os.getcwd()}\n")
        mf.write(f"bench_seconds={args.bench_seconds}\n")
        mf.write(f"samples={args.samples}\n")
        mf.write(f"resolved_samples={samples_root}\n")

    if args.single_file:
        samples = resolve_single_sample(args.single_file, str(samples_root), samples)

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
                "File", "FreqPoint", "CPUFreqTarget_MHz" if use_mhz_mode else "CPUFreqTargetPct", "GPUFreqTarget_MHz" if use_mhz_mode else "GPUFreqTargetPct",
                "Engine", "Threads_LSZ", "BlockSize", "HashLog", "Acceleration", "Ratio%",
                "CompMBs", "DecMBs", "CompTotalMBs", "DecTotalMBs",
                "CompTime_s", "DecTime_s",
                "CPUFreqAvgKernel_MHz", "GPUFreqAvgKernel_MHz",
                "CompCPUEnergy_J", "CompGPUEnergy_J", "CompCPUPower_W", "CompGPUPower_W",
                "Roundtrip_OK"
            ])

            if use_mhz_mode:
                point_idx = 0

                if run_cpu:
                    print(f"[Phase 1/3] CPU-only freq sweep ({len(cpu_freq_mhz_points)} points)")
                    for cpu_freq_target in cpu_freq_mhz_points:
                        gpu_freq_target = None
                        point_idx += 1
                        cpu_freq_apply = apply_freq_mhz(CPU_CONTROL_SCRIPT, cpu_freq_target)
                        print(f"[FreqPoint {point_idx}] CPU={cpu_freq_target}MHz apply={cpu_freq_apply}; GPU=N/A apply=skipped")

                        for sample in samples:
                            sample_key = str(sample)
                            if sample_key not in hash_cache:
                                hash_cache[sample_key] = compute_sha256(sample_key)
                            orig_hash = hash_cache[sample_key]

                            for t in cpu_threads:
                                for bs in cpu_block_sizes:
                                    cpu_stats = run_lz4_cpu(sample, bs, t, orig_hash, telemetry=telemetry, bench_seconds=args.bench_seconds)
                                    writer.writerow([
                                        sample.name,
                                        point_idx,
                                        "" if cpu_freq_target is None else cpu_freq_target,
                                        "" if gpu_freq_target is None else gpu_freq_target,
                                        "CPU", t, bs, "N/A", "N/A", fmtf(cpu_stats['ratio'], 2),
                                        fmtf(cpu_stats['comp_mbs'], 2), fmtf(cpu_stats['dec_mbs'], 2),
                                        fmtf(cpu_stats.get('comp_total_mbs', 0), 2),
                                        fmtf(cpu_stats.get('dec_total_mbs', 0), 2),
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

                if run_gpu:
                    print(f"[Phase 2/3] GPU-only freq sweep ({len(gpu_freq_mhz_points)} points)")
                    for gpu_freq_target in gpu_freq_mhz_points:
                        cpu_freq_target = None
                        point_idx += 1
                        gpu_freq_apply = apply_freq_mhz(GPU_CONTROL_SCRIPT, gpu_freq_target)
                        print(f"[FreqPoint {point_idx}] CPU=N/A apply=skipped; GPU={gpu_freq_target}MHz apply={gpu_freq_apply}")

                        for sample in samples:
                            sample_key = str(sample)
                            if sample_key not in hash_cache:
                                hash_cache[sample_key] = compute_sha256(sample_key)
                            orig_hash = hash_cache[sample_key]

                            for bs in gpu_block_sizes:
                                for lsz in local_sizes:
                                    for accel in gpu_accels:
                                        gpu_stats = run_lz4_gpu(sample, bs, lsz, accel, orig_hash, telemetry=telemetry, bench_seconds=args.bench_seconds, use_daemon=use_gpu_daemon)
                                        writer.writerow([
                                            sample.name,
                                            point_idx,
                                            "" if cpu_freq_target is None else cpu_freq_target,
                                            "" if gpu_freq_target is None else gpu_freq_target,
                                            "GPU", lsz, bs, 14, accel, fmtf(gpu_stats['ratio'], 2),
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

                                        gpu_cfg_label = f"FP={point_idx};BS={bs};LSZ={lsz};ACC={accel}"
                                        emit_case_average(sample.name, "GPU", gpu_cfg_label, gpu_stats)
                                        if gpu_stats.get("roundtrip_verified", False):
                                            summary_records.append(build_summary_record("GPU", gpu_cfg_label, gpu_stats))

                if run_hybrid:
                    if hybrid_freq_pairs:
                        hybrid_combos = hybrid_freq_pairs
                    else:
                        hybrid_combos = [(c, g) for c in cpu_freq_mhz_points for g in gpu_freq_mhz_points]
                    print(f"[Phase 3/3] Hybrid freq sweep ({len(hybrid_combos)} points)")
                    for cpu_freq_target, gpu_freq_target in hybrid_combos:
                        point_idx += 1
                        cpu_freq_apply = apply_freq_mhz(CPU_CONTROL_SCRIPT, cpu_freq_target)
                        gpu_freq_apply = apply_freq_mhz(GPU_CONTROL_SCRIPT, gpu_freq_target)
                        print(f"[FreqPoint {point_idx}] CPU={cpu_freq_target}MHz apply={cpu_freq_apply}; GPU={gpu_freq_target}MHz apply={gpu_freq_apply}")

                        for sample in samples:
                            sample_key = str(sample)
                            if sample_key not in hash_cache:
                                hash_cache[sample_key] = compute_sha256(sample_key)
                            orig_hash = hash_cache[sample_key]

                            for bs in hybrid_block_sizes:
                                for hlsz in hybrid_local_sizes:
                                    for split_mode in hybrid_split_modes:
                                        # Adaptive mode: scheduler decides ratio automatically, sweep is meaningless.
                                        # Use gpu_ratio=1.0 as upper-bound hint and let the algorithm adjust.
                                        effective_ratios = [1.0] if split_mode == "adaptive" else hybrid_gpu_ratios
                                        for ratio in effective_ratios:
                                            for ht in hybrid_cpu_threads:
                                                for accel in hybrid_accels:
                                                    hybrid_stats = run_lz4_hybrid(
                                                        sample,
                                                        bs,
                                                        ratio,
                                                        ht,
                                                        hlsz,
                                                        accel,
                                                        orig_hash,
                                                        telemetry=telemetry,
                                                        bench_seconds=args.bench_seconds,
                                                        split_mode=split_mode,
                                                    )
                                                    writer.writerow([
                                                        sample.name,
                                                        point_idx,
                                                        "" if cpu_freq_target is None else cpu_freq_target,
                                                        "" if gpu_freq_target is None else gpu_freq_target,
                                                        "HYBRID", f"{split_mode}:R{ratio}_T{ht}_L{hlsz}_A{accel}", bs, 14, accel, fmtf(hybrid_stats['ratio'], 2),
                                                        fmtf(hybrid_stats['comp_mbs'], 2), fmtf(hybrid_stats['dec_mbs'], 2),
                                                        fmtf(hybrid_stats.get('comp_total_mbs', 0), 2),
                                                        fmtf(hybrid_stats.get('dec_total_mbs', 0), 2),
                                                        fmtf(hybrid_stats['comp_time_s'], 6), fmtf(hybrid_stats['dec_time_s'], 6),
                                                        fmtf(hybrid_stats['cpu_freq_avg_mhz'], 2),
                                                        fmtf(hybrid_stats['gpu_freq_avg_mhz'], 2),
                                                        fmtf(hybrid_stats['cpu_energy_j'], 6),
                                                        fmtf(hybrid_stats['gpu_energy_j'], 6),
                                                        fmtf(hybrid_stats['comp_cpu_power_w'], 6),
                                                        fmtf(hybrid_stats['comp_gpu_power_w'], 6),
                                                        "yes" if hybrid_stats.get('roundtrip_verified') else "no",
                                                    ])
                                                    f.flush()

                                                    hybrid_cfg_label = f"FP={point_idx};BS={bs};M={split_mode};R={ratio};T={ht};LSZ={hlsz};ACC={accel}"
                                                    emit_case_average(sample.name, "HYBRID", hybrid_cfg_label, hybrid_stats)
                                                    if hybrid_stats.get("roundtrip_verified", False):
                                                        summary_records.append(build_summary_record("HYBRID", hybrid_cfg_label, hybrid_stats))
                                                    if (not run_gpu and split_mode == "fixed" and abs(float(ratio) - 1.0) < 1e-9):
                                                        emit_gpu_row_from_hybrid(sample, point_idx, cpu_freq_target, gpu_freq_target, bs, hlsz, accel, hybrid_stats)
            else:
                freq_combos = [(fp, fp) for fp in freq_points]

                for point_idx, (cpu_freq_target, gpu_freq_target) in enumerate(freq_combos, start=1):
                    cpu_freq_apply = apply_freq_percent(CPU_CONTROL_SCRIPT, cpu_freq_target)
                    gpu_freq_apply = apply_freq_percent(GPU_CONTROL_SCRIPT, gpu_freq_target)
                    print(f"[FreqPoint {point_idx}] CPU={cpu_freq_target}% apply={cpu_freq_apply}; GPU={gpu_freq_target}% apply={gpu_freq_apply}")

                    for sample in samples:
                        sample_key = str(sample)
                        if sample_key not in hash_cache:
                            hash_cache[sample_key] = compute_sha256(sample_key)
                        orig_hash = hash_cache[sample_key]

                        # CPU Sweep: block size isn't always supported by CPU tool; convert and use benchmark flag
                        if run_cpu:
                            for t in cpu_threads:
                                for bs in cpu_block_sizes:
                                    cpu_stats = run_lz4_cpu(sample, bs, t, orig_hash, telemetry=telemetry, bench_seconds=args.bench_seconds)
                                    writer.writerow([
                                        sample.name,
                                        point_idx,
                                        "" if cpu_freq_target is None else cpu_freq_target,
                                        "" if gpu_freq_target is None else gpu_freq_target,
                                        "CPU", t, bs, "N/A", "N/A", fmtf(cpu_stats['ratio'], 2),
                                        fmtf(cpu_stats['comp_mbs'], 2), fmtf(cpu_stats['dec_mbs'], 2),
                                        fmtf(cpu_stats.get('comp_total_mbs', 0), 2),
                                        fmtf(cpu_stats.get('dec_total_mbs', 0), 2),
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
                        if run_gpu:
                            for bs in gpu_block_sizes:
                                for lsz in local_sizes:
                                    for accel in gpu_accels:
                                        gpu_stats = run_lz4_gpu(sample, bs, lsz, accel, orig_hash, telemetry=telemetry, bench_seconds=args.bench_seconds, use_daemon=use_gpu_daemon)
                                        writer.writerow([
                                            sample.name,
                                            point_idx,
                                            "" if cpu_freq_target is None else cpu_freq_target,
                                            "" if gpu_freq_target is None else gpu_freq_target,
                                            "GPU", lsz, bs, 14, accel, fmtf(gpu_stats['ratio'], 2),
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

                                        gpu_cfg_label = f"FP={point_idx};BS={bs};LSZ={lsz};ACC={accel}"
                                        emit_case_average(sample.name, "GPU", gpu_cfg_label, gpu_stats)
                                        if gpu_stats.get("roundtrip_verified", False):
                                            summary_records.append(build_summary_record("GPU", gpu_cfg_label, gpu_stats))

                        if run_hybrid:
                             for bs in hybrid_block_sizes:
                                for hlsz in hybrid_local_sizes:
                                    for split_mode in hybrid_split_modes:
                                        # Adaptive mode: scheduler decides ratio automatically, sweep is meaningless.
                                        # Use gpu_ratio=1.0 as upper-bound hint and let the algorithm adjust.
                                        effective_ratios = [1.0] if split_mode == "adaptive" else hybrid_gpu_ratios
                                        for ratio in effective_ratios:
                                            for ht in hybrid_cpu_threads:
                                                for accel in hybrid_accels:
                                                    hybrid_stats = run_lz4_hybrid(
                                                        sample,
                                                        bs,
                                                        ratio,
                                                        ht,
                                                        hlsz,
                                                        accel,
                                                        orig_hash,
                                                        telemetry=telemetry,
                                                        bench_seconds=args.bench_seconds,
                                                        split_mode=split_mode,
                                                    )
                                                    writer.writerow([
                                                        sample.name,
                                                        point_idx,
                                                        "" if cpu_freq_target is None else cpu_freq_target,
                                                        "" if gpu_freq_target is None else gpu_freq_target,
                                                        "HYBRID", f"{split_mode}:R{ratio}_T{ht}_L{hlsz}_A{accel}", bs, 14, accel, fmtf(hybrid_stats['ratio'], 2),
                                                        fmtf(hybrid_stats['comp_mbs'], 2), fmtf(hybrid_stats['dec_mbs'], 2),
                                                        fmtf(hybrid_stats.get('comp_total_mbs', 0), 2),
                                                        fmtf(hybrid_stats.get('dec_total_mbs', 0), 2),
                                                        fmtf(hybrid_stats['comp_time_s'], 6), fmtf(hybrid_stats['dec_time_s'], 6),
                                                        fmtf(hybrid_stats['cpu_freq_avg_mhz'], 2),
                                                        fmtf(hybrid_stats['gpu_freq_avg_mhz'], 2),
                                                        fmtf(hybrid_stats['cpu_energy_j'], 6),
                                                        fmtf(hybrid_stats['gpu_energy_j'], 6),
                                                        fmtf(hybrid_stats['comp_cpu_power_w'], 6),
                                                        fmtf(hybrid_stats['comp_gpu_power_w'], 6),
                                                        "yes" if hybrid_stats.get('roundtrip_verified') else "no",
                                                    ])
                                                    f.flush()

                                                    hybrid_cfg_label = f"FP={point_idx};BS={bs};M={split_mode};R={ratio};T={ht};LSZ={hlsz};ACC={accel}"
                                                    emit_case_average(sample.name, "HYBRID", hybrid_cfg_label, hybrid_stats)
                                                    if hybrid_stats.get("roundtrip_verified", False):
                                                        summary_records.append(build_summary_record("HYBRID", hybrid_cfg_label, hybrid_stats))
                                                    if (not run_gpu and split_mode == "fixed" and abs(float(ratio) - 1.0) < 1e-9):
                                                        emit_gpu_row_from_hybrid(sample, point_idx, cpu_freq_target, gpu_freq_target, bs, hlsz, accel, hybrid_stats)

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

        cpu_reset = run_control_action(CPU_CONTROL_SCRIPT, 'reset')
        gpu_reset = run_control_action(GPU_CONTROL_SCRIPT, 'reset')
        print(f"[Cleanup] CPU reset={cpu_reset}; GPU reset={gpu_reset}")

if __name__ == "__main__":
    main()
