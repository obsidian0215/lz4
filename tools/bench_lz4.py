#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
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
BASELINE_DIR = REPO_ROOT / "exp_results" / "baseline"

# Configuration Space
CPU_BLOCK_SIZES = ["64K", "1M"]
GPU_BLOCK_SIZES = ["32K", "64K"]
CPU_THREADS = [1, 2, 4]
LOCAL_SIZES = [1]
GPU_ACCELS = [1, 2]
HYBRID_BLOCK_SIZES = ["64K", "256K"]
HYBRID_GPU_RATIOS = [0.0, 0.3, 0.5, 0.7, 1.0]
HYBRID_CPU_THREADS = [1, 2, 4]
HYBRID_SPLIT_MODES = ["adaptive"]
HYBRID_SPLIT_LAYOUTS = ["prefix"]
HYBRID_LOCAL_SIZES = [1]
HYBRID_ACCELS = [1, 2]

# Default frequency configs (for intel iGPU)
DEFAULT_CPU_FREQ_MHZ = "800,1900,3000,5000"
DEFAULT_GPU_FREQ_MHZ = "500,1000,1500"

BASELINE_IDLE_PKG_W = None
BASELINE_IDLE_CORE_W = None
BASELINE_IDLE_GPU_W = None
_LZ4_HYBRID_SUPPORTS_SPLIT_LAYOUT = None

BASELINE_FILE_HEADER = [
    "File",
    "Rows",
    "CompTotalMean_MBps",
    "CompTotalMedian_MBps",
    "DecTotalMean_MBps",
    "DecTotalMedian_MBps",
    "RatioMean_pct",
    "RatioMedian_pct",
    "CompPowerMean_W",
    "DecPowerMean_W",
    "CompEffMean_MBpsPerW",
    "DecEffMean_MBpsPerW",
]

ENGINE_SUMMARY_HEADER = [
    "Engine",
    "Rows",
    "CompTotalMean_MBps",
    "CompTotalMedian_MBps",
    "DecTotalMean_MBps",
    "DecTotalMedian_MBps",
    "RatioMean_pct",
    "RatioMedian_pct",
    "CompPowerMean_W",
    "DecPowerMean_W",
    "CompEffMean_MBpsPerW",
    "DecEffMean_MBpsPerW",
]

ENGINE_VS_CPU_HEADER = [
    "File",
    "CPU_CompTotalMean",
    "CPU_DecTotalMean",
    "CPU_RatioMean",
    "GPU_CompDelta_pct",
    "GPU_DecDelta_pct",
    "GPU_RatioDelta_pctpt",
    "HYB_CompDelta_pct",
    "HYB_DecDelta_pct",
    "HYB_RatioDelta_pctpt",
]


def refresh_idle_baseline(telemetry, duration_s=2.0, label=""):
    global BASELINE_IDLE_PKG_W, BASELINE_IDLE_CORE_W, BASELINE_IDLE_GPU_W
    if telemetry is None:
        return
    sec = float(duration_s if duration_s and duration_s > 0 else 2.0)
    BASELINE_IDLE_PKG_W = telemetry.measure_idle_pkg_power_w(sec)
    BASELINE_IDLE_CORE_W = telemetry.measure_idle_core_power_w(sec)
    BASELINE_IDLE_GPU_W = telemetry.measure_idle_gpu_power_w(sec)
    tag = f"[{label}] " if label else ""
    print(
        f"[TelemetryBaseline] {tag}"
        f"cpu_pkg_idle={BASELINE_IDLE_PKG_W:.3f}W "
        f"cpu_core_idle={BASELINE_IDLE_CORE_W:.3f}W "
        f"gpu_idle={BASELINE_IDLE_GPU_W:.3f}W"
    )


def build_freq_target_env(cpu_freq_target, gpu_freq_target, use_mhz_mode):
    env = {}
    if use_mhz_mode:
        if cpu_freq_target is not None:
            env["HYBRID_CPU_FREQ_TARGET_MHZ"] = str(cpu_freq_target)
            env["LZ4_HYBRID_CPU_FREQ_TARGET_MHZ"] = str(cpu_freq_target)
        if gpu_freq_target is not None:
            env["HYBRID_GPU_FREQ_TARGET_MHZ"] = str(gpu_freq_target)
            env["LZ4_HYBRID_GPU_FREQ_TARGET_MHZ"] = str(gpu_freq_target)
    else:
        if cpu_freq_target is not None:
            env["HYBRID_CPU_FREQ_TARGET_PCT"] = str(cpu_freq_target)
            env["LZ4_HYBRID_CPU_FREQ_TARGET_PCT"] = str(cpu_freq_target)
        if gpu_freq_target is not None:
            env["HYBRID_GPU_FREQ_TARGET_PCT"] = str(gpu_freq_target)
            env["LZ4_HYBRID_GPU_FREQ_TARGET_PCT"] = str(gpu_freq_target)
    return env


def freq_cfg_prefix(cpu_freq_target, gpu_freq_target, use_mhz_mode):
    if use_mhz_mode:
        c = "NA" if cpu_freq_target is None else f"{cpu_freq_target}MHz"
        g = "NA" if gpu_freq_target is None else f"{gpu_freq_target}MHz"
    else:
        c = "NA" if cpu_freq_target is None else f"{cpu_freq_target}%"
        g = "NA" if gpu_freq_target is None else f"{gpu_freq_target}%"
    return f"CF={c};GF={g}"


class _DropFreqPointWriter:
    def __init__(self, base_writer):
        self._base = base_writer

    def writerow(self, row):
        data = list(row)
        if len(data) > 1:
            del data[1]
        self._base.writerow(data)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)



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
        "dec_cpu_energy_j",
        "dec_gpu_energy_j",
        "comp_cpu_power_w",
        "comp_gpu_power_w",
        "dec_cpu_power_w",
        "dec_gpu_power_w",
        "comp_eff_mbps_per_w",
        "dec_eff_mbps_per_w",
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
        "dec_cpu_energy_j",
        "dec_gpu_energy_j",
        "comp_cpu_power_w",
        "comp_gpu_power_w",
        "dec_cpu_power_w",
        "dec_gpu_power_w",
        "comp_eff_mbps_per_w",
        "dec_eff_mbps_per_w",
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


def print_split_layout_summary(summary_records):
    by_layout = {}
    for rec in summary_records:
        if rec.get("engine") != "HYBRID":
            continue
        cfg = str(rec.get("config", ""))
        m = re.search(r"(?:^|;)SL=([^;]+)", cfg)
        if not m:
            continue
        layout = m.group(1)
        by_layout.setdefault(layout, {"comp": [], "dec": []})
        c = rec.get("comp_total")
        d = rec.get("dec_total")
        if c is not None and float(c) > 0:
            by_layout[layout]["comp"].append(float(c))
        if d is not None and float(d) > 0:
            by_layout[layout]["dec"].append(float(d))

    if not by_layout:
        return

    print("\n===== Hybrid Split Layout Summary =====")
    for layout in sorted(by_layout.keys()):
        comp_mean = safe_mean(by_layout[layout]["comp"]) if by_layout[layout]["comp"] else 0.0
        dec_mean = safe_mean(by_layout[layout]["dec"]) if by_layout[layout]["dec"] else 0.0
        print(f"[SplitLayoutSummary] layout={layout} comp_total_mean={comp_mean:.2f} MB/s dec_total_mean={dec_mean:.2f} MB/s")

    if "prefix" in by_layout and "striped" in by_layout:
        p_comp = safe_mean(by_layout["prefix"]["comp"]) if by_layout["prefix"]["comp"] else 0.0
        s_comp = safe_mean(by_layout["striped"]["comp"]) if by_layout["striped"]["comp"] else 0.0
        p_dec = safe_mean(by_layout["prefix"]["dec"]) if by_layout["prefix"]["dec"] else 0.0
        s_dec = safe_mean(by_layout["striped"]["dec"]) if by_layout["striped"]["dec"] else 0.0
        comp_uplift = ((p_comp / s_comp) - 1.0) * 100.0 if s_comp > 0 else 0.0
        dec_uplift = ((p_dec / s_dec) - 1.0) * 100.0 if s_dec > 0 else 0.0
        print(f"[SplitLayoutSummary] prefix_vs_striped comp_uplift={comp_uplift:.2f}% dec_uplift={dec_uplift:.2f}%")


def fmtf(v, digits):
    if v is None or v == "":
        return ""
    return f"{float(v):.{digits}f}"


def prepare_results_paths(results_root, csv_name, summary_name):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_root) / "runs" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, str(run_dir / csv_name), str(run_dir / summary_name)


def path_is_under(path, ancestor):
    try:
        Path(path).resolve().relative_to(Path(ancestor).resolve())
        return True
    except Exception:
        return False


def _to_float(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _fmt6(value):
    return f"{float(value):.6f}"


def _roundtrip_yes(row):
    return str(row.get("Roundtrip_OK", "")).strip().lower() == "yes"


def _read_csv_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _hybrid_mode_of_row(row):
    token = str(row.get("Threads_LSZ", "") or "").strip().lower()
    if token.startswith("adaptive:") or "adaptive" in token:
        return "adaptive"
    return "fixed"


def _select_engine_rows(rows, engine, hybrid_mode=None):
    out = []
    for row in rows:
        if str(row.get("Engine", "")).strip().upper() != engine:
            continue
        if engine == "HYBRID" and hybrid_mode:
            if _hybrid_mode_of_row(row) != hybrid_mode:
                continue
        out.append(row)
    return out


def _prefer_roundtrip_rows(rows):
    yes_rows = [r for r in rows if _roundtrip_yes(r)]
    if yes_rows:
        return yes_rows, "roundtrip_yes"
    return rows, "fallback_all_rows"


def _aggregate_baseline_records(rows):
    per_file = {}
    for row in rows:
        file_name = str(row.get("File", "") or "").strip()
        if not file_name:
            continue
        per_file.setdefault(file_name, []).append(row)

    records = []
    for file_name in sorted(per_file.keys()):
        fr = per_file[file_name]
        comp_total = [_to_float(r.get("CompTotalMBs")) for r in fr]
        dec_total = [_to_float(r.get("DecTotalMBs")) for r in fr]
        ratio_vals = [_to_float(r.get("Ratio%")) for r in fr]
        comp_power = [_to_float(r.get("CompCPUPower_W")) for r in fr]
        dec_power = [_to_float(r.get("DecCPUPower_W")) for r in fr]
        comp_eff = [_to_float(r.get("CompEff_MBpsPerW")) for r in fr]
        dec_eff = [_to_float(r.get("DecEff_MBpsPerW")) for r in fr]

        comp_total = [v for v in comp_total if v is not None]
        dec_total = [v for v in dec_total if v is not None]
        ratio_vals = [v for v in ratio_vals if v is not None]
        comp_power = [v for v in comp_power if v is not None]
        dec_power = [v for v in dec_power if v is not None]
        comp_eff = [v for v in comp_eff if v is not None]
        dec_eff = [v for v in dec_eff if v is not None]

        records.append({
            "File": file_name,
            "Rows": len(fr),
            "CompTotalMean_MBps": safe_mean(comp_total),
            "CompTotalMedian_MBps": safe_median(comp_total),
            "DecTotalMean_MBps": safe_mean(dec_total),
            "DecTotalMedian_MBps": safe_median(dec_total),
            "RatioMean_pct": safe_mean(ratio_vals),
            "RatioMedian_pct": safe_median(ratio_vals),
            "CompPowerMean_W": safe_mean(comp_power),
            "DecPowerMean_W": safe_mean(dec_power),
            "CompEffMean_MBpsPerW": safe_mean(comp_eff),
            "DecEffMean_MBpsPerW": safe_mean(dec_eff),
        })
    return records


def _write_baseline_records_csv(out_path, records):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(BASELINE_FILE_HEADER)
        for rec in records:
            writer.writerow([
                rec["File"],
                int(rec["Rows"]),
                _fmt6(rec["CompTotalMean_MBps"]),
                _fmt6(rec["CompTotalMedian_MBps"]),
                _fmt6(rec["DecTotalMean_MBps"]),
                _fmt6(rec["DecTotalMedian_MBps"]),
                _fmt6(rec["RatioMean_pct"]),
                _fmt6(rec["RatioMedian_pct"]),
                _fmt6(rec["CompPowerMean_W"]),
                _fmt6(rec["DecPowerMean_W"]),
                _fmt6(rec["CompEffMean_MBpsPerW"]),
                _fmt6(rec["DecEffMean_MBpsPerW"]),
            ])
    return len(records)


def _copy_file(src, dst):
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def _pct_delta(new_v, base_v):
    if base_v is None or abs(float(base_v)) < 1e-12:
        return 0.0
    return ((float(new_v) - float(base_v)) / float(base_v)) * 100.0


def _aggregate_engine_stats(rows):
    comp_total = [_to_float(r.get("CompTotalMBs")) for r in rows]
    dec_total = [_to_float(r.get("DecTotalMBs")) for r in rows]
    ratio_vals = [_to_float(r.get("Ratio%")) for r in rows]
    comp_power = [_to_float(r.get("CompCPUPower_W")) for r in rows]
    dec_power = [_to_float(r.get("DecCPUPower_W")) for r in rows]
    comp_eff = [_to_float(r.get("CompEff_MBpsPerW")) for r in rows]
    dec_eff = [_to_float(r.get("DecEff_MBpsPerW")) for r in rows]

    comp_total = [v for v in comp_total if v is not None]
    dec_total = [v for v in dec_total if v is not None]
    ratio_vals = [v for v in ratio_vals if v is not None]
    comp_power = [v for v in comp_power if v is not None]
    dec_power = [v for v in dec_power if v is not None]
    comp_eff = [v for v in comp_eff if v is not None]
    dec_eff = [v for v in dec_eff if v is not None]

    return {
        "Rows": len(rows),
        "CompTotalMean_MBps": safe_mean(comp_total),
        "CompTotalMedian_MBps": safe_median(comp_total),
        "DecTotalMean_MBps": safe_mean(dec_total),
        "DecTotalMedian_MBps": safe_median(dec_total),
        "RatioMean_pct": safe_mean(ratio_vals),
        "RatioMedian_pct": safe_median(ratio_vals),
        "CompPowerMean_W": safe_mean(comp_power),
        "DecPowerMean_W": safe_mean(dec_power),
        "CompEffMean_MBpsPerW": safe_mean(comp_eff),
        "DecEffMean_MBpsPerW": safe_mean(dec_eff),
    }


def _write_engine_summary_csv(out_path, engine_rows):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    order = ["CPU", "GPU", "HYBRID"]
    written = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ENGINE_SUMMARY_HEADER)
        for eng in order:
            rows = engine_rows.get(eng) or []
            if not rows:
                continue
            stats = _aggregate_engine_stats(rows)
            writer.writerow([
                eng,
                int(stats["Rows"]),
                _fmt6(stats["CompTotalMean_MBps"]),
                _fmt6(stats["CompTotalMedian_MBps"]),
                _fmt6(stats["DecTotalMean_MBps"]),
                _fmt6(stats["DecTotalMedian_MBps"]),
                _fmt6(stats["RatioMean_pct"]),
                _fmt6(stats["RatioMedian_pct"]),
                _fmt6(stats["CompPowerMean_W"]),
                _fmt6(stats["DecPowerMean_W"]),
                _fmt6(stats["CompEffMean_MBpsPerW"]),
                _fmt6(stats["DecEffMean_MBpsPerW"]),
            ])
            written += 1
    return written


def _per_file_means(rows):
    grouped = {}
    for row in rows:
        fn = str(row.get("File", "") or "").strip()
        if not fn:
            continue
        grouped.setdefault(fn, []).append(row)

    out = {}
    for fn, rr in grouped.items():
        comp_total = [_to_float(r.get("CompTotalMBs")) for r in rr]
        dec_total = [_to_float(r.get("DecTotalMBs")) for r in rr]
        ratio_vals = [_to_float(r.get("Ratio%")) for r in rr]
        comp_total = [v for v in comp_total if v is not None]
        dec_total = [v for v in dec_total if v is not None]
        ratio_vals = [v for v in ratio_vals if v is not None]
        out[fn] = {
            "comp_total_mean": safe_mean(comp_total),
            "dec_total_mean": safe_mean(dec_total),
            "ratio_mean": safe_mean(ratio_vals),
        }
    return out


def _write_engine_vs_cpu_file_summary(out_path, cpu_rows, gpu_rows, hybrid_rows):
    cpu_map = _per_file_means(cpu_rows)
    gpu_map = _per_file_means(gpu_rows) if gpu_rows else {}
    hyb_map = _per_file_means(hybrid_rows) if hybrid_rows else {}
    if not cpu_map:
        return 0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ENGINE_VS_CPU_HEADER)
        for fn in sorted(cpu_map.keys()):
            cpu = cpu_map[fn]
            gpu = gpu_map.get(fn)
            hyb = hyb_map.get(fn)
            writer.writerow([
                fn,
                _fmt6(cpu["comp_total_mean"]),
                _fmt6(cpu["dec_total_mean"]),
                _fmt6(cpu["ratio_mean"]),
                "" if gpu is None else _fmt6(_pct_delta(gpu["comp_total_mean"], cpu["comp_total_mean"])),
                "" if gpu is None else _fmt6(_pct_delta(gpu["dec_total_mean"], cpu["dec_total_mean"])),
                "" if gpu is None else _fmt6(gpu["ratio_mean"] - cpu["ratio_mean"]),
                "" if hyb is None else _fmt6(_pct_delta(hyb["comp_total_mean"], cpu["comp_total_mean"])),
                "" if hyb is None else _fmt6(_pct_delta(hyb["dec_total_mean"], cpu["dec_total_mean"])),
                "" if hyb is None else _fmt6(hyb["ratio_mean"] - cpu["ratio_mean"]),
            ])
            written += 1
    return written


def run_baseline_generation(strict_csv, history_tag=""):
    strict_csv_path = Path(strict_csv).resolve()
    if not strict_csv_path.exists():
        print(f"[Baseline] skip: strict csv not found: {strict_csv_path}")
        return

    rows = _read_csv_rows(strict_csv_path)
    if not rows:
        print(f"[Baseline] skip: strict csv empty: {strict_csv_path}")
        return

    strict_total_rows = len(rows)
    strict_roundtrip_yes_rows = sum(1 for r in rows if _roundtrip_yes(r))
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    tag = (history_tag or "").strip() or strict_csv_path.parent.name

    baseline_dir = BASELINE_DIR
    manifests_dir = baseline_dir / "manifests"
    history_root = baseline_dir / "history" / tag
    manifests_dir.mkdir(parents=True, exist_ok=True)
    history_root.mkdir(parents=True, exist_ok=True)

    outputs = []
    history_copies = []
    filter_modes = {}

    def register_output(key, path_obj, records_count):
        p = Path(path_obj).resolve()
        outputs.append({
            "key": key,
            "path": str(p),
            "sha256": compute_sha256(str(p)),
            "rows": int(records_count),
        })

        if path_is_under(p, baseline_dir):
            rel = p.relative_to(baseline_dir.resolve())
            hist_path = history_root / rel
            _copy_file(p, hist_path)
            history_copies.append({"from": str(p), "to": str(hist_path.resolve())})

    cpu_rows_all = _select_engine_rows(rows, "CPU")
    gpu_rows_all = _select_engine_rows(rows, "GPU")
    hyb_rows_all = _select_engine_rows(rows, "HYBRID")
    hyb_fixed_all = _select_engine_rows(rows, "HYBRID", hybrid_mode="fixed")
    hyb_adaptive_all = _select_engine_rows(rows, "HYBRID", hybrid_mode="adaptive")

    cpu_rows_final = []
    gpu_rows_final = []
    hyb_rows_final = []

    if cpu_rows_all:
        cpu_rows_final, filter_modes["cpu_latest"] = _prefer_roundtrip_rows(cpu_rows_all)
        cpu_records = _aggregate_baseline_records(cpu_rows_final)
        cpu_latest = baseline_dir / "cpu" / "latest" / "baseline.csv"
        n = _write_baseline_records_csv(cpu_latest, cpu_records)
        register_output("cpu_latest", cpu_latest, n)

        cpu_compat = baseline_dir / "cpu_baseline.csv"
        _copy_file(cpu_latest, cpu_compat)
        register_output("cpu_compat", cpu_compat, n)

    if gpu_rows_all:
        gpu_rows_final, filter_modes["gpu_latest"] = _prefer_roundtrip_rows(gpu_rows_all)
        gpu_records = _aggregate_baseline_records(gpu_rows_final)
        gpu_latest = baseline_dir / "gpu" / "latest" / "baseline.csv"
        n = _write_baseline_records_csv(gpu_latest, gpu_records)
        register_output("gpu_latest", gpu_latest, n)

        gpu_compat = baseline_dir / "gpu_baseline.csv"
        _copy_file(gpu_latest, gpu_compat)
        register_output("gpu_compat", gpu_compat, n)

    if hyb_rows_all:
        hyb_rows_final, filter_modes["hybrid_latest"] = _prefer_roundtrip_rows(hyb_rows_all)
        hyb_records = _aggregate_baseline_records(hyb_rows_final)
        hyb_latest = baseline_dir / "hybrid" / "latest" / "baseline.csv"
        n = _write_baseline_records_csv(hyb_latest, hyb_records)
        register_output("hybrid_latest", hyb_latest, n)

        hyb_compat = baseline_dir / "hybrid_baseline.csv"
        _copy_file(hyb_latest, hyb_compat)
        register_output("hybrid_compat", hyb_compat, n)

    if hyb_fixed_all:
        hyb_fixed_rows, filter_modes["hybrid_fixed_latest"] = _prefer_roundtrip_rows(hyb_fixed_all)
        fixed_records = _aggregate_baseline_records(hyb_fixed_rows)
        fixed_latest = baseline_dir / "hybrid" / "fixed" / "latest" / "baseline.csv"
        n = _write_baseline_records_csv(fixed_latest, fixed_records)
        register_output("hybrid_fixed_latest", fixed_latest, n)

    if hyb_adaptive_all:
        hyb_adaptive_rows, filter_modes["hybrid_adaptive_latest"] = _prefer_roundtrip_rows(hyb_adaptive_all)
        adaptive_records = _aggregate_baseline_records(hyb_adaptive_rows)
        adaptive_latest = baseline_dir / "hybrid" / "adaptive" / "latest" / "baseline.csv"
        n = _write_baseline_records_csv(adaptive_latest, adaptive_records)
        register_output("hybrid_adaptive_latest", adaptive_latest, n)

    engine_rows = {
        "CPU": cpu_rows_final,
        "GPU": gpu_rows_final,
        "HYBRID": hyb_rows_final,
    }
    if any(engine_rows.values()):
        engine_summary = baseline_dir / "lz4_baseline_engine_summary.csv"
        written = _write_engine_summary_csv(engine_summary, engine_rows)
        if written > 0:
            register_output("engine_summary", engine_summary, written)

    if cpu_rows_final and (gpu_rows_final or hyb_rows_final):
        vs_cpu_summary = baseline_dir / "lz4_engine_vs_cpu_file_summary.csv"
        written = _write_engine_vs_cpu_file_summary(vs_cpu_summary, cpu_rows_final, gpu_rows_final, hyb_rows_final)
        if written > 0:
            register_output("engine_vs_cpu_summary", vs_cpu_summary, written)

    manifest = {
        "repo_root": str(REPO_ROOT.resolve()),
        "generated_at": timestamp,
        "history_tag": tag,
        "strict_csv": str(strict_csv_path),
        "strict_csv_sha256": compute_sha256(str(strict_csv_path)),
        "strict_total_rows": strict_total_rows,
        "strict_roundtrip_yes_rows": strict_roundtrip_yes_rows,
        "filters": filter_modes,
        "outputs": outputs,
        "history_copies": history_copies,
    }

    manifest_path = manifests_dir / f"baseline_update_{tag}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    latest_manifest_path = manifests_dir / "latest.json"
    _copy_file(manifest_path, latest_manifest_path)

    print(
        f"[Baseline] generated: tag={tag} strict_rows={strict_total_rows} "
        f"roundtrip_yes={strict_roundtrip_yes_rows} outputs={len(outputs)}"
    )
    print(f"[Baseline] manifest={manifest_path}")


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


def lz4_hybrid_supports_split_layout():
    global _LZ4_HYBRID_SUPPORTS_SPLIT_LAYOUT
    if _LZ4_HYBRID_SUPPORTS_SPLIT_LAYOUT is not None:
        return _LZ4_HYBRID_SUPPORTS_SPLIT_LAYOUT

    try:
        res = subprocess.run([LZ4_HYBRID_BIN, "--help"], capture_output=True, text=True, check=False)
        help_text = ((res.stdout or "") + "\n" + (res.stderr or "")).lower()
        _LZ4_HYBRID_SUPPORTS_SPLIT_LAYOUT = ("--split-prefix" in help_text) and ("--split-striped" in help_text)
    except Exception:
        _LZ4_HYBRID_SUPPORTS_SPLIT_LAYOUT = False

    return _LZ4_HYBRID_SUPPORTS_SPLIT_LAYOUT


def run_command_with_telemetry(cmd, telemetry=None, env=None, sample_interval_s=0.2):
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
        "cpu_pkg_avg_power_w": float(summary.get("cpu_pkg_avg_power_w", 0.0) or 0.0),
        "cpu_core_avg_power_w": float(summary.get("cpu_core_avg_power_w", 0.0) or 0.0),
        "gpu_avg_power_w": float(summary.get("gpu_avg_power_w", 0.0) or 0.0),
    }
    completed = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    return completed, tel


def run_command_with_telemetry_cwd(cmd, cwd=None, telemetry=None, env=None, sample_interval_s=0.2):
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
        "cpu_pkg_avg_power_w": float(summary.get("cpu_pkg_avg_power_w", 0.0) or 0.0),
        "cpu_core_avg_power_w": float(summary.get("cpu_core_avg_power_w", 0.0) or 0.0),
        "gpu_avg_power_w": float(summary.get("gpu_avg_power_w", 0.0) or 0.0),
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


def _derive_active_power(avg_power_w, energy_j, elapsed_s, peak_power_w):
    avg_v = float(avg_power_w or 0.0)
    if avg_v > 0.0:
        return avg_v
    energy_v = float(energy_j or 0.0)
    if energy_v > 0.0 and elapsed_s > 0.0:
        return energy_v / elapsed_s
    peak_v = float(peak_power_w or 0.0)
    if peak_v > 0.0:
        return peak_v * 0.6
    return 0.0


def apply_wall_energy(stats, tel_window, comp_elapsed_s, dec_elapsed_s, energy_source,
                      idle_pkg_power_w=0.0, idle_core_power_w=0.0, idle_gpu_power_w=0.0,
                      gpu_share_hint=0.0):
    stats['cpu_freq_avg_mhz'] = float(tel_window.get('cpu_freq_avg_mhz', 0.0) or 0.0)
    stats['gpu_freq_avg_mhz'] = float(tel_window.get('gpu_freq_avg_mhz', 0.0) or 0.0)
    elapsed_s = float(tel_window.get('elapsed_s', 0.0) or 0.0)
    if elapsed_s <= 0.0:
        stats['cpu_energy_j'] = 0.0
        stats['gpu_energy_j'] = 0.0
        stats['dec_cpu_energy_j'] = 0.0
        stats['dec_gpu_energy_j'] = 0.0
        stats['comp_cpu_power_w'] = 0.0
        stats['comp_gpu_power_w'] = 0.0
        stats['dec_cpu_power_w'] = None
        stats['dec_gpu_power_w'] = None
        stats['comp_eff_mbps_per_w'] = 0.0
        stats['dec_eff_mbps_per_w'] = 0.0
        stats['energy_source'] = energy_source
        return

    pkg_total_energy = float(tel_window.get('cpu_energy_j', 0.0) or 0.0)
    core_total_energy = float(tel_window.get('core_energy_j', 0.0) or 0.0)
    gpu_total_energy = float(tel_window.get('gpu_energy_j', 0.0) or 0.0)

    pkg_peak_power_w = float(tel_window.get('cpu_pkg_peak_power_w', 0.0) or 0.0)
    core_peak_power_w = float(tel_window.get('cpu_core_peak_power_w', 0.0) or 0.0)
    gpu_peak_power_w = float(tel_window.get('gpu_peak_power_w', 0.0) or 0.0)

    pkg_avg_power_w = float(tel_window.get('cpu_pkg_avg_power_w', 0.0) or 0.0)
    core_avg_power_w = float(tel_window.get('cpu_core_avg_power_w', 0.0) or 0.0)
    gpu_avg_power_w = float(tel_window.get('gpu_avg_power_w', 0.0) or 0.0)

    pkg_active_power_w = _derive_active_power(pkg_avg_power_w, pkg_total_energy, elapsed_s, pkg_peak_power_w)
    core_active_power_w = _derive_active_power(core_avg_power_w, core_total_energy, elapsed_s, core_peak_power_w)
    gpu_active_power_w = _derive_active_power(gpu_avg_power_w, gpu_total_energy, elapsed_s, gpu_peak_power_w)

    has_pkg = (pkg_total_energy > 0.0) or (pkg_avg_power_w > 0.0) or (pkg_peak_power_w > 0.0)
    has_core = (core_total_energy > 0.0) or (core_avg_power_w > 0.0) or (core_peak_power_w > 0.0)

    gpu_hint = _clamp01(gpu_share_hint)
    inferred_gpu_active_w = 0.0
    if pkg_active_power_w > 0.0 and core_active_power_w > 0.0:
        inferred_gpu_active_w = max(0.0, pkg_active_power_w - core_active_power_w)

    if has_pkg and has_core:
        if gpu_active_power_w <= 0.0:
            if inferred_gpu_active_w > 0.0:
                gpu_active_power_w = inferred_gpu_active_w
            elif gpu_hint > 0.0:
                gpu_active_power_w = pkg_active_power_w * min(0.40, max(0.10, gpu_hint * 0.45))

        if pkg_active_power_w > 0.0:
            max_gpu_w = pkg_active_power_w * 0.45
            if gpu_active_power_w > max_gpu_w:
                gpu_active_power_w = max_gpu_w

        cpu_active_power_w = max(core_active_power_w, pkg_active_power_w - gpu_active_power_w)
        cpu_idle_power_w = float(idle_core_power_w or idle_pkg_power_w or 0.0)
        inferred_gpu_idle_w = max(0.0, float(idle_pkg_power_w or 0.0) - float(idle_core_power_w or 0.0))
        gpu_idle_power_w = float(idle_gpu_power_w if idle_gpu_power_w and idle_gpu_power_w > 0.0 else inferred_gpu_idle_w)
        cpu_peak_power_w = core_peak_power_w if core_peak_power_w > 0.0 else max(pkg_peak_power_w - gpu_peak_power_w, cpu_active_power_w)
    elif has_pkg:
        if gpu_active_power_w <= 0.0 and gpu_hint > 0.0:
            gpu_active_power_w = pkg_active_power_w * min(0.40, max(0.10, gpu_hint * 0.45))
        if pkg_active_power_w > 0.0 and gpu_active_power_w > pkg_active_power_w * 0.45:
            gpu_active_power_w = pkg_active_power_w * 0.45
        cpu_active_power_w = max(0.0, pkg_active_power_w - gpu_active_power_w)
        cpu_idle_power_w = float(idle_pkg_power_w or idle_core_power_w or 0.0)
        gpu_idle_power_w = float(idle_gpu_power_w or 0.0)
        cpu_peak_power_w = pkg_peak_power_w if pkg_peak_power_w > 0.0 else cpu_active_power_w
    elif has_core:
        if gpu_active_power_w <= 0.0 and gpu_hint > 0.0:
            gpu_active_power_w = core_active_power_w * min(0.35, max(0.08, gpu_hint * 0.40))
        if gpu_active_power_w > core_active_power_w * 0.90:
            gpu_active_power_w = core_active_power_w * 0.90
        cpu_active_power_w = max(core_active_power_w, gpu_active_power_w * 1.05)
        cpu_idle_power_w = float(idle_core_power_w or idle_pkg_power_w or 0.0)
        gpu_idle_power_w = float(idle_gpu_power_w or 0.0)
        cpu_peak_power_w = core_peak_power_w if core_peak_power_w > 0.0 else cpu_active_power_w
    else:
        cpu_active_power_w = 0.0
        cpu_idle_power_w = float(idle_pkg_power_w or idle_core_power_w or 0.0)
        gpu_idle_power_w = float(idle_gpu_power_w or 0.0)
        cpu_peak_power_w = 0.0

    if gpu_hint <= 0.01:
        gpu_active_power_w = 0.0

    if cpu_active_power_w > 0.0 and gpu_active_power_w >= cpu_active_power_w:
        gpu_active_power_w = cpu_active_power_w * 0.90

    if gpu_peak_power_w <= 0.0 and pkg_peak_power_w > 0.0 and core_peak_power_w > 0.0:
        gpu_peak_power_w = max(0.0, pkg_peak_power_w - core_peak_power_w)
    if gpu_peak_power_w <= 0.0:
        gpu_peak_power_w = gpu_active_power_w

    cpu_dynamic_power_w = max(0.0, cpu_active_power_w - cpu_idle_power_w)
    gpu_dynamic_power_w = max(0.0, gpu_active_power_w - gpu_idle_power_w)
    cpu_export_power_w = max(cpu_active_power_w, cpu_idle_power_w)
    gpu_export_power_w = min(
        max(gpu_active_power_w if gpu_active_power_w > 0.0 else gpu_dynamic_power_w, 0.0),
        cpu_export_power_w * 0.95 if cpu_export_power_w > 0.0 else float("inf")
    )

    comp_phase_s = max(0.0, float(comp_elapsed_s or 0.0))
    dec_phase_s = max(0.0, float(dec_elapsed_s or 0.0))
    if comp_phase_s <= 0.0 and dec_phase_s <= 0.0:
        comp_phase_s = elapsed_s

    # 兼容现有 CSV 字段：保留列名不变，但承载 idle/peak 功率
    stats['cpu_energy_j'] = cpu_idle_power_w
    stats['gpu_energy_j'] = gpu_idle_power_w
    stats['dec_cpu_energy_j'] = cpu_peak_power_w
    stats['dec_gpu_energy_j'] = gpu_peak_power_w

    # 对外功率字段：动态功率优先；当前无法稳定区分 dec phase 时，回填 comp power。
    stats['comp_cpu_power_w'] = cpu_export_power_w
    stats['comp_gpu_power_w'] = gpu_export_power_w
    stats['dec_cpu_power_w'] = cpu_export_power_w
    stats['dec_gpu_power_w'] = gpu_export_power_w
    stats['cpu_peak_power_w'] = cpu_peak_power_w
    stats['gpu_peak_power_w'] = gpu_peak_power_w
    stats['cpu_idle_power_w'] = cpu_idle_power_w
    stats['gpu_idle_power_w'] = gpu_idle_power_w
    stats['cpu_active_power_w'] = cpu_active_power_w
    stats['gpu_active_power_w'] = gpu_active_power_w

    comp_total_power = max(0.0, cpu_export_power_w + gpu_export_power_w)
    dec_total_power = comp_total_power
    comp_total_mbs = float(stats.get('comp_total_mbs', 0.0) or 0.0)
    dec_total_mbs = float(stats.get('dec_total_mbs', 0.0) or 0.0)
    stats['comp_eff_mbps_per_w'] = (comp_total_mbs / comp_total_power) if comp_total_power > 0.0 and comp_total_mbs > 0.0 else 0.0
    stats['dec_eff_mbps_per_w'] = (dec_total_mbs / dec_total_power) if dec_total_power > 0.0 and dec_total_mbs > 0.0 else 0.0
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


def apply_total_throughput_from_elapsed(stats, input_bytes, comp_elapsed_s, dec_elapsed_s):
    in_mb = (float(input_bytes) / (1024.0 * 1024.0)) if input_bytes and input_bytes > 0 else 0.0
    c_elapsed = float(comp_elapsed_s or 0.0)
    d_elapsed = float(dec_elapsed_s or 0.0)

    if in_mb > 0.0 and c_elapsed > 0.0:
        stats['comp_time_s'] = c_elapsed
        stats['comp_total_mbs'] = in_mb / c_elapsed
    elif in_mb > 0.0 and float(stats.get('comp_total_mbs', 0.0) or 0.0) > 0.0:
        stats['comp_time_s'] = in_mb / float(stats['comp_total_mbs'])

    if in_mb > 0.0 and d_elapsed > 0.0:
        stats['dec_time_s'] = d_elapsed
        stats['dec_total_mbs'] = in_mb / d_elapsed
    elif in_mb > 0.0 and float(stats.get('dec_total_mbs', 0.0) or 0.0) > 0.0:
        stats['dec_time_s'] = in_mb / float(stats['dec_total_mbs'])


def parse_adaptive_bench_info(output):
    text = output or ""

    m = re.search(
        r"Bench\s+Adaptive\s*:\s*gpu_ratio=([0-9]+(?:\.[0-9]+)?)\s*objective=([^\s]+)",
        text,
        re.IGNORECASE,
    )
    if m:
        return {
            "gpu_ratio": float(m.group(1)),
            "objective": m.group(2),
        }

    m = re.search(
        r"Bench\s+Adaptive\s*:\s*gpu_ratio_mean=([0-9]+(?:\.[0-9]+)?)\s+min=([0-9]+(?:\.[0-9]+)?)\s+max=([0-9]+(?:\.[0-9]+)?)\s+samples=(\d+)",
        text,
        re.IGNORECASE,
    )
    if m:
        return {
            "gpu_ratio": float(m.group(1)),
            "objective": "perf_energy_ratio",
        }

    ratios = re.findall(r"effective_gpu_ratio=([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if ratios:
        return {
            "gpu_ratio": float(ratios[-1]),
            "objective": "runtime_trace",
        }

    return None


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


def run_lz4_cpu(file_path, bs, threads, orig_hash, telemetry=None, bench_seconds=3.0, run_env=None):
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
        'dec_cpu_power_w': 0.0,
        'dec_gpu_power_w': 0.0,
        'dec_cpu_energy_j': 0.0,
        'dec_gpu_energy_j': 0.0,
        'comp_eff_mbps_per_w': 0.0,
        'dec_eff_mbps_per_w': 0.0,
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
            if BASELINE_IDLE_PKG_W is not None:
                idle_pkg_power_w = float(BASELINE_IDLE_PKG_W or 0.0)
                idle_core_power_w = float(BASELINE_IDLE_CORE_W or idle_pkg_power_w)
                idle_gpu_power_w = float(BASELINE_IDLE_GPU_W or 0.0)
            else:
                idle_pkg_power_w = telemetry.measure_idle_pkg_power_w(0.2)
                idle_core_power_w = telemetry.measure_idle_core_power_w(0.2)
                idle_gpu_power_w = telemetry.measure_idle_gpu_power_w(0.2)

        # Use native lz4 bench output only (no temp file workflow)
        cmd_bench = [LZ4_BIN, "-q", f"-B{str(bs).upper()}", f"-T{exec_threads}", "-b1", sample_path]
        bench_res, tel_window = run_command_with_telemetry(cmd_bench, telemetry=telemetry, env=run_env)
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
            comp_elapsed_s = 0.0
            dec_elapsed_s = 0.0
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
                comp_total_res, _ = run_command_with_telemetry(cmd_comp_total, telemetry=None, env=run_env)
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
                dec_total_res, _ = run_command_with_telemetry(cmd_dec_total, telemetry=None, env=run_env)
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


def run_lz4_gpu(file_path, bs, lsz, accel, orig_hash, telemetry=None, bench_seconds=3.0, use_daemon=False, run_env=None):
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
        'dec_cpu_power_w': 0.0,
        'dec_gpu_power_w': 0.0,
        'dec_cpu_energy_j': 0.0,
        'dec_gpu_energy_j': 0.0,
        'comp_eff_mbps_per_w': 0.0,
        'dec_eff_mbps_per_w': 0.0,
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
            if BASELINE_IDLE_PKG_W is not None:
                idle_pkg_power_w = float(BASELINE_IDLE_PKG_W or 0.0)
                idle_core_power_w = float(BASELINE_IDLE_CORE_W or idle_pkg_power_w)
                idle_gpu_power_w = float(BASELINE_IDLE_GPU_W or 0.0)
            else:
                idle_pkg_power_w = telemetry.measure_idle_pkg_power_w(0.2)
                idle_core_power_w = telemetry.measure_idle_core_power_w(0.2)
                idle_gpu_power_w = telemetry.measure_idle_gpu_power_w(0.2)

        gpu_env = build_gpu_subprocess_env()
        if run_env:
            gpu_env.update(run_env)

        bench_cmd = [
            LZ4_GPU_BIN,
            "--bench", str(bench_seconds),
            "-B", bs_arg,
            "--local", str(lsz),
            "-a", str(accel),
            sample_path,
        ]
        bench_res, tel_window = run_command_with_telemetry_cwd(bench_cmd, cwd=gpu_dir, telemetry=telemetry, env=gpu_env)
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
            if in_sz > 0:
                if float(stats.get('comp_total_mbs', 0.0) or 0.0) > 0.0:
                    stats['comp_time_s'] = in_sz / (float(stats['comp_total_mbs']) * 1024.0 * 1024.0)
                if float(stats.get('dec_total_mbs', 0.0) or 0.0) > 0.0:
                    stats['dec_time_s'] = in_sz / (float(stats['dec_total_mbs']) * 1024.0 * 1024.0)

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
                comp_total_res, comp_total_tel = run_command_with_telemetry_cwd(cmd_comp_total, cwd=gpu_dir, telemetry=telemetry, env=gpu_env)

                cmd_dec_total = [LZ4_GPU_BIN]
                if use_daemon:
                    cmd_dec_total.append("--use-daemon")
                cmd_dec_total.extend([
                    "-v",
                    "-d",
                    "-o", tmp_dec,
                    tmp_comp,
                ])
                dec_total_res, dec_total_tel = run_command_with_telemetry_cwd(cmd_dec_total, cwd=gpu_dir, telemetry=telemetry, env=gpu_env)

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


def run_lz4_hybrid(file_path, bs, gpu_ratio, cpu_threads, local_size, accel, orig_hash, telemetry=None, bench_seconds=3.0, split_mode="adaptive", split_layout="prefix", sample_blocks=8, run_env=None):
    if split_mode == "adaptive":
        print(f"Bench_HYBRID: {file_path.name} BS={bs} mode={split_mode} layout={split_layout} T={cpu_threads} LSZ={local_size} A={accel}")
    else:
        print(f"Bench_HYBRID: {file_path.name} BS={bs} mode={split_mode} layout={split_layout} R={gpu_ratio} T={cpu_threads} LSZ={local_size} A={accel}")
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
        'dec_cpu_power_w': 0.0,
        'dec_gpu_power_w': 0.0,
        'dec_cpu_energy_j': 0.0,
        'dec_gpu_energy_j': 0.0,
        'comp_eff_mbps_per_w': 0.0,
        'dec_eff_mbps_per_w': 0.0,
        'adaptive_gpu_ratio': None,
        'adaptive_objective': '',
        'energy_source': 'none',
    }
    idle_pkg_power_w = 0.0
    idle_core_power_w = 0.0
    idle_gpu_power_w = 0.0
    try:
        in_sz = file_path.stat().st_size
        hybrid_dir = str(Path(LZ4_HYBRID_BIN).resolve().parent)
        supports_split_layout = lz4_hybrid_supports_split_layout()

        if telemetry:
            if BASELINE_IDLE_PKG_W is not None:
                idle_pkg_power_w = float(BASELINE_IDLE_PKG_W or 0.0)
                idle_core_power_w = float(BASELINE_IDLE_CORE_W or idle_pkg_power_w)
                idle_gpu_power_w = float(BASELINE_IDLE_GPU_W or 0.0)
            else:
                idle_pkg_power_w = telemetry.measure_idle_pkg_power_w(5)
                idle_core_power_w = telemetry.measure_idle_core_power_w(5)
                idle_gpu_power_w = telemetry.measure_idle_gpu_power_w(5)

        hybrid_env = build_gpu_subprocess_env()
        if run_env:
            hybrid_env.update(run_env)

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
            bench_cmd[1:1] = ["--adaptive", "--sample-blocks", str(sample_blocks)]
        else:
            bench_cmd[1:1] = ["--gpu-ratio", str(gpu_ratio)]
        if supports_split_layout:
            bench_cmd[1:1] = ["--split-striped" if split_layout == "striped" else "--split-prefix"]

        bench_res, _ = run_command_with_telemetry_cwd(bench_cmd, cwd=hybrid_dir, telemetry=telemetry, env=hybrid_env)
        bench_output = (bench_res.stdout or "") + (bench_res.stderr or "")
        stable = parse_stable_bench_output(bench_output)
        if stable:
            stats['ratio'] = stable['ratio']
            stats['comp_mbs'] = stable['comp_kernel_tp']
            stats['dec_mbs'] = stable['dec_kernel_tp']
            stats['comp_total_mbs'] = stable.get('comp_total_tp', 0.0)
            stats['dec_total_mbs'] = stable.get('dec_total_tp', 0.0)
            adaptive_info = parse_adaptive_bench_info(bench_output)
            if adaptive_info:
                stats['adaptive_gpu_ratio'] = adaptive_info['gpu_ratio']
                stats['adaptive_objective'] = adaptive_info['objective']
            elif split_mode == 'adaptive':
                stats['adaptive_gpu_ratio'] = 0.0
                stats['adaptive_objective'] = 'fallback_cpu_only_or_no_trace'
            if in_sz > 0:
                if float(stats.get('comp_total_mbs', 0.0) or 0.0) > 0.0:
                    stats['comp_time_s'] = in_sz / (float(stats['comp_total_mbs']) * 1024.0 * 1024.0)
                if float(stats.get('dec_total_mbs', 0.0) or 0.0) > 0.0:
                    stats['dec_time_s'] = in_sz / (float(stats['dec_total_mbs']) * 1024.0 * 1024.0)

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
                    cmd_comp_total[1:1] = ["--adaptive", "--sample-blocks", str(sample_blocks)]
                else:
                    cmd_comp_total[1:1] = ["--gpu-ratio", str(gpu_ratio)]
                if supports_split_layout:
                    cmd_comp_total[1:1] = ["--split-striped" if split_layout == "striped" else "--split-prefix"]
                comp_total_res, comp_total_tel = run_command_with_telemetry_cwd(cmd_comp_total, cwd=hybrid_dir, telemetry=telemetry, env=hybrid_env)

                cmd_dec_total = [
                    LZ4_HYBRID_BIN,
                    "-d",
                    "-o", tmp_dec,
                    tmp_comp,
                ]
                if split_mode == "adaptive":
                    cmd_dec_total[1:1] = ["--adaptive", "--sample-blocks", str(sample_blocks), "-T", str(cpu_threads)]
                else:
                    cmd_dec_total[1:1] = ["--gpu-ratio", str(gpu_ratio), "-T", str(cpu_threads)]
                if supports_split_layout:
                    cmd_dec_total[1:1] = ["--split-striped" if split_layout == "striped" else "--split-prefix"]
                dec_total_res, dec_total_tel = run_command_with_telemetry_cwd(cmd_dec_total, cwd=hybrid_dir, telemetry=telemetry, env=hybrid_env)

                comp_elapsed_s = float(comp_total_tel.get('elapsed_s', 0.0) or 0.0)
                dec_elapsed_s = float(dec_total_tel.get('elapsed_s', 0.0) or 0.0)

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
                if split_mode == 'adaptive':
                    adaptive_share = stats.get('adaptive_gpu_ratio')
                    if adaptive_share is None:
                        adaptive_share = 0.5
                else:
                    adaptive_share = gpu_ratio if gpu_ratio is not None else 0.0
                apply_wall_energy(
                    stats,
                    total_tel,
                    float(stats.get('comp_time_s', 0.0) or 0.0),
                    float(stats.get('dec_time_s', 0.0) or 0.0),
                    telemetry.describe_sources(),
                    idle_pkg_power_w=idle_pkg_power_w,
                    idle_core_power_w=idle_core_power_w,
                    idle_gpu_power_w=idle_gpu_power_w,
                    gpu_share_hint=float(adaptive_share),
                )
        else:
            stats['throughput_semantics'] = 'stable_kernel_bench_parse_failed'
            stats['roundtrip_verified'] = False
            print(f"  [HYBRID] stable bench parse failed for {file_path} (BS={bs} mode={split_mode} R={gpu_ratio} T={cpu_threads} LSZ={local_size} A={accel})", flush=True)
    except Exception as exc:
        print(f"HYBRID Error: {exc}")

    return stats

def main():
    global BASELINE_IDLE_PKG_W, BASELINE_IDLE_CORE_W, BASELINE_IDLE_GPU_W
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
    parser.add_argument('--hybrid-split-layouts', default=','.join(HYBRID_SPLIT_LAYOUTS), help='Hybrid split layouts, comma-separated (prefix,striped)')
    parser.add_argument('--hybrid-local-sizes', default=','.join(str(x) for x in HYBRID_LOCAL_SIZES), help='Hybrid local sizes, comma-separated')
    parser.add_argument('--hybrid-accels', default=','.join(str(x) for x in HYBRID_ACCELS), help='Hybrid acceleration factors, comma-separated')
    parser.add_argument('--freq-percent', type=int, default=None, help='Set both CPU and GPU to one shared frequency percent (0-100)')
    parser.add_argument('--freq-points', default='', help='Shared CPU/GPU frequency points, comma-separated (e.g. 40,70,100)')
    parser.add_argument('--cpu-freq-points', default=DEFAULT_CPU_FREQ_MHZ, help='CPU frequency points in MHz, comma-separated (default: %(default)s)')
    parser.add_argument('--gpu-freq-points', default=DEFAULT_GPU_FREQ_MHZ, help='GPU frequency points in MHz, comma-separated (default: %(default)s)')
    parser.add_argument('--hybrid-freq-pairs', default='', help='Hybrid CPU+GPU freq pairs in MHz, semicolon-separated (e.g. "800,300;800,1500;5000,300;5000,1500")')
    parser.add_argument('--no-freq-scan', action='store_true', help='Disable frequency sweep and run a single baseline point with default clocks')
    parser.add_argument('--single-file', default='', help='Only benchmark one file (path or basename under samples dir)')
    parser.add_argument('--no-telemetry', action='store_true', help='Disable freq/power telemetry collection')
    parser.add_argument('--bench-seconds', type=float, default=3.0, help='Benchmark duration in seconds for timed bench paths (default: 3.0)')
    parser.add_argument('--results-dir', default=RESULTS_DIR, help='Directory for benchmark outputs')
    parser.add_argument('--baseline', action='store_true', help='Generate baseline latest/history/manifests directly from this run\'s CSV')
    parser.add_argument('--baseline-history-tag', default='', help='Override history tag for --baseline snapshots')
    args = parser.parse_args()

    only_flags = sum([args.cpu_only, args.gpu_only, args.hybrid_only])
    if only_flags > 1:
        raise SystemExit('Cannot combine --cpu-only, --gpu-only, --hybrid-only')

    run_cpu = not args.gpu_only and not args.hybrid_only
    run_gpu = not args.cpu_only and not args.hybrid_only
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
            fmtf(hybrid_stats.get('dec_cpu_energy_j', 0), 6),
            fmtf(hybrid_stats.get('dec_gpu_energy_j', 0), 6),
            fmtf(hybrid_stats['comp_cpu_power_w'], 6),
            fmtf(hybrid_stats['comp_gpu_power_w'], 6),
            fmtf(hybrid_stats.get('dec_cpu_power_w', 0), 6),
            fmtf(hybrid_stats.get('dec_gpu_power_w', 0), 6),
            fmtf(hybrid_stats.get('comp_eff_mbps_per_w', 0), 6),
            fmtf(hybrid_stats.get('dec_eff_mbps_per_w', 0), 6),
            "",
            "",
            "yes" if hybrid_stats.get('roundtrip_verified') else "no",
        ])
        f.flush()

        gpu_cfg_label = f"{freq_cfg_prefix(cpu_freq_target, gpu_freq_target, use_mhz_mode)};BS={bs};LSZ={lsz};ACC={accel}"
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
    hybrid_split_layouts = parse_str_list(args.hybrid_split_layouts, HYBRID_SPLIT_LAYOUTS)
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

    if args.no_freq_scan:
        freq_points = [None]
        cpu_freq_mhz_points = [None]
        gpu_freq_mhz_points = [None]
        hybrid_freq_pairs = []
        print("[FreqControl] --no-freq-scan enabled: using single default clock point.")

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
        refresh_idle_baseline(telemetry, duration_s=3.0, label="startup")

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
            writer = _DropFreqPointWriter(csv.writer(f))
            writer.writerow([
                "File", "FreqPoint", "CPUFreqTarget_MHz" if use_mhz_mode else "CPUFreqTargetPct", "GPUFreqTarget_MHz" if use_mhz_mode else "GPUFreqTargetPct",
                "Engine", "Threads_LSZ", "BlockSize", "HashLog", "Acceleration", "Ratio%",
                "CompMBs", "DecMBs", "CompTotalMBs", "DecTotalMBs",
                "CompTime_s", "DecTime_s",
                "CPUFreqAvgKernel_MHz", "GPUFreqAvgKernel_MHz",
                "CPUIdlePower_W", "GPUIdlePower_W", "CPUPeakPower_W", "GPUPeakPower_W",
                "CompCPUPower_W", "CompGPUPower_W", "DecCPUPower_W", "DecGPUPower_W",
                "CompEff_MBpsPerW", "DecEff_MBpsPerW", "AdaptiveGpuRatio", "AdaptiveObjective",
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
                        point_env = build_freq_target_env(cpu_freq_target, gpu_freq_target, use_mhz_mode)
                        refresh_idle_baseline(telemetry, duration_s=2.0, label=f"CF={cpu_freq_target}MHz")
                        print(f"[FreqPoint {point_idx}] CPU={cpu_freq_target}MHz apply={cpu_freq_apply}; GPU=N/A apply=skipped")

                        for sample in samples:
                            sample_key = str(sample)
                            if sample_key not in hash_cache:
                                hash_cache[sample_key] = compute_sha256(sample_key)
                            orig_hash = hash_cache[sample_key]

                            for t in cpu_threads:
                                for bs in cpu_block_sizes:
                                    cpu_stats = run_lz4_cpu(sample, bs, t, orig_hash, telemetry=telemetry, bench_seconds=args.bench_seconds, run_env=point_env)
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
                                        fmtf(cpu_stats.get('dec_cpu_energy_j', 0), 6),
                                        fmtf(cpu_stats.get('dec_gpu_energy_j', 0), 6),
                                        fmtf(cpu_stats['comp_cpu_power_w'], 6),
                                        fmtf(cpu_stats['comp_gpu_power_w'], 6),
                                        fmtf(cpu_stats.get('dec_cpu_power_w', 0), 6),
                                        fmtf(cpu_stats.get('dec_gpu_power_w', 0), 6),
                                        fmtf(cpu_stats.get('comp_eff_mbps_per_w', 0), 6),
                                        fmtf(cpu_stats.get('dec_eff_mbps_per_w', 0), 6),
                                        "",
                                        "",
                                        "yes" if cpu_stats.get('roundtrip_verified') else "no",
                                    ])
                                    f.flush()

                                    cpu_cfg_label = f"{freq_cfg_prefix(cpu_freq_target, gpu_freq_target, use_mhz_mode)};BS={bs};T={t}"
                                    emit_case_average(sample.name, "CPU", cpu_cfg_label, cpu_stats)
                                    if cpu_stats.get("roundtrip_verified", False):
                                        summary_records.append(build_summary_record("CPU", cpu_cfg_label, cpu_stats))

                if run_gpu:
                    print(f"[Phase 2/3] GPU-only freq sweep ({len(gpu_freq_mhz_points)} points)")
                    for gpu_freq_target in gpu_freq_mhz_points:
                        cpu_freq_target = None
                        point_idx += 1
                        gpu_freq_apply = apply_freq_mhz(GPU_CONTROL_SCRIPT, gpu_freq_target)
                        point_env = build_freq_target_env(cpu_freq_target, gpu_freq_target, use_mhz_mode)
                        refresh_idle_baseline(telemetry, duration_s=2.0, label=f"GF={gpu_freq_target}MHz")
                        print(f"[FreqPoint {point_idx}] CPU=N/A apply=skipped; GPU={gpu_freq_target}MHz apply={gpu_freq_apply}")

                        for sample in samples:
                            sample_key = str(sample)
                            if sample_key not in hash_cache:
                                hash_cache[sample_key] = compute_sha256(sample_key)
                            orig_hash = hash_cache[sample_key]

                            for bs in gpu_block_sizes:
                                for lsz in local_sizes:
                                    for accel in gpu_accels:
                                        gpu_stats = run_lz4_gpu(sample, bs, lsz, accel, orig_hash, telemetry=telemetry, bench_seconds=args.bench_seconds, use_daemon=use_gpu_daemon, run_env=point_env)
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
                                            fmtf(gpu_stats.get('dec_cpu_energy_j', 0), 6),
                                            fmtf(gpu_stats.get('dec_gpu_energy_j', 0), 6),
                                            fmtf(gpu_stats['comp_cpu_power_w'], 6),
                                            fmtf(gpu_stats['comp_gpu_power_w'], 6),
                                            fmtf(gpu_stats.get('dec_cpu_power_w', 0), 6),
                                            fmtf(gpu_stats.get('dec_gpu_power_w', 0), 6),
                                            fmtf(gpu_stats.get('comp_eff_mbps_per_w', 0), 6),
                                            fmtf(gpu_stats.get('dec_eff_mbps_per_w', 0), 6),
                                            "",
                                            "",
                                            "yes" if gpu_stats.get('roundtrip_verified') else "no",
                                        ])
                                        f.flush()

                                        gpu_cfg_label = f"{freq_cfg_prefix(cpu_freq_target, gpu_freq_target, use_mhz_mode)};BS={bs};LSZ={lsz};ACC={accel}"
                                        emit_case_average(sample.name, "GPU", gpu_cfg_label, gpu_stats)
                                        if gpu_stats.get("roundtrip_verified", False):
                                            summary_records.append(build_summary_record("GPU", gpu_cfg_label, gpu_stats))

                if run_hybrid:
                    if hybrid_freq_pairs:
                        hybrid_combos = hybrid_freq_pairs
                    elif freq_points != [None]:
                        hybrid_combos = [(fp, fp) for fp in freq_points]
                    else:
                        hybrid_combos = [(c, g) for c in cpu_freq_mhz_points for g in gpu_freq_mhz_points]
                    print(f"[Phase 3/3] Hybrid freq sweep ({len(hybrid_combos)} points)")
                    for cpu_freq_target, gpu_freq_target in hybrid_combos:
                        point_idx += 1
                        if freq_points != [None]:
                            cpu_freq_apply = apply_freq_percent(CPU_CONTROL_SCRIPT, cpu_freq_target)
                            gpu_freq_apply = apply_freq_percent(GPU_CONTROL_SCRIPT, gpu_freq_target)
                            print(f"[FreqPoint {point_idx}] CPU={cpu_freq_target}% apply={cpu_freq_apply}; GPU={gpu_freq_target}% apply={gpu_freq_apply}")
                        else:
                            cpu_freq_apply = apply_freq_mhz(CPU_CONTROL_SCRIPT, cpu_freq_target)
                            gpu_freq_apply = apply_freq_mhz(GPU_CONTROL_SCRIPT, gpu_freq_target)
                            print(f"[FreqPoint {point_idx}] CPU={cpu_freq_target}MHz apply={cpu_freq_apply}; GPU={gpu_freq_target}MHz apply={gpu_freq_apply}")
                        point_env = build_freq_target_env(cpu_freq_target, gpu_freq_target, use_mhz_mode)
                        label_unit = "MHz" if use_mhz_mode else "%"
                        refresh_idle_baseline(telemetry, duration_s=2.0, label=f"CF={cpu_freq_target}{label_unit};GF={gpu_freq_target}{label_unit}")

                        for sample in samples:
                            sample_key = str(sample)
                            if sample_key not in hash_cache:
                                hash_cache[sample_key] = compute_sha256(sample_key)
                            orig_hash = hash_cache[sample_key]

                            for hlsz in hybrid_local_sizes:
                                for split_layout in hybrid_split_layouts:
                                    for split_mode in hybrid_split_modes:
                                        ratio_candidates = [None] if split_mode == "adaptive" else hybrid_gpu_ratios
                                        for ratio in ratio_candidates:
                                            bs_candidates = cpu_block_sizes if (split_mode == "fixed" and ratio is not None and abs(float(ratio)) < 1e-9) else hybrid_block_sizes
                                            for bs in bs_candidates:
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
                                                            split_layout=split_layout,
                                                            run_env=point_env,
                                                        )
                                                        ratio_label = "auto" if split_mode == "adaptive" else str(ratio)
                                                        writer.writerow([
                                                            sample.name,
                                                            point_idx,
                                                            "" if cpu_freq_target is None else cpu_freq_target,
                                                            "" if gpu_freq_target is None else gpu_freq_target,
                                                            "HYBRID", f"{split_mode}:{split_layout}:R{ratio_label}_T{ht}_L{hlsz}_A{accel}", bs, 14, accel, fmtf(hybrid_stats['ratio'], 2),
                                                            fmtf(hybrid_stats['comp_mbs'], 2), fmtf(hybrid_stats['dec_mbs'], 2),
                                                            fmtf(hybrid_stats.get('comp_total_mbs', 0), 2),
                                                            fmtf(hybrid_stats.get('dec_total_mbs', 0), 2),
                                                            fmtf(hybrid_stats['comp_time_s'], 6), fmtf(hybrid_stats['dec_time_s'], 6),
                                                            fmtf(hybrid_stats['cpu_freq_avg_mhz'], 2),
                                                            fmtf(hybrid_stats['gpu_freq_avg_mhz'], 2),
                                                            fmtf(hybrid_stats['cpu_energy_j'], 6),
                                                            fmtf(hybrid_stats['gpu_energy_j'], 6),
                                                            fmtf(hybrid_stats.get('dec_cpu_energy_j', 0), 6),
                                                            fmtf(hybrid_stats.get('dec_gpu_energy_j', 0), 6),
                                                            fmtf(hybrid_stats['comp_cpu_power_w'], 6),
                                                            fmtf(hybrid_stats['comp_gpu_power_w'], 6),
                                                            fmtf(hybrid_stats.get('dec_cpu_power_w', 0), 6),
                                                            fmtf(hybrid_stats.get('dec_gpu_power_w', 0), 6),
                                                            fmtf(hybrid_stats.get('comp_eff_mbps_per_w', 0), 6),
                                                            fmtf(hybrid_stats.get('dec_eff_mbps_per_w', 0), 6),
                                                            fmtf(hybrid_stats.get('adaptive_gpu_ratio'), 4),
                                                            hybrid_stats.get('adaptive_objective', ''),
                                                            "yes" if hybrid_stats.get('roundtrip_verified') else "no",
                                                        ])
                                                        f.flush()

                                                        hybrid_cfg_label = f"{freq_cfg_prefix(cpu_freq_target, gpu_freq_target, use_mhz_mode)};BS={bs};M={split_mode};SL={split_layout};R={ratio_label};T={ht};LSZ={hlsz};ACC={accel}"
                                                        emit_case_average(sample.name, "HYBRID", hybrid_cfg_label, hybrid_stats)
                                                        if hybrid_stats.get("roundtrip_verified", False):
                                                            summary_records.append(build_summary_record("HYBRID", hybrid_cfg_label, hybrid_stats))
                                                        if (not run_gpu and split_mode == "fixed" and split_layout == "prefix" and ratio is not None and abs(float(ratio) - 1.0) < 1e-9 and ht == hybrid_cpu_threads[0]):
                                                            emit_gpu_row_from_hybrid(sample, point_idx, cpu_freq_target, gpu_freq_target, bs, hlsz, accel, hybrid_stats)
            else:
                freq_combos = [(fp, fp) for fp in freq_points]

                for point_idx, (cpu_freq_target, gpu_freq_target) in enumerate(freq_combos, start=1):
                    cpu_freq_apply = apply_freq_percent(CPU_CONTROL_SCRIPT, cpu_freq_target)
                    gpu_freq_apply = apply_freq_percent(GPU_CONTROL_SCRIPT, gpu_freq_target)
                    point_env = build_freq_target_env(cpu_freq_target, gpu_freq_target, use_mhz_mode)
                    refresh_idle_baseline(telemetry, duration_s=2.0, label=f"CF={cpu_freq_target}%;GF={gpu_freq_target}%")
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
                                    cpu_stats = run_lz4_cpu(sample, bs, t, orig_hash, telemetry=telemetry, bench_seconds=args.bench_seconds, run_env=point_env)
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
                                        fmtf(cpu_stats.get('dec_cpu_energy_j', 0), 6),
                                        fmtf(cpu_stats.get('dec_gpu_energy_j', 0), 6),
                                        fmtf(cpu_stats['comp_cpu_power_w'], 6),
                                        fmtf(cpu_stats['comp_gpu_power_w'], 6),
                                        fmtf(cpu_stats.get('dec_cpu_power_w', 0), 6),
                                        fmtf(cpu_stats.get('dec_gpu_power_w', 0), 6),
                                        fmtf(cpu_stats.get('comp_eff_mbps_per_w', 0), 6),
                                        fmtf(cpu_stats.get('dec_eff_mbps_per_w', 0), 6),
                                        "",
                                        "",
                                        "yes" if cpu_stats.get('roundtrip_verified') else "no",
                                    ])
                                    f.flush()

                                    cpu_cfg_label = f"{freq_cfg_prefix(cpu_freq_target, gpu_freq_target, use_mhz_mode)};BS={bs};T={t}"
                                    emit_case_average(sample.name, "CPU", cpu_cfg_label, cpu_stats)
                                    if cpu_stats.get("roundtrip_verified", False):
                                        summary_records.append(build_summary_record("CPU", cpu_cfg_label, cpu_stats))

                        # GPU Sweep
                        if run_gpu:
                            for bs in gpu_block_sizes:
                                for lsz in local_sizes:
                                    for accel in gpu_accels:
                                        gpu_stats = run_lz4_gpu(sample, bs, lsz, accel, orig_hash, telemetry=telemetry, bench_seconds=args.bench_seconds, use_daemon=use_gpu_daemon, run_env=point_env)
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
                                            fmtf(gpu_stats.get('dec_cpu_energy_j', 0), 6),
                                            fmtf(gpu_stats.get('dec_gpu_energy_j', 0), 6),
                                            fmtf(gpu_stats['comp_cpu_power_w'], 6),
                                            fmtf(gpu_stats['comp_gpu_power_w'], 6),
                                            fmtf(gpu_stats.get('dec_cpu_power_w', 0), 6),
                                            fmtf(gpu_stats.get('dec_gpu_power_w', 0), 6),
                                            fmtf(gpu_stats.get('comp_eff_mbps_per_w', 0), 6),
                                            fmtf(gpu_stats.get('dec_eff_mbps_per_w', 0), 6),
                                            "",
                                            "",
                                            "yes" if gpu_stats.get('roundtrip_verified') else "no",
                                        ])
                                        f.flush()

                                        gpu_cfg_label = f"{freq_cfg_prefix(cpu_freq_target, gpu_freq_target, use_mhz_mode)};BS={bs};LSZ={lsz};ACC={accel}"
                                        emit_case_average(sample.name, "GPU", gpu_cfg_label, gpu_stats)
                                        if gpu_stats.get("roundtrip_verified", False):
                                            summary_records.append(build_summary_record("GPU", gpu_cfg_label, gpu_stats))

                        if run_hybrid:
                            for hlsz in hybrid_local_sizes:
                                for split_layout in hybrid_split_layouts:
                                    for split_mode in hybrid_split_modes:
                                        ratio_candidates = [None] if split_mode == "adaptive" else hybrid_gpu_ratios
                                        for ratio in ratio_candidates:
                                            bs_candidates = cpu_block_sizes if (split_mode == "fixed" and ratio is not None and abs(float(ratio)) < 1e-9) else hybrid_block_sizes
                                            for bs in bs_candidates:
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
                                                            split_layout=split_layout,
                                                            run_env=point_env,
                                                        )
                                                        ratio_label = "auto" if split_mode == "adaptive" else str(ratio)
                                                        writer.writerow([
                                                            sample.name,
                                                            point_idx,
                                                            "" if cpu_freq_target is None else cpu_freq_target,
                                                            "" if gpu_freq_target is None else gpu_freq_target,
                                                            "HYBRID", f"{split_mode}:{split_layout}:R{ratio_label}_T{ht}_L{hlsz}_A{accel}", bs, 14, accel, fmtf(hybrid_stats['ratio'], 2),
                                                            fmtf(hybrid_stats['comp_mbs'], 2), fmtf(hybrid_stats['dec_mbs'], 2),
                                                            fmtf(hybrid_stats.get('comp_total_mbs', 0), 2),
                                                            fmtf(hybrid_stats.get('dec_total_mbs', 0), 2),
                                                            fmtf(hybrid_stats['comp_time_s'], 6), fmtf(hybrid_stats['dec_time_s'], 6),
                                                            fmtf(hybrid_stats['cpu_freq_avg_mhz'], 2),
                                                            fmtf(hybrid_stats['gpu_freq_avg_mhz'], 2),
                                                            fmtf(hybrid_stats['cpu_energy_j'], 6),
                                                            fmtf(hybrid_stats['gpu_energy_j'], 6),
                                                            fmtf(hybrid_stats.get('dec_cpu_energy_j', 0), 6),
                                                            fmtf(hybrid_stats.get('dec_gpu_energy_j', 0), 6),
                                                            fmtf(hybrid_stats['comp_cpu_power_w'], 6),
                                                            fmtf(hybrid_stats['comp_gpu_power_w'], 6),
                                                            fmtf(hybrid_stats.get('dec_cpu_power_w', 0), 6),
                                                            fmtf(hybrid_stats.get('dec_gpu_power_w', 0), 6),
                                                            fmtf(hybrid_stats.get('comp_eff_mbps_per_w', 0), 6),
                                                            fmtf(hybrid_stats.get('dec_eff_mbps_per_w', 0), 6),
                                                            fmtf(hybrid_stats.get('adaptive_gpu_ratio'), 4),
                                                            hybrid_stats.get('adaptive_objective', ''),
                                                            "yes" if hybrid_stats.get('roundtrip_verified') else "no",
                                                        ])
                                                        f.flush()

                                                        hybrid_cfg_label = f"{freq_cfg_prefix(cpu_freq_target, gpu_freq_target, use_mhz_mode)};BS={bs};M={split_mode};SL={split_layout};R={ratio_label};T={ht};LSZ={hlsz};ACC={accel}"
                                                        emit_case_average(sample.name, "HYBRID", hybrid_cfg_label, hybrid_stats)
                                                        if hybrid_stats.get("roundtrip_verified", False):
                                                            summary_records.append(build_summary_record("HYBRID", hybrid_cfg_label, hybrid_stats))
                                                        if (not run_gpu and split_mode == "fixed" and split_layout == "prefix" and ratio is not None and abs(float(ratio) - 1.0) < 1e-9 and ht == hybrid_cpu_threads[0]):
                                                            emit_gpu_row_from_hybrid(sample, point_idx, cpu_freq_target, gpu_freq_target, bs, hlsz, accel, hybrid_stats)

        print_and_save_config_summary(summary_records, results_summary_csv)
        print_split_layout_summary(summary_records)
        if args.baseline:
            run_baseline_generation(results_csv, history_tag=args.baseline_history_tag)
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
