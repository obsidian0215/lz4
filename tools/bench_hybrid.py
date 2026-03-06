#!/usr/bin/env python3
import csv
import itertools
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hw_telemetry import TelemetryProbe, apply_freq_percent


LZ4_HYBRID_BIN = "/root/lz4/lz4_hybrid/lz4_hybrid"
SAMPLES_DIR = "/root/samples"
OUT_DIR = "/root/lz4/exp_results/hybrid_bench"

BLOCK_SIZE = "16K"
HASH_LOG = 14
LOCAL_SIZE = 1
BENCH_SECONDS = 3

GPU_RATIOS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
CPU_THREADS = [1, 2]
ACCELS = [1, 3]

COMP_RE = re.compile(
    r"Bench\s+Compress\s*:\s*kernel_tp=([0-9]+(?:\.[0-9]+)?)\s*MB/s\s+"
    r"total_tp=([0-9]+(?:\.[0-9]+)?)\s*MB/s\s+ratio=([0-9]+(?:\.[0-9]+)?)%",
    re.IGNORECASE,
)
DEC_RE = re.compile(
    r"Bench\s+Decompress\s*:\s*kernel_tp=([0-9]+(?:\.[0-9]+)?)\s*MB/s\s+"
    r"total_tp=([0-9]+(?:\.[0-9]+)?)\s*MB/s\s+verify=(OK|FAIL)",
    re.IGNORECASE,
)


def run_command_with_telemetry(cmd, telemetry, sample_interval_s=0.05):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
    completed = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    return completed, delta


def parse_output(stdout_text):
    comp = COMP_RE.search(stdout_text or "")
    dec = DEC_RE.search(stdout_text or "")
    if not comp or not dec:
        return None
    return {
        "comp_kernel_tp": float(comp.group(1)),
        "comp_total_tp": float(comp.group(2)),
        "ratio_pct": float(comp.group(3)),
        "dec_kernel_tp": float(dec.group(1)),
        "dec_total_tp": float(dec.group(2)),
        "verify_ok": dec.group(3).upper() == "OK",
    }


def mean_or_zero(values):
    vals = [float(v) for v in values if v is not None]
    return float(statistics.fmean(vals)) if vals else 0.0


def print_summary(all_rows):
    grouped = defaultdict(list)
    for row in all_rows:
        key = (row["GPURatio"], row["CPUThreads"], row["Acceleration"])
        grouped[key].append(row)

    print("\n=== Mean kernel throughput/ratio per config (across all files) ===")
    print("GPURatio CPUThreads Accel  MeanCompKernel(MB/s) MeanDecKernel(MB/s) MeanRatio(%)  N")

    summary_rows = []
    for key in sorted(grouped.keys()):
        rows = grouped[key]
        mean_comp = mean_or_zero([r["CompKernelTP_MBs"] for r in rows])
        mean_dec = mean_or_zero([r["DecKernelTP_MBs"] for r in rows])
        mean_ratio = mean_or_zero([r["Ratio_pct"] for r in rows])
        summary_rows.append({
            "key": key,
            "mean_comp": mean_comp,
            "mean_dec": mean_dec,
            "mean_ratio": mean_ratio,
            "n": len(rows),
        })
        gr, t, a = key
        print(f"{gr:>7.1f} {t:>10d} {a:>5d} {mean_comp:>20.2f} {mean_dec:>19.2f} {mean_ratio:>11.2f} {len(rows):>3d}")

    valid = [s for s in summary_rows if s["n"] > 0]
    best_comp = max(valid, key=lambda x: x["mean_comp"])
    best_dec = max(valid, key=lambda x: x["mean_dec"])

    print("\n=== Best configs ===")
    bgr, bt, ba = best_comp["key"]
    dgr, dt, da = best_dec["key"]
    print(
        "Best compression: "
        f"gpu_ratio={bgr:.1f}, T={bt}, a={ba}, "
        f"mean_comp_kernel={best_comp['mean_comp']:.2f} MB/s"
    )
    print(
        "Best decompression: "
        f"gpu_ratio={dgr:.1f}, T={dt}, a={da}, "
        f"mean_dec_kernel={best_dec['mean_dec']:.2f} MB/s"
    )

    pure_gpu = [s for s in summary_rows if s["key"][0] == 1.0]
    pure_cpu = [s for s in summary_rows if s["key"][0] == 0.0 and s["key"][1] == 1]
    hybrids = [s for s in summary_rows if 0.0 < s["key"][0] < 1.0]

    best_pure_gpu = max(pure_gpu, key=lambda x: x["mean_comp"], default=None)
    best_pure_cpu = max(pure_cpu, key=lambda x: x["mean_comp"], default=None)
    best_hybrid = max(hybrids, key=lambda x: x["mean_comp"], default=None)

    def fmt_cfg(s):
        gr, t, a = s["key"]
        return (
            f"gpu_ratio={gr:.1f}, T={t}, a={a}, "
            f"comp={s['mean_comp']:.2f} MB/s, dec={s['mean_dec']:.2f} MB/s, ratio={s['mean_ratio']:.2f}%"
        )

    print("\n=== Pure GPU vs Pure CPU(T=1) vs Best Hybrid (by compression) ===")
    if best_pure_gpu:
        print(f"Pure GPU best:  {fmt_cfg(best_pure_gpu)}")
    else:
        print("Pure GPU best:  N/A")

    if best_pure_cpu:
        print(f"Pure CPU best:  {fmt_cfg(best_pure_cpu)}")
    else:
        print("Pure CPU best:  N/A")

    if best_hybrid:
        print(f"Best hybrid:    {fmt_cfg(best_hybrid)}")
    else:
        print("Best hybrid:    N/A")


def main():
    if not (os.path.isfile(LZ4_HYBRID_BIN) and os.access(LZ4_HYBRID_BIN, os.X_OK)):
        raise SystemExit(f"Missing or non-executable binary: {LZ4_HYBRID_BIN}")

    samples = sorted([p for p in Path(SAMPLES_DIR).iterdir() if p.is_file()])
    if not samples:
        raise SystemExit(f"No sample files found in {SAMPLES_DIR}")

    configs = list(itertools.product(GPU_RATIOS, CPU_THREADS, ACCELS))
    expected_rows = len(samples) * len(configs)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = Path(OUT_DIR) / f"hybrid_bench_{ts}.csv"
    latest_csv_path = Path(OUT_DIR) / "hybrid_bench_latest.csv"

    telemetry = TelemetryProbe()
    print(f"Telemetry sources: {telemetry.describe_sources()}")

    cpu_apply = apply_freq_percent("/root/lz4/tools/cpu_control.sh", 100)
    gpu_apply = apply_freq_percent("/root/lz4/tools/gpu_control.sh", 100)
    print(f"Frequency apply: CPU={cpu_apply}, GPU={gpu_apply}")

    csv_columns = [
        "File",
        "GPURatio",
        "CPUThreads",
        "Acceleration",
        "CompKernelTP_MBs",
        "CompTotalTP_MBs",
        "DecKernelTP_MBs",
        "DecTotalTP_MBs",
        "Ratio_pct",
        "VerifyOK",
        "CPUEnergy_J",
        "GPUEnergy_J",
    ]

    all_rows = []
    failures = 0

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()

            total_runs = expected_rows
            done = 0

            for sample in samples:
                for gpu_ratio, threads, accel in configs:
                    done += 1
                    cmd = [
                        LZ4_HYBRID_BIN,
                        "--bench", str(BENCH_SECONDS),
                        "-b", BLOCK_SIZE,
                        "-H", str(HASH_LOG),
                        "-l", str(LOCAL_SIZE),
                        "-a", str(accel),
                        "-T", str(threads),
                        "--gpu-ratio", str(gpu_ratio),
                        str(sample),
                    ]

                    res, tel_delta = run_command_with_telemetry(cmd, telemetry)
                    merged_output = (res.stdout or "") + "\n" + (res.stderr or "")
                    parsed = parse_output(merged_output)

                    row = {
                        "File": sample.name,
                        "GPURatio": f"{gpu_ratio:.1f}",
                        "CPUThreads": threads,
                        "Acceleration": accel,
                        "CompKernelTP_MBs": "",
                        "CompTotalTP_MBs": "",
                        "DecKernelTP_MBs": "",
                        "DecTotalTP_MBs": "",
                        "Ratio_pct": "",
                        "VerifyOK": "false",
                        "CPUEnergy_J": f"{float(tel_delta.get('cpu_energy_j', 0.0) or 0.0):.6f}",
                        "GPUEnergy_J": f"{float(tel_delta.get('gpu_energy_j', 0.0) or 0.0):.6f}",
                    }

                    if parsed is not None:
                        row["CompKernelTP_MBs"] = f"{parsed['comp_kernel_tp']:.2f}"
                        row["CompTotalTP_MBs"] = f"{parsed['comp_total_tp']:.2f}"
                        row["DecKernelTP_MBs"] = f"{parsed['dec_kernel_tp']:.2f}"
                        row["DecTotalTP_MBs"] = f"{parsed['dec_total_tp']:.2f}"
                        row["Ratio_pct"] = f"{parsed['ratio_pct']:.2f}"
                        row["VerifyOK"] = "true" if parsed["verify_ok"] and res.returncode == 0 else "false"
                    else:
                        failures += 1

                    writer.writerow(row)
                    f.flush()

                    all_rows.append({
                        "GPURatio": gpu_ratio,
                        "CPUThreads": threads,
                        "Acceleration": accel,
                        "CompKernelTP_MBs": float(row["CompKernelTP_MBs"]) if row["CompKernelTP_MBs"] else None,
                        "DecKernelTP_MBs": float(row["DecKernelTP_MBs"]) if row["DecKernelTP_MBs"] else None,
                        "Ratio_pct": float(row["Ratio_pct"]) if row["Ratio_pct"] else None,
                    })

                    if done % 25 == 0 or done == total_runs:
                        print(f"Progress: {done}/{total_runs} ({100.0 * done / total_runs:.1f}%)")

        with open(csv_path, "r", encoding="utf-8") as src, open(latest_csv_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())

        print(f"\nCSV written: {csv_path}")
        print(f"CSV latest:  {latest_csv_path}")
        print(f"Samples: {len(samples)}, configs/file: {len(configs)}, expected data rows: {expected_rows}")
        print(f"Parse failures: {failures}")

        print_summary(all_rows)

    finally:
        subprocess.run(["/root/lz4/tools/cpu_control.sh", "reset"], capture_output=True, text=True, check=False)
        subprocess.run(["/root/lz4/tools/gpu_control.sh", "reset"], capture_output=True, text=True, check=False)
        print("Frequency reset: CPU/GPU scripts invoked with 'reset'.")


if __name__ == "__main__":
    main()
