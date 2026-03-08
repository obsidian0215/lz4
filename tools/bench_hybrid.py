#!/usr/bin/env python3
import argparse
import csv
import hashlib
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
DEFAULT_SAMPLES_DIR = "/root/samples_subset"
OUT_DIR = "/root/lz4/exp_results/hybrid_bench"

BLOCK_SIZES = ["16K", "32K", "64K"]
HASH_LOG = 14
LOCAL_SIZE = 1
BENCH_SECONDS = 3

GPU_RATIOS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
CPU_THREADS = [1, 2]
ACCELS = [1, 3]
SPLIT_MODES = ["fixed", "adaptive"]


def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_matches_hash(path, expected_hash):
    if not path or not expected_hash or not os.path.exists(path):
        return False
    return compute_sha256(path) == expected_hash

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


def apply_wall_energy(tel_window, comp_elapsed_s, dec_elapsed_s):
    elapsed_s = float(tel_window.get('elapsed_s', 0.0) or 0.0)
    if elapsed_s <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    total_phase_s = max(0.0, float(comp_elapsed_s or 0.0)) + max(0.0, float(dec_elapsed_s or 0.0))
    if total_phase_s <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    phase_scale = min(1.0, total_phase_s / elapsed_s)
    comp_share = float(comp_elapsed_s or 0.0) / total_phase_s
    cpu_energy = float(tel_window.get('cpu_energy_j', 0.0) or 0.0) * phase_scale * comp_share
    gpu_energy = float(tel_window.get('gpu_energy_j', 0.0) or 0.0) * phase_scale * comp_share
    cpu_power = (cpu_energy / comp_elapsed_s) if comp_elapsed_s and comp_elapsed_s > 0.0 else 0.0
    gpu_power = (gpu_energy / comp_elapsed_s) if comp_elapsed_s and comp_elapsed_s > 0.0 else 0.0
    return cpu_energy, gpu_energy, cpu_power, gpu_power


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
        key = (row["BlockSize"], row["SplitMode"], row["GPURatio"], row["CPUThreads"], row["Acceleration"])
        grouped[key].append(row)

    print("\n=== Mean kernel throughput/ratio per config (across all files) ===")
    print("BlockSize Mode GPURatio CPUThreads Accel MeanCompKernel(MB/s) MeanDecKernel(MB/s) MeanRatio(%)  N")

    summary_rows = []
    for key in sorted(grouped.keys()):
        rows = grouped[key]
        mean_comp = mean_or_zero([r["CompKernelTP_MBs"] for r in rows])
        mean_dec = mean_or_zero([r["DecKernelTP_MBs"] for r in rows])
        mean_comp_total = mean_or_zero([r["CompTotalTP_MBs"] for r in rows])
        mean_dec_total = mean_or_zero([r["DecTotalTP_MBs"] for r in rows])
        mean_ratio = mean_or_zero([r["Ratio_pct"] for r in rows])
        summary_rows.append({
            "key": key,
            "mean_comp": mean_comp,
            "mean_dec": mean_dec,
            "mean_comp_total": mean_comp_total,
            "mean_dec_total": mean_dec_total,
            "mean_ratio": mean_ratio,
            "n": len(rows),
        })
        block_size, mode, gr, t, a = key
        print(f"{block_size:>8s} {mode:>8s} {gr:>7.1f} {t:>10d} {a:>5d} {mean_comp:>20.2f} {mean_dec:>19.2f} {mean_ratio:>11.2f} {len(rows):>3d}")

    valid = [s for s in summary_rows if s["n"] > 0]
    best_comp = max(valid, key=lambda x: x["mean_comp_total"])
    best_dec = max(valid, key=lambda x: x["mean_dec_total"])

    print("\n=== Best configs ===")
    bbs, bm, bgr, bt, ba = best_comp["key"]
    dbs, dm, dgr, dt, da = best_dec["key"]
    print(
        "Best compression: "
            f"block={bbs}, mode={bm}, gpu_ratio={bgr:.1f}, T={bt}, a={ba}, "
        f"mean_comp_total={best_comp['mean_comp_total']:.2f} MB/s"
    )
    print(
        "Best decompression: "
            f"block={dbs}, mode={dm}, gpu_ratio={dgr:.1f}, T={dt}, a={da}, "
        f"mean_dec_total={best_dec['mean_dec_total']:.2f} MB/s"
    )

    pure_gpu = [s for s in summary_rows if s["key"][2] == 1.0]
    pure_cpu = [s for s in summary_rows if s["key"][2] == 0.0 and s["key"][3] == 1]
    hybrids = [s for s in summary_rows if 0.0 < s["key"][2] < 1.0]

    best_pure_gpu = max(pure_gpu, key=lambda x: x["mean_comp_total"], default=None)
    best_pure_cpu = max(pure_cpu, key=lambda x: x["mean_comp_total"], default=None)
    best_hybrid = max(hybrids, key=lambda x: x["mean_comp_total"], default=None)

    def fmt_cfg(s):
        block_size, mode, gr, t, a = s["key"]
        return (
            f"block={block_size}, mode={mode}, gpu_ratio={gr:.1f}, T={t}, a={a}, "
            f"comp_total={s['mean_comp_total']:.2f} MB/s, dec_total={s['mean_dec_total']:.2f} MB/s, ratio={s['mean_ratio']:.2f}%"
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
    parser = argparse.ArgumentParser(description="Benchmark lz4_hybrid fixed/adaptive split modes")
    parser.add_argument("--samples", default=DEFAULT_SAMPLES_DIR, help=f"Samples directory (default: {DEFAULT_SAMPLES_DIR})")
    parser.add_argument("--bench-seconds", type=float, default=BENCH_SECONDS, help=f"Benchmark seconds per config (default: {BENCH_SECONDS})")
    args = parser.parse_args()

    if not (os.path.isfile(LZ4_HYBRID_BIN) and os.access(LZ4_HYBRID_BIN, os.X_OK)):
        raise SystemExit(f"Missing or non-executable binary: {LZ4_HYBRID_BIN}")

    samples_dir = Path(args.samples)
    samples = sorted([p for p in samples_dir.iterdir() if p.is_file()])
    if not samples:
        raise SystemExit(f"No sample files found in {samples_dir}")

    configs = list(itertools.product(BLOCK_SIZES, SPLIT_MODES, GPU_RATIOS, CPU_THREADS, ACCELS))
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
        "BlockSize",
        "SplitMode",
        "GPURatio",
        "CPUThreads",
        "Acceleration",
        "CompKernelTP_MBs",
        "CompTotalTP_MBs",
        "DecKernelTP_MBs",
        "DecTotalTP_MBs",
        "CompTime_s",
        "DecTime_s",
        "Ratio_pct",
        "VerifyOK",
        "CPUEnergy_J",
        "GPUEnergy_J",
        "CPUPower_W",
        "GPUPower_W",
    ]

    all_rows = []
    failures = 0

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()

            hash_cache = {}
            total_runs = expected_rows
            done = 0

            for sample in samples:
                sample_key = str(sample)
                if sample_key not in hash_cache:
                    hash_cache[sample_key] = compute_sha256(sample_key)
                orig_hash = hash_cache[sample_key]
                for block_size, split_mode, gpu_ratio, threads, accel in configs:
                    done += 1
                    bench_cmd = [
                        LZ4_HYBRID_BIN,
                        "--bench", str(args.bench_seconds),
                        "--bench-io",
                        "-b", block_size,
                        "-H", str(HASH_LOG),
                        "-l", str(LOCAL_SIZE),
                        "-a", str(accel),
                        "-T", str(threads),
                        "--gpu-ratio", str(gpu_ratio),
                        str(sample),
                    ]
                    if split_mode == "adaptive":
                        bench_cmd[1:1] = ["--adaptive", "--sample-blocks", "8"]

                    res, tel_delta = run_command_with_telemetry(bench_cmd, telemetry)
                    merged_output = (res.stdout or "") + "\n" + (res.stderr or "")
                    parsed = parse_output(merged_output)

                    comp_elapsed_s = 0.0
                    dec_elapsed_s = 0.0
                    cpu_energy_j = 0.0
                    gpu_energy_j = 0.0
                    cpu_power_w = 0.0
                    gpu_power_w = 0.0
                    total_ok = False

                    if parsed is not None:
                        in_sz_mb = sample.stat().st_size / (1024.0 * 1024.0)
                        if parsed['comp_total_tp'] > 0:
                            comp_elapsed_s = in_sz_mb / parsed['comp_total_tp']
                        if parsed['dec_total_tp'] > 0:
                            dec_elapsed_s = in_sz_mb / parsed['dec_total_tp']
                        total_ok = parsed["verify_ok"] and res.returncode == 0
                        cpu_energy_j, gpu_energy_j, cpu_power_w, gpu_power_w = apply_wall_energy(tel_delta, comp_elapsed_s, dec_elapsed_s)

                    row = {
                        "File": sample.name,
                        "BlockSize": block_size,
                        "SplitMode": split_mode,
                        "GPURatio": f"{gpu_ratio:.1f}",
                        "CPUThreads": threads,
                        "Acceleration": accel,
                        "CompKernelTP_MBs": "",
                        "CompTotalTP_MBs": "",
                        "DecKernelTP_MBs": "",
                        "DecTotalTP_MBs": "",
                        "CompTime_s": f"{comp_elapsed_s:.6f}",
                        "DecTime_s": f"{dec_elapsed_s:.6f}",
                        "Ratio_pct": "",
                        "VerifyOK": "false",
                        "CPUEnergy_J": f"{cpu_energy_j:.6f}",
                        "GPUEnergy_J": f"{gpu_energy_j:.6f}",
                        "CPUPower_W": f"{cpu_power_w:.6f}",
                        "GPUPower_W": f"{gpu_power_w:.6f}",
                    }

                    if parsed is not None:
                        row["CompKernelTP_MBs"] = f"{parsed['comp_kernel_tp']:.2f}"
                        row["CompTotalTP_MBs"] = f"{parsed['comp_total_tp']:.2f}"
                        row["DecKernelTP_MBs"] = f"{parsed['dec_kernel_tp']:.2f}"
                        row["DecTotalTP_MBs"] = f"{parsed['dec_total_tp']:.2f}"
                        row["Ratio_pct"] = f"{parsed['ratio_pct']:.2f}"
                        row["VerifyOK"] = "true" if parsed["verify_ok"] and res.returncode == 0 and total_ok else "false"
                    else:
                        failures += 1

                    writer.writerow(row)
                    f.flush()

                    all_rows.append({
                        "SplitMode": split_mode,
                        "BlockSize": block_size,
                        "GPURatio": gpu_ratio,
                        "CPUThreads": threads,
                        "Acceleration": accel,
                        "CompKernelTP_MBs": float(row["CompKernelTP_MBs"]) if row["CompKernelTP_MBs"] else None,
                        "CompTotalTP_MBs": float(row["CompTotalTP_MBs"]) if row["CompTotalTP_MBs"] else None,
                        "DecKernelTP_MBs": float(row["DecKernelTP_MBs"]) if row["DecKernelTP_MBs"] else None,
                        "DecTotalTP_MBs": float(row["DecTotalTP_MBs"]) if row["DecTotalTP_MBs"] else None,
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
