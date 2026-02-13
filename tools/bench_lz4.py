#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

# Paths
LZ4_BIN = "/root/lz4/lz4"
LZ4_GPU_BIN = "/root/lz4/lz4_gpu/lz4_gpu"
SAMPLES_DIR = "/root/samples"
RESULTS_DIR = "/root/lz4/exp_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "lz4_param_sweep.csv")

# Configuration Space
BLOCK_SIZES = ["16K", "32K", "64K", "256K", "1M"]
CPU_THREADS = [1, 2, 4]
HASH_LOGS = [12, 13, 14, 15, 16]
LOCAL_SIZES = [1, 8, 64]

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


def run_lz4_cpu(file_path, bs, threads, orig_hash):
    print(f"Bench_CPU: {file_path.name} BS={bs} T={threads}")
    bs_bytes = parse_size_to_bytes(bs)
    stats = {
        'ratio': 0,
        'comp_overall_tp': 0,
        'dec_overall_tp': 0,
        'comp_kernel_tp': None,
        'dec_kernel_tp': None,
        'throughput_semantics': 'cpu_overall_wallclock'
    }
    verified = False
    tmp_comp = None
    tmp_decomp = None
    try:
        tmp_comp = tempfile.NamedTemporaryFile(prefix="bench_lz4_cpu_", suffix=".lz4", delete=False)
        tmp_comp.close()
        t0 = time.time()
        subprocess.run([LZ4_BIN, "-f", "-1", f"-T{threads}", f"-B{bs_bytes}", str(file_path), tmp_comp.name], check=True, capture_output=True)
        t1 = time.time()
        in_sz = file_path.stat().st_size
        out_sz = os.path.getsize(tmp_comp.name)
        elapsed = max(1e-6, t1 - t0)
        stats['ratio'] = (out_sz / in_sz) * 100.0
        stats['comp_overall_tp'] = in_sz / elapsed / (1024.0 * 1024.0)

        tmp_decomp = tempfile.NamedTemporaryFile(prefix="bench_lz4_cpu_dec_", delete=False)
        tmp_decomp.close()
        t0 = time.time()
        subprocess.run([LZ4_BIN, "-f", "-d", tmp_comp.name, tmp_decomp.name], check=True, capture_output=True)
        t1 = time.time()
        elapsed = max(1e-6, t1 - t0)
        stats['dec_overall_tp'] = in_sz / elapsed / (1024.0 * 1024.0)
        verified = file_matches_hash(tmp_decomp.name, orig_hash)
        if not verified:
            print(f"  [CPU] Roundtrip mismatch for {file_path}", flush=True)
    except Exception as e:
        print(f"CPU error: {e}")
    finally:
        safe_remove(tmp_comp.name if tmp_comp else None)
        safe_remove(tmp_decomp.name if tmp_decomp else None)
    stats['roundtrip_verified'] = verified
    return stats


def run_lz4_gpu(file_path, bs, hl, lsz, orig_hash):
    print(f"Bench_GPU: {file_path.name} BS={bs} HL={hl} LSZ={lsz}")
    bs_arg = str(bs).lower()
    stats = {
        'ratio': 0,
        'comp_kernel_tp': 0,
        'dec_kernel_tp': 0,
        'comp_overall_tp': 0,
        'dec_overall_tp': 0,
        'throughput_semantics': 'gpu_kernel_and_inclusive',
        'roundtrip_verified': False
    }
    tmp_lz4 = None
    tmp_dec_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="bench_lz4_tmp_", suffix=".lz4", delete=False) as tf:
            tmp_lz4 = tf.name
        res_c = subprocess.run(
            [LZ4_GPU_BIN, "-v", "-b", bs_arg, "-H", str(hl), "-l", str(lsz), str(file_path), "-o", tmp_lz4],
            capture_output=True, text=True, check=False
        )
        stats_c = parse_gpu_output((res_c.stdout or "") + (res_c.stderr or ""))
        stats['comp_kernel_tp'] = stats_c.get('kernel_tp', 0)
        stats['comp_overall_tp'] = stats_c.get('inclusive_tp', 0)

        in_sz = file_path.stat().st_size
        out_sz = os.path.getsize(tmp_lz4) if tmp_lz4 and os.path.exists(tmp_lz4) else 0
        if in_sz > 0 and out_sz > 0:
            stats['ratio'] = (out_sz / in_sz) * 100.0

        tmp_dec = tempfile.NamedTemporaryFile(prefix="bench_lz4_gpu_dec_", suffix="", delete=False)
        tmp_dec.close()
        tmp_dec_path = tmp_dec.name
        cmd_d = [LZ4_GPU_BIN, "-v", "-d", tmp_lz4, "-o", tmp_dec.name, "-l", str(lsz), "-H", str(hl)]
        res_d = subprocess.run(cmd_d, capture_output=True, text=True, check=False)
        stats_d = parse_gpu_output((res_d.stdout or "") + (res_d.stderr or ""))
        stats['dec_kernel_tp'] = stats_d.get('kernel_tp', 0)
        stats['dec_overall_tp'] = stats_d.get('inclusive_tp', 0)

        verified = file_matches_hash(tmp_dec.name, orig_hash)
        stats['roundtrip_verified'] = verified
        if not verified:
            print(f"  [GPU] Roundtrip mismatch for {file_path} (BS={bs} HL={hl} LSZ={lsz})", flush=True)
    except Exception as exc:
        print(f"GPU error: {exc}")
    finally:
        safe_remove(tmp_dec_path)
        safe_remove(tmp_lz4)
    return stats

def main():
    parser = argparse.ArgumentParser(description='Bench LZ4 CPU/GPU sweep (supports --limit for quick runs)')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of samples (0 = all)')
    parser.add_argument('--samples', default=SAMPLES_DIR, help='Samples directory (default: /root/samples)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    samples = sorted([p for p in Path(args.samples).glob("*") if p.is_file()])

    if not samples:
        print(f"No samples found in {args.samples}")
        return

    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    hash_cache = {}

    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "File", "Engine", "Threads_LSZ", "BlockSize", "HashLog", "Ratio%",
            "CompOverall_MBs", "DecOverall_MBs", "CompKernel_MBs", "DecKernel_MBs",
            "ThroughputSemantics", "Roundtrip_OK"
        ])

        for sample in samples:
            sample_key = str(sample)
            if sample_key not in hash_cache:
                hash_cache[sample_key] = compute_sha256(sample_key)
            orig_hash = hash_cache[sample_key]

            # CPU Sweep: block size isn't always supported by CPU tool; convert and use benchmark flag
            for t in CPU_THREADS:
                for bs in BLOCK_SIZES:
                    cpu_stats = run_lz4_cpu(sample, bs, t, orig_hash)
                    writer.writerow([
                        sample.name, "CPU", t, bs, "N/A", f"{cpu_stats['ratio']:.2f}",
                        f"{cpu_stats['comp_overall_tp']:.2f}", f"{cpu_stats['dec_overall_tp']:.2f}",
                        "", "", cpu_stats.get('throughput_semantics', 'cpu_overall_wallclock'),
                        "yes" if cpu_stats.get('roundtrip_verified') else "no"
                    ])
                    f.flush()

            # GPU Sweep
            for bs in BLOCK_SIZES:
                for hl in HASH_LOGS:
                    for lsz in LOCAL_SIZES:
                        gpu_stats = run_lz4_gpu(sample, bs, hl, lsz, orig_hash)
                        writer.writerow([
                            sample.name, "GPU", lsz, bs, hl, f"{gpu_stats['ratio']:.2f}",
                            f"{gpu_stats['comp_overall_tp']:.2f}", f"{gpu_stats['dec_overall_tp']:.2f}",
                            f"{gpu_stats['comp_kernel_tp']:.2f}", f"{gpu_stats['dec_kernel_tp']:.2f}",
                            gpu_stats.get('throughput_semantics', 'gpu_kernel_and_inclusive'),
                            "yes" if gpu_stats.get('roundtrip_verified') else "no"
                        ])
                        f.flush()

if __name__ == "__main__":
    main()
