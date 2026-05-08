#!/usr/bin/env python3
import argparse
import csv
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tools.hw_telemetry import TelemetryProbe, apply_freq_mhz, apply_freq_percent

CPU_CONTROL = SCRIPT_DIR / "cpu_control.sh"
GPU_CONTROL = SCRIPT_DIR / "gpu_control.sh"
BENCH_LZ4 = SCRIPT_DIR / "bench_lz4.py"

DEFAULT_CPU_FREQ_MHZ = "1900,3500,NA"
DEFAULT_GPU_FREQ_MHZ = "1000,NA"


def parse_frequency_points(text):
    if text is None or str(text).strip() == "":
        return [{"kind": "none", "value": None, "label": "NA"}]
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item or item.upper() in ("NA", "NONE", "-"):
            out.append({"kind": "none", "value": None, "label": "NA"})
        elif item.endswith("%"):
            value = int(item[:-1].strip())
            if value < 0 or value > 100:
                raise SystemExit(f"invalid percent frequency point: {item}")
            out.append({"kind": "percent", "value": value, "label": f"{value}pct"})
        else:
            raw = item[:-3] if item.lower().endswith("mhz") else item
            value = int(raw.strip())
            if value < 1:
                raise SystemExit(f"invalid MHz frequency point: {item}")
            out.append({"kind": "mhz", "value": value, "label": f"{value}mhz"})
    return out or [{"kind": "none", "value": None, "label": "NA"}]


def run_control_reset(script):
    if os.name == "nt":
        return "unsupported_on_windows"
    cmd = [str(script), "reset"]
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return "ok"
        msg = ((res.stderr or "") + "\n" + (res.stdout or "")).strip().splitlines()
        return f"failed:{res.returncode}:{msg[0][:100] if msg else ''}"
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def apply_frequency(script, point):
    kind = point["kind"]
    value = point["value"]
    if kind == "none":
        return run_control_reset(script)
    if kind == "percent":
        return apply_freq_percent(str(script), value)
    if kind == "mhz":
        return apply_freq_mhz(str(script), value)
    return f"invalid_kind:{kind}"


def with_option(args_list, option, value=None):
    out = []
    skip_next = False
    for item in args_list:
        if skip_next:
            skip_next = False
            continue
        if item == option:
            if value is None:
                continue
            skip_next = True
            continue
        if item.startswith(option + "="):
            continue
        out.append(item)
    if value is None:
        out.append(option)
    else:
        out += [option, str(value)]
    return out


def bench_arg_value(args_list, option):
    for idx, item in enumerate(args_list):
        if item == option and idx + 1 < len(args_list):
            return args_list[idx + 1]
        if item.startswith(option + "="):
            return item.split("=", 1)[1]
    return None


def split_csv_text(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def discover_samples(samples_dir, limit, single_file):
    root = Path(samples_dir)
    if not root.exists():
        raise SystemExit(f"samples dir not found: {root}")
    files = sorted([p for p in root.iterdir() if p.is_file()])
    if not files:
        child_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        if len(child_dirs) == 1:
            files = sorted([p for p in child_dirs[0].iterdir() if p.is_file()])

    if single_file:
        direct = Path(single_file)
        if direct.is_file():
            return [direct]
        matches = [p for p in files if p.name == str(single_file)]
        if len(matches) != 1:
            raise SystemExit(f"single file not found or ambiguous: {single_file}")
        return matches

    if int(limit or 0) > 0:
        return files[: int(limit)]
    return files


def bench_samples_from_args(args_list):
    samples_dir = bench_arg_value(args_list, "--samples") or str(REPO_ROOT.parent / "samples")
    single_file = bench_arg_value(args_list, "--single-file") or ""
    limit_text = bench_arg_value(args_list, "--limit")
    try:
        limit = int(limit_text) if limit_text is not None and str(limit_text).strip() else 0
    except Exception:
        limit = 0
    return discover_samples(samples_dir, limit, single_file)


def effective_engines(args_list):
    text = bench_arg_value(args_list, "--engines")
    if text:
        return set(split_csv_text(text))
    cpu_only = "--cpu-only" in args_list
    gpu_only = "--gpu-only" in args_list
    hybrid_only = "--hybrid-only" in args_list
    if cpu_only:
        return {"native_cpu"}
    if gpu_only:
        return {"gpu"}
    if hybrid_only:
        return {"hybrid"}
    return {"gpu", "native_cpu"}


def effective_hybrid_ratios(args_list):
    text = bench_arg_value(args_list, "--hybrid-gpu-ratios")
    if not text:
        return ["0.3", "0.5", "0.7"]
    return split_csv_text(text)


def is_cpu_only_ratio(ratio):
    try:
        return float(ratio) <= 0.0
    except Exception:
        return False


def is_gpu_only_ratio(ratio):
    try:
        return float(ratio) >= 1.0
    except Exception:
        return False


def strip_engine_flags(args_list):
    out = list(args_list)
    for opt in ("--cpu-only", "--gpu-only", "--hybrid-only"):
        out = [x for x in out if x != opt]
    return out


def detect_run_modes(args_list):
    cpu_only = "--cpu-only" in args_list
    gpu_only = "--gpu-only" in args_list
    hybrid_only = "--hybrid-only" in args_list

    only_flags = sum([cpu_only, gpu_only, hybrid_only])
    if only_flags > 1:
        raise SystemExit("Cannot combine --cpu-only, --gpu-only, --hybrid-only in wrapper pass-through args")

    run_cpu = not gpu_only and not hybrid_only
    run_gpu = not cpu_only and not hybrid_only
    run_hybrid = (not cpu_only and not gpu_only) or hybrid_only
    return run_cpu, run_gpu, run_hybrid


def build_work_items(args, cpu_points, gpu_points):
    run_cpu, run_gpu, run_hybrid = detect_run_modes(args.bench_args)
    base = strip_engine_flags(args.bench_args)
    engines = effective_engines(base)
    hybrid_ratios = effective_hybrid_ratios(base)

    items = []
    seen = set()
    none_cpu = {"kind": "none", "value": None, "label": "NA"}
    none_gpu = {"kind": "none", "value": None, "label": "NA"}

    def add(label, cpu_point, gpu_point, bench_args):
        key = (label, cpu_point["label"], gpu_point["label"], tuple(bench_args))
        if key in seen:
            return
        seen.add(key)
        items.append({"label": label, "cpu": cpu_point, "gpu": gpu_point, "bench_args": bench_args})

    if run_cpu and "native_cpu" in engines:
        bench_args = with_option(with_option(base, "--engines", "native_cpu"), "--cpu-only", None)
        for cpu_point in cpu_points:
            add("cpu", cpu_point, none_gpu, bench_args)

    if run_gpu and "gpu" in engines:
        bench_args = with_option(with_option(base, "--engines", "gpu"), "--gpu-only", None)
        for gpu_point in gpu_points:
            add("gpu", none_cpu, gpu_point, bench_args)

    if run_hybrid and "hybrid" in engines:
        for ratio in hybrid_ratios:
            bench_args = with_option(with_option(with_option(base, "--engines", "hybrid"), "--hybrid-gpu-ratios", ratio), "--hybrid-only", None)
            if str(ratio).strip().lower() == "adaptive":
                for cpu_point in cpu_points:
                    for gpu_point in gpu_points:
                        add(f"hybrid_ratio_{ratio}", cpu_point, gpu_point, bench_args)
            elif is_cpu_only_ratio(ratio):
                for cpu_point in cpu_points:
                    add(f"hybrid_ratio_{ratio}", cpu_point, none_gpu, bench_args)
            elif is_gpu_only_ratio(ratio):
                for gpu_point in gpu_points:
                    add(f"hybrid_ratio_{ratio}", none_cpu, gpu_point, bench_args)
            else:
                for cpu_point in cpu_points:
                    for gpu_point in gpu_points:
                        add(f"hybrid_ratio_{ratio}", cpu_point, gpu_point, bench_args)

    return items


def run_bench(args, item):
    cpu_point = item["cpu"]
    gpu_point = item["gpu"]
    safe_label = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in item["label"])
    run_dir = Path(args.output_dir) / f"{safe_label}_cpu_{cpu_point['label']}_gpu_{gpu_point['label']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    bench_args = list(item["bench_args"])
    bench_args = with_option(bench_args, "--results-dir", str(run_dir))
    bench_args = with_option(bench_args, "--no-freq-scan", None)
    bench_args = with_option(bench_args, "--no-telemetry", None)

    cmd = [sys.executable, str(BENCH_LZ4)]
    cmd.extend(bench_args)

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc, cmd, run_dir, start


def collect_until_done(proc, telemetry, interval_s):
    samples = []
    lines = []
    stop = threading.Event()

    def sampler():
        while not stop.wait(interval_s):
            try:
                samples.append(telemetry.snapshot())
            except Exception:
                pass
            if proc.poll() is not None:
                break

    thread = threading.Thread(target=sampler, daemon=True)
    try:
        samples.append(telemetry.snapshot())
    except Exception:
        pass
    thread.start()
    for line in proc.stdout:
        lines.append(line)
        print(line, end="")
    rc = proc.wait()
    stop.set()
    thread.join(timeout=max(1.0, interval_s * 2.0))
    try:
        samples.append(telemetry.snapshot())
    except Exception:
        pass
    return rc, "".join(lines), samples


def summarize_samples(telemetry, samples):
    if not samples:
        return telemetry.summarize_samples([])
    return telemetry.summarize_samples(samples)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_per_file_summary(run_dir):
    run_path = Path(run_dir)
    candidates = sorted(run_path.rglob("per_file_summary.csv"))
    return candidates[-1] if candidates else None


def read_csv_rows(path):
    if not path or not Path(path).exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv):
    parser = argparse.ArgumentParser(
        description="External power/frequency wrapper for bench_lz4.py. Arguments after -- are passed to bench_lz4.py."
    )
    parser.add_argument("--platform-id", default=os.environ.get("LZ4_PLATFORM_ID", "local"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "exp_results" / "power_freq_runs" / time.strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--cpu-frequencies", "--cpu-points", default=DEFAULT_CPU_FREQ_MHZ, help='Comma list: "2100,3400,NA" or "50%,100%,NA"')
    parser.add_argument("--gpu-frequencies", "--gpu-points", default=DEFAULT_GPU_FREQ_MHZ, help='Comma list: "1000,NA" or "70%,NA"')
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--no-reset-before", action="store_true")
    parser.add_argument("bench_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.bench_args and args.bench_args[0] == "--":
        args.bench_args = args.bench_args[1:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry = TelemetryProbe()
    cpu_points = parse_frequency_points(args.cpu_frequencies)
    gpu_points = parse_frequency_points(args.gpu_frequencies)
    points = build_work_items(args, cpu_points, gpu_points)

    print(f"[wrapper] output_dir={output_dir}")
    print(f"[wrapper] telemetry={telemetry.describe_sources()}")
    print(f"[wrapper] points={[(p['label'], p['cpu']['label'], p['gpu']['label'], shlex.join(p['bench_args'])) for p in points]}")

    records = []
    per_file_records = []
    any_fail = False
    print("[wrapper] start")

    for idx, item in enumerate(points, start=1):
        cpu_point = item["cpu"]
        gpu_point = item["gpu"]
        rc = -1
        run_dir = ""
        elapsed = 0.0
        summary = {}

        try:
            if not args.no_reset_before:
                run_control_reset(CPU_CONTROL)
                run_control_reset(GPU_CONTROL)

            apply_frequency(CPU_CONTROL, cpu_point)
            apply_frequency(GPU_CONTROL, gpu_point)

            proc, cmd, run_dir_path, start = run_bench(args, item)
            rc, output, samples = collect_until_done(proc, telemetry, max(0.05, args.sample_interval))
            elapsed = max(0.0, time.perf_counter() - start)
            run_dir = str(run_dir_path)
            summary = summarize_samples(telemetry, samples)

            with open(run_dir_path / "wrapper_command.txt", "w", encoding="utf-8") as handle:
                handle.write(" ".join(str(x) for x in cmd) + "\n")
            with open(run_dir_path / "wrapper_output.log", "w", encoding="utf-8") as handle:
                handle.write(output)
        finally:
            run_control_reset(CPU_CONTROL)
            run_control_reset(GPU_CONTROL)

        record = {
            "workload": item["label"],
            "elapsed_s": elapsed,
            "run_dir": run_dir,
            "bench_args": shlex.join(item["bench_args"]),
            "telemetry_sources": telemetry.describe_sources(),
            "cpu_freq_avg_mhz": summary.get("cpu_freq_avg_mhz", 0.0),
            "gpu_freq_avg_mhz": summary.get("gpu_freq_avg_mhz", 0.0),
            "cpu_energy_j": summary.get("cpu_energy_j", 0.0),
            "core_energy_j": summary.get("core_energy_j", 0.0),
            "dram_energy_j": summary.get("dram_energy_j", 0.0),
            "gpu_energy_j": summary.get("gpu_energy_j", 0.0),
            "cpu_pkg_avg_power_w": summary.get("cpu_pkg_avg_power_w", 0.0),
            "cpu_core_avg_power_w": summary.get("cpu_core_avg_power_w", 0.0),
            "dram_avg_power_w": summary.get("dram_avg_power_w", 0.0),
            "gpu_avg_power_w": summary.get("gpu_avg_power_w", 0.0),
            "cpu_pkg_peak_power_w": summary.get("cpu_pkg_peak_power_w", 0.0),
            "cpu_core_peak_power_w": summary.get("cpu_core_peak_power_w", 0.0),
            "dram_peak_power_w": summary.get("dram_peak_power_w", 0.0),
            "gpu_peak_power_w": summary.get("gpu_peak_power_w", 0.0),
        }
        records.append(record)
        if rc != 0:
            any_fail = True
        write_csv(output_dir / "power_frequency_summary.csv", records, list(record.keys()))

        per_file_path = find_per_file_summary(run_dir)
        for pf in read_csv_rows(per_file_path):
            pf_record = {
                "workload": item["label"],
                "sample": pf.get("sample", ""),
                "engine": pf.get("engine", ""),
                "ratio_pct_median": pf.get("ratio_pct_median", ""),
                "manual_comp_no_ocl_mbs_mean": pf.get("manual_comp_no_ocl_mbs_mean", ""),
                "manual_dec_no_ocl_mbs_mean": pf.get("manual_dec_no_ocl_mbs_mean", ""),
                "elapsed_s": elapsed,
                "run_dir": run_dir,
                "cpu_freq_avg_mhz": summary.get("cpu_freq_avg_mhz", 0.0),
                "gpu_freq_avg_mhz": summary.get("gpu_freq_avg_mhz", 0.0),
                "cpu_energy_j": summary.get("cpu_energy_j", 0.0),
                "core_energy_j": summary.get("core_energy_j", 0.0),
                "dram_energy_j": summary.get("dram_energy_j", 0.0),
                "gpu_energy_j": summary.get("gpu_energy_j", 0.0),
                "cpu_pkg_avg_power_w": summary.get("cpu_pkg_avg_power_w", 0.0),
                "cpu_core_avg_power_w": summary.get("cpu_core_avg_power_w", 0.0),
                "dram_avg_power_w": summary.get("dram_avg_power_w", 0.0),
                "gpu_avg_power_w": summary.get("gpu_avg_power_w", 0.0),
                "cpu_pkg_peak_power_w": summary.get("cpu_pkg_peak_power_w", 0.0),
                "cpu_core_peak_power_w": summary.get("cpu_core_peak_power_w", 0.0),
                "dram_peak_power_w": summary.get("dram_peak_power_w", 0.0),
                "gpu_peak_power_w": summary.get("gpu_peak_power_w", 0.0),
            }
            per_file_records.append(pf_record)
        if per_file_records:
            write_csv(output_dir / "per_file_power_summary.csv", per_file_records, list(per_file_records[0].keys()))

        print(f"[wrapper] point={idx} rc={rc}")

    print(f"[wrapper] summary={output_dir / 'power_frequency_summary.csv'}")
    return 0 if not any_fail else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
