#!/usr/bin/env python3
import argparse
import csv
import os
import shlex
import subprocess
import sys
import tempfile
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

DEFAULT_CPU_FREQ_MHZ = "NA" if os.name == "nt" else "1200,1800,2400,3000,3600,NA"
DEFAULT_GPU_FREQ_MHZ = "NA" if os.name == "nt" else "400,700,1000,1300,NA"
MAX_IDLE_SHARE = 0.95


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
    return {"gpu", "native_cpu", "hybrid"}


def effective_hybrid_ratios(args_list):
    text = bench_arg_value(args_list, "--hybrid-gpu-ratios")
    if not text:
        return ["0", "0.25", "0.5", "0.75", "1"]
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


def build_work_items(args, cpu_points, gpu_points):
    base = list(args.bench_args)
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

    if "native_cpu" in engines:
        bench_args = with_option(base, "--engines", "native_cpu")
        for cpu_point in cpu_points:
            add("cpu", cpu_point, none_gpu, bench_args)

    if "gpu" in engines:
        bench_args = with_option(base, "--engines", "gpu")
        for gpu_point in gpu_points:
            add("gpu", none_cpu, gpu_point, bench_args)

    if "hybrid" in engines:
        for ratio in hybrid_ratios:
            bench_args = with_option(with_option(base, "--engines", "hybrid"), "--hybrid-gpu-ratios", ratio)
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


def run_bench(args, item, bench_args_override=None, sample_tag=""):
    cpu_point = item["cpu"]
    gpu_point = item["gpu"]
    safe_label = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in item["label"])
    base_dir = Path(args.output_dir) / f"{safe_label}_cpu_{cpu_point['label']}_gpu_{gpu_point['label']}"
    if sample_tag:
        safe_sample = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(sample_tag))
        run_dir = base_dir / safe_sample
    else:
        run_dir = base_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    bench_args = list(bench_args_override if bench_args_override is not None else item["bench_args"])
    bench_args = with_option(bench_args, "--results-dir", str(run_dir))

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

def telemetry_worker_main(worker_out: Path, worker_stop: Path, interval_s: float):
    telemetry = TelemetryProbe()
    worker_out.parent.mkdir(parents=True, exist_ok=True)
    with worker_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "cpu_power_w", "gpu_power_w", "cpu_freq_mhz", "gpu_freq_mhz", "cpu_energy_j", "core_energy_j", "gpu_energy_j"])
        while not worker_stop.exists():
            snap = telemetry.snapshot()
            w.writerow([
                f"{float(snap.get('ts') or 0.0):.9f}",
                f"{float(snap.get('cpu_power_w') or 0.0):.9f}",
                f"{float(snap.get('gpu_power_w') or 0.0):.9f}",
                f"{float(snap.get('cpu_freq_mhz') or 0.0):.6f}",
                f"{float(snap.get('gpu_freq_mhz') or 0.0):.6f}",
                "" if snap.get("cpu_energy_j") is None else f"{float(snap.get('cpu_energy_j') or 0.0):.9f}",
                "" if snap.get("core_energy_j") is None else f"{float(snap.get('core_energy_j') or 0.0):.9f}",
                "" if snap.get("gpu_energy_j") is None else f"{float(snap.get('gpu_energy_j') or 0.0):.9f}",
            ])
            f.flush()
            time.sleep(max(interval_s, 0.05))
    return 0


def start_telemetry_worker(interval_s: float):
    fd_csv, csv_path = tempfile.mkstemp(prefix="lz4_wrap_worker_", suffix=".csv")
    os.close(fd_csv)
    fd_stop, stop_path = tempfile.mkstemp(prefix="lz4_wrap_worker_", suffix=".stop")
    os.close(fd_stop)
    worker_csv = Path(csv_path)
    worker_stop = Path(stop_path)
    worker_stop.unlink(missing_ok=True)

    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--telemetry-worker",
            "--worker-out",
            str(worker_csv),
            "--worker-stop",
            str(worker_stop),
            "--worker-interval",
            str(interval_s),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + 8.0
    while time.time() < deadline:
        try:
            if worker_csv.exists() and worker_csv.stat().st_size > 24:
                return proc, worker_csv, worker_stop
        except Exception:
            pass
        if proc.poll() is not None:
            break
        time.sleep(0.02)
    raise RuntimeError("failed to start telemetry worker")


def stop_telemetry_worker(proc, worker_csv: Path, worker_stop: Path):
    samples = []
    try:
        worker_stop.write_text("stop", encoding="utf-8")
    except Exception:
        pass
    try:
        proc.wait(timeout=5.0)
    except Exception:
        proc.kill()

    if worker_csv.exists():
        with worker_csv.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    samples.append({
                        "ts": float(row.get("ts") or 0.0),
                        "cpu_power_w": float(row.get("cpu_power_w") or 0.0),
                        "gpu_power_w": float(row.get("gpu_power_w") or 0.0),
                        "cpu_freq_mhz": float(row.get("cpu_freq_mhz") or 0.0),
                        "gpu_freq_mhz": float(row.get("gpu_freq_mhz") or 0.0),
                        "cpu_energy_j": None if row.get("cpu_energy_j", "") == "" else float(row.get("cpu_energy_j") or 0.0),
                        "core_energy_j": None if row.get("core_energy_j", "") == "" else float(row.get("core_energy_j") or 0.0),
                        "gpu_energy_j": None if row.get("gpu_energy_j", "") == "" else float(row.get("gpu_energy_j") or 0.0),
                    })
                except Exception:
                    continue

    worker_csv.unlink(missing_ok=True)
    worker_stop.unlink(missing_ok=True)
    return samples


def collect_until_done(proc, telemetry, interval_s, workload_label):
    lines = []
    worker_proc, worker_csv, worker_stop = start_telemetry_worker(interval_s)
    for line in proc.stdout:
        lines.append(line)
        print(line, end="")
    rc = proc.wait()
    samples = stop_telemetry_worker(worker_proc, worker_csv, worker_stop)
    return rc, "".join(lines), samples


def summarize_samples(telemetry, samples):
    if not samples:
        return telemetry.summarize_samples([])
    return telemetry.summarize_samples(samples)


def measure_idle_baseline(telemetry, idle_seconds, sample_interval):
    idle_seconds = float(idle_seconds or 0.0)
    if idle_seconds <= 0:
        return {
            "cpu_pkg_idle_power_w": 0.0,
            "cpu_core_idle_power_w": 0.0,
            "gpu_idle_power_w": 0.0,
        }
    if hasattr(telemetry, "measure_idle_pkg_power_w") and hasattr(telemetry, "measure_idle_gpu_power_w"):
        cpu_idle = float(telemetry.measure_idle_pkg_power_w(idle_seconds) or 0.0)
        core_idle = float(getattr(telemetry, "measure_idle_core_power_w", telemetry.measure_idle_pkg_power_w)(idle_seconds) or 0.0)
        gpu_idle = float(telemetry.measure_idle_gpu_power_w(idle_seconds) or 0.0)
        return {
            "cpu_pkg_idle_power_w": cpu_idle,
            "cpu_core_idle_power_w": core_idle,
            "gpu_idle_power_w": gpu_idle,
        }
    samples = []
    deadline = time.perf_counter() + idle_seconds
    while time.perf_counter() < deadline:
        samples.append(telemetry.snapshot())
        time.sleep(max(0.05, float(sample_interval or 0.2)))
    if not samples:
        return {
            "cpu_pkg_idle_power_w": 0.0,
            "cpu_core_idle_power_w": 0.0,
            "gpu_idle_power_w": 0.0,
        }
    return {
        "cpu_pkg_idle_power_w": sum(float(s.get("cpu_power_w") or 0.0) for s in samples) / len(samples),
        "cpu_core_idle_power_w": sum(float(s.get("core_power_w") or 0.0) for s in samples) / len(samples),
        "gpu_idle_power_w": sum(float(s.get("gpu_power_w") or 0.0) for s in samples) / len(samples),
    }


def net_power(avg_power, idle_power):
    avg = float(avg_power or 0.0)
    idle = max(0.0, float(idle_power or 0.0))
    if avg <= 0.0:
        return 0.0
    capped_idle = min(idle, avg * MAX_IDLE_SHARE)
    return max(0.0, avg - capped_idle)


def power_main_domain(workload):
    wl = str(workload or "")
    if wl.startswith("cpu"):
        return "cpu"
    if wl.startswith("hybrid_ratio_"):
        ratio = wl[len("hybrid_ratio_"):].split("_", 1)[0]
        if is_cpu_only_ratio(ratio):
            return "cpu"
        if is_gpu_only_ratio(ratio):
            return "gpu"
        return "mixed"
    return "gpu"


def power_main_value(workload, summary):
    domain = power_main_domain(workload)
    if domain == "cpu":
        return float(summary.get("cpu_pkg_avg_power_w", 0.0) or 0.0)
    return float(summary.get("gpu_avg_power_w", 0.0) or 0.0)


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
    if "--telemetry-worker" in argv:
        worker_parser = argparse.ArgumentParser(description="telemetry worker")
        worker_parser.add_argument("--telemetry-worker", action="store_true")
        worker_parser.add_argument("--worker-out", required=True)
        worker_parser.add_argument("--worker-stop", required=True)
        worker_parser.add_argument("--worker-interval", type=float, default=0.2)
        wargs = worker_parser.parse_args(argv)
        return telemetry_worker_main(Path(wargs.worker_out), Path(wargs.worker_stop), max(0.05, float(wargs.worker_interval)))

    parser = argparse.ArgumentParser(
        description="External power/frequency wrapper for bench_lz4.py. Arguments after -- are passed to bench_lz4.py."
    )
    parser.add_argument("--platform-id", default=os.environ.get("LZ4_PLATFORM_ID", "local"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "exp_results" / "power_freq_runs" / time.strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--cpu-frequencies", "--cpu-points", default=DEFAULT_CPU_FREQ_MHZ, help='Comma list: "2100,3400,NA" or "50%%,100%%,NA"')
    parser.add_argument("--gpu-frequencies", "--gpu-points", default=DEFAULT_GPU_FREQ_MHZ, help='Comma list: "1000,NA" or "70%%,NA"')
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--idle-seconds", type=float, default=2.0, help="Idle baseline sampling seconds before each workload point")
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
        item_samples = bench_samples_from_args(item["bench_args"])

        if not args.no_reset_before:
            run_control_reset(CPU_CONTROL)
            run_control_reset(GPU_CONTROL)
        apply_frequency(CPU_CONTROL, cpu_point)
        apply_frequency(GPU_CONTROL, gpu_point)

        idle_baseline = measure_idle_baseline(telemetry, args.idle_seconds, args.sample_interval)
        print(
            f"[wrapper] idle_baseline cpu_pkg={idle_baseline['cpu_pkg_idle_power_w']:.3f}W "
            f"cpu_core={idle_baseline['cpu_core_idle_power_w']:.3f}W gpu={idle_baseline['gpu_idle_power_w']:.3f}W"
        )

        try:
            for sample_idx, sample_path in enumerate(item_samples, start=1):
                rc = -1
                run_dir = ""
                elapsed = 0.0
                summary = {}
                sample_name = Path(sample_path).name
                bench_args_single = with_option(item["bench_args"], "--single-file", str(sample_path))

                max_attempts = 8
                for attempt in range(1, max_attempts + 1):
                    try:
                        proc, cmd, run_dir_path, start = run_bench(args, item, bench_args_override=bench_args_single, sample_tag=sample_name)
                        rc, output, samples = collect_until_done(proc, telemetry, max(0.05, args.sample_interval), item["label"])
                        elapsed = max(0.0, time.perf_counter() - start)
                        run_dir = str(run_dir_path)
                        summary = summarize_samples(telemetry, samples)
                        main_raw = power_main_value(item["label"], summary)
                        if rc == 0 and main_raw <= 0.0:
                            raise RuntimeError("invalid telemetry: main raw average power is zero")

                        with open(run_dir_path / "wrapper_command.txt", "w", encoding="utf-8") as handle:
                            handle.write(" ".join(str(x) for x in cmd) + "\n")
                        with open(run_dir_path / "wrapper_output.log", "w", encoding="utf-8") as handle:
                            handle.write(output)
                        break
                    except Exception as exc:
                        output = f"wrapper run exception (attempt {attempt}/{max_attempts}): {type(exc).__name__}: {exc}"
                        rc = 1
                        if attempt < max_attempts:
                            time.sleep(1.0)

                cpu_pkg_avg = float(summary.get("cpu_pkg_avg_power_w", 0.0) or 0.0)
                cpu_core_avg = float(summary.get("cpu_core_avg_power_w", 0.0) or 0.0)
                gpu_avg = float(summary.get("gpu_avg_power_w", 0.0) or 0.0)

                cpu_pkg_idle = float(idle_baseline.get("cpu_pkg_idle_power_w", 0.0) or 0.0)
                cpu_core_idle = float(idle_baseline.get("cpu_core_idle_power_w", 0.0) or 0.0)
                gpu_idle = float(idle_baseline.get("gpu_idle_power_w", 0.0) or 0.0)

                cpu_pkg_net = net_power(cpu_pkg_avg, cpu_pkg_idle)
                cpu_core_net = net_power(cpu_core_avg, cpu_core_idle)
                gpu_net = net_power(gpu_avg, gpu_idle)
                main_domain = power_main_domain(item["label"])

                record = {
                    "workload": item["label"],
                    "sample": sample_name,
                    "elapsed_s": elapsed,
                    "run_dir": run_dir,
                    "bench_args": shlex.join(bench_args_single),
                    "telemetry_sources": telemetry.describe_sources(),
                    "cpu_freq_avg_mhz": summary.get("cpu_freq_avg_mhz", 0.0),
                    "gpu_freq_avg_mhz": summary.get("gpu_freq_avg_mhz", 0.0),
                    "cpu_energy_j": summary.get("cpu_energy_j", 0.0),
                    "core_energy_j": summary.get("core_energy_j", 0.0),
                    "dram_energy_j": summary.get("dram_energy_j", 0.0),
                    "gpu_energy_j": summary.get("gpu_energy_j", 0.0),
                    "cpu_pkg_avg_power_w": cpu_pkg_avg,
                    "cpu_core_avg_power_w": cpu_core_avg,
                    "gpu_avg_power_w": gpu_avg,
                    "cpu_pkg_idle_power_w": cpu_pkg_idle,
                    "cpu_core_idle_power_w": cpu_core_idle,
                    "gpu_idle_power_w": gpu_idle,
                    "cpu_pkg_net_power_w": cpu_pkg_net,
                    "cpu_core_net_power_w": cpu_core_net,
                    "gpu_net_power_w": gpu_net,
                    "cpu_pkg_peak_power_w": summary.get("cpu_pkg_peak_power_w", 0.0),
                    "cpu_core_peak_power_w": summary.get("cpu_core_peak_power_w", 0.0),
                    "gpu_peak_power_w": summary.get("gpu_peak_power_w", 0.0),
                    "cpu_pkg_power_stdev_w": summary.get("cpu_pkg_power_stdev_w", 0.0),
                    "cpu_core_power_stdev_w": summary.get("cpu_core_power_stdev_w", 0.0),
                    "gpu_power_stdev_w": summary.get("gpu_power_stdev_w", 0.0),
                    "cpu_pkg_power_cv_pct": summary.get("cpu_pkg_power_cv_pct", 0.0),
                    "cpu_core_power_cv_pct": summary.get("cpu_core_power_cv_pct", 0.0),
                    "gpu_power_cv_pct": summary.get("gpu_power_cv_pct", 0.0),
                    "power_main_domain": main_domain,
                    "power_main_avg_w": (
                        cpu_pkg_net if main_domain == "cpu" else
                        gpu_net if main_domain == "gpu" else
                        cpu_pkg_net + gpu_net
                    ),
                    "power_main_raw_avg_w": (
                        cpu_pkg_avg if main_domain == "cpu" else
                        gpu_avg if main_domain == "gpu" else
                        cpu_pkg_avg + gpu_avg
                    ),
                    "rc": rc,
                }
                records.append(record)
                if rc != 0:
                    any_fail = True
                write_csv(output_dir / "power_frequency_summary.csv", records, list(record.keys()))

                per_file_path = find_per_file_summary(run_dir)
                for pf in read_csv_rows(per_file_path):
                    pf_record = {
                        "workload": item["label"],
                        "sample": pf.get("sample", sample_name),
                        "engine": pf.get("engine", ""),
                        "block": pf.get("block", ""),
                        "local_size": pf.get("local_size", ""),
                        "accel": pf.get("accel", ""),
                        "d_bits": pf.get("d_bits", ""),
                        "gpu_ratio": pf.get("gpu_ratio", ""),
                        "cpu_threads": pf.get("cpu_threads", ""),
                        "ratio_pct_median": pf.get("ratio_pct_median", ""),
                        "comp_mbs_mean": pf.get("comp_mbs_mean", pf.get("manual_comp_mbs_mean", "")),
                        "comp_mbs_median": pf.get("comp_mbs_median", pf.get("manual_comp_mbs_median", "")),
                        "dec_mbs_mean": pf.get("dec_mbs_mean", pf.get("manual_dec_mbs_mean", "")),
                        "dec_mbs_median": pf.get("dec_mbs_median", pf.get("manual_dec_mbs_median", "")),
                        "e2e_comp_mbs_mean": pf.get("e2e_comp_mbs_mean", pf.get("manual_e2e_comp_mbs_mean", "")),
                        "e2e_comp_mbs_median": pf.get("e2e_comp_mbs_median", pf.get("manual_e2e_comp_mbs_median", "")),
                        "e2e_dec_mbs_mean": pf.get("e2e_dec_mbs_mean", pf.get("manual_e2e_dec_mbs_mean", "")),
                        "e2e_dec_mbs_median": pf.get("e2e_dec_mbs_median", pf.get("manual_e2e_dec_mbs_median", "")),
                        "elapsed_s": elapsed,
                        "run_dir": run_dir,
                        "cpu_freq_avg_mhz": summary.get("cpu_freq_avg_mhz", 0.0),
                        "gpu_freq_avg_mhz": summary.get("gpu_freq_avg_mhz", 0.0),
                        "cpu_pkg_avg_power_w": cpu_pkg_avg,
                        "cpu_core_avg_power_w": cpu_core_avg,
                        "gpu_avg_power_w": gpu_avg,
                        "cpu_pkg_idle_power_w": cpu_pkg_idle,
                        "cpu_core_idle_power_w": cpu_core_idle,
                        "gpu_idle_power_w": gpu_idle,
                        "cpu_pkg_net_power_w": cpu_pkg_net,
                        "cpu_core_net_power_w": cpu_core_net,
                        "gpu_net_power_w": gpu_net,
                        "cpu_pkg_power_increment_w": cpu_pkg_net,
                        "cpu_core_power_increment_w": cpu_core_net,
                        "gpu_power_increment_w": gpu_net,
                        "cpu_pkg_power_stdev_w": summary.get("cpu_pkg_power_stdev_w", 0.0),
                        "cpu_core_power_stdev_w": summary.get("cpu_core_power_stdev_w", 0.0),
                        "gpu_power_stdev_w": summary.get("gpu_power_stdev_w", 0.0),
                        "cpu_pkg_power_cv_pct": summary.get("cpu_pkg_power_cv_pct", 0.0),
                        "cpu_core_power_cv_pct": summary.get("cpu_core_power_cv_pct", 0.0),
                        "gpu_power_cv_pct": summary.get("gpu_power_cv_pct", 0.0),
                        "power_main_domain": main_domain,
                        "power_main_avg_w": (
                            cpu_pkg_net if main_domain == "cpu" else
                            gpu_net if main_domain == "gpu" else
                            cpu_pkg_net + gpu_net
                        ),
                        "power_main_raw_avg_w": (
                            cpu_pkg_avg if main_domain == "cpu" else
                            gpu_avg if main_domain == "gpu" else
                            cpu_pkg_avg + gpu_avg
                        ),
                        "telemetry_note": "file_level_power(net=avg-idle)",
                    }
                    per_file_records.append(pf_record)
                if per_file_records:
                    write_csv(output_dir / "per_file_power_summary.csv", per_file_records, list(per_file_records[0].keys()))

                print(f"[wrapper] point={idx} sample={sample_idx}/{len(item_samples)} rc={rc}")
        finally:
            run_control_reset(CPU_CONTROL)
            run_control_reset(GPU_CONTROL)

    print(f"[wrapper] summary={output_dir / 'power_frequency_summary.csv'}")
    return 0 if not any_fail else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
