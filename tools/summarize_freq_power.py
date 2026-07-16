#!/usr/bin/env python3
import argparse
import csv
import statistics
from pathlib import Path


NUMERIC_FIELDS = [
    "ratio_pct_median",
    "comp_mbs_median",
    "dec_mbs_median",
    "e2e_comp_mbs_median",
    "e2e_dec_mbs_median",
    "cpu_pkg_power_increment_w",
    "cpu_core_power_increment_w",
    "gpu_power_increment_w",
    "power_main_avg_w",
    "cpu_freq_avg_mhz",
    "gpu_freq_avg_mhz",
]

EXTRA_FIELDS = [
    "workload",
    "sample",
    "local_size",
    "accel",
    "elapsed_s",
    "run_dir",
    "cpu_pkg_avg_power_w",
    "cpu_core_avg_power_w",
    "gpu_avg_power_w",
    "cpu_pkg_idle_power_w",
    "cpu_core_idle_power_w",
    "gpu_idle_power_w",
    "cpu_pkg_net_power_w",
    "cpu_core_net_power_w",
    "gpu_net_power_w",
    "cpu_pkg_power_increment_w",
    "cpu_core_power_increment_w",
    "gpu_power_increment_w",
    "cpu_pkg_power_stdev_w",
    "cpu_core_power_stdev_w",
    "gpu_power_stdev_w",
    "cpu_pkg_power_cv_pct",
    "cpu_core_power_cv_pct",
    "gpu_power_cv_pct",
    "power_main_domain",
    "power_main_raw_avg_w",
    "telemetry_note",
]


def read_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("per_file_power_summary.csv")):
        codec = "lzo" if "/lzo/" in path.as_posix() else "lz4"
        scan = path.parent.name
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["codec"] = codec
                row["scan"] = scan
                row = normalize_power_main(row)
                rows.append(row)
    return rows


def fnum(row, key):
    try:
        text = row.get(key, "")
        if text in ("", None):
            return None
        return float(text)
    except Exception:
        return None


def corrected_power_domain(row):
    workload = str(row.get("workload", "") or "")
    ratio = str(row.get("gpu_ratio", "") or "")
    if workload.startswith("cpu"):
        return "cpu"
    if workload.startswith("gpu"):
        return "gpu"
    if workload.startswith("hybrid_ratio_"):
        text = workload[len("hybrid_ratio_"):].split("_", 1)[0].strip().lower()
        if text in ("0", "0.0", "0.00"):
            return "cpu"
        if text in ("1", "1.0", "1.00"):
            return "gpu"
        try:
            value = float(ratio)
            if value <= 0.0:
                return "cpu"
            if value >= 1.0:
                return "gpu"
        except Exception:
            pass
        return "mixed"
    return row.get("power_main_domain", "") or "gpu"


def normalize_power_main(row):
    domain = corrected_power_domain(row)
    row["power_main_domain"] = domain
    cpu_pkg_net = fnum(row, "cpu_pkg_net_power_w")
    gpu_net = fnum(row, "gpu_net_power_w")
    cpu_pkg_avg = fnum(row, "cpu_pkg_avg_power_w")
    gpu_avg = fnum(row, "gpu_avg_power_w")
    if domain == "cpu":
        if cpu_pkg_net is not None:
            row["power_main_avg_w"] = cpu_pkg_net
        if cpu_pkg_avg is not None:
            row["power_main_raw_avg_w"] = cpu_pkg_avg
    elif domain == "gpu":
        if gpu_net is not None:
            row["power_main_avg_w"] = gpu_net
        if gpu_avg is not None:
            row["power_main_raw_avg_w"] = gpu_avg
    else:
        if cpu_pkg_net is not None or gpu_net is not None:
            row["power_main_avg_w"] = (cpu_pkg_net or 0.0) + (gpu_net or 0.0)
        if cpu_pkg_avg is not None or gpu_avg is not None:
            row["power_main_raw_avg_w"] = (cpu_pkg_avg or 0.0) + (gpu_avg or 0.0)
    return row


def median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else ""


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else ""


def stdev(vals):
    vals = [v for v in vals if v is not None]
    return statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else "")


def ci95(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    if len(vals) == 1:
        return 0.0
    return 1.96 * statistics.stdev(vals) / (len(vals) ** 0.5)


def cv_pct(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    avg = sum(vals) / len(vals)
    if avg == 0:
        return 0.0
    return 100.0 * statistics.stdev(vals) / avg if len(vals) > 1 else 0.0


def group_key(row):
    return (
        row.get("codec", ""),
        row.get("scan", ""),
        row.get("workload", ""),
        row.get("engine", ""),
        row.get("block", ""),
        row.get("local_size", ""),
        row.get("accel", ""),
        row.get("alg", ""),
        row.get("level", ""),
        row.get("d_bits", ""),
        row.get("gpu_ratio", ""),
        row.get("cpu_threads", ""),
        row.get("power_main_domain", ""),
    )


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)

    out = []
    for key, items in sorted(grouped.items()):
        first = items[0]
        record = {
            "codec": first.get("codec", ""),
            "scan": first.get("scan", ""),
            "workload": first.get("workload", ""),
            "engine": first.get("engine", ""),
            "block": first.get("block", ""),
            "local_size": first.get("local_size", ""),
            "accel": first.get("accel", ""),
            "alg": first.get("alg", ""),
            "level": first.get("level", ""),
            "d_bits": first.get("d_bits", ""),
            "gpu_ratio": first.get("gpu_ratio", ""),
            "cpu_threads": first.get("cpu_threads", ""),
            "power_main_domain": first.get("power_main_domain", ""),
            "samples": len(items),
        }
        for field in NUMERIC_FIELDS:
            vals = [fnum(row, field) for row in items]
            record[f"{field}_mean"] = mean(vals)
            record[f"{field}_median"] = median(vals)
            record[f"{field}_stdev"] = stdev(vals)
            record[f"{field}_ci95_half"] = ci95(vals)
            record[f"{field}_cv_pct"] = cv_pct(vals)
        out.append(record)
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for field in ["codec", "scan", "workload", "engine", "block", "local_size", "accel", "alg", "level", "d_bits", "gpu_ratio", "cpu_threads", "power_main_domain", "samples", *EXTRA_FIELDS, *NUMERIC_FIELDS]:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in row.keys():
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize LZ4/LZO frequency-power scan results.")
    parser.add_argument("--root", default="/root/compress_freq_power_results")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args(argv)

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "summary"
    rows = read_rows(root)
    write_csv(out_dir / "all_per_file_power.csv", rows)
    write_csv(out_dir / "aggregate_power_summary.csv", summarize(rows))
    print(f"rows={len(rows)} out={out_dir}")


if __name__ == "__main__":
    main()
