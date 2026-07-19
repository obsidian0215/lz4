#!/usr/bin/env python3
"""Benchmark fixed/adaptive/oracle LZ4TP1 policies on registered real samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


FORMAL_N = (1, 2, 4, 8)
CANON_SELECTOR_MANIFEST_SHA256 = "3447f8ff3a04eb1f92285f472eb700efccdc1560ee4c5bc58fc050d739697fa1"
CANON_CALIBRATION_MANIFEST_SHA256 = "c63be7e92c94cab029beed414f58c4034407a5001e95397523dba74bd385aba9"
RAW_FIELDS = (
    "sample",
    "policy",
    "operation",
    "n",
    "chunk_blocks",
    "repetition",
    "kernel_mbs",
    "no_ocl_mbs",
    "total_mbs",
    "ratio_pct",
    "roundtrip_ok",
    "reference_ok",
    "derived",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "heterolz.samples.v1":
        raise ValueError(f"invalid manifest schema: {path}")
    if not isinstance(data.get("files"), list) or not data["files"]:
        raise ValueError(f"empty manifest: {path}")
    if "file_count" in data and data["file_count"] != len(data["files"]):
        raise ValueError(f"manifest file_count does not match files: {path}")
    seen: set[str] = set()
    for index, record in enumerate(data["files"]):
        if not isinstance(record, dict):
            raise ValueError(f"manifest record {index} must be an object: {path}")
        relative_path = record.get("relative_path")
        size = record.get("size")
        sha256 = record.get("sha256")
        relative = Path(relative_path) if isinstance(relative_path, str) else None
        if (not isinstance(relative_path, str) or not relative_path or
                relative is None or relative.is_absolute() or ".." in relative.parts or
                not isinstance(size, int) or size < 0 or
                not isinstance(sha256, str) or len(sha256) != 64 or
                any(char not in "0123456789abcdef" for char in sha256)):
            raise ValueError(f"invalid manifest record {index}: {path}")
        if relative_path in seen:
            raise ValueError(f"duplicate sample path in manifest: {relative_path}")
        seen.add(relative_path)
    return data


def require_executable(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} is not executable: {resolved}")
    return resolved


def validate_sample(root: Path, record: dict[str, object]) -> Path:
    relative_path = record.get("relative_path")
    size = record.get("size")
    expected_sha256 = record.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(size, int) or not isinstance(expected_sha256, str):
        raise ValueError("incomplete sample record")
    sample = (root / relative_path).resolve()
    resolved_root = root.resolve()
    if resolved_root not in sample.parents:
        raise ValueError(f"sample path escapes root: {relative_path}")
    if not sample.is_file() or sample.stat().st_size != size:
        raise RuntimeError(f"sample size mismatch: {relative_path}")
    if sha256_file(sample) != expected_sha256:
        raise RuntimeError(f"sample SHA256 mismatch: {relative_path}")
    return sample


class Runner:
    def __init__(self, commands_log: Path, raw_stdout: Path, venue: str, profile: Path) -> None:
        self.commands_log = commands_log
        self.raw_stdout = raw_stdout
        self.venue = venue
        self.profile = profile
        commands_log.parent.mkdir(parents=True, exist_ok=True)
        raw_stdout.parent.mkdir(parents=True, exist_ok=True)
        commands_log.write_text("", encoding="utf-8")
        raw_stdout.write_text("", encoding="utf-8")

    def run(self, argv: list[str], use_profile: bool = False) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["FORCE_OPENCL_DEVICE"] = self.venue
        if use_profile:
            env["LZ4TP_PROFILE"] = str(self.profile)
        prefix = f"FORCE_OPENCL_DEVICE={self.venue} "
        if use_profile:
            prefix += f"LZ4TP_PROFILE={shlex.quote(str(self.profile))} "
        command_text = prefix + shlex.join(argv)
        with self.commands_log.open("a", encoding="utf-8") as handle:
            handle.write(command_text + "\n")
        completed = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with self.raw_stdout.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {command_text}\n{completed.stdout}{completed.stderr}[exit={completed.returncode}]\n")
        if completed.returncode != 0:
            raise RuntimeError(f"command failed: {command_text}")
        return completed


def load_metric(path: Path, operation: str) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "heterolz.operation-metric.v1" or data.get("operation") != operation:
        raise RuntimeError(f"invalid {operation} metric: {path}")
    for field in ("input_bytes", "output_bytes", "n", "block_size", "hash_log", "chunk_blocks",
                  "kernel_us", "no_ocl_us", "total_us", "ocl_setup_us", "ratio_pct"):
        if not isinstance(data.get(field), (int, float)):
            raise RuntimeError(f"metric field missing: {field}")
    if data["n"] not in FORMAL_N or data["block_size"] <= 0 or not 11 <= data["hash_log"] <= 15 or data["chunk_blocks"] <= 0:
        raise RuntimeError(f"invalid codec dimensions in {path}")
    if data["input_bytes"] <= 0 or data["output_bytes"] <= 0:
        raise RuntimeError(f"invalid byte counts in {path}")
    if data["kernel_us"] <= 0 or data["no_ocl_us"] <= 0 or data["total_us"] <= 0 or data["ocl_setup_us"] < 0:
        raise RuntimeError(f"non-positive timing in {path}")
    if not data["kernel_us"] <= data["no_ocl_us"] <= data["total_us"]:
        raise RuntimeError(f"timing scopes are inconsistent in {path}")
    return data


def metric_row(sample: str, policy: str, operation: str, repetition: int,
               metric: dict[str, object], roundtrip_ok: bool, reference_ok: bool,
               derived: bool = False) -> dict[str, object]:
    original_bytes = int(metric["input_bytes"] if operation == "compress" else metric["output_bytes"])
    return {
        "sample": sample,
        "policy": policy,
        "operation": operation,
        "n": int(metric["n"]),
        "chunk_blocks": int(metric["chunk_blocks"]),
        "repetition": repetition,
        "kernel_mbs": original_bytes / float(metric["kernel_us"]),
        "no_ocl_mbs": original_bytes / float(metric["no_ocl_us"]),
        "total_mbs": original_bytes / float(metric["total_us"]),
        "ratio_pct": float(metric["ratio_pct"]),
        "roundtrip_ok": str(roundtrip_ok).lower(),
        "reference_ok": str(reference_ok).lower(),
        "derived": str(derived).lower(),
    }


def median_row(rows: list[dict[str, object]]) -> dict[str, object]:
    first = rows[0]
    result = dict(first)
    result["repetition"] = 0
    result["derived"] = "true"
    for field in ("kernel_mbs", "no_ocl_mbs", "total_mbs", "ratio_pct"):
        result[field] = statistics.median(float(row[field]) for row in rows)
    return result


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    policies = sorted({str(row["policy"]) for row in rows})
    for policy in policies:
        for operation in ("compress", "decompress"):
            selected = [row for row in rows if row["policy"] == policy and row["operation"] == operation]
            if not selected:
                continue
            by_sample: dict[str, list[dict[str, object]]] = {}
            for row in selected:
                by_sample.setdefault(str(row["sample"]), []).append(row)
            sample_medians = [median_row(values) for values in by_sample.values()]
            for metric in ("kernel_mbs", "no_ocl_mbs", "total_mbs"):
                summary.append(
                    {
                        "policy": policy,
                        "metric": f"{operation}_{metric}_median_of_files",
                        "value": statistics.median(float(row[metric]) for row in sample_medians),
                    }
                )
            if operation == "compress":
                summary.append(
                    {
                        "policy": policy,
                        "metric": "ratio_pct_median_of_files",
                        "value": statistics.median(float(row["ratio_pct"]) for row in sample_medians),
                    }
                )

    adaptive = [row for row in rows if row["policy"] == "adaptive" and row["operation"] == "compress"]
    oracle = [row for row in rows if row["policy"] == "oracle" and row["operation"] == "compress"]
    adaptive_groups: dict[str, list[dict[str, object]]] = {}
    for row in adaptive:
        adaptive_groups.setdefault(str(row["sample"]), []).append(row)
    adaptive_by_sample = {
        name: float(median_row(values)["no_ocl_mbs"])
        for name, values in adaptive_groups.items()
    }
    oracle_by_sample = {str(row["sample"]): float(row["no_ocl_mbs"]) for row in oracle}
    ratios = [adaptive_by_sample[name] / oracle_by_sample[name]
              for name in sorted(set(adaptive_by_sample) & set(oracle_by_sample))
              if adaptive_by_sample[name] > 0 and oracle_by_sample[name] > 0]
    if ratios:
        geomean = math.exp(sum(math.log(value) for value in ratios) / len(ratios)) * 100.0
        summary.append({"policy": "adaptive", "metric": "oracle_attainment_no_ocl_pct", "value": geomean})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, default=Path("/root/samples"))
    parser.add_argument("--binary", type=Path, default=Path("./lz4_gpu"))
    parser.add_argument("--reference-decoder", type=Path, default=Path("./tp_ref_decode"))
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--commands-log", type=Path, required=True)
    parser.add_argument("--raw-stdout", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--venue", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--n", default="1,2,4,8")
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--hash-log", type=int, default=14)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-adaptive", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    try:
        n_values = tuple(int(value) for value in args.n.split(",") if value)
        if (not n_values or len(set(n_values)) != len(n_values) or
                any(value not in FORMAL_N for value in n_values)):
            raise ValueError("--n must contain values from 1,2,4,8")
        verification = json.loads(args.verification.read_text(encoding="utf-8"))
        checks = verification.get("checks")
        if verification.get("status") != "passed" or verification.get("all_inputs_passed") is not True:
            raise ValueError("formal correctness verification has not passed")
        if args.formal:
            if args.repetitions < 9 or n_values != FORMAL_N or args.limit is not None or args.skip_adaptive:
                raise ValueError("formal mode requires >=9 repetitions, all N values, full manifest, and adaptive")
            if args.block_size != 65536 or args.hash_log != 14:
                raise ValueError("formal mode requires block_size=65536 and hash_log=14")
            if not isinstance(checks, dict) or any(checks.get(name) != "passed" for name in
                                                   ("roundtrip", "reference_decode", "bridge_interop", "determinism", "cross_venue")):
                raise ValueError("formal mode requires the complete correctness gate")
        if args.repetitions < 1:
            raise ValueError("repetitions must be positive")

        output_paths = {
            args.raw_results.resolve(),
            args.summary.resolve(),
            args.commands_log.resolve(),
            args.raw_stdout.resolve(),
        }
        if len(output_paths) != 4:
            raise ValueError("raw results, summary, commands log, and raw stdout must use distinct paths")
        protected_paths = {
            args.manifest.resolve(),
            args.calibration_manifest.resolve(),
            args.verification.resolve(),
            args.binary.resolve(),
            args.reference_decoder.resolve(),
        }
        if output_paths & protected_paths:
            raise ValueError("benchmark output paths must not replace inputs or executables")
        sample_root_resolved = args.sample_root.resolve()
        if args.formal and any(path == sample_root_resolved or sample_root_resolved in path.parents
                               for path in [*output_paths, args.work_root.resolve()]):
            raise ValueError("formal benchmark artifacts and work files must stay outside the sample root")

        manifest = load_manifest(args.manifest)
        calibration_manifest = load_manifest(args.calibration_manifest)
        if args.formal and (
                sha256_file(args.manifest) != CANON_SELECTOR_MANIFEST_SHA256 or
                sha256_file(args.calibration_manifest) != CANON_CALIBRATION_MANIFEST_SHA256):
            raise ValueError("formal mode requires the registered selector and calibration manifests")
        records = manifest["files"]
        calibration_records = calibration_manifest["files"]
        validation_paths = {item.get("relative_path") for item in records if isinstance(item, dict)}
        calibration_paths = {item.get("relative_path") for item in calibration_records if isinstance(item, dict)}
        if validation_paths & calibration_paths:
            raise ValueError("calibration and selector-validation samples must be disjoint")
        if args.formal and (manifest.get("selection_name") != "selector_validation" or
                            calibration_manifest.get("selection_name") != "calibration"):
            raise ValueError("formal mode requires the registered calibration and selector-validation manifests")
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            records = records[: args.limit]
        if len(calibration_records) != 1:
            raise ValueError("current one-time calibration requires exactly one registered real sample")
        expected_verification_inputs = {
            str(item["relative_path"]): (int(item["size"]), str(item["sha256"]))
            for item in [*records, *calibration_records]
            if isinstance(item, dict) and isinstance(item.get("relative_path"), str) and
            isinstance(item.get("size"), int) and isinstance(item.get("sha256"), str)
        }
        verification_inputs_obj = verification.get("inputs")
        if not isinstance(verification_inputs_obj, list):
            raise ValueError("correctness verification does not identify its real inputs")
        verification_inputs = {
            str(item.get("relative_path")): (item.get("size"), item.get("sha256"))
            for item in verification_inputs_obj if isinstance(item, dict)
        }
        if len(verification_inputs) != len(verification_inputs_obj):
            raise ValueError("correctness verification contains duplicate or invalid inputs")
        if args.formal:
            if verification_inputs != expected_verification_inputs:
                raise ValueError("correctness verification must exactly cover calibration and selector-validation inputs")
        elif any(verification_inputs.get(path) != value for path, value in expected_verification_inputs.items()):
            raise ValueError("correctness verification does not cover the requested real inputs")
        if verification.get("parameters") != {"block_size": args.block_size, "hash_log": args.hash_log}:
            raise ValueError("correctness verification parameters do not match the benchmark")
        verified_n = verification.get("n_values")
        if not isinstance(verified_n, list) or (args.formal and verified_n != list(FORMAL_N)) or \
                (not args.formal and not set(n_values).issubset(set(verified_n))):
            raise ValueError("correctness verification does not cover the requested N values")

        binary = require_executable(args.binary, "binary")
        reference_decoder = require_executable(args.reference_decoder, "reference decoder")
        args.work_root.mkdir(parents=True, exist_ok=True)
        calibration_sample = validate_sample(args.sample_root, calibration_records[0])

        rows: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="heterolz-real-benchmark-", dir=args.work_root) as temp_root:
            root = Path(temp_root)
            profile = root / "lz4tp.profile"
            runner = Runner(args.commands_log, args.raw_stdout, args.venue, profile)
            calibration_start = time.perf_counter_ns()
            runner.run(
                [str(binary), "--calibrate", "-B", str(args.block_size), "--d-bits", str(args.hash_log),
                 str(calibration_sample), "-o", str(profile)]
            )
            calibration_us = (time.perf_counter_ns() - calibration_start) // 1000
            with args.raw_stdout.open("a", encoding="utf-8") as handle:
                handle.write(f"CALIBRATION_TOTAL_US={calibration_us}\n")

            for sample_index, record in enumerate(records):
                if not isinstance(record, dict):
                    raise ValueError(f"manifest record {sample_index} must be an object")
                sample = validate_sample(args.sample_root, record)
                relative_path = str(record["relative_path"])
                expected_sha256 = str(record["sha256"])
                key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
                sample_dir = root / f"{Path(relative_path).name}.{key}"
                sample_dir.mkdir()
                try:
                    policy_specs = [(f"fixed_n{n}", n, False) for n in n_values]
                    if not args.skip_adaptive:
                        policy_specs.append(("adaptive", None, True))
                    for policy, fixed_n, adaptive in policy_specs:
                        reference_ok = False
                        for repetition in range(1, args.repetitions + 1):
                            frame = sample_dir / f"{policy}.r{repetition}.lz4tp"
                            restored = sample_dir / f"{policy}.r{repetition}.restored"
                            comp_metric_path = sample_dir / f"{policy}.r{repetition}.compress.json"
                            dec_metric_path = sample_dir / f"{policy}.r{repetition}.decompress.json"
                            common = ["-B", str(args.block_size), "--d-bits", str(args.hash_log)]
                            if adaptive:
                                runner.run(
                                    [str(binary), "--auto", *common, "--metrics-json", str(comp_metric_path),
                                     str(sample), "-o", str(frame)],
                                    use_profile=True,
                                )
                            else:
                                runner.run(
                                    [str(binary), "--twophase", "-N", str(fixed_n), *common,
                                     "--metrics-json", str(comp_metric_path), str(sample), "-o", str(frame)]
                                )
                            runner.run(
                                [str(binary), "--twophase", "-d", *common, "--metrics-json", str(dec_metric_path),
                                 str(frame), "-o", str(restored)]
                            )
                            roundtrip_ok = restored.stat().st_size == sample.stat().st_size and sha256_file(restored) == expected_sha256
                            if not roundtrip_ok:
                                raise RuntimeError(f"benchmark roundtrip failed: {relative_path} {policy} rep={repetition}")
                            if repetition == 1:
                                runner.run([str(reference_decoder), str(frame), str(sample)])
                                reference_ok = True
                            comp_metric = load_metric(comp_metric_path, "compress")
                            dec_metric = load_metric(dec_metric_path, "decompress")
                            if comp_metric["n"] != dec_metric["n"]:
                                raise RuntimeError(f"metric N mismatch: {relative_path} {policy}")
                            if fixed_n is not None and comp_metric["n"] != fixed_n:
                                raise RuntimeError(f"fixed policy selected the wrong N: {relative_path} {policy}")
                            for metric in (comp_metric, dec_metric):
                                if metric["block_size"] != args.block_size or metric["hash_log"] != args.hash_log:
                                    raise RuntimeError(f"metric parameters mismatch: {relative_path} {policy}")
                            if (comp_metric["input_bytes"] != sample.stat().st_size or
                                    comp_metric["output_bytes"] != frame.stat().st_size or
                                    dec_metric["input_bytes"] != frame.stat().st_size or
                                    dec_metric["output_bytes"] != sample.stat().st_size):
                                raise RuntimeError(f"metric byte counts mismatch: {relative_path} {policy}")
                            rows.append(metric_row(relative_path, policy, "compress", repetition,
                                                   comp_metric, roundtrip_ok, reference_ok))
                            rows.append(metric_row(relative_path, policy, "decompress", repetition,
                                                   dec_metric, roundtrip_ok, reference_ok))
                            for path in (frame, restored, comp_metric_path, dec_metric_path):
                                path.unlink(missing_ok=True)
                finally:
                    shutil.rmtree(sample_dir, ignore_errors=True)

            fixed_policies = {f"fixed_n{n}" for n in n_values}
            for relative_path in sorted({str(row["sample"]) for row in rows}):
                fixed_comp = [row for row in rows if row["sample"] == relative_path and
                              row["policy"] in fixed_policies and row["operation"] == "compress"]
                medians_by_policy: dict[str, dict[str, object]] = {}
                for policy in fixed_policies:
                    values = [row for row in fixed_comp if row["policy"] == policy]
                    if values:
                        medians_by_policy[policy] = median_row(values)
                oracle_policy = max(
                    medians_by_policy,
                    key=lambda name: (
                        float(medians_by_policy[name]["no_ocl_mbs"]),
                        -int(name.removeprefix("fixed_n")),
                    ),
                )
                oracle_n = int(medians_by_policy[oracle_policy]["n"])
                for operation in ("compress", "decompress"):
                    values = [row for row in rows if row["sample"] == relative_path and
                              row["policy"] == oracle_policy and row["operation"] == operation]
                    oracle = median_row(values)
                    oracle["policy"] = "oracle"
                    oracle["n"] = oracle_n
                    rows.append(oracle)

        write_csv(args.raw_results, RAW_FIELDS, rows)
        summary_rows = summarize(rows)
        summary_rows.append({"policy": "calibration", "metric": "total_us", "value": calibration_us})
        write_csv(args.summary, ("policy", "metric", "value"), summary_rows)
        return 0
    except Exception as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
