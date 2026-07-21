#!/usr/bin/env python3
"""Assemble self-contained external-baseline evidence for a formal run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from gpulz_external_adapter import (
    SCHEMA as GPULZ_RUN_SCHEMA,
    UPSTREAM_COMMIT,
    UPSTREAM_PARAMETERS,
    UPSTREAM_REPOSITORY,
    sha256_text,
    write_json_atomic,
)
from nvcomp_external_adapter import (
    CHUNK_SIZE as NVCOMP_CHUNK_SIZE,
    DECOMPRESS_BACKEND as NVCOMP_DECOMPRESS_BACKEND,
    ITERATIONS as NVCOMP_ITERATIONS,
    NVCOMP_VERSION,
    SCHEMA as NVCOMP_RUN_SCHEMA,
    STREAM_MODE as NVCOMP_STREAM_MODE,
    SYNCHRONIZATION_MODE as NVCOMP_SYNCHRONIZATION_MODE,
    UPSTREAM_SAMPLES_COMMIT,
    UPSTREAM_SAMPLES_REPOSITORY,
    WARMUP as NVCOMP_WARMUP,
    WARM_E2E_SCOPE as NVCOMP_WARM_E2E_SCOPE,
)


REGISTRY_SCHEMA = "heterolz.baseline-registry.v1"
COVERAGE_SCHEMA = "heterolz.baseline-coverage.v1"
GPULZ_COVERAGE_SCHEMA = "heterolz.gpulz-coverage.v1"
CANON_PAPER_MANIFEST_SHA256 = (
    "82bb8fe07ed49660b9df30cfe2f9c6b1256fecf18754a5e34187d5857dc51d0e"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sample_map(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    if manifest.get("schema") != "heterolz.samples.v1":
        raise ValueError("sample manifest schema must be heterolz.samples.v1")
    selection_name = manifest.get("selection_name")
    files = manifest.get("files")
    if not isinstance(selection_name, str) or not selection_name:
        raise ValueError("sample manifest must have a selection_name")
    if not isinstance(files, list) or not files:
        raise ValueError("sample manifest must contain files")
    result: dict[str, dict[str, object]] = {}
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise ValueError(f"sample manifest entry {index} must be an object")
        relative_path = item.get("relative_path")
        size = item.get("size")
        digest = item.get("sha256")
        if (not isinstance(relative_path, str) or not relative_path or
                relative_path in result or not isinstance(size, int) or size < 0 or
                not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None):
            raise ValueError(f"invalid or duplicate sample manifest entry {index}")
        result[relative_path] = {
            "relative_path": relative_path,
            "size": size,
            "sha256": digest,
        }
    return result


def require_positive_number(value: object, label: str) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or
            not math.isfinite(float(value)) or float(value) <= 0):
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def validate_gpulz_coverage(
    coverage: dict[str, object],
    manifest_sha256: str,
    selection_name: str,
    samples: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    if coverage.get("schema") != GPULZ_COVERAGE_SCHEMA:
        raise ValueError("GPULZ coverage has an invalid schema")
    if (coverage.get("baseline") != "GPULZ" or
            coverage.get("repository") != UPSTREAM_REPOSITORY or
            coverage.get("commit") != UPSTREAM_COMMIT):
        raise ValueError("GPULZ coverage has an invalid baseline identity")
    if (coverage.get("sample_manifest_sha256") != manifest_sha256 or
            coverage.get("selection_name") != selection_name):
        raise ValueError("GPULZ coverage does not match the sample manifest")
    files = coverage.get("files")
    if not isinstance(files, list) or len(files) != len(samples):
        raise ValueError("GPULZ coverage must contain every sample exactly once")
    result: dict[str, dict[str, object]] = {}
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise ValueError(f"GPULZ coverage entry {index} must be an object")
        relative_path = item.get("relative_path")
        if not isinstance(relative_path, str) or relative_path in result:
            raise ValueError(f"GPULZ coverage entry {index} has an invalid path")
        expected = samples.get(relative_path)
        if expected is None or any(item.get(key) != expected[key] for key in ("size", "sha256")):
            raise ValueError(f"GPULZ coverage identity mismatch: {relative_path}")
        supported = int(expected["size"]) > 0 and int(expected["size"]) % 4 == 0
        expected_status = "supported" if supported else "unsupported"
        expected_reason = None if supported else (
            "empty_input" if int(expected["size"]) == 0
            else "input_size_not_multiple_of_uint32"
        )
        if item.get("status") != expected_status or item.get("reason") != expected_reason:
            raise ValueError(f"GPULZ coverage support decision mismatch: {relative_path}")
        result[relative_path] = dict(item)
    supported_count = sum(item["status"] == "supported" for item in result.values())
    if (coverage.get("total") != len(samples) or
            coverage.get("supported") != supported_count or
            coverage.get("unsupported") != len(samples) - supported_count):
        raise ValueError("GPULZ coverage counters are inconsistent")
    return result


def validate_gpulz_record(
    record: dict[str, object],
    manifest_sha256: str,
    selection_name: str,
    samples: dict[str, dict[str, object]],
) -> tuple[str, int]:
    if record.get("schema") != GPULZ_RUN_SCHEMA or record.get("status") != "complete":
        raise ValueError("GPULZ record must be a complete adapter result")
    repetition = record.get("repetition")
    if not isinstance(repetition, int) or isinstance(repetition, bool) or repetition <= 0:
        raise ValueError("GPULZ record repetition must be a positive integer")
    sample = record.get("sample")
    if not isinstance(sample, dict) or not isinstance(sample.get("relative_path"), str):
        raise ValueError("GPULZ record has an invalid sample")
    relative_path = sample["relative_path"]
    expected = samples.get(relative_path)
    if expected is None or any(sample.get(key) != expected[key] for key in ("size", "sha256")):
        raise ValueError(f"GPULZ record sample identity mismatch: {relative_path}")
    if (sample.get("manifest_sha256") != manifest_sha256 or
            sample.get("selection_name") != selection_name):
        raise ValueError(f"GPULZ record manifest identity mismatch: {relative_path}")
    support = record.get("support")
    if support != {"supported": True, "reason": None}:
        raise ValueError(f"GPULZ complete record must be supported: {relative_path}")
    baseline = record.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("GPULZ record baseline identity is missing")
    if (baseline.get("name") != "GPULZ" or
            baseline.get("repository") != UPSTREAM_REPOSITORY or
            baseline.get("commit") != UPSTREAM_COMMIT or
            baseline.get("parameters") != UPSTREAM_PARAMETERS or
            not isinstance(baseline.get("git_tree"), str) or
            re.fullmatch(r"[0-9a-f]{40}", str(baseline.get("git_tree"))) is None or
            not isinstance(baseline.get("binary_sha256"), str) or
            SHA256_RE.fullmatch(str(baseline.get("binary_sha256"))) is None):
        raise ValueError("GPULZ record baseline identity is invalid")
    environment = record.get("environment")
    if not isinstance(environment, dict) or not isinstance(environment.get("gpu"), dict):
        raise ValueError("GPULZ record environment identity is missing")
    for key in ("cuda_arch", "cuda_release", "nvcc_sha256", "nvidia_smi_sha256"):
        if not isinstance(environment.get(key), str) or not environment[key]:
            raise ValueError(f"GPULZ record environment field is missing: {key}")
    process = record.get("process")
    if not isinstance(process, dict) or process.get("returncode") != 0:
        raise ValueError("GPULZ record process did not complete successfully")
    stdout = process.get("stdout")
    stderr = process.get("stderr")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise ValueError("GPULZ record must preserve raw stdout and stderr")
    if (process.get("stdout_sha256") != sha256_text(stdout) or
            process.get("stderr_sha256") != sha256_text(stderr)):
        raise ValueError("GPULZ record raw-output hash mismatch")
    require_positive_number(process.get("cold_total_ns"), "GPULZ cold_total_ns")
    timing = record.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("GPULZ record timing is missing")
    require_positive_number(
        timing.get("upstream_compression_pipeline_gbps"),
        "GPULZ compression throughput",
    )
    require_positive_number(
        timing.get("upstream_decompression_pipeline_gbps"),
        "GPULZ decompression throughput",
    )
    if timing.get("process_cold_total_ns") != process.get("cold_total_ns"):
        raise ValueError("GPULZ cold timing fields disagree")
    compression = record.get("compression")
    if not isinstance(compression, dict):
        raise ValueError("GPULZ record compression fields are missing")
    ratio = require_positive_number(
        compression.get("upstream_original_over_compressed_ratio"),
        "GPULZ compression ratio",
    )
    percent = require_positive_number(
        compression.get("compressed_over_original_percent"),
        "GPULZ compressed percentage",
    )
    if not math.isclose(percent, 100.0 / ratio, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("GPULZ compression ratio conversion is inconsistent")
    if (record.get("correctness_scope") != "upstream_internal_roundtrip_only" or
            record.get("compressed_size_exact_available") is not False or
            record.get("verification") != {
                "upstream_failure_text_absent": True,
                "input_tail_fully_covered": True,
            }):
        raise ValueError("GPULZ correctness scope is invalid")
    return relative_path, repetition


def validate_nvcomp_record(
    record: dict[str, object],
    manifest_sha256: str,
    selection_name: str,
    samples: dict[str, dict[str, object]],
) -> tuple[str, int]:
    if record.get("schema") != NVCOMP_RUN_SCHEMA or record.get("status") != "complete":
        raise ValueError("nvCOMP record must be a complete adapter result")
    repetition = record.get("repetition")
    if not isinstance(repetition, int) or isinstance(repetition, bool) or repetition <= 0:
        raise ValueError("nvCOMP record repetition must be a positive integer")
    sample = record.get("sample")
    if not isinstance(sample, dict) or not isinstance(sample.get("relative_path"), str):
        raise ValueError("nvCOMP record has an invalid sample")
    relative_path = sample["relative_path"]
    expected = samples.get(relative_path)
    if expected is None or any(sample.get(key) != expected[key] for key in ("size", "sha256")):
        raise ValueError(f"nvCOMP record sample identity mismatch: {relative_path}")
    if (sample.get("manifest_sha256") != manifest_sha256 or
            sample.get("selection_name") != selection_name):
        raise ValueError(f"nvCOMP record manifest identity mismatch: {relative_path}")
    baseline = record.get("baseline")
    expected_parameters = {
        "chunk_size": NVCOMP_CHUNK_SIZE,
        "data_type": "char",
        "bitshuffle_mode": 0,
        "decompress_backend": NVCOMP_DECOMPRESS_BACKEND,
        "warmup": NVCOMP_WARMUP,
        "iterations": NVCOMP_ITERATIONS,
        "stream": NVCOMP_STREAM_MODE,
        "synchronization": NVCOMP_SYNCHRONIZATION_MODE,
        "warm_e2e_scope": NVCOMP_WARM_E2E_SCOPE,
    }
    if not isinstance(baseline, dict):
        raise ValueError("nvCOMP record baseline identity is missing")
    if (baseline.get("name") != "nvCOMP" or
            baseline.get("version") != NVCOMP_VERSION or
            baseline.get("official_samples_repository") != UPSTREAM_SAMPLES_REPOSITORY or
            baseline.get("official_samples_commit") != UPSTREAM_SAMPLES_COMMIT or
            baseline.get("distribution_mode") != "external-fixed-binary-package" or
            baseline.get("api") != "low-level-batched-cpp" or
            baseline.get("algorithm") != "LZ4" or
            baseline.get("parameters") != expected_parameters):
        raise ValueError("nvCOMP record baseline identity is invalid")
    for field in (
        "header_sha256", "library_sha256", "version_file_sha256",
        "license_file_sha256", "ldd_sha256", "ldd_stdout_sha256",
        "binary_sha256", "driver_source_sha256",
    ):
        if not isinstance(baseline.get(field), str) or SHA256_RE.fullmatch(baseline[field]) is None:
            raise ValueError(f"nvCOMP record baseline hash is invalid: {field}")
    if baseline.get("resolved_nvcomp_library") != baseline.get("library_path"):
        raise ValueError("nvCOMP record dynamic library identity is inconsistent")
    environment = record.get("environment")
    if not isinstance(environment, dict) or not isinstance(environment.get("gpu"), dict):
        raise ValueError("nvCOMP record environment identity is missing")
    process = record.get("process")
    if not isinstance(process, dict) or process.get("returncode") != 0:
        raise ValueError("nvCOMP record process did not complete successfully")
    stdout = process.get("stdout")
    stderr = process.get("stderr")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise ValueError("nvCOMP record must preserve raw stdout and stderr")
    if (process.get("stdout_sha256") != sha256_text(stdout) or
            process.get("stderr_sha256") != sha256_text(stderr)):
        raise ValueError("nvCOMP record raw-output hash mismatch")
    cold_total_ns = require_positive_number(process.get("cold_total_ns"), "nvCOMP cold_total_ns")
    chunking = record.get("chunking")
    input_bytes = int(expected["size"])
    expected_batch = (input_bytes - 1) // NVCOMP_CHUNK_SIZE + 1
    if chunking != {"chunk_size": NVCOMP_CHUNK_SIZE, "batch_size": expected_batch}:
        raise ValueError("nvCOMP chunking does not match the registered sample")
    compression = record.get("compression")
    if not isinstance(compression, dict):
        raise ValueError("nvCOMP record compression fields are missing")
    compressed_bytes = compression.get("compressed_bytes")
    percent = compression.get("compressed_over_original_percent")
    if (not isinstance(compressed_bytes, int) or isinstance(compressed_bytes, bool) or
            compressed_bytes <= 0):
        raise ValueError("nvCOMP compressed_bytes must be positive")
    actual_percent = require_positive_number(percent, "nvCOMP compressed percentage")
    if not math.isclose(
        actual_percent, compressed_bytes / input_bytes * 100.0,
        rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise ValueError("nvCOMP compression percentage is inconsistent")
    timing = record.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("nvCOMP record timing is missing")
    comp_ms = require_positive_number(timing.get("compression_kernel_ms"), "nvCOMP compression kernel")
    decomp_ms = require_positive_number(timing.get("decompression_kernel_ms"), "nvCOMP decompression kernel")
    comp_warm_ns = require_positive_number(timing.get("compression_warm_e2e_ns"), "nvCOMP compression warm")
    decomp_warm_ns = require_positive_number(timing.get("decompression_warm_e2e_ns"), "nvCOMP decompression warm")
    if timing.get("process_cold_total_ns") != process.get("cold_total_ns"):
        raise ValueError("nvCOMP cold timing fields disagree")
    expected_throughputs = {
        "compression_kernel_gbps": input_bytes / (comp_ms * 1_000_000.0),
        "decompression_kernel_gbps": input_bytes / (decomp_ms * 1_000_000.0),
        "compression_warm_e2e_gbps": input_bytes / comp_warm_ns,
        "decompression_warm_e2e_gbps": input_bytes / decomp_warm_ns,
    }
    for field, expected_value in expected_throughputs.items():
        actual = require_positive_number(timing.get(field), f"nvCOMP {field}")
        if not math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"nvCOMP throughput is inconsistent: {field}")
    if record.get("verification") != {
        "roundtrip": "passed",
        "scope": "nvcomp_internal_batched_roundtrip",
    }:
        raise ValueError("nvCOMP correctness scope is invalid")
    if cold_total_ns <= max(comp_warm_ns, decomp_warm_ns):
        raise ValueError("nvCOMP cold process time must exceed each warm phase")
    return relative_path, repetition


def build_gpulz_contract(
    manifest_path: Path,
    coverage_path: Path,
    record_paths: list[Path],
    repetitions: int,
    *,
    formal: bool,
    nvcomp_record_paths: list[Path] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    manifest = load_object(manifest_path)
    samples = sample_map(manifest)
    manifest_sha256 = sha256_file(manifest_path)
    selection_name = str(manifest["selection_name"])
    if formal and (
        manifest_sha256 != CANON_PAPER_MANIFEST_SHA256 or
        selection_name != "paper16" or len(samples) != 16 or repetitions < 9
    ):
        raise ValueError("formal GPULZ evidence requires registered paper16 and at least 9 repetitions")
    source_coverage = load_object(coverage_path)
    coverage_records = validate_gpulz_coverage(
        source_coverage, manifest_sha256, selection_name, samples
    )

    records: list[dict[str, object]] = []
    observed: dict[str, set[int]] = {path: set() for path in samples}
    baseline_identity: dict[str, object] | None = None
    environment_identity: dict[str, object] | None = None
    for path in record_paths:
        record = load_object(path)
        relative_path, repetition = validate_gpulz_record(
            record, manifest_sha256, selection_name, samples
        )
        if coverage_records[relative_path]["status"] != "supported":
            raise ValueError(f"GPULZ record exists for unsupported sample: {relative_path}")
        if repetition in observed[relative_path]:
            raise ValueError(f"duplicate GPULZ repetition: {relative_path} repetition {repetition}")
        observed[relative_path].add(repetition)
        current_baseline = record["baseline"]
        current_environment = record["environment"]
        if baseline_identity is None:
            baseline_identity = current_baseline
            environment_identity = current_environment
        elif current_baseline != baseline_identity or current_environment != environment_identity:
            raise ValueError("GPULZ identity or environment changed within the evidence set")
        records.append(record)

    expected_repetitions = set(range(1, repetitions + 1))
    normalized_files: list[dict[str, object]] = []
    for relative_path, sample in samples.items():
        source = coverage_records[relative_path]
        completed = sorted(observed[relative_path])
        if source["status"] == "supported" and set(completed) != expected_repetitions:
            raise ValueError(f"GPULZ repetitions are incomplete: {relative_path}")
        if source["status"] == "unsupported" and completed:
            raise ValueError(f"GPULZ unsupported sample has measured records: {relative_path}")
        normalized_files.append({
            **sample,
            "status": "measured" if source["status"] == "supported" else "unsupported",
            "reason": source["reason"],
            "required_repetitions": repetitions if source["status"] == "supported" else 0,
            "completed_repetitions": completed,
        })
    if baseline_identity is None or environment_identity is None:
        raise ValueError("GPULZ evidence contains no measured records")
    records.sort(key=lambda item: (str(item["sample"]["relative_path"]), int(item["repetition"])))
    measured_count = sum(item["status"] == "measured" for item in normalized_files)
    registry_baselines: list[dict[str, object]] = [{
        "name": "GPULZ",
        "identity": baseline_identity,
        "environment": environment_identity,
        "timing_scope": {
            "upstream_pipeline": (
                "CUDA events after input H2D and before result D2H; "
                "compression and decompression execute in one process"
            ),
            "cold_total": "adapter-observed process wall time",
        },
        "correctness_scope": "upstream_internal_roundtrip_only",
        "compressed_size_exact_available": False,
        "records": records,
    }]
    coverage_baselines: list[dict[str, object]] = [{
        "name": "GPULZ",
        "total": len(normalized_files),
        "measured": measured_count,
        "unsupported": len(normalized_files) - measured_count,
        "files": normalized_files,
    }]

    nvcomp_paths = nvcomp_record_paths or []
    if formal and not nvcomp_paths:
        raise ValueError("formal baseline evidence requires nvCOMP C++ batched records")
    if nvcomp_paths:
        nvcomp_records: list[dict[str, object]] = []
        nvcomp_observed: dict[str, set[int]] = {path: set() for path in samples}
        nvcomp_identity: dict[str, object] | None = None
        nvcomp_environment: dict[str, object] | None = None
        for path in nvcomp_paths:
            record = load_object(path)
            relative_path, repetition = validate_nvcomp_record(
                record, manifest_sha256, selection_name, samples
            )
            if repetition in nvcomp_observed[relative_path]:
                raise ValueError(
                    f"duplicate nvCOMP repetition: {relative_path} repetition {repetition}"
                )
            nvcomp_observed[relative_path].add(repetition)
            current_identity = record["baseline"]
            current_environment = record["environment"]
            if nvcomp_identity is None:
                nvcomp_identity = current_identity
                nvcomp_environment = current_environment
            elif (current_identity != nvcomp_identity or
                  current_environment != nvcomp_environment):
                raise ValueError("nvCOMP identity or environment changed within the evidence set")
            nvcomp_records.append(record)
        for relative_path in samples:
            if nvcomp_observed[relative_path] != expected_repetitions:
                raise ValueError(f"nvCOMP repetitions are incomplete: {relative_path}")
        if nvcomp_identity is None or nvcomp_environment is None:
            raise ValueError("nvCOMP evidence contains no measured records")
        nvcomp_records.sort(
            key=lambda item: (str(item["sample"]["relative_path"]), int(item["repetition"]))
        )
        registry_baselines.append({
            "name": "nvCOMP",
            "identity": nvcomp_identity,
            "environment": nvcomp_environment,
            "timing_scope": {
                "kernel": "CUDA events around one low-level batched API operation",
                "warm_e2e": "resident process with H2D, operation, synchronization, and exact D2H",
                "cold_total": "adapter-observed process wall time",
            },
            "correctness_scope": "nvcomp_internal_batched_roundtrip",
            "records": nvcomp_records,
        })
        coverage_baselines.append({
            "name": "nvCOMP",
            "total": len(samples),
            "measured": len(samples),
            "unsupported": 0,
            "files": [
                {
                    **sample,
                    "status": "measured",
                    "reason": None,
                    "required_repetitions": repetitions,
                    "completed_repetitions": sorted(nvcomp_observed[relative_path]),
                }
                for relative_path, sample in samples.items()
            ],
        })
    registry = {
        "schema": REGISTRY_SCHEMA,
        "sample_manifest_sha256": manifest_sha256,
        "selection_name": selection_name,
        "repetitions": repetitions,
        "baselines": registry_baselines,
    }
    coverage = {
        "schema": COVERAGE_SCHEMA,
        "sample_manifest_sha256": manifest_sha256,
        "selection_name": selection_name,
        "repetitions": repetitions,
        "baselines": coverage_baselines,
    }
    return registry, coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-manifest", type=Path, required=True)
    parser.add_argument("--gpulz-coverage", type=Path, required=True)
    parser.add_argument("--gpulz-record", type=Path, action="append", required=True)
    parser.add_argument("--nvcomp-record", type=Path, action="append")
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    try:
        outputs = (args.registry_output, args.coverage_output)
        if outputs[0].resolve(strict=False) == outputs[1].resolve(strict=False):
            raise ValueError("baseline registry and coverage outputs must be distinct")
        if any(path.exists() for path in outputs):
            raise FileExistsError("baseline evidence assembler refuses to overwrite output")
        registry, coverage = build_gpulz_contract(
            args.paper_manifest.resolve(strict=True),
            args.gpulz_coverage.resolve(strict=True),
            [path.resolve(strict=True) for path in args.gpulz_record],
            args.repetitions,
            formal=args.formal,
            nvcomp_record_paths=(
                [path.resolve(strict=True) for path in args.nvcomp_record]
                if args.nvcomp_record else None
            ),
        )
        write_json_atomic(args.registry_output, registry)
        write_json_atomic(args.coverage_output, coverage)
        return 0
    except Exception as exc:
        print(f"baseline evidence assembly failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
