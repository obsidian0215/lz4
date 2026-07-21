#!/usr/bin/env python3
"""Run and identify the formal nvCOMP C++ batched LZ4 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from gpulz_external_adapter import (
    inspect_cuda_environment,
    load_registered_sample,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_text_atomic,
)


SCHEMA = "heterolz.nvcomp-run.v1"
UPSTREAM_SAMPLES_REPOSITORY = "https://github.com/NVIDIA/CUDALibrarySamples.git"
UPSTREAM_SAMPLES_COMMIT = "b7bc246a3b655c9da7f3deb1b7e13a57b43b8a8e"
NVCOMP_VERSION = "5.3.0"
CHUNK_SIZE = 65536
WARMUP = 3
ITERATIONS = 1
DECOMPRESS_BACKEND = 0
STREAM_MODE = "explicit-created-default-flags"
SYNCHRONIZATION_MODE = "cuda-event-kernel-and-stream-sync-phase-boundaries"
WARM_E2E_SCOPE = "h2d-operation-synchronization-exact-d2h"
EXPECTED_FIELDS = {
    "schema",
    "api",
    "algorithm",
    "data_type",
    "bitshuffle_mode",
    "decompress_backend",
    "input_bytes",
    "compressed_bytes",
    "chunk_size",
    "batch_size",
    "warmup",
    "iterations",
    "compression_kernel_ms",
    "decompression_kernel_ms",
    "compression_warm_e2e_ns",
    "decompression_warm_e2e_ns",
    "roundtrip_ok",
    "stream",
    "synchronization",
    "warm_e2e_scope",
}
FAILURE_MARKERS = (
    "failed",
    "cuda error",
    "invalid device function",
    "illegal memory access",
    "out of memory",
    "segmentation fault",
    "core dumped",
)


def parse_version_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"(?:PACKAGE_VERSION|NVCOMP_VERSION)[^0-9]*([0-9]+\.[0-9]+\.[0-9]+)",
        text,
        re.IGNORECASE,
    )
    unique = sorted(set(matches))
    if unique != [NVCOMP_VERSION]:
        raise ValueError(f"nvCOMP version file must identify exactly {NVCOMP_VERSION}")
    return unique[0]


def inspect_nvcomp_package(header: Path, library: Path, version_file: Path,
                           license_file: Path) -> dict[str, object]:
    header_path = header.resolve(strict=True)
    library_path = library.resolve(strict=True)
    version_path = version_file.resolve(strict=True)
    license_path = license_file.resolve(strict=True)
    for path in (header_path, library_path, version_path, license_path):
        if not path.is_file():
            raise ValueError(f"nvCOMP package identity path is not a file: {path}")
    return {
        "name": "nvCOMP",
        "version": parse_version_file(version_path),
        "distribution_mode": "external-fixed-binary-package",
        "official_samples_repository": UPSTREAM_SAMPLES_REPOSITORY,
        "official_samples_commit": UPSTREAM_SAMPLES_COMMIT,
        "header_path": str(header_path),
        "header_sha256": sha256_file(header_path),
        "library_path": str(library_path),
        "library_sha256": sha256_file(library_path),
        "version_file_path": str(version_path),
        "version_file_sha256": sha256_file(version_path),
        "license_file_path": str(license_path),
        "license_file_sha256": sha256_file(license_path),
    }


def inspect_dynamic_linkage(ldd: Path, binary: Path,
                            library: Path) -> dict[str, object]:
    ldd_path = ldd.resolve(strict=True)
    completed = subprocess.run(
        [str(ldd_path), str(binary)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30.0,
    )
    if completed.returncode != 0:
        raise RuntimeError("ldd failed for the nvCOMP baseline binary")
    candidates: list[Path] = []
    for line in completed.stdout.splitlines():
        if "libnvcomp" not in line:
            continue
        match = re.search(r"=>\s+(\S+)", line)
        if match and match.group(1) != "not":
            candidates.append(Path(match.group(1)))
    resolved = [path.resolve(strict=True) for path in candidates]
    expected = library.resolve(strict=True)
    if resolved.count(expected) != 1:
        raise ValueError("nvCOMP baseline binary is not linked to the registered library")
    return {
        "ldd_path": str(ldd_path),
        "ldd_sha256": sha256_file(ldd_path),
        "resolved_nvcomp_library": str(expected),
        "ldd_stdout_sha256": sha256_text(completed.stdout),
    }


def parse_int(value: str, name: str, *, allow_zero: bool = False) -> int:
    if re.fullmatch(r"[0-9]+", value) is None:
        raise RuntimeError(f"nvCOMP returned invalid integer {name}: {value}")
    parsed = int(value)
    if parsed < 0 or (parsed == 0 and not allow_zero):
        raise RuntimeError(f"nvCOMP returned invalid integer {name}: {value}")
    return parsed


def parse_positive_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"nvCOMP returned invalid float {name}: {value}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise RuntimeError(f"nvCOMP returned invalid float {name}: {value}")
    return parsed


def parse_output(stdout: str, stderr: str, expected_input_size: int) -> dict[str, object]:
    combined = f"{stdout}\n{stderr}".lower()
    for marker in FAILURE_MARKERS:
        if marker in combined:
            raise RuntimeError(f"nvCOMP baseline reported failure marker: {marker}")
    fields: dict[str, str] = {}
    for index, line in enumerate(stdout.splitlines(), 1):
        if not line or "=" not in line:
            raise RuntimeError(f"nvCOMP output has an invalid line {index}")
        name, value = line.split("=", 1)
        if not name or name in fields:
            raise RuntimeError(f"nvCOMP output has a duplicate or empty field at line {index}")
        fields[name] = value
    if set(fields) != EXPECTED_FIELDS:
        raise RuntimeError("nvCOMP output fields do not match the fixed contract")
    expected_strings = {
        "schema": "heterolz.nvcomp-lz4-batched.v1",
        "api": "low-level-batched-cpp",
        "algorithm": "LZ4",
        "data_type": "char",
        "bitshuffle_mode": "0",
        "decompress_backend": str(DECOMPRESS_BACKEND),
        "chunk_size": str(CHUNK_SIZE),
        "warmup": str(WARMUP),
        "iterations": str(ITERATIONS),
        "roundtrip_ok": "true",
        "stream": STREAM_MODE,
        "synchronization": SYNCHRONIZATION_MODE,
        "warm_e2e_scope": WARM_E2E_SCOPE,
    }
    if any(fields[name] != expected for name, expected in expected_strings.items()):
        raise RuntimeError("nvCOMP output configuration does not match the formal baseline")
    input_bytes = parse_int(fields["input_bytes"], "input_bytes")
    compressed_bytes = parse_int(fields["compressed_bytes"], "compressed_bytes")
    batch_size = parse_int(fields["batch_size"], "batch_size")
    if input_bytes != expected_input_size:
        raise RuntimeError("nvCOMP output input size does not match the registered sample")
    expected_batch = (input_bytes - 1) // CHUNK_SIZE + 1
    if batch_size != expected_batch:
        raise RuntimeError("nvCOMP output batch size does not match chunking")
    compression_kernel_ms = parse_positive_float(
        fields["compression_kernel_ms"], "compression_kernel_ms"
    )
    decompression_kernel_ms = parse_positive_float(
        fields["decompression_kernel_ms"], "decompression_kernel_ms"
    )
    compression_warm_ns = parse_int(
        fields["compression_warm_e2e_ns"], "compression_warm_e2e_ns"
    )
    decompression_warm_ns = parse_int(
        fields["decompression_warm_e2e_ns"], "decompression_warm_e2e_ns"
    )
    return {
        "input_bytes": input_bytes,
        "compressed_bytes": compressed_bytes,
        "batch_size": batch_size,
        "compression_kernel_ms": compression_kernel_ms,
        "decompression_kernel_ms": decompression_kernel_ms,
        "compression_warm_e2e_ns": compression_warm_ns,
        "decompression_warm_e2e_ns": decompression_warm_ns,
    }


def execute(binary: Path, input_path: Path, gpu_index: int,
            timeout_seconds: float) -> tuple[subprocess.CompletedProcess[str], int]:
    argv = [
        str(binary),
        "--input", str(input_path),
        "--chunk-size", str(CHUNK_SIZE),
        "--warmup", str(WARMUP),
        "--iterations", str(ITERATIONS),
        "--gpu-index", str(gpu_index),
        "--decompress-backend", str(DECOMPRESS_BACKEND),
    ]
    started = time.perf_counter_ns()
    completed = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        env=os.environ.copy(),
    )
    return completed, time.perf_counter_ns() - started


def throughput_gbps(input_bytes: int, nanoseconds: float) -> float:
    return input_bytes / nanoseconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--driver-source", type=Path, required=True)
    parser.add_argument("--nvcomp-header", type=Path, required=True)
    parser.add_argument("--nvcomp-library", type=Path, required=True)
    parser.add_argument("--nvcomp-version-file", type=Path, required=True)
    parser.add_argument("--nvcomp-license-file", type=Path, required=True)
    parser.add_argument("--ldd", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--nvidia-smi", type=Path, required=True)
    parser.add_argument("--cuda-arch", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    try:
        if args.repetition <= 0:
            raise ValueError("repetition must be a positive integer")
        if args.gpu_index < 0:
            raise ValueError("GPU index must be nonnegative")
        if args.timeout_seconds <= 0 or not math.isfinite(args.timeout_seconds):
            raise ValueError("timeout must be a positive finite value")
        outputs = (args.metrics, args.stdout, args.stderr)
        if len({path.resolve(strict=False) for path in outputs}) != len(outputs):
            raise ValueError("nvCOMP adapter outputs must be distinct")
        if any(path.exists() for path in outputs):
            raise FileExistsError("nvCOMP adapter refuses to overwrite output")
        binary = args.binary.resolve(strict=True)
        driver_source = args.driver_source.resolve(strict=True)
        input_path = args.input.resolve(strict=True)
        if not binary.is_file() or not driver_source.is_file():
            raise ValueError("nvCOMP baseline binary or driver source is missing")
        sample = load_registered_sample(
            input_path, args.samples_root, args.sample_manifest.resolve(strict=True)
        )
        if int(sample["size"]) <= 0:
            raise ValueError("nvCOMP formal baseline requires a non-empty sample")
        package = inspect_nvcomp_package(
            args.nvcomp_header, args.nvcomp_library,
            args.nvcomp_version_file, args.nvcomp_license_file,
        )
        cuda = inspect_cuda_environment(
            args.nvcc, args.nvidia_smi, args.gpu_index, args.cuda_arch
        )
        linkage = inspect_dynamic_linkage(args.ldd, binary, args.nvcomp_library)
        completed, cold_total_ns = execute(
            binary, input_path, args.gpu_index, args.timeout_seconds
        )
        write_text_atomic(args.stdout, completed.stdout)
        write_text_atomic(args.stderr, completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"nvCOMP baseline exited with status {completed.returncode}")
        parsed = parse_output(completed.stdout, completed.stderr, int(sample["size"]))
        input_bytes = int(parsed["input_bytes"])
        record = {
            "schema": SCHEMA,
            "status": "complete",
            "repetition": args.repetition,
            "baseline": {
                **package,
                **linkage,
                "binary_path": str(binary),
                "binary_sha256": sha256_file(binary),
                "driver_source_path": str(driver_source),
                "driver_source_sha256": sha256_file(driver_source),
                "api": "low-level-batched-cpp",
                "algorithm": "LZ4",
                "parameters": {
                    "chunk_size": CHUNK_SIZE,
                    "data_type": "char",
                    "bitshuffle_mode": 0,
                    "decompress_backend": DECOMPRESS_BACKEND,
                    "warmup": WARMUP,
                    "iterations": ITERATIONS,
                    "stream": STREAM_MODE,
                    "synchronization": SYNCHRONIZATION_MODE,
                    "warm_e2e_scope": WARM_E2E_SCOPE,
                },
            },
            "environment": cuda,
            "sample": sample,
            "process": {
                "returncode": completed.returncode,
                "cold_total_ns": cold_total_ns,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "stdout_sha256": sha256_text(completed.stdout),
                "stderr_sha256": sha256_text(completed.stderr),
            },
            "chunking": {
                "chunk_size": CHUNK_SIZE,
                "batch_size": parsed["batch_size"],
            },
            "compression": {
                "compressed_bytes": parsed["compressed_bytes"],
                "compressed_over_original_percent": (
                    int(parsed["compressed_bytes"]) / input_bytes * 100.0
                ),
            },
            "timing": {
                "compression_kernel_ms": parsed["compression_kernel_ms"],
                "decompression_kernel_ms": parsed["decompression_kernel_ms"],
                "compression_warm_e2e_ns": parsed["compression_warm_e2e_ns"],
                "decompression_warm_e2e_ns": parsed["decompression_warm_e2e_ns"],
                "process_cold_total_ns": cold_total_ns,
                "compression_kernel_gbps": throughput_gbps(
                    input_bytes, float(parsed["compression_kernel_ms"]) * 1_000_000.0
                ),
                "decompression_kernel_gbps": throughput_gbps(
                    input_bytes, float(parsed["decompression_kernel_ms"]) * 1_000_000.0
                ),
                "compression_warm_e2e_gbps": throughput_gbps(
                    input_bytes, float(parsed["compression_warm_e2e_ns"])
                ),
                "decompression_warm_e2e_gbps": throughput_gbps(
                    input_bytes, float(parsed["decompression_warm_e2e_ns"])
                ),
            },
            "verification": {
                "roundtrip": "passed",
                "scope": "nvcomp_internal_batched_roundtrip",
            },
        }
        write_json_atomic(args.metrics, record)
        return 0
    except subprocess.TimeoutExpired as exc:
        print(f"nvCOMP adapter failed: timeout after {exc.timeout}s", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"nvCOMP adapter failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
