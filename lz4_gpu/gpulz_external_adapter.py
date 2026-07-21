#!/usr/bin/env python3
"""Run the fixed GPULZ baseline without vendoring its upstream source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


SCHEMA = "heterolz.gpulz-run.v1"
UPSTREAM_REPOSITORY = "https://github.com/hpdps-group/ICS23-GPULZ.git"
UPSTREAM_COMMIT = "314d6cfcfc6c8dbd5ee173859b672528f29d31ab"
UPSTREAM_PARAMETERS = {
    "block_size_bytes": 2048,
    "thread_size": 128,
    "window_size_symbols": 32,
    "input_type": "uint32_t",
}
NUMBER = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
FIELD_PATTERNS = {
    "original_over_compressed_ratio": re.compile(
        rf"^compression ratio:\s*({NUMBER})\s*$", re.MULTILINE
    ),
    "compression_pipeline_gbps": re.compile(
        rf"^compression e2e throughput:\s*({NUMBER})\s*GB/s\s*$", re.MULTILINE
    ),
    "decompression_pipeline_gbps": re.compile(
        rf"^decompression e2e throughput:\s*({NUMBER})\s*GB/s\s*$", re.MULTILINE
    ),
}
FAILURE_PATTERNS = (
    "verification failed",
    "cuda error",
    "invalid device function",
    "unspecified launch failure",
    "illegal memory access",
    "out of memory",
    "segmentation fault",
    "core dumped",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(value, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run_git(source_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"GPULZ source identity query failed: {' '.join(args)}")
    return completed.stdout.strip()


def inspect_source(source_root: Path) -> dict[str, object]:
    root = source_root.resolve(strict=True)
    commit = run_git(root, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"GPULZ commit mismatch: expected {UPSTREAM_COMMIT}, got {commit}")
    remote = run_git(root, "remote", "get-url", "origin")
    accepted_remotes = {UPSTREAM_REPOSITORY, UPSTREAM_REPOSITORY.removesuffix(".git")}
    if remote not in accepted_remotes:
        raise ValueError(f"GPULZ origin mismatch: {remote}")
    if run_git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("GPULZ tracked source files contain local modifications")
    tree = run_git(root, "rev-parse", "HEAD^{tree}")
    return {
        "repository": UPSTREAM_REPOSITORY,
        "origin": remote,
        "commit": commit,
        "git_tree": tree,
        "distribution_mode": "external-fixed-artifact",
        "license_status": "no-license-file-at-fixed-commit",
    }


def run_identity_tool(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"identity command failed: {argv[0]}")
    return completed.stdout


def inspect_cuda_environment(nvcc: Path, nvidia_smi: Path, gpu_index: int,
                             cuda_arch: str) -> dict[str, object]:
    if re.fullmatch(r"sm_[0-9]{2,3}", cuda_arch) is None:
        raise ValueError(f"invalid CUDA architecture: {cuda_arch}")
    nvcc_path = nvcc.resolve(strict=True)
    smi_path = nvidia_smi.resolve(strict=True)
    nvcc_output = run_identity_tool([str(nvcc_path), "--version"])
    release = re.findall(r"\brelease\s+([0-9]+(?:\.[0-9]+)*)", nvcc_output)
    if len(release) != 1:
        raise RuntimeError("nvcc output does not contain one CUDA release")
    smi_output = run_identity_tool([
        str(smi_path),
        "--query-gpu=index,name,uuid,pci.bus_id,driver_version",
        "--format=csv,noheader,nounits",
    ])
    rows = list(csv.reader(io.StringIO(smi_output)))
    selected = [row for row in rows if len(row) == 5 and row[0].strip() == str(gpu_index)]
    if len(selected) != 1:
        raise RuntimeError(f"nvidia-smi did not return exactly one GPU index {gpu_index}")
    row = [value.strip() for value in selected[0]]
    if any(not value for value in row):
        raise RuntimeError("nvidia-smi returned an empty GPU identity field")
    return {
        "cuda_arch": cuda_arch,
        "cuda_release": release[0],
        "nvcc_path": str(nvcc_path),
        "nvcc_sha256": sha256_file(nvcc_path),
        "nvidia_smi_path": str(smi_path),
        "nvidia_smi_sha256": sha256_file(smi_path),
        "gpu": {
            "index": int(row[0]),
            "name": row[1],
            "uuid": row[2],
            "pci_bus_id": row[3],
            "driver_version": row[4],
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def load_registered_sample(input_path: Path, samples_root: Path,
                           manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if manifest.get("schema") != "heterolz.samples.v1" or not isinstance(files, list):
        raise ValueError(f"invalid sample manifest: {manifest_path}")
    input_resolved = input_path.resolve(strict=True)
    root_resolved = samples_root.resolve(strict=True)
    matches: list[dict[str, object]] = []
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("relative_path"), str):
            raise ValueError(f"invalid sample entry in {manifest_path}")
        candidate = (root_resolved / item["relative_path"]).resolve(strict=False)
        if candidate == input_resolved:
            matches.append(item)
    if len(matches) != 1:
        raise ValueError(f"input is not uniquely registered by {manifest_path}: {input_path}")
    item = matches[0]
    size = input_resolved.stat().st_size
    digest = sha256_file(input_resolved)
    if item.get("size") != size or item.get("sha256") != digest:
        raise ValueError(f"registered sample identity mismatch: {input_path}")
    return {
        "relative_path": item["relative_path"],
        "size": size,
        "sha256": digest,
        "manifest_sha256": sha256_file(manifest_path),
        "selection_name": manifest.get("selection_name"),
    }


def support_status(sample: dict[str, object]) -> tuple[bool, str | None]:
    size = int(sample["size"])
    if size == 0:
        return False, "empty_input"
    if size % 4 != 0:
        return False, "input_size_not_multiple_of_uint32"
    return True, None


def build_coverage(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if manifest.get("schema") != "heterolz.samples.v1" or not isinstance(files, list):
        raise ValueError(f"invalid sample manifest: {manifest_path}")
    coverage: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError(f"invalid sample entry in {manifest_path}")
        relative_path = item.get("relative_path")
        size = item.get("size")
        digest = item.get("sha256")
        if (not isinstance(relative_path, str) or not relative_path or
                relative_path in seen or not isinstance(size, int) or size < 0 or
                not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError(f"invalid or duplicate sample entry in {manifest_path}")
        seen.add(relative_path)
        supported, reason = support_status({"size": size})
        coverage.append({
            "relative_path": relative_path,
            "size": size,
            "sha256": digest,
            "status": "supported" if supported else "unsupported",
            "reason": reason,
        })
    supported_count = sum(item["status"] == "supported" for item in coverage)
    return {
        "schema": "heterolz.gpulz-coverage.v1",
        "baseline": "GPULZ",
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "restriction": "input size must be a positive multiple of uint32_t",
        "sample_manifest_sha256": sha256_file(manifest_path),
        "selection_name": manifest.get("selection_name"),
        "total": len(coverage),
        "supported": supported_count,
        "unsupported": len(coverage) - supported_count,
        "files": coverage,
    }


def parse_output(stdout: str, stderr: str) -> dict[str, float]:
    combined = f"{stdout}\n{stderr}".lower()
    for marker in FAILURE_PATTERNS:
        if marker in combined:
            raise RuntimeError(f"GPULZ reported failure marker: {marker}")
    values: dict[str, float] = {}
    for name, pattern in FIELD_PATTERNS.items():
        matches = pattern.findall(stdout)
        if len(matches) != 1:
            raise RuntimeError(f"GPULZ output requires exactly one {name}; found {len(matches)}")
        value = float(matches[0])
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"GPULZ returned invalid {name}: {value}")
        values[name] = value
    return values


def ensure_output_paths(paths: tuple[Path, ...], protected: tuple[Path, ...]) -> None:
    resolved = [path.resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("GPULZ output paths must be distinct")
    protected_resolved = {path.resolve(strict=False) for path in protected}
    if any(path in protected_resolved for path in resolved):
        raise ValueError("GPULZ output path aliases an input artifact")
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("GPULZ adapter refuses to overwrite: " + ",".join(existing))


def execute(binary: Path, input_path: Path, cwd: Path, timeout_seconds: float,
            environment: dict[str, str] | None = None,
            binary_args: tuple[str, ...] = ()) -> tuple[subprocess.CompletedProcess[str], int]:
    started = time.perf_counter_ns()
    completed = subprocess.run(
        [str(binary), *binary_args, "-i", str(input_path)],
        cwd=cwd,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
    )
    elapsed = time.perf_counter_ns() - started
    return completed, elapsed


def make_record(source: dict[str, object], cuda: dict[str, object], binary: Path,
                sample: dict[str, object], repetition: int, supported: bool,
                reason: str | None) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "status": "pending" if supported else "unsupported",
        "repetition": repetition,
        "baseline": {
            "name": "GPULZ",
            **source,
            "binary_sha256": sha256_file(binary),
            "parameters": UPSTREAM_PARAMETERS,
        },
        "environment": cuda,
        "sample": sample,
        "support": {"supported": supported, "reason": reason},
        "correctness_scope": "upstream_internal_roundtrip_only",
        "compressed_size_exact_available": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--binary", type=Path, required=True)
    run_parser.add_argument("--source-root", type=Path, required=True)
    run_parser.add_argument("--nvcc", type=Path, required=True)
    run_parser.add_argument("--nvidia-smi", type=Path, required=True)
    run_parser.add_argument("--gpu-index", type=int, required=True)
    run_parser.add_argument("--cuda-arch", required=True)
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--samples-root", type=Path, required=True)
    run_parser.add_argument("--sample-manifest", type=Path, required=True)
    run_parser.add_argument("--repetition", type=int, required=True)
    run_parser.add_argument("--metrics", type=Path, required=True)
    run_parser.add_argument("--stdout", type=Path, required=True)
    run_parser.add_argument("--stderr", type=Path, required=True)
    run_parser.add_argument("--timeout-seconds", type=float, default=300.0)
    coverage_parser = subparsers.add_parser("coverage")
    coverage_parser.add_argument("--sample-manifest", type=Path, required=True)
    coverage_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "coverage":
            if args.output.exists():
                raise FileExistsError(f"GPULZ adapter refuses to overwrite: {args.output}")
            manifest = args.sample_manifest.resolve(strict=True)
            write_json_atomic(args.output, build_coverage(manifest))
            return 0
        if args.timeout_seconds <= 0 or not math.isfinite(args.timeout_seconds):
            raise ValueError("timeout must be a positive finite value")
        if args.repetition <= 0:
            raise ValueError("repetition must be a positive integer")
        binary = args.binary.resolve(strict=True)
        if not binary.is_file():
            raise ValueError(f"GPULZ binary is not a file: {binary}")
        source_root = args.source_root.resolve(strict=True)
        input_path = args.input.resolve(strict=True)
        ensure_output_paths(
            (args.metrics, args.stdout, args.stderr),
            (binary, input_path, args.sample_manifest, source_root),
        )
        source = inspect_source(source_root)
        cuda = inspect_cuda_environment(
            args.nvcc, args.nvidia_smi, args.gpu_index, args.cuda_arch
        )
        sample = load_registered_sample(input_path, args.samples_root, args.sample_manifest)
        supported, reason = support_status(sample)
        record = make_record(
            source, cuda, binary, sample, args.repetition, supported, reason
        )
        if not supported:
            write_text_atomic(args.stdout, "")
            write_text_atomic(args.stderr, "")
            write_json_atomic(args.metrics, record)
            return 0

        completed, cold_total_ns = execute(
            binary, input_path, source_root, args.timeout_seconds, os.environ.copy()
        )
        write_text_atomic(args.stdout, completed.stdout)
        write_text_atomic(args.stderr, completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"GPULZ exited with status {completed.returncode}")
        parsed = parse_output(completed.stdout, completed.stderr)
        ratio = parsed["original_over_compressed_ratio"]
        record.update({
            "status": "complete",
            "process": {
                "returncode": completed.returncode,
                "cold_total_ns": cold_total_ns,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "stdout_sha256": sha256_text(completed.stdout),
                "stderr_sha256": sha256_text(completed.stderr),
            },
            "timing": {
                "upstream_compression_pipeline_gbps": parsed["compression_pipeline_gbps"],
                "upstream_decompression_pipeline_gbps": parsed["decompression_pipeline_gbps"],
                "upstream_boundary": (
                    "CUDA events after input H2D and before result D2H; "
                    "compression and decompression execute in one process"
                ),
                "process_cold_total_ns": cold_total_ns,
            },
            "compression": {
                "upstream_original_over_compressed_ratio": ratio,
                "compressed_over_original_percent": 100.0 / ratio,
            },
            "verification": {
                "upstream_failure_text_absent": True,
                "input_tail_fully_covered": True,
            },
        })
        write_json_atomic(args.metrics, record)
        return 0
    except subprocess.TimeoutExpired as exc:
        print(f"GPULZ adapter failed: timeout after {exc.timeout}s", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"GPULZ adapter failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
