#!/usr/bin/env python3
"""Verify LZ4TP1 correctness and interoperability on registered real samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


FORMAL_N = (1, 2, 4, 8)
CANON_CORRECTNESS_MANIFEST_SHA256 = "56d52716825c68bade8e032237fa9787a05606d177f733bfc2a14334ac837153"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


class CommandRunner:
    def __init__(self, commands_log: Path | None, gate_log: Path | None) -> None:
        self.commands_log = commands_log
        self.gate_log = gate_log
        for path in (commands_log, gate_log):
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

    def append(self, path: Path | None, text: str) -> None:
        if path:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(text)

    def run(self, argv: list[str], venue: str | None = None) -> None:
        env = os.environ.copy()
        prefix = ""
        if venue:
            env["FORCE_OPENCL_DEVICE"] = venue
            prefix = f"FORCE_OPENCL_DEVICE={venue} "
        command_text = prefix + shlex.join(argv)
        self.append(self.commands_log, command_text + "\n")
        completed = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.append(
            self.gate_log,
            f"$ {command_text}\n{completed.stdout}{completed.stderr}[exit={completed.returncode}]\n",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"command failed: {command_text}")


def load_manifest(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "heterolz.samples.v1":
        raise ValueError("sample manifest schema must be heterolz.samples.v1")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("sample manifest files must be a non-empty list")
    if "file_count" in data and data["file_count"] != len(files):
        raise ValueError("sample manifest file_count does not match files")
    seen: set[str] = set()
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise ValueError(f"manifest record {index} must be an object")
        relative_path = record.get("relative_path")
        size = record.get("size")
        sha256 = record.get("sha256")
        relative = Path(relative_path) if isinstance(relative_path, str) else None
        if (not isinstance(relative_path, str) or not relative_path or
                relative is None or relative.is_absolute() or ".." in relative.parts or
                not isinstance(size, int) or size < 0 or
                not isinstance(sha256, str) or len(sha256) != 64 or
                any(char not in "0123456789abcdef" for char in sha256)):
            raise ValueError(f"manifest record {index} is incomplete")
        if relative_path in seen:
            raise ValueError(f"duplicate sample path in manifest: {relative_path}")
        seen.add(relative_path)
    return data


def require_executable(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} is not executable: {resolved}")
    return resolved


def verify_equal(original: Path, restored: Path, expected_sha256: str) -> None:
    if original.stat().st_size != restored.stat().st_size:
        raise RuntimeError(f"roundtrip size mismatch: {original.name}")
    if sha256_file(restored) != expected_sha256:
        raise RuntimeError(f"roundtrip hash mismatch: {original.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, default=Path("/root/samples"))
    parser.add_argument("--binary", type=Path, default=Path("./lz4_gpu"))
    parser.add_argument("--reference-decoder", type=Path, default=Path("./tp_ref_decode"))
    parser.add_argument("--bridge", type=Path, default=Path("./tp_to_lz4"))
    parser.add_argument("--fingerprint", type=Path)
    parser.add_argument("--device-info", type=Path)
    parser.add_argument("--stock-lz4", default="lz4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commands-log", type=Path)
    parser.add_argument("--gate-log", type=Path)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--n", default="1,2,4,8")
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--hash-log", type=int, default=14)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-determinism", action="store_true")
    parser.add_argument("--skip-cross-venue", action="store_true")
    parser.add_argument("--skip-bridge", action="store_true")
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    result: dict[str, object] = {
        "schema": "heterolz.verification.v1",
        "status": "failed",
        "all_inputs_passed": False,
        "checks": {
            "roundtrip": "failed",
            "reference_decode": "failed",
            "bridge_interop": "skipped" if args.skip_bridge else "failed",
            "determinism": "skipped" if args.skip_determinism else "failed",
            "cross_venue": "skipped" if args.skip_cross_venue else "failed",
        },
        "failures": [],
        "inputs": [],
        "samples": [],
    }

    try:
        n_values = tuple(int(value) for value in args.n.split(",") if value)
        if (not n_values or len(set(n_values)) != len(n_values) or
                any(value not in FORMAL_N for value in n_values)):
            raise ValueError("--n must contain values from 1,2,4,8")
        if args.block_size <= 0 or not 11 <= args.hash_log <= 15:
            raise ValueError("invalid block size or hash log")
        if args.formal:
            if n_values != FORMAL_N or args.limit is not None:
                raise ValueError("formal mode requires N=1,2,4,8 and the complete manifest")
            if args.block_size != 65536 or args.hash_log != 14:
                raise ValueError("formal mode requires block_size=65536 and hash_log=14")
            if args.skip_determinism or args.skip_cross_venue or args.skip_bridge:
                raise ValueError("formal mode requires determinism, cross-venue, and bridge checks")
            if not args.fingerprint or not args.device_info:
                raise ValueError("formal mode requires fingerprint and device-info evidence")

        artifact_paths = [path.resolve() for path in (args.output, args.commands_log, args.gate_log) if path]
        if len(set(artifact_paths)) != len(artifact_paths):
            raise ValueError("verification output, commands log, and gate log must use distinct paths")
        protected_paths = {
            args.manifest.resolve(),
            args.binary.resolve(),
            args.reference_decoder.resolve(),
            args.bridge.resolve(),
        }
        if args.fingerprint:
            protected_paths.add(args.fingerprint.resolve())
        if args.device_info:
            protected_paths.add(args.device_info.resolve())
        if set(artifact_paths) & protected_paths:
            raise ValueError("verification artifacts must not replace inputs or executables")
        sample_root_resolved = args.sample_root.resolve()
        if args.formal and any(path == sample_root_resolved or sample_root_resolved in path.parents
                               for path in [*artifact_paths, args.work_root.resolve()]):
            raise ValueError("formal verification artifacts and work files must stay outside the sample root")

        manifest = load_manifest(args.manifest)
        if args.formal and sha256_file(args.manifest) != CANON_CORRECTNESS_MANIFEST_SHA256:
            raise ValueError("formal mode requires the registered 25-file correctness manifest")
        records = manifest["files"]
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            records = records[: args.limit]
        normalized_inputs: list[dict[str, object]] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"manifest record {index} must be an object")
            normalized_inputs.append(
                {
                    "relative_path": record.get("relative_path"),
                    "size": record.get("size"),
                    "sha256": record.get("sha256"),
                }
            )
        result["manifest_sha256"] = sha256_file(args.manifest)
        result["selection_name"] = manifest.get("selection_name")
        result["parameters"] = {"block_size": args.block_size, "hash_log": args.hash_log}
        result["n_values"] = list(n_values)
        result["inputs"] = normalized_inputs
        result["venues"] = ["GPU", "CPU"] if not args.skip_cross_venue else ["GPU"]
        binary = require_executable(args.binary, "binary")
        reference_decoder = require_executable(args.reference_decoder, "reference decoder")
        bridge = require_executable(args.bridge, "bridge") if not args.skip_bridge else None
        stock_lz4 = shutil.which(args.stock_lz4)
        if not args.skip_bridge and not stock_lz4:
            raise ValueError(f"stock lz4 CLI unavailable: {args.stock_lz4}")
        if args.formal:
            fingerprint = json.loads(args.fingerprint.read_text(encoding="utf-8"))
            device_info = json.loads(args.device_info.read_text(encoding="utf-8"))
            if (fingerprint.get("schema") != "heterolz.source-fingerprint.v1" or
                    device_info.get("schema") != "heterolz.device-matrix.v1"):
                raise ValueError("invalid formal source or device identity evidence")
            source_fingerprint = fingerprint.get("source_fingerprint")
            git_commit = fingerprint.get("git_commit")
            if (not isinstance(source_fingerprint, str) or len(source_fingerprint) != 64 or
                    not isinstance(git_commit, str) or len(git_commit) != 40):
                raise ValueError("incomplete formal source identity evidence")
            venue_devices = device_info.get("venues")
            if not isinstance(venue_devices, dict) or set(venue_devices) != {"GPU", "CPU"}:
                raise ValueError("device-info evidence must identify GPU and CPU venues")
            for venue in ("GPU", "CPU"):
                record = venue_devices.get(venue)
                if not isinstance(record, dict) or record.get("schema") != "heterolz.device-info.v1":
                    raise ValueError(f"device-info evidence is invalid for {venue}")
                for field in ("device_name", "platform_name", "driver_version"):
                    if not isinstance(record.get(field), str) or not record[field]:
                        raise ValueError(f"device-info evidence is missing {venue}.{field}")
            result["source_identity"] = {
                "git_commit": git_commit,
                "source_fingerprint": source_fingerprint,
                "binary_sha256": sha256_file(binary),
                "device_info_sha256": sha256_file(args.device_info),
                "venue_devices": venue_devices,
            }
        args.work_root.mkdir(parents=True, exist_ok=True)
        runner = CommandRunner(args.commands_log, args.gate_log)

        with tempfile.TemporaryDirectory(prefix="heterolz-real-verify-", dir=args.work_root) as temp_root:
            root = Path(temp_root)
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    raise ValueError(f"manifest record {index} must be an object")
                relative_path = record.get("relative_path")
                expected_size = record.get("size")
                expected_sha256 = record.get("sha256")
                if not isinstance(relative_path, str) or not isinstance(expected_size, int) or not isinstance(expected_sha256, str):
                    raise ValueError(f"manifest record {index} is incomplete")
                sample = (args.sample_root / relative_path).resolve()
                sample_root = args.sample_root.resolve()
                if sample_root not in sample.parents:
                    raise ValueError(f"sample path escapes root: {relative_path}")
                if not sample.is_file() or sample.stat().st_size != expected_size:
                    raise RuntimeError(f"sample size mismatch: {relative_path}")
                if sha256_file(sample) != expected_sha256:
                    raise RuntimeError(f"sample SHA256 mismatch: {relative_path}")

                sample_key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
                sample_dir = root / f"{Path(relative_path).name}.{sample_key}"
                sample_dir.mkdir()
                sample_result = {
                    "relative_path": relative_path,
                    "size": expected_size,
                    "sha256": expected_sha256,
                    "n": {},
                }
                for n in n_values:
                    frame_gpu = sample_dir / f"n{n}.gpu.lz4tp"
                    restored_gpu = sample_dir / f"n{n}.gpu.restored"
                    common = ["-B", str(args.block_size), "--d-bits", str(args.hash_log)]
                    runner.run([str(binary), "--twophase", "-N", str(n), *common, str(sample), "-o", str(frame_gpu)], "GPU")
                    runner.run([str(binary), "--twophase", "-d", *common, str(frame_gpu), "-o", str(restored_gpu)], "GPU")
                    verify_equal(sample, restored_gpu, expected_sha256)
                    runner.run([str(reference_decoder), str(frame_gpu), str(sample)])

                    n_result: dict[str, object] = {
                        "gpu_frame_sha256": sha256_file(frame_gpu),
                        "roundtrip": "passed",
                        "reference_decode": "passed",
                    }
                    if not args.skip_determinism:
                        repeat = sample_dir / f"n{n}.gpu.repeat.lz4tp"
                        runner.run([str(binary), "--twophase", "-N", str(n), *common, str(sample), "-o", str(repeat)], "GPU")
                        if sha256_file(repeat) != n_result["gpu_frame_sha256"]:
                            raise RuntimeError(f"same-device determinism failed: {relative_path} N={n}")
                        n_result["determinism"] = "passed"

                    if not args.skip_cross_venue:
                        restored_cpu = sample_dir / f"n{n}.gpu-to-cpu.restored"
                        runner.run([str(binary), "--twophase", "-d", *common, str(frame_gpu), "-o", str(restored_cpu)], "CPU")
                        verify_equal(sample, restored_cpu, expected_sha256)
                        frame_cpu = sample_dir / f"n{n}.cpu.lz4tp"
                        restored_cross = sample_dir / f"n{n}.cpu-to-gpu.restored"
                        runner.run([str(binary), "--twophase", "-N", str(n), *common, str(sample), "-o", str(frame_cpu)], "CPU")
                        runner.run([str(binary), "--twophase", "-d", *common, str(frame_cpu), "-o", str(restored_cross)], "GPU")
                        verify_equal(sample, restored_cross, expected_sha256)
                        runner.run([str(reference_decoder), str(frame_cpu), str(sample)])
                        cpu_sha256 = sha256_file(frame_cpu)
                        if cpu_sha256 != n_result["gpu_frame_sha256"]:
                            raise RuntimeError(f"cross-venue determinism failed: {relative_path} N={n}")
                        n_result["cpu_frame_sha256"] = cpu_sha256
                        n_result["cross_venue"] = "passed"

                    if n == 1 and not args.skip_bridge:
                        standard_frame = sample_dir / "n1.standard.lz4"
                        stock_restored = sample_dir / "n1.stock.restored"
                        runner.run([str(bridge), str(frame_gpu), str(standard_frame)])
                        runner.run([str(stock_lz4), "-q", "-d", "-f", str(standard_frame), str(stock_restored)])
                        verify_equal(sample, stock_restored, expected_sha256)
                        n_result["bridge_interop"] = "passed"
                    sample_result["n"][str(n)] = n_result
                result["samples"].append(sample_result)
                shutil.rmtree(sample_dir)

        result["status"] = "passed"
        result["all_inputs_passed"] = True
        result["checks"]["roundtrip"] = "passed"
        result["checks"]["reference_decode"] = "passed"
        if not args.skip_bridge:
            result["checks"]["bridge_interop"] = "passed"
        if not args.skip_determinism:
            result["checks"]["determinism"] = "passed"
        if not args.skip_cross_venue:
            result["checks"]["cross_venue"] = "passed"
        if args.gate_log:
            with args.gate_log.open("a", encoding="utf-8") as handle:
                handle.write("FORMAL-GATE-PASS\n" if args.formal else "QUICK-GATE-PASS\n")
    except Exception as exc:
        result["failures"].append(str(exc))
        if args.gate_log:
            args.gate_log.parent.mkdir(parents=True, exist_ok=True)
            with args.gate_log.open("a", encoding="utf-8") as handle:
                handle.write(f"GATE-FAIL: {exc}\n")
    write_json_atomic(args.output, result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
