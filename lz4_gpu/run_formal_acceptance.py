#!/usr/bin/env python3
"""Build one HeteroLZ formal run and promote it after fixed audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
except ImportError:  # Formal runs are Linux-only; keep --help and syntax checks portable.
    fcntl = None


ADMISSION_ARTIFACTS = (
    "binary.sha256",
    "calibration_manifest.json",
    "commands.log",
    "compile.log",
    "correctness_manifest.json",
    "device_info.json",
    "expected_artifacts.txt",
    "fingerprint.json",
    "gate.log",
    "result_audit.py",
    "run_manifest.json",
    "run_meta.txt",
    "samples_manifest.json",
    "verification.json",
)

PERFORMANCE_ARTIFACTS = tuple(sorted({
    *ADMISSION_ARTIFACTS,
    "admission_audit.json",
    "admission_manifest.json",
    "raw.stdout",
    "raw_results.csv",
    "summary.csv",
}))

CANON_SELECTOR_MANIFEST_SHA256 = "3447f8ff3a04eb1f92285f472eb700efccdc1560ee4c5bc58fc050d739697fa1"
CANON_CALIBRATION_MANIFEST_SHA256 = "c63be7e92c94cab029beed414f58c4034407a5001e95397523dba74bd385aba9"
CANON_CORRECTNESS_MANIFEST_SHA256 = "56d52716825c68bade8e032237fa9787a05606d177f733bfc2a14334ac837153"
CANON_RESULT_AUDITOR_SHA256 = "f47af8171bb5e18cdef4cfd3bff41fda5ebd9ff22e5ad9f3b7251d013a4dcf36"
CANON_RESULTS_ROOT = Path("/root/heterolz-formal-results")
RUN_ID_RE = re.compile(r"heterolz-(?:admission|performance)-\d{8}T\d{12}Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json_atomic(path: Path, data: object) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def run(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None,
        stdout_path: Path | None = None, append: bool = False) -> None:
    mode = "a" if append else "w"
    output = stdout_path.open(mode, encoding="utf-8") if stdout_path else subprocess.DEVNULL
    try:
        completed = subprocess.run(argv, cwd=cwd, env=env, text=True, stdout=output,
                                   stderr=subprocess.STDOUT)
    finally:
        if stdout_path:
            output.close()
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit {completed.returncode}: {argv}")


def load_samples(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files")
    if data.get("schema") != "heterolz.samples.v1" or not isinstance(files, list) or not files:
        raise ValueError(f"invalid sample manifest: {path}")
    return files


def query_device_info(binary: Path, cwd: Path, venue: str) -> dict[str, object]:
    env = os.environ.copy()
    env["FORCE_OPENCL_DEVICE"] = venue
    completed = subprocess.run(
        [str(binary), "--device-info"], cwd=cwd, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{venue} device identity query failed: {completed.stderr.strip()}")
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{venue} device identity query returned invalid JSON") from exc
    if data.get("schema") != "heterolz.device-info.v1":
        raise RuntimeError(f"{venue} device identity query returned an invalid schema")
    for field in ("device_name", "platform_name", "driver_version"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise RuntimeError(f"{venue} device identity is missing {field}")
    return data


def release_lock(handle: object | None, path: Path | None) -> None:
    del path  # The stable /tmp lock inode must remain to avoid unlock/unlink races.
    if handle is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def validate_results_root(root: Path) -> None:
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_dir() or RUN_ID_RE.fullmatch(entry.name) is None:
            raise ValueError(f"formal result root contains an unexpected entry: {entry.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--sample-root", type=Path, default=Path("/root/samples"))
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--correctness-manifest", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=CANON_RESULTS_ROOT)
    parser.add_argument("--venue", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--admission-dir", type=Path)
    args = parser.parse_args()

    staging: Path | None = None
    staging_root: Path | None = None
    results_root: Path | None = None
    lz4_gpu: Path | None = None
    lock_handle: object | None = None
    lock_path: Path | None = None
    try:
        if args.performance and args.repetitions < 9:
            raise ValueError("formal performance requires at least nine repetitions")
        if not args.performance and args.admission_dir:
            raise ValueError("--admission-dir is only valid with --performance")
        if args.performance and not args.admission_dir:
            raise ValueError("formal performance requires an audited admission directory")
        repo = args.repo.resolve()
        lz4_gpu = repo / "lz4_gpu"
        if fcntl is None:
            raise RuntimeError("formal runs require Linux file locking")
        lock_key = hashlib.sha256(str(repo).encode("utf-8")).hexdigest()[:16]
        lock_path = Path("/tmp") / f"heterolz-formal-run-{lock_key}.lock"
        lock_handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_handle.close()
            lock_handle = None
            lock_path = None
            raise RuntimeError("another formal run is using this source tree") from exc
        for path in (args.selector_manifest, args.calibration_manifest,
                     args.correctness_manifest, args.auditor):
            if not path.resolve().is_file():
                raise ValueError(f"required input is missing: {path}")
        if (sha256_file(args.selector_manifest.resolve()) != CANON_SELECTOR_MANIFEST_SHA256 or
                sha256_file(args.calibration_manifest.resolve()) != CANON_CALIBRATION_MANIFEST_SHA256 or
                sha256_file(args.correctness_manifest.resolve()) != CANON_CORRECTNESS_MANIFEST_SHA256 or
                sha256_file(args.auditor.resolve()) != CANON_RESULT_AUDITOR_SHA256):
            raise ValueError("formal run requires the registered manifests and fixed result auditor")
        sample_root = args.sample_root.resolve()
        if sample_root != Path("/root/samples"):
            raise ValueError("formal runs require the canonical /root/samples root")
        results_root = args.results_root.resolve()
        if results_root != CANON_RESULTS_ROOT:
            raise ValueError(f"formal runs require the canonical {CANON_RESULTS_ROOT} result root")
        if (results_root == sample_root or sample_root in results_root.parents or
                results_root in sample_root.parents or results_root == repo or
                repo in results_root.parents or results_root in repo.parents):
            raise ValueError("formal results must stay outside the sample and source trees")
        results_root.mkdir(parents=True, exist_ok=True)
        validate_results_root(results_root)
        run_kind = "performance" if args.performance else "admission"
        run_id = f"heterolz-{run_kind}-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        final = results_root / run_id
        if final.exists():
            raise RuntimeError(f"run path already exists: {run_id}")
        staging_root = Path(tempfile.mkdtemp(prefix=f".{run_id}.staging-", dir=results_root))
        staging = staging_root / run_id
        staging.mkdir()

        compile_log = staging / "compile.log"
        run(["make", "clean"], cwd=lz4_gpu, stdout_path=compile_log)
        run(
            ["make", "CFLAGS=-O2 -Wall -Wextra -Werror -DCL_TARGET_OPENCL_VERSION=200 "
             "-I. -I../lib -I.. -pthread", "all", "interop-tools"],
            cwd=lz4_gpu,
            stdout_path=compile_log,
            append=True,
        )
        with compile_log.open("a", encoding="utf-8") as handle:
            handle.write("BUILD-OK\n")

        binary = lz4_gpu / "lz4_gpu"
        reference_decoder = lz4_gpu / "tp_ref_decode"
        bridge = lz4_gpu / "tp_to_lz4"
        binary_sha = sha256_file(binary)
        write_text_atomic(staging / "binary.sha256", f"{binary_sha}  lz4_gpu\n")

        fingerprint_path = staging / "fingerprint.json"
        run(
            [sys.executable, str(lz4_gpu / "source_fingerprint.py"), "--repo", str(repo),
             "--output", str(fingerprint_path), "--require-clean"],
            cwd=repo,
        )
        fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        commit = fingerprint["git_commit"]

        venue_devices = {
            venue: query_device_info(binary, lz4_gpu, venue)
            for venue in ("GPU", "CPU")
        }
        write_json_atomic(
            staging / "device_info.json",
            {"schema": "heterolz.device-matrix.v1", "venues": venue_devices},
        )
        device_info = venue_devices[args.venue]
        env = os.environ.copy()
        env["FORCE_OPENCL_DEVICE"] = args.venue

        shutil.copyfile(args.selector_manifest.resolve(), staging / "samples_manifest.json")
        shutil.copyfile(args.calibration_manifest.resolve(), staging / "calibration_manifest.json")
        shutil.copyfile(args.auditor.resolve(), staging / "result_audit.py")
        correctness_copy = staging / "correctness_manifest.json"
        shutil.copyfile(args.correctness_manifest.resolve(), correctness_copy)

        gate_log = staging / "gate.log"
        admission_provenance: dict[str, object] | None = None
        if args.performance:
            admission_dir = args.admission_dir.resolve()
            admission_auditor = admission_dir / "result_audit.py"
            admission_manifest = admission_dir / "run_manifest.json"
            admission_verification = admission_dir / "verification.json"
            admission_commands = admission_dir / "commands.log"
            admission_gate = admission_dir / "gate.log"
            if (not admission_auditor.is_file() or not admission_manifest.is_file() or
                    not admission_verification.is_file() or
                    not admission_commands.is_file() or not admission_gate.is_file()):
                raise ValueError(f"admission evidence is incomplete: {admission_dir}")
            if sha256_file(admission_auditor) != CANON_RESULT_AUDITOR_SHA256:
                raise ValueError("admission directory does not contain the fixed result auditor")
            admission_audit = subprocess.run(
                [sys.executable, str(args.auditor.resolve()), str(admission_dir)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            try:
                admission_audit_data = json.loads(admission_audit.stdout)
            except json.JSONDecodeError as exc:
                raise ValueError("fixed auditor returned invalid admission JSON") from exc
            if admission_audit.returncode != 0 or admission_audit_data.get("status") != "complete":
                raise ValueError("admission directory does not pass its fixed auditor")
            admission_manifest_data = json.loads(admission_manifest.read_text(encoding="utf-8"))
            if admission_manifest_data.get("run_kind") != "formal_admission":
                raise ValueError("performance requires a formal_admission directory")
            shutil.copyfile(admission_manifest, staging / "admission_manifest.json")
            write_json_atomic(staging / "admission_audit.json", admission_audit_data)
            shutil.copyfile(admission_verification, staging / "verification.json")
            admission_data = json.loads((staging / "verification.json").read_text(encoding="utf-8"))
            identity = admission_data.get("source_identity")
            if (admission_data.get("status") != "passed" or not isinstance(identity, dict) or
                    identity.get("git_commit") != commit or
                    identity.get("source_fingerprint") != fingerprint["source_fingerprint"] or
                    identity.get("binary_sha256") != binary_sha or
                    identity.get("device_info_sha256") != sha256_file(staging / "device_info.json")):
                raise ValueError("admission verification identity does not match this formal build")
            admission_provenance = {
                "run_id": admission_manifest_data.get("run_id"),
                "run_manifest_sha256": sha256_file(staging / "admission_manifest.json"),
                "auditor_sha256": sha256_file(admission_auditor),
                "verification_sha256": sha256_file(staging / "verification.json"),
            }
            shutil.copyfile(admission_gate, gate_log)
            shutil.copyfile(admission_commands, staging / "commands.log")
            run(
                [sys.executable, str(lz4_gpu / "benchmark_real_samples.py"),
                 "--manifest", str(staging / "samples_manifest.json"),
                 "--calibration-manifest", str(staging / "calibration_manifest.json"),
                 "--verification", str(staging / "verification.json"),
                 "--sample-root", str(args.sample_root), "--binary", str(binary),
                 "--reference-decoder", str(reference_decoder),
                 "--raw-results", str(staging / "raw_results.csv"),
                 "--summary", str(staging / "summary.csv"),
                 "--commands-log", str(staging / "benchmark_commands.log"),
                 "--raw-stdout", str(staging / "raw.stdout"),
                 "--work-root", str(staging), "--venue", args.venue,
                 "--repetitions", str(args.repetitions), "--formal"],
                cwd=lz4_gpu,
                env=env,
            )
            with (staging / "commands.log").open("a", encoding="utf-8") as destination, \
                    (staging / "benchmark_commands.log").open(encoding="utf-8") as source:
                destination.write(source.read())
            (staging / "benchmark_commands.log").unlink()
            with gate_log.open("a", encoding="utf-8") as handle:
                handle.write("ADMISSION-AUDIT-COMPLETE\nFORMAL-PERFORMANCE-PASS\nFORMAL-GATE-PASS\n")
        else:
            run(
                [sys.executable, str(lz4_gpu / "verify_real_samples.py"),
                 "--manifest", str(correctness_copy), "--sample-root", str(args.sample_root),
                 "--binary", str(binary), "--reference-decoder", str(reference_decoder),
                 "--bridge", str(bridge), "--fingerprint", str(fingerprint_path),
                 "--device-info", str(staging / "device_info.json"),
                 "--output", str(staging / "verification.json"),
                 "--commands-log", str(staging / "commands.log"), "--gate-log", str(gate_log),
                 "--work-root", str(staging), "--formal"],
                cwd=lz4_gpu,
                env=env,
            )
        selector_inputs = load_samples(staging / "samples_manifest.json")
        calibration_inputs = load_samples(staging / "calibration_manifest.json")
        correctness_inputs = load_samples(correctness_copy)
        run(["make", "clean"], cwd=lz4_gpu, stdout_path=compile_log, append=True)
        with compile_log.open("a", encoding="utf-8") as handle:
            handle.write("CLEAN-OK\n")
        expected_artifacts = PERFORMANCE_ARTIFACTS if args.performance else ADMISSION_ARTIFACTS
        write_text_atomic(staging / "expected_artifacts.txt", "\n".join(expected_artifacts) + "\n")
        write_text_atomic(
            staging / "run_meta.txt",
            f"run_id={run_id}\nhost={os.uname().nodename}\nvenue={args.venue}\n"
            f"git_commit={commit}\nrepetitions={args.repetitions if args.performance else 0}\n",
        )
        manifest = {
            "schema": "heterolz.run.v1",
            "run_id": run_id,
            "run_kind": "formal_performance" if args.performance else "formal_admission",
            "topic": "lz4_two_phase_selector_performance" if args.performance else "lz4_two_phase_correctness_admission",
            "git_commit": commit,
            "source_fingerprint": fingerprint["source_fingerprint"],
            "binary_sha256": binary_sha,
            "device_info_sha256": sha256_file(staging / "device_info.json"),
            "auditor_sha256": sha256_file(staging / "result_audit.py"),
            "samples_manifest_sha256": sha256_file(staging / "samples_manifest.json"),
            "calibration_manifest_sha256": sha256_file(staging / "calibration_manifest.json"),
            "correctness_manifest_sha256": sha256_file(correctness_copy),
            "input_source": "registered_real_samples",
            "host": os.uname().nodename,
            "device": device_info["device_name"],
            "opencl_platform": device_info["platform_name"],
            "driver": device_info["driver_version"],
            "venue_devices": venue_devices,
            "inputs": selector_inputs,
            "calibration_inputs": calibration_inputs,
            "correctness_inputs": correctness_inputs,
            "admission": admission_provenance,
            "parameters": {"block_size": 65536, "hash_log": 14, "n": [1, 2, 4, 8],
                           "venue": args.venue},
            "timing": {"kernel": True, "no_ocl": True, "total": True},
            "repetitions": args.repetitions if args.performance else 0,
            "expected_artifacts": list(expected_artifacts),
            "actual_artifacts": list(expected_artifacts),
            "exit_status": 0,
        }
        write_json_atomic(staging / "run_manifest.json", manifest)
        audit_run = subprocess.run(
            [sys.executable, str(args.auditor.resolve()), str(staging)], cwd=staging,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            audit = json.loads(audit_run.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("fixed result auditor returned invalid final JSON") from exc
        if audit_run.returncode != 0 or audit.get("status") != "complete":
            raise RuntimeError("fixed result auditor did not return complete")
        os.replace(staging, final)
        staging = None
        staging_root.rmdir()
        staging_root = None
        release_lock(lock_handle, lock_path)
        lock_handle = None
        lock_path = None
        print(final)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"formal run failed: {exc}", file=sys.stderr)
        if staging_root and staging_root.exists():
            if results_root and staging_root.parent == results_root:
                shutil.rmtree(staging_root)
                print("failed staging directory cleaned", file=sys.stderr)
        if lz4_gpu and lz4_gpu.is_dir():
            subprocess.run(["make", "clean"], cwd=lz4_gpu,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        release_lock(lock_handle, lock_path)
        return 1


if __name__ == "__main__":
    sys.exit(main())
