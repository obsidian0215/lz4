#!/usr/bin/env python3
"""Promote one audited HeteroLZ run from the remote host to the local result root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath


DEFAULT_REMOTE = "root@192.168.2.225"
DEFAULT_REMOTE_RESULTS_ROOT = "/root/heterolz-formal-results"
DEFAULT_LOCAL_RESULTS_ROOT = Path(r"C:\Users\obsid\Desktop\博士毕设\heterolz\exp_results\runs")
DEFAULT_AUDITOR = Path(
    r"C:\Users\obsid\Desktop\博士毕设\heterolz\阶段文档\交接工具\heterolz_result_audit.py"
)
DEFAULT_SOURCE_REGISTRY = Path(
    r"C:\Users\obsid\Desktop\博士毕设\heterolz\阶段文档\交接工具\formal_source_fingerprint.json"
)
CANON_RESULT_AUDITOR_SHA256 = "f5ef9cb1568ee79795e4703adba4cc29113105e23e26147919901d09d6c7ed22"
RUN_ID_RE = re.compile(r"heterolz-(?:admission|performance)-\d{8}T\d{12}Z")
REMOTE_RE = re.compile(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9_][A-Za-z0-9_.-]*")
SAFE_POSIX_RE = re.compile(r"/[A-Za-z0-9_./-]+")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_EXTRACTED_BYTES = 8 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_id(value: str) -> str:
    if not RUN_ID_RE.fullmatch(value):
        raise ValueError("run id must match heterolz-admission/performance-<UTC timestamp>")
    return value


def validate_remote(value: str) -> str:
    if not REMOTE_RE.fullmatch(value):
        raise ValueError("remote host contains unsupported characters")
    return value


def validate_remote_root(value: str) -> str:
    path = PurePosixPath(value)
    if (not path.is_absolute() or ".." in path.parts or not SAFE_POSIX_RE.fullmatch(value)
            or str(path) != value or value == "/"):
        raise ValueError("remote result root must be a normalized absolute POSIX path")
    return value


def validate_formal_endpoint(remote: str, remote_root: str) -> tuple[str, str]:
    remote = validate_remote(remote)
    remote_root = validate_remote_root(remote_root)
    if remote != DEFAULT_REMOTE or remote_root != DEFAULT_REMOTE_RESULTS_ROOT:
        raise ValueError("formal result promotion is locked to the registered 225 result root")
    return remote, remote_root


def validate_local_results_root(path: Path) -> Path:
    if os.name != "nt":
        raise RuntimeError("formal result promotion must run from the registered Windows host")
    resolved = path.resolve()
    expected = DEFAULT_LOCAL_RESULTS_ROOT.resolve()
    if resolved != expected:
        raise ValueError(f"formal result promotion requires the canonical local result root: {expected}")
    if not resolved.is_dir():
        raise ValueError(f"canonical local result root is missing: {resolved}")
    validate_result_root_contents(resolved, allow_readme=True)
    return resolved


def validate_result_root_contents(root: Path, *, allow_readme: bool) -> None:
    for entry in root.iterdir():
        if (allow_readme and entry.name == "README.md" and entry.is_file()
                and not entry.is_symlink()):
            continue
        if entry.is_symlink() or not entry.is_dir() or RUN_ID_RE.fullmatch(entry.name) is None:
            raise ValueError(f"formal result root contains an unexpected entry: {entry.name}")


def parse_sha256sum(output: str) -> str:
    value = output.strip().split(maxsplit=1)[0] if output.strip() else ""
    if not SHA256_RE.fullmatch(value):
        raise RuntimeError("remote sha256sum returned an invalid digest")
    return value


def run_checked(argv: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed with exit {completed.returncode}: {argv!r}: {detail}")
    return completed


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"required transfer tool is unavailable: {name}")
    return path


def ensure_auditor(path: Path, expected_sha256: str | None = None) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"fixed result auditor is missing: {resolved}")
    actual = sha256_file(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"fixed result auditor SHA256 mismatch: {actual}")
    return resolved


def validate_formal_auditor(path: Path) -> Path:
    resolved = path.resolve()
    expected = DEFAULT_AUDITOR.resolve()
    if resolved != expected:
        raise ValueError(f"formal result promotion requires the canonical local auditor: {expected}")
    return ensure_auditor(resolved, CANON_RESULT_AUDITOR_SHA256)


def validate_source_registry(path: Path, expected_path: Path | None = DEFAULT_SOURCE_REGISTRY) -> Path:
    resolved = path.resolve()
    expected = expected_path.resolve() if expected_path is not None else None
    if (expected is not None and resolved != expected) or not resolved.is_file():
        raise ValueError(f"formal result promotion requires the canonical source registry: {expected}")
    return resolved


def validate_archive(archive: tarfile.TarFile, run_id: str) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members:
        raise ValueError("result archive is empty")
    seen: set[str] = set()
    extracted_bytes = 0
    files = 0
    for member in members:
        if "\\" in member.name:
            raise ValueError(f"archive member uses a non-POSIX separator: {member.name}")
        path = PurePosixPath(member.name)
        parts = path.parts
        if (path.is_absolute() or ".." in parts or not parts or parts[0] != run_id
                or len(parts) > 2):
            raise ValueError(f"archive member escapes the formal run directory: {member.name}")
        normalized = path.as_posix()
        if normalized in seen:
            raise ValueError(f"archive contains a duplicate member: {normalized}")
        seen.add(normalized)
        if len(parts) == 1:
            if not member.isdir():
                raise ValueError("archive run root must be a directory")
            continue
        if not member.isfile():
            raise ValueError(f"archive member must be a regular file: {member.name}")
        extracted_bytes += member.size
        files += 1
        if extracted_bytes > MAX_EXTRACTED_BYTES:
            raise ValueError("result archive expands beyond the safety limit")
    if files == 0:
        raise ValueError("result archive contains no evidence files")
    return members


def extract_archive(archive_path: Path, staging_root: Path, run_id: str) -> Path:
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("result archive exceeds the transfer safety limit")
    destination = staging_root / run_id
    destination.mkdir()
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = validate_archive(archive, run_id)
        for member in members:
            path = PurePosixPath(member.name)
            if len(path.parts) == 1:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"archive member cannot be read: {member.name}")
            target = destination / path.name
            with source, target.open("xb") as handle:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
    return destination


def audit_result(run_dir: Path, auditor: Path, source_registry: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(auditor), str(run_dir),
         "--source-registry", str(source_registry)],
        cwd=run_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("fixed result auditor returned invalid JSON") from exc
    if completed.returncode != 0 or result.get("status") != "complete":
        errors = result.get("errors")
        raise RuntimeError(f"fixed result audit is incomplete: {errors}")
    return result


def promote_local_archive(
    archive_path: Path,
    run_id: str,
    local_results_root: Path,
    auditor_path: Path,
    source_registry_path: Path | None = None,
    *,
    expected_auditor_sha256: str | None = None,
) -> tuple[Path, dict[str, object]]:
    run_id = validate_run_id(run_id)
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise ValueError(f"downloaded result archive is missing: {archive_path}")
    auditor = ensure_auditor(auditor_path, expected_auditor_sha256)
    source_registry = validate_source_registry(
        source_registry_path if source_registry_path is not None else DEFAULT_SOURCE_REGISTRY,
        DEFAULT_SOURCE_REGISTRY if source_registry_path is None else None,
    )
    local_results_root = local_results_root.resolve()
    local_results_root.mkdir(parents=True, exist_ok=True)
    validate_result_root_contents(local_results_root, allow_readme=True)
    final = local_results_root / run_id
    if final.exists():
        raise FileExistsError(f"local formal run already exists: {final}")
    with tempfile.TemporaryDirectory(
        prefix=f".{run_id}.staging-", dir=local_results_root
    ) as staging_text:
        staging_root = Path(staging_text)
        extracted = extract_archive(archive_path, staging_root, run_id)
        audit = audit_result(extracted, auditor, source_registry)
        os.replace(extracted, final)
    return final, audit


def remote_archive_path(run_id: str) -> str:
    token = uuid.uuid4().hex[:12]
    return f"/tmp/heterolz-promote-{run_id}-{token}.tar.gz"


def create_remote_archive(
    ssh: str, remote: str, remote_root: str, run_id: str, archive_path: str
) -> str:
    run_checked([ssh, remote, "tar", "-C", remote_root, "-czf", archive_path, "--", run_id])
    checksum = run_checked([ssh, remote, "sha256sum", "--", archive_path])
    return parse_sha256sum(checksum.stdout)


def cleanup_remote_archive(ssh: str, remote: str, archive_path: str) -> None:
    run_checked([ssh, remote, "rm", "-f", "--", archive_path])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--local-results-root", type=Path, default=DEFAULT_LOCAL_RESULTS_ROOT)
    parser.add_argument("--auditor", type=Path, default=DEFAULT_AUDITOR)
    parser.add_argument("--source-registry", type=Path, default=DEFAULT_SOURCE_REGISTRY)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--remote-results-root", default=DEFAULT_REMOTE_RESULTS_ROOT)
    args = parser.parse_args()

    local_archive: Path | None = None
    remote_archive: str | None = None
    remote_archive_exists = False
    try:
        run_id = validate_run_id(args.run_id)
        remote, remote_root = validate_formal_endpoint(args.remote, args.remote_results_root)
        local_results_root = validate_local_results_root(args.local_results_root)
        auditor = validate_formal_auditor(args.auditor)
        source_registry = validate_source_registry(args.source_registry, DEFAULT_SOURCE_REGISTRY)
        auditor_sha256 = sha256_file(auditor)
        final = local_results_root / run_id
        if final.exists():
            raise FileExistsError(f"local formal run already exists: {final}")
        ssh = require_tool("ssh")
        scp = require_tool("scp")
        fd, local_name = tempfile.mkstemp(prefix=f"{run_id}-", suffix=".tar.gz")
        os.close(fd)
        local_archive = Path(local_name)
        remote_archive = remote_archive_path(run_id)
        remote_archive_exists = True
        remote_sha256 = create_remote_archive(ssh, remote, remote_root, run_id, remote_archive)
        run_checked([scp, f"{remote}:{remote_archive}", str(local_archive)])
        local_sha256 = sha256_file(local_archive)
        if local_sha256 != remote_sha256:
            raise RuntimeError("transferred result archive SHA256 does not match the remote archive")
        cleanup_remote_archive(ssh, remote, remote_archive)
        remote_archive_exists = False
        promoted, audit = promote_local_archive(
            local_archive, run_id, local_results_root, auditor, source_registry,
            expected_auditor_sha256=auditor_sha256,
        )
        print(json.dumps({
            "status": "complete",
            "run_id": run_id,
            "archive_sha256": local_sha256,
            "local_run": str(promoted),
            "audit_status": audit.get("status"),
        }, ensure_ascii=False, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"formal result promotion failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if remote_archive_exists and remote_archive:
            try:
                cleanup_remote_archive(require_tool("ssh"), validate_remote(args.remote), remote_archive)
            except Exception as cleanup_exc:
                print(f"warning: remote archive cleanup failed: {cleanup_exc}", file=sys.stderr)
        if local_archive:
            local_archive.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
