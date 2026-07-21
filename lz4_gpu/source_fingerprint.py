#!/usr/bin/env python3
"""Generate the controlled-source fingerprint used by HeteroLZ formal runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


CONTROLLED_LZ4_GPU_FILES = {
    "Makefile",
    "baseline_evidence.py",
    "benchmark_real_samples.py",
    "formal_cpu_budget.py",
    "formal_resource_probe.py",
    "gpulz_external_adapter.py",
    "lz4_gpu.c",
    "lz4_gpu.cl",
    "lz4_gpu_client.c",
    "lz4_gpu_core.c",
    "lz4_gpu_core.h",
    "lz4_gpu_daemon.c",
    "lz4_gpu_debug.h",
    "native_lz4_blocks.c",
    "nvcomp_external_adapter.py",
    "nvcomp_lz4_batched.cu",
    "lz4_gpu_protocol.h",
    "lz4_gpu_utils.c",
    "lz4_gpu_utils.h",
    "lz4_tp_profile.c",
    "lz4_tp_profile.h",
    "promote_formal_result.py",
    "run_formal_acceptance.py",
    "source_fingerprint.py",
    "test_promote_formal_result.py",
    "test_baseline_evidence.py",
    "test_formal_run_controls.py",
    "test_formal_resource_probe.py",
    "test_gpulz_external_adapter.py",
    "test_nvcomp_external_adapter.py",
    "timing.h",
    "tp_ref_decode.c",
    "tp_to_lz4.c",
    "verify_daemon_twophase.sh",
    "verify_linux_early.sh",
    "verify_real_samples.py",
    "verify_twophase.sh",
}
LIB_FILES = {
    "lib/lz4.c",
    "lib/lz4.h",
    "lib/lz4hc.c",
    "lib/lz4hc.h",
    "lib/xxhash.c",
    "lib/xxhash.h",
}
PROGRAM_FILES = {
    "programs/lz4conf.h",
    "programs/threadpool.c",
    "programs/threadpool.h",
    "programs/timefn.c",
    "programs/timefn.h",
}


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def controlled_files(repo: Path) -> list[str]:
    candidates = git(
        repo,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ).stdout.split("\0")
    selected: list[str] = []
    for value in candidates:
        if not value:
            continue
        path = Path(value)
        if value in LIB_FILES or value in PROGRAM_FILES:
            selected.append(path.as_posix())
        elif len(path.parts) == 2 and path.parts[0] == "lz4_gpu":
            if path.name in CONTROLLED_LZ4_GPU_FILES:
                selected.append(path.as_posix())
    return sorted(selected)


def validate_contract(paths: list[str]) -> None:
    expected = set(LIB_FILES) | set(PROGRAM_FILES) | {
        f"lz4_gpu/{name}" for name in CONTROLLED_LZ4_GPU_FILES
    }
    actual = set(paths)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise RuntimeError("controlled source contract mismatch: " + "; ".join(details))


def generate(repo: Path, require_clean: bool) -> dict[str, object]:
    repo = repo.resolve()
    commit = git(repo, "rev-parse", "HEAD").stdout.strip()
    paths = controlled_files(repo)
    if not paths:
        raise RuntimeError("controlled source list is empty")
    validate_contract(paths)
    if require_clean:
        controlled = set(paths)
        status = git(repo, "status", "--porcelain", "-z").stdout.split("\0")
        dirty_paths: list[str] = []
        for entry in status:
            if not entry:
                continue
            path = entry[3:] if len(entry) >= 4 else entry
            if path in controlled:
                dirty_paths.append(path)
        if dirty_paths:
            raise RuntimeError("formal source tree contains uncommitted project files: " +
                               ",".join(sorted(dirty_paths)))

    files: list[dict[str, object]] = []
    for relative_path in paths:
        path = repo / relative_path
        if not path.is_file():
            raise RuntimeError(f"controlled source missing: {relative_path}")
        files.append(
            {
                "path": relative_path,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    material = "".join(
        f"{item['sha256']}  {item['size']}  {item['path']}\n" for item in files
    )
    return {
        "schema": "heterolz.source-fingerprint.v1",
        "git_commit": commit,
        "source_fingerprint": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    try:
        result = generate(args.repo, args.require_clean)
    except Exception as exc:
        print(f"source fingerprint failed: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_name(args.output.name + ".tmp")
        try:
            temp.write_text(text, encoding="utf-8")
            os.replace(temp, args.output)
        finally:
            temp.unlink(missing_ok=True)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
