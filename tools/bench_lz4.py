#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from contextlib import contextmanager

IS_WINDOWS = os.name == "nt"
EXEEXT = ".exe" if IS_WINDOWS else ""

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_SAMPLES = REPO_ROOT.parent / "samples"
DEFAULT_RESULTS = REPO_ROOT / "exp_results"
DEFAULT_CPU_BIN = REPO_ROOT / "programs" / f"lz4{EXEEXT}"
DEFAULT_GPU_BIN = REPO_ROOT / "lz4_gpu" / f"lz4_gpu{EXEEXT}"
DEFAULT_HYBRID_BIN = REPO_ROOT / "lz4_hybrid" / f"lz4_hybrid{EXEEXT}"

DEFAULT_BENCH_SECONDS = 3
DEFAULT_MANUAL_ROUNDS = 6
# DEFAULT_ENGINES = ["gpu", "native_cpu", "hybrid"]
DEFAULT_ENGINES = ["gpu", "native_cpu"]
DEFAULT_CPU_THREADS = [1]
DEFAULT_CPU_BLOCK_SIZES = ["64K"]
DEFAULT_GPU_BLOCK_SIZES = ["64K"]
DEFAULT_LOCAL_SIZES = [1]
DEFAULT_GPU_ACCELS = [1]
DEFAULT_HYBRID_BLOCK_SIZES = ["64K"]
DEFAULT_HYBRID_GPU_RATIOS = [0.3, 0.5, 0.7]
DEFAULT_HYBRID_CPU_THREADS = [1]
DEFAULT_HYBRID_LOCAL_SIZES = [1]
DEFAULT_HYBRID_ACCELS = [1]

BENCH_COMP_RE = re.compile(
    r"Bench\s+Compress\s*:\s*kernel_tp=([0-9]+(?:\.[0-9]+)?)\s*MB/s.*?ratio=([0-9]+(?:\.[0-9]+)?)%",
    re.IGNORECASE,
)
BENCH_DEC_RE = re.compile(
    r"Bench\s+Decompress\s*:\s*kernel_tp=([0-9]+(?:\.[0-9]+)?)\s*MB/s.*?verify=(OK|FAIL)",
    re.IGNORECASE,
)
CPU_BENCH_RE = re.compile(
    r"^\s*-\d+\s+(\d+)\s+\(([0-9]+(?:\.[0-9]+)?)\)\s+([0-9]+(?:\.[0-9]+)?)\s+MB/s\s+([0-9]+(?:\.[0-9]+)?)\s+MB/s",
    re.MULTILINE,
)
TOTAL_RE = re.compile(r"TOTAL\s+INCLUSIVE\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
DONE_RE = re.compile(r"Done\s+in\s*([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)
OCI_RE = re.compile(r"OCI\s+Setup\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
GENERIC_TP_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*MB/s", re.IGNORECASE)

LZ4_DAEMON_SOCKET = Path("/tmp/lz4_gpu_daemon.sock")
LZ4_DAEMON_PIDFILE = Path("/tmp/lz4_gpu_daemon.pid")

RAW_FIELDS = [
    "sample", "engine", "phase", "round",
    "block", "local_size", "accel", "gpu_ratio", "cpu_threads",
    "input_bytes", "compressed_bytes", "ratio_pct", "verify_ok",
    "bench_comp_kernel_mbs", "bench_dec_kernel_mbs",
    "manual_comp_kernel_mbs", "manual_dec_kernel_mbs",
    "manual_comp_no_ocl_mbs", "manual_dec_no_ocl_mbs",
    "manual_comp_seconds", "manual_dec_seconds",
    "status", "error",
]


def _is_executable_file(path):
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def _resolve_make_executable():
    if IS_WINDOWS:
        for cand in ("mingw32-make", "make"):
            if shutil.which(cand):
                return cand
        return "mingw32-make"
    return "make"


def _try_build_lz4_cpu_binary():
    build_plans = []
    if IS_WINDOWS:
        mk = _resolve_make_executable()
        build_plans.extend([
            ([mk, "-C", str(REPO_ROOT), "lz4"], [str(REPO_ROOT / "programs" / "lz4"), str(REPO_ROOT / "programs" / "lz4.exe"), str(REPO_ROOT / "lz4"), str(REPO_ROOT / "lz4.exe")]),
            ([mk, "-C", str(REPO_ROOT / "programs"), "lz4"], [str(REPO_ROOT / "programs" / "lz4"), str(REPO_ROOT / "programs" / "lz4.exe")]),
        ])
    else:
        build_plans.extend([
            (["make", "-C", "/root/lz4/programs", "lz4", "-j8"], ["/root/lz4/programs/lz4"]),
            (["make", "-C", "/root/lz4", "lz4", "-j8"], ["/root/lz4/lz4"]),
        ])

    for cmd, out_bins in build_plans:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                for out_bin in out_bins:
                    if _is_executable_file(out_bin):
                        return out_bin
        except Exception:
            continue
    return None


def resolve_lz4_cpu_binary(preferred):
    candidates = []
    env_bin = os.environ.get("LZ4_CPU_BIN")
    if env_bin:
        candidates.append(env_bin)
    if preferred:
        candidates.append(str(preferred))
    candidates.extend([
        str(REPO_ROOT / "programs" / "lz4"),
        str(REPO_ROOT / "programs" / f"lz4{EXEEXT}"),
        str(REPO_ROOT / "lz4"),
        str(REPO_ROOT / f"lz4{EXEEXT}"),
        "/root/lz4/programs/lz4",
        "/root/lz4/lz4",
    ])

    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if _is_executable_file(cand):
            return cand

    built = _try_build_lz4_cpu_binary()
    if built:
        return built

    raise FileNotFoundError(f"Cannot find/build LZ4 CPU binary. Tried: {candidates}")


def str_list(value):
    return [x.strip() for x in str(value).split(",") if x.strip()]


def int_list(value):
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def float_list(value):
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def mb(byte_count):
    return float(byte_count or 0) / (1024.0 * 1024.0)


def median(values):
    vals = [float(v) for v in values if v not in (None, "")]
    return statistics.median(vals) if vals else None


def mean(values):
    vals = [float(v) for v in values if v not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def stdev(values):
    vals = [float(v) for v in values if v not in (None, "")]
    return statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None)


def host_id():
    return platform.node() or os.environ.get("COMPUTERNAME") or "unknown"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd, cwd=None, env=None, timeout=None):
    merged = os.environ.copy()
    if IS_WINDOWS:
        merged["PATH"] = r"C:\msys64\ucrt64\bin;C:\msys64\usr\bin;" + merged.get("PATH", "")
    if env:
        merged.update(env)
    start = time.perf_counter()
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        env=merged,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - start
    return proc.returncode, proc.stdout or "", elapsed


def safe_unlink(path):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def daemon_is_running():
    return LZ4_DAEMON_SOCKET.exists()


def running_daemon_exe():
    try:
        if not LZ4_DAEMON_PIDFILE.exists():
            return None
        pid_text = LZ4_DAEMON_PIDFILE.read_text(encoding="utf-8", errors="ignore").strip()
        if not pid_text:
            return None
        pid = int(pid_text.splitlines()[0].strip())
        exe_link = Path(f"/proc/{pid}/exe")
        if not exe_link.exists():
            return None
        return exe_link.resolve()
    except Exception:
        return None


def stop_daemon(exe_path):
    exe = Path(exe_path)
    cwd = exe.parent
    try:
        rc, out, _ = run_cmd([exe, "--stop-daemon"], cwd=cwd, timeout=10)
        return rc == 0, out
    except Exception as exc:
        return False, str(exc)


def start_daemon(exe_path, timeout_s=15.0):
    exe = Path(exe_path)
    cwd = exe.parent
    proc = subprocess.Popen(
        [str(exe), "--daemon"],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    deadline = time.time() + max(1.0, float(timeout_s))
    logs = []
    while time.time() < deadline:
        if daemon_is_running():
            return proc, "".join(logs)
        if proc.poll() is not None:
            logs.append(proc.stdout.read() if proc.stdout else "")
            break
        time.sleep(0.1)
    if proc.poll() is None:
        proc.terminate()
    try:
        out = proc.stdout.read() if proc.stdout else ""
    except Exception:
        out = ""
    logs.append(out)
    raise RuntimeError(f"failed to start daemon for {exe}: {''.join(logs)[-500:]}")


@contextmanager
def daemon_session(exe_path, enabled):
    if (not enabled) or IS_WINDOWS:
        yield False
        return

    started_here = False
    proc = None
    expected_exe = Path(exe_path).resolve()
    try:
        if daemon_is_running():
            running_exe = running_daemon_exe()
            if running_exe is not None and running_exe == expected_exe:
                yield True
                return
            # Socket is occupied by another daemon binary; switch to the expected one.
            stop_daemon(exe_path)
            time.sleep(0.1)
            if daemon_is_running():
                # Best effort cleanup before starting expected daemon.
                try:
                    LZ4_DAEMON_SOCKET.unlink(missing_ok=True)
                except OSError:
                    pass
        proc, _ = start_daemon(exe_path)
        started_here = True
        yield True
    finally:
        if started_here:
            stop_daemon(exe_path)
        if proc and proc.poll() is None:
            proc.terminate()


def parse_total_seconds(text, wall_seconds, subtract_oci=False):
    total = None
    m = TOTAL_RE.findall(text or "")
    if m:
        total = float(m[-1]) / 1000.0
    if total is None:
        m2 = DONE_RE.findall(text or "")
        if m2:
            total = float(m2[-1])
    if not total or total <= 0.0:
        total = float(wall_seconds or 0.0)
    if subtract_oci:
        mo = OCI_RE.search(text or "")
        if mo:
            oci_s = float(mo.group(1)) / 1000.0
            if 0.0 < oci_s < total:
                total -= oci_s
    return max(0.0, total)


def parse_bench_cpu(text, input_bytes):
    matches = list(CPU_BENCH_RE.finditer(text or ""))
    if not matches:
        return None
    m = matches[-1]
    c_bytes = int(m.group(1))
    ratio_pct = (100.0 * c_bytes / input_bytes) if input_bytes > 0 else 0.0
    return {
        "ratio_pct": ratio_pct,
        "comp_kernel_mbs": float(m.group(3)),
        "dec_kernel_mbs": float(m.group(4)),
        "verify_ok": True,
        "compressed_bytes": c_bytes,
    }


def parse_bench_stable(text):
    comp = BENCH_COMP_RE.search(text or "")
    dec = BENCH_DEC_RE.search(text or "")
    if not comp or not dec:
        return None
    return {
        "ratio_pct": float(comp.group(2)),
        "comp_kernel_mbs": float(comp.group(1)),
        "dec_kernel_mbs": float(dec.group(1)),
        "verify_ok": dec.group(2).upper() == "OK",
    }


def parse_manual_kernel_tp(text):
    vals = GENERIC_TP_RE.findall(text or "")
    if not vals:
        return None
    try:
        return float(vals[-1])
    except Exception:
        return None


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
        matches = [p for p in files if p.name == single_file]
        if len(matches) != 1:
            raise SystemExit(f"single file not found or ambiguous: {single_file}")
        return matches
    if limit > 0:
        return files[:limit]
    return files


def base_row(args, sample, engine, phase, round_id, cfg):
    return {
        "sample": sample.name,
        "engine": engine,
        "phase": phase,
        "round": round_id,
        "block": cfg.get("block", ""),
        "local_size": cfg.get("local_size", ""),
        "accel": cfg.get("accel", ""),
        "gpu_ratio": cfg.get("gpu_ratio", ""),
        "cpu_threads": cfg.get("cpu_threads", ""),
        "input_bytes": sample.stat().st_size,
        "compressed_bytes": "",
        "ratio_pct": "",
        "verify_ok": "",
        "bench_comp_kernel_mbs": "",
        "bench_dec_kernel_mbs": "",
        "manual_comp_kernel_mbs": "",
        "manual_dec_kernel_mbs": "",
        "manual_comp_no_ocl_mbs": "",
        "manual_dec_no_ocl_mbs": "",
        "manual_comp_seconds": "",
        "manual_dec_seconds": "",
        "status": "ok",
        "error": "",
    }


def run_cpu_case(args, sample, threads, block, tmp_root):
    exe = Path(args.cpu_bin)
    cfg = {"block": block, "cpu_threads": threads}
    rows = []

    input_bytes = sample.stat().st_size
    original_hash = sha256(sample)

    bench_cmd = [exe, "-q", f"-B{block}", f"-T{threads}", "-b1", sample]
    row = base_row(args, sample, "lz4_cpu", "bench", 0, cfg)
    try:
        rc, out, _ = run_cmd(bench_cmd, timeout=args.timeout)
        parsed = parse_bench_cpu(out, input_bytes)
        if rc != 0 or not parsed:
            raise RuntimeError(f"cpu bench failed rc={rc}: {out[-500:]}")
        row["ratio_pct"] = parsed["ratio_pct"]
        row["compressed_bytes"] = parsed["compressed_bytes"]
        row["verify_ok"] = parsed["verify_ok"]
        row["bench_comp_kernel_mbs"] = parsed["comp_kernel_mbs"]
        row["bench_dec_kernel_mbs"] = parsed["dec_kernel_mbs"]
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    rows.append(row)

    for round_id in range(args.manual_rounds):
        comp_path = tmp_root / f"{sample.name}.cpu.{round_id}.lz4"
        dec_path = tmp_root / f"{sample.name}.cpu.{round_id}.dec"
        row = base_row(args, sample, "lz4_cpu", "manual", round_id, cfg)
        try:
            comp_cmd = [exe, "-v", "-z", "-f", f"-B{block}", f"-T{threads}", sample, comp_path]
            rc_c, out_c, wall_c = run_cmd(comp_cmd, timeout=args.timeout)
            if rc_c != 0 or not comp_path.exists():
                raise RuntimeError(f"cpu compress failed rc={rc_c}: {out_c[-500:]}")

            dec_cmd = [exe, "-v", "-f", "-d", comp_path, dec_path]
            rc_d, out_d, wall_d = run_cmd(dec_cmd, timeout=args.timeout)
            if rc_d != 0 or not dec_path.exists():
                raise RuntimeError(f"cpu decompress failed rc={rc_d}: {out_d[-500:]}")

            verify_ok = sha256(dec_path) == original_hash
            c_bytes = comp_path.stat().st_size
            comp_s = parse_total_seconds(out_c, wall_c, subtract_oci=False)
            dec_s = parse_total_seconds(out_d, wall_d, subtract_oci=False)
            row["compressed_bytes"] = c_bytes
            row["ratio_pct"] = 100.0 * c_bytes / input_bytes if input_bytes else 0.0
            row["verify_ok"] = verify_ok
            row["manual_comp_kernel_mbs"] = parse_manual_kernel_tp(out_c)
            row["manual_dec_kernel_mbs"] = parse_manual_kernel_tp(out_d)
            row["manual_comp_no_ocl_mbs"] = mb(input_bytes) / comp_s if comp_s > 0 else ""
            row["manual_dec_no_ocl_mbs"] = mb(input_bytes) / dec_s if dec_s > 0 else ""
            row["manual_comp_seconds"] = comp_s
            row["manual_dec_seconds"] = dec_s
            if not verify_ok:
                row["status"] = "verify_failed"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        finally:
            safe_unlink(comp_path)
            safe_unlink(dec_path)
        rows.append(row)

    return rows


def run_gpu_case(args, sample, block, local_size, accel, tmp_root, use_daemon=False):
    exe = Path(args.gpu_bin)
    cfg = {"block": block, "local_size": local_size, "accel": accel, "gpu_ratio": "1", "cpu_threads": ""}
    rows = []

    input_bytes = sample.stat().st_size
    original_hash = sha256(sample)
    cwd = exe.parent
    daemon_prefix = ["--use-daemon"] if use_daemon else []

    row = base_row(args, sample, "lz4_gpu", "bench", 0, cfg)
    try:
        bench_cmd = [exe, "--bench", args.bench_seconds, "-B", block, "--local", local_size, "-a", accel, sample]
        rc, out, _ = run_cmd(bench_cmd, cwd=cwd, timeout=args.timeout)
        parsed = parse_bench_stable(out)
        if rc != 0 or not parsed:
            raise RuntimeError(f"gpu bench failed rc={rc}: {out[-500:]}")
        row["ratio_pct"] = parsed["ratio_pct"]
        row["verify_ok"] = parsed["verify_ok"]
        row["bench_comp_kernel_mbs"] = parsed["comp_kernel_mbs"]
        row["bench_dec_kernel_mbs"] = parsed["dec_kernel_mbs"]
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    rows.append(row)

    for round_id in range(args.manual_rounds):
        comp_path = tmp_root / f"{sample.name}.gpu.{round_id}.lz4"
        dec_path = tmp_root / f"{sample.name}.gpu.{round_id}.dec"
        row = base_row(args, sample, "lz4_gpu", "manual", round_id, cfg)
        try:
            comp_cmd = [exe, *daemon_prefix, "-v", "-B", block, "--local", local_size, "-a", accel, "-o", comp_path, sample]
            rc_c, out_c, wall_c = run_cmd(comp_cmd, cwd=cwd, timeout=args.timeout)
            if rc_c != 0 or not comp_path.exists():
                raise RuntimeError(f"gpu compress failed rc={rc_c}: {out_c[-500:]}")

            dec_cmd = [exe, *daemon_prefix, "-v", "-d", "-o", dec_path, comp_path]
            rc_d, out_d, wall_d = run_cmd(dec_cmd, cwd=cwd, timeout=args.timeout)
            if rc_d != 0 or not dec_path.exists():
                raise RuntimeError(f"gpu decompress failed rc={rc_d}: {out_d[-500:]}")

            verify_ok = sha256(dec_path) == original_hash
            c_bytes = comp_path.stat().st_size
            comp_s = parse_total_seconds(out_c, wall_c, subtract_oci=True)
            dec_s = parse_total_seconds(out_d, wall_d, subtract_oci=True)
            row["compressed_bytes"] = c_bytes
            row["ratio_pct"] = 100.0 * c_bytes / input_bytes if input_bytes else 0.0
            row["verify_ok"] = verify_ok
            row["manual_comp_kernel_mbs"] = parse_manual_kernel_tp(out_c)
            row["manual_dec_kernel_mbs"] = parse_manual_kernel_tp(out_d)
            row["manual_comp_no_ocl_mbs"] = mb(input_bytes) / comp_s if comp_s > 0 else ""
            row["manual_dec_no_ocl_mbs"] = mb(input_bytes) / dec_s if dec_s > 0 else ""
            row["manual_comp_seconds"] = comp_s
            row["manual_dec_seconds"] = dec_s
            if not verify_ok:
                row["status"] = "verify_failed"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        finally:
            safe_unlink(comp_path)
            safe_unlink(dec_path)
        rows.append(row)

    return rows


def run_hybrid_case(args, sample, block, local_size, accel, ratio, cpu_threads, tmp_root, use_daemon=False):
    exe = Path(args.hybrid_bin)
    cfg = {
        "block": block,
        "local_size": local_size,
        "accel": accel,
        "gpu_ratio": ratio,
        "cpu_threads": cpu_threads,
    }
    rows = []

    input_bytes = sample.stat().st_size
    original_hash = sha256(sample)
    cwd = exe.parent
    daemon_prefix = ["--use-daemon"] if use_daemon else []

    row = base_row(args, sample, "lz4_hybrid", "bench", 0, cfg)
    try:
        bench_cmd = [exe, "--bench", args.bench_seconds, "-b", block.lower(), "-l", local_size, "-a", accel, "-T", cpu_threads]
        bench_cmd += ["--gpu-ratio", ratio]
        bench_cmd += [sample]
        rc, out, _ = run_cmd(bench_cmd, cwd=cwd, timeout=args.timeout)
        parsed = parse_bench_stable(out)
        if rc != 0 or not parsed:
            raise RuntimeError(f"hybrid bench failed rc={rc}: {out[-500:]}")
        row["ratio_pct"] = parsed["ratio_pct"]
        row["verify_ok"] = parsed["verify_ok"]
        row["bench_comp_kernel_mbs"] = parsed["comp_kernel_mbs"]
        row["bench_dec_kernel_mbs"] = parsed["dec_kernel_mbs"]
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    rows.append(row)

    for round_id in range(args.manual_rounds):
        comp_path = tmp_root / f"{sample.name}.hybrid.{round_id}.lz4"
        dec_path = tmp_root / f"{sample.name}.hybrid.{round_id}.dec"
        row = base_row(args, sample, "lz4_hybrid", "manual", round_id, cfg)
        try:
            if use_daemon:
                comp_cmd = [exe, *daemon_prefix, "-v", "-B", block, "--local", local_size, "-a", accel, "--cpu-threads", cpu_threads]
                comp_cmd += ["--gpu-ratio", ratio]
            else:
                comp_cmd = [exe, *daemon_prefix, "-v", "-b", block.lower(), "-l", local_size, "-a", accel, "-T", cpu_threads]
                comp_cmd += ["--gpu-ratio", ratio]
            comp_cmd += ["-o", comp_path, sample]
            rc_c, out_c, wall_c = run_cmd(comp_cmd, cwd=cwd, timeout=args.timeout)
            if rc_c != 0 or not comp_path.exists():
                raise RuntimeError(f"hybrid compress failed rc={rc_c}: {out_c[-500:]}")

            if use_daemon:
                dec_cmd = [exe, *daemon_prefix, "-v", "-d", "--cpu-threads", cpu_threads]
                dec_cmd += ["--gpu-ratio", ratio]
            else:
                dec_cmd = [exe, *daemon_prefix, "-v", "-d", "-T", cpu_threads]
                dec_cmd += ["--gpu-ratio", ratio]
            dec_cmd += ["-o", dec_path, comp_path]
            rc_d, out_d, wall_d = run_cmd(dec_cmd, cwd=cwd, timeout=args.timeout)
            if rc_d != 0 or not dec_path.exists():
                raise RuntimeError(f"hybrid decompress failed rc={rc_d}: {out_d[-500:]}")

            verify_ok = sha256(dec_path) == original_hash
            c_bytes = comp_path.stat().st_size
            comp_s = parse_total_seconds(out_c, wall_c, subtract_oci=True)
            dec_s = parse_total_seconds(out_d, wall_d, subtract_oci=True)
            row["compressed_bytes"] = c_bytes
            row["ratio_pct"] = 100.0 * c_bytes / input_bytes if input_bytes else 0.0
            row["verify_ok"] = verify_ok
            row["manual_comp_kernel_mbs"] = parse_manual_kernel_tp(out_c)
            row["manual_dec_kernel_mbs"] = parse_manual_kernel_tp(out_d)
            row["manual_comp_no_ocl_mbs"] = mb(input_bytes) / comp_s if comp_s > 0 else ""
            row["manual_dec_no_ocl_mbs"] = mb(input_bytes) / dec_s if dec_s > 0 else ""
            row["manual_comp_seconds"] = comp_s
            row["manual_dec_seconds"] = dec_s
            if not verify_ok:
                row["status"] = "verify_failed"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        finally:
            safe_unlink(comp_path)
            safe_unlink(dec_path)
        rows.append(row)

    return rows


def case_key(row):
    return (
        row["sample"], row["engine"],
        row["block"], row["local_size"], row["accel"],
        row["gpu_ratio"], row["cpu_threads"],
    )


def summarize(raw_rows):
    grouped = {}
    for row in raw_rows:
        grouped.setdefault(case_key(row), []).append(row)

    per_file = []
    for _, rows in sorted(grouped.items()):
        bench_rows = [r for r in rows if r["phase"] == "bench" and r["status"] == "ok"]
        manual_rows = [r for r in rows if r["phase"] == "manual" and r["status"] == "ok"]
        first = rows[0]
        out = {field: first.get(field, "") for field in RAW_FIELDS if field not in {
            "phase", "round", "status", "error",
            "bench_comp_kernel_mbs", "bench_dec_kernel_mbs",
            "manual_comp_kernel_mbs", "manual_dec_kernel_mbs",
            "manual_comp_no_ocl_mbs", "manual_dec_no_ocl_mbs",
            "manual_comp_seconds", "manual_dec_seconds",
        }}
        out.update({
            "bench_rounds": len(bench_rows),
            "manual_rounds": len(manual_rows),
            "ok": all(str(r.get("verify_ok", "")).lower() in ("true", "yes", "1") for r in manual_rows) if manual_rows else False,
            "bench_comp_kernel_mbs_median": median([r["bench_comp_kernel_mbs"] for r in bench_rows]),
            "bench_dec_kernel_mbs_median": median([r["bench_dec_kernel_mbs"] for r in bench_rows]),
            "manual_comp_kernel_mbs_median": median([r["manual_comp_kernel_mbs"] for r in manual_rows]),
            "manual_dec_kernel_mbs_median": median([r["manual_dec_kernel_mbs"] for r in manual_rows]),
            "manual_comp_no_ocl_mbs_median": median([r["manual_comp_no_ocl_mbs"] for r in manual_rows]),
            "manual_dec_no_ocl_mbs_median": median([r["manual_dec_no_ocl_mbs"] for r in manual_rows]),
            "manual_comp_no_ocl_mbs_mean": mean([r["manual_comp_no_ocl_mbs"] for r in manual_rows]),
            "manual_dec_no_ocl_mbs_mean": mean([r["manual_dec_no_ocl_mbs"] for r in manual_rows]),
            "manual_comp_no_ocl_mbs_stdev": stdev([r["manual_comp_no_ocl_mbs"] for r in manual_rows]),
            "manual_dec_no_ocl_mbs_stdev": stdev([r["manual_dec_no_ocl_mbs"] for r in manual_rows]),
            "ratio_pct_median": median([r["ratio_pct"] for r in manual_rows] or [r["ratio_pct"] for r in bench_rows]),
        })
        per_file.append(out)

    aggregate_grouped = {}
    for row in per_file:
        key = (
            row["engine"],
            row["block"], row["local_size"], row["accel"],
            row["gpu_ratio"], row["cpu_threads"],
        )
        aggregate_grouped.setdefault(key, []).append(row)

    aggregate = []
    for _, rows in sorted(aggregate_grouped.items()):
        first = rows[0]
        aggregate.append({
            "engine": first["engine"],
            "block": first["block"],
            "local_size": first["local_size"],
            "accel": first["accel"],
            "gpu_ratio": first["gpu_ratio"],
            "cpu_threads": first["cpu_threads"],
            "samples": len(rows),
            "verify_all": all(bool(r["ok"]) for r in rows),
            "ratio_pct_median_of_files": median([r["ratio_pct_median"] for r in rows]),
            "bench_comp_kernel_mbs_median_of_files": median([r["bench_comp_kernel_mbs_median"] for r in rows]),
            "bench_dec_kernel_mbs_median_of_files": median([r["bench_dec_kernel_mbs_median"] for r in rows]),
            "manual_comp_kernel_mbs_median_of_files": median([r["manual_comp_kernel_mbs_median"] for r in rows]),
            "manual_dec_kernel_mbs_median_of_files": median([r["manual_dec_kernel_mbs_median"] for r in rows]),
            "manual_comp_no_ocl_mbs_median_of_files": median([r["manual_comp_no_ocl_mbs_median"] for r in rows]),
            "manual_dec_no_ocl_mbs_median_of_files": median([r["manual_dec_no_ocl_mbs_median"] for r in rows]),
        })
    return per_file, aggregate


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def make_run_dir(results_dir):
    run_dir = Path(results_dir).resolve() / "runs" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_all(args):
    samples = discover_samples(args.samples, args.limit, args.single_file)
    if not samples:
        raise SystemExit("no samples found")

    cpu_threads = [str(x) for x in int_list(args.cpu_threads)]
    cpu_blocks = str_list(args.cpu_block_sizes)
    gpu_blocks = str_list(args.gpu_block_sizes)
    local_sizes = [str(x) for x in int_list(args.local_sizes)]
    gpu_accels = [str(x) for x in int_list(args.gpu_accels)]

    hybrid_blocks = str_list(args.hybrid_block_sizes)
    hybrid_ratios = [str(x) for x in float_list(args.hybrid_gpu_ratios)]
    hybrid_cpu_threads = [str(x) for x in int_list(args.hybrid_cpu_threads)]
    hybrid_local_sizes = [str(x) for x in int_list(args.hybrid_local_sizes)]
    hybrid_accels = [str(x) for x in int_list(args.hybrid_accels)]

    engines = set(str_list(args.engines)) if str(args.engines).strip() else {"gpu", "native_cpu", "hybrid"}
    run_cpu = "native_cpu" in engines
    run_gpu = "gpu" in engines
    run_hybrid = "hybrid" in engines

    if args.use_daemon and not IS_WINDOWS:
        print("[bench_lz4] use_daemon=on (check daemon, start if missing, auto-stop if started by this run)")

    if run_cpu:
        resolved_cpu = resolve_lz4_cpu_binary(args.cpu_bin)
        args.cpu_bin = str(resolved_cpu)
        print(f"[bench_lz4] cpu_bin={args.cpu_bin}")

    run_dir = make_run_dir(args.results_dir)
    tmp_root = run_dir / "tmp"
    raw_rows = []

    print(f"[bench_lz4] run_dir={run_dir}")
    print(f"[bench_lz4] samples={len(samples)} bench_seconds={args.bench_seconds} manual_rounds={args.manual_rounds}")

    try:
        tmp_root.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            if run_cpu:
                for block in cpu_blocks:
                    for threads in cpu_threads:
                        raw_rows.extend(run_cpu_case(args, sample, threads, block, tmp_root))

            if run_gpu:
                with daemon_session(args.gpu_bin, args.use_daemon) as gpu_daemon_active:
                    for block in gpu_blocks:
                        for local_size in local_sizes:
                            for accel in gpu_accels:
                                raw_rows.extend(run_gpu_case(args, sample, block, local_size, accel, tmp_root, use_daemon=gpu_daemon_active))

            if run_hybrid:
                with daemon_session(args.hybrid_bin, args.use_daemon) as hybrid_daemon_active:
                    for block in hybrid_blocks:
                        for ratio in hybrid_ratios:
                            for cpu_threads_h in hybrid_cpu_threads:
                                for local_size in hybrid_local_sizes:
                                    for accel in hybrid_accels:
                                        raw_rows.extend(
                                            run_hybrid_case(
                                                args,
                                                sample,
                                                block,
                                                local_size,
                                                accel,
                                                ratio,
                                                cpu_threads_h,
                                                tmp_root,
                                                use_daemon=hybrid_daemon_active,
                                            )
                                        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    per_file, aggregate = summarize(raw_rows)

    raw_path = run_dir / "raw.csv"
    per_file_path = run_dir / "per_file_summary.csv"
    aggregate_path = run_dir / "aggregate.csv"

    write_csv(raw_path, raw_rows, RAW_FIELDS)
    if per_file:
        write_csv(per_file_path, per_file, list(per_file[0].keys()))
    if aggregate:
        write_csv(aggregate_path, aggregate, list(aggregate[0].keys()))

    with open(run_dir / "run_meta.txt", "w", encoding="utf-8") as handle:
        handle.write(f"argv={' '.join(sys.argv)}\n")
        handle.write(f"platform_id={args.platform_id}\n")
        handle.write(f"bench_seconds={args.bench_seconds}\n")
        handle.write(f"manual_rounds={args.manual_rounds}\n")
        handle.write(f"samples={args.samples}\n")
        handle.write(f"run_cpu={run_cpu} run_gpu={run_gpu} run_hybrid={run_hybrid}\n")

    print(f"[bench_lz4] raw={raw_path}")
    print(f"[bench_lz4] per_file={per_file_path}")
    print(f"[bench_lz4] aggregate={aggregate_path}")
    return run_dir


def parse_args(argv):
    parser = argparse.ArgumentParser(description="LZ4 benchmark/manual runner; power/frequency scans belong to external wrappers.")
    parser.add_argument("--platform-id", default=platform.node() or "local")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--single-file", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--bench-seconds", default=str(DEFAULT_BENCH_SECONDS))
    parser.add_argument("--manual-rounds", type=int, default=DEFAULT_MANUAL_ROUNDS)
    parser.add_argument("--engines", default=",".join(DEFAULT_ENGINES), help="Comma list: gpu,native_cpu,hybrid")

    parser.add_argument("--cpu-threads", default=",".join(str(x) for x in DEFAULT_CPU_THREADS))
    parser.add_argument("--cpu-block-sizes", default=",".join(DEFAULT_CPU_BLOCK_SIZES))
    parser.add_argument("--gpu-block-sizes", default=",".join(DEFAULT_GPU_BLOCK_SIZES))
    parser.add_argument("--local-sizes", default=",".join(str(x) for x in DEFAULT_LOCAL_SIZES))
    parser.add_argument("--gpu-accels", default=",".join(str(x) for x in DEFAULT_GPU_ACCELS))

    parser.add_argument("--hybrid-block-sizes", default=",".join(DEFAULT_HYBRID_BLOCK_SIZES))
    parser.add_argument("--hybrid-gpu-ratios", default=",".join(str(x) for x in DEFAULT_HYBRID_GPU_RATIOS))
    parser.add_argument("--hybrid-cpu-threads", default=",".join(str(x) for x in DEFAULT_HYBRID_CPU_THREADS))
    parser.add_argument("--hybrid-local-sizes", default=",".join(str(x) for x in DEFAULT_HYBRID_LOCAL_SIZES))
    parser.add_argument("--hybrid-accels", default=",".join(str(x) for x in DEFAULT_HYBRID_ACCELS))

    parser.add_argument("--use-daemon", action="store_true", help="Linux only: use GPU daemon; check running, start if missing, stop only if started by this run")

    parser.add_argument("--cpu-bin", default=str(DEFAULT_CPU_BIN))
    parser.add_argument("--gpu-bin", default=str(DEFAULT_GPU_BIN))
    parser.add_argument("--hybrid-bin", default=str(DEFAULT_HYBRID_BIN))

    args = parser.parse_args(argv)

    return args


if __name__ == "__main__":
    run_all(parse_args(sys.argv[1:]))
