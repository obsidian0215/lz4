#!/usr/bin/env python3
import glob
import hashlib
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

OUT_JSON = "/tmp/robust_dual_reval_report.json"
REPEATS = 2
THRESH = 5.0

SAMPLES = [
    "sample_226mb_random_4.txt",
    "sample_53.18mb_random_3.txt",
    "sample_6.86mb_random_1.txt",
    "sample_99mb_structured_5.txt",
    "sample_83.02mb_mixed_3.txt",
    "sample_59.45mb_zero_3.txt",
    "elasticsearch_parent_0_pages_img.tar",
    "redis-video__migrate__parent_6__pages-1.img",
    "redis-video__migrate__parent_8__pages-1.img",
    "yolo_parent_0_pages_img.tar",
    "dickens",
    "webster",
    "xml",
    "nci",
    "ooffice",
    "mozilla",
]

LZ4 = {
    "name": "lz4",
    "root_cand": "/root/lz4",
    "root_base": "/tmp/lz4_base",
    "bin_rel": "lz4_gpu/lz4_gpu",
    "sock": "/tmp/lz4_gpu_daemon.sock",
    "pid": "/tmp/lz4_gpu_daemon.pid",
    "groups": {
        "G1_kernel_cl_only": ["lz4_gpu/lz4_gpu.cl"],
        "G2_host_core_only": [
            "lz4_gpu/lz4_gpu_core.c",
            "lz4_gpu/lz4_gpu_core.h",
            "lz4_gpu/lz4_gpu.c",
            "lz4_gpu/lz4_gpu_daemon.c",
            "lz4_gpu/Makefile",
        ],
        "G3_cli_bench_only": [
            "tools/bench_lz4.py",
            "programs/bench.c",
            "programs/bench.h",
            "programs/lz4cli.c",
        ],
    },
}

LZO = {
    "name": "lzo",
    "root_cand": "/root/lzo-2.10",
    "root_base": "/tmp/lzo_base",
    "bin_rel": "lzo_gpu/lzo_gpu",
    "sock": "/tmp/lzo_gpu_daemon.sock",
    "pid": "/tmp/lzo_gpu_daemon.pid",
    "groups": {
        "G1_kernel_compress_only": ["lzo_gpu/lzo1x.cl", "lzo_gpu/lzo1y.cl"],
        "G2_host_runtime_only": [
            "lzo_gpu/lzo_gpu_core.c",
            "lzo_gpu/lzo_gpu_core.h",
            "lzo_gpu/lzo_gpu_utils.c",
            "lzo_gpu/lzo_gpu_daemon.c",
            "lzo_gpu/lzo_defaults.h",
            "lzo_gpu/Makefile",
        ],
        "G3_bench_tooling_only": ["tools/bench_lzo.py", "tools/hw_telemetry.py"],
    },
}

INC_RE = re.compile(r"Inclusive Throughput\s*:\s*([0-9]+\.?[0-9]*)\s*MB/s", re.I)


def run(cmd, cwd=None, env=None, check=True, timeout=180):
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        if check:
            raise RuntimeError(f"timeout: {' '.join(cmd)}")
        return SimpleNamespace(returncode=124, stdout=e.stdout or "", stderr=(e.stderr or "") + "\n[TIMEOUT]")
    if check and r.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def parse_mbs(out):
    m = INC_RE.search(out or "")
    return float(m.group(1)) if m else 0.0


def cleanup_tmp():
    for pat in [
        "/tmp/rblz4_*", "/tmp/rblz4d_*", "/tmp/rblzo_*", "/tmp/rblzod_*",
        "/tmp/lz4ab_*", "/tmp/lz4ab_dec_*", "/tmp/bench_lz4_*", "/tmp/bench_lzo_gpu_*",
    ]:
        for p in glob.glob(pat):
            try:
                os.remove(p)
            except OSError:
                pass


def wait_sock(sock, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(sock):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(0.2)
                s.connect(sock)
                s.close()
                return True
            except Exception:
                pass
        time.sleep(0.1)
    return False


def start_daemon(bin_path, sock, pid, env):
    run([bin_path, "--stop-daemon"], env=env, check=False)
    for p in (sock, pid):
        try:
            os.remove(p)
        except OSError:
            pass
    proc = subprocess.Popen([bin_path, "--daemon"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    if not wait_sock(sock):
        try:
            proc.kill()
        except Exception:
            pass
        raise RuntimeError(f"daemon not ready: {bin_path}")
    return proc


def stop_daemon(bin_path, proc, env):
    run([bin_path, "--stop-daemon"], env=env, check=False)
    if proc is not None:
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def ensure_worktree(repo):
    if os.path.isdir(repo["root_base"]):
        return
    run(["git", "worktree", "add", "--detach", repo["root_base"], "HEAD"], cwd=repo["root_cand"])


def rewrite_lz4_base_paths(base_root):
    for rel in ["lz4_gpu/lz4_gpu_core.c", "lz4_gpu/lz4_gpu_daemon.c"]:
        p = os.path.join(base_root, rel)
        if not os.path.isfile(p):
            continue
        s = Path(p).read_text(encoding="utf-8")
        s2 = s.replace("/root/lz4/lz4_gpu/", "/tmp/lz4_base/lz4_gpu/")
        if s2 != s:
            Path(p).write_text(s2, encoding="utf-8")


def build(repo_name, root):
    if repo_name == "lz4":
        run(["make", "-C", os.path.join(root, "lz4_gpu"), "lz4_gpu", "-j8"])
    else:
        run(["make", "-C", os.path.join(root, "lzo_gpu"), "lzo_gpu", "-j8"])


def reset_base(repo):
    run(["git", "reset", "--hard", "HEAD"], cwd=repo["root_base"])
    if repo["name"] == "lz4":
        rewrite_lz4_base_paths(repo["root_base"])


def copy_files(repo, rel_files):
    for rel in rel_files:
        src = os.path.join(repo["root_cand"], rel)
        dst = os.path.join(repo["root_base"], rel)
        if not os.path.isfile(src):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def measure_lz4(bin_path, sample, src_hash, env):
    with tempfile.NamedTemporaryFile(prefix="rblz4_", suffix=".lz4", delete=False) as tf:
        cpath = tf.name
    with tempfile.NamedTemporaryFile(prefix="rblz4d_", suffix=".bin", delete=False) as tf2:
        dpath = tf2.name
    try:
        rc = run([bin_path, "--use-daemon", "-v", "-b", "32k", "-H", "14", "-l", "1", "-a", "1", sample, "-o", cpath], env=env, check=False)
        rd = run([bin_path, "--use-daemon", "-v", "-d", cpath, "-o", dpath, "-H", "14", "-l", "1"], env=env, check=False)
        ok = rc.returncode == 0 and rd.returncode == 0 and os.path.isfile(dpath) and sha256(dpath) == src_hash
        return {
            "ok": bool(ok),
            "comp": parse_mbs((rc.stdout or "") + "\n" + (rc.stderr or "")),
            "dec": parse_mbs((rd.stdout or "") + "\n" + (rd.stderr or "")),
        }
    finally:
        for p in (cpath, dpath):
            try:
                os.remove(p)
            except OSError:
                pass


def measure_lzo(bin_path, sample, src_hash, alg, env):
    with tempfile.NamedTemporaryFile(prefix="rblzo_", suffix=".lzo", delete=False) as tf:
        cpath = tf.name
    with tempfile.NamedTemporaryFile(prefix="rblzod_", suffix=".bin", delete=False) as tf2:
        dpath = tf2.name
    try:
        # LZO baseline daemon path can fail to resolve decomp kernels for some trees;
        # use standalone mode to keep A/B apples-to-apples and deterministic.
        rc = run([bin_path, "-v", "-a", alg, "-L", "12", "-B", "16k", "--local", "1", sample, "-o", cpath], env=env, check=False)
        rd = run([bin_path, "-v", "-d", "-a", alg, cpath, "-o", dpath, "--local", "1"], env=env, check=False)
        ok = rc.returncode == 0 and rd.returncode == 0 and os.path.isfile(dpath) and sha256(dpath) == src_hash
        return {
            "ok": bool(ok),
            "comp": parse_mbs((rc.stdout or "") + "\n" + (rc.stderr or "")),
            "dec": parse_mbs((rd.stdout or "") + "\n" + (rd.stderr or "")),
        }
    finally:
        for p in (cpath, dpath):
            try:
                os.remove(p)
            except OSError:
                pass


def run_suite(repo, bin_path, env):
    cleanup_tmp()
    out = {}
    if repo["name"] == "lz4":
        proc = start_daemon(bin_path, repo["sock"], repo["pid"], env)
        try:
            for s in SAMPLES:
                sp = os.path.join("/root/samples", s)
                if not os.path.isfile(sp):
                    continue
                h = sha256(sp)
                key = f"{s}::lz4"
                out[key] = [measure_lz4(bin_path, sp, h, env) for _ in range(REPEATS)]
        finally:
            stop_daemon(bin_path, proc, env)
            cleanup_tmp()
        return out

    # LZO: standalone path for baseline/candidate consistency.
    for s in SAMPLES:
        sp = os.path.join("/root/samples", s)
        if not os.path.isfile(sp):
            continue
        h = sha256(sp)
        for alg in ("lzo1x", "lzo1y"):
            key = f"{s}::{alg}"
            out[key] = [measure_lzo(bin_path, sp, h, alg, env) for _ in range(REPEATS)]
    cleanup_tmp()
    return out


def delta(c, b):
    return None if b <= 0 else (c / b - 1.0) * 100.0


def summarize_rows(rows):
    vals = [r["avg_delta_pct"] for r in rows if r.get("avg_delta_pct") is not None]
    if not vals:
        return {"decision": "delete", "reason": "no_valid_rows"}

    med = statistics.median(vals)
    abs_dev = [abs(v - med) for v in vals]
    mad = statistics.median(abs_dev) if abs_dev else 0.0
    scale = max(1e-9, 1.4826 * mad)

    reg = [r for r in rows if r.get("avg_delta_pct") is not None and r["avg_delta_pct"] <= -THRESH]
    imp = [r for r in rows if r.get("avg_delta_pct") is not None and r["avg_delta_pct"] >= THRESH]

    for r in rows:
        v = r.get("avg_delta_pct")
        if v is None:
            r["robust_z"] = None
            r["isolated_extreme"] = False
        else:
            rz = abs(v - med) / scale
            r["robust_z"] = rz
            r["isolated_extreme"] = False

    for r in reg:
        if len(reg) <= 1 and (r.get("robust_z") or 0.0) >= 3.5:
            r["isolated_extreme"] = True
    for r in imp:
        if len(imp) <= 1 and (r.get("robust_z") or 0.0) >= 3.5:
            r["isolated_extreme"] = True

    non_isolated_reg = sum(1 for r in reg if not r.get("isolated_extreme"))
    if non_isolated_reg > 0:
        decision, reason = "delete", "non_isolated_regression"
    elif med >= THRESH and len(reg) == 0:
        decision, reason = "keep", "strong_gain_without_regression"
    elif med > 0 and len(reg) <= 1 and len(imp) >= max(2, int(0.2 * len(vals))):
        decision, reason = "keep_with_tail_risk", "mostly_positive_with_isolated_tail"
    elif len(reg) == 0 and len(imp) == 0:
        decision, reason = "neutral", "within_noise_band"
    else:
        decision, reason = "delete", "insufficient_robust_gain"

    return {
        "n": len(vals),
        "median_avg_delta_pct": med,
        "reg_ge5_count": len(reg),
        "imp_ge5_count": len(imp),
        "isolated_reg_ge5_count": sum(1 for r in reg if r.get("isolated_extreme")),
        "isolated_imp_ge5_count": sum(1 for r in imp if r.get("isolated_extreme")),
        "decision": decision,
        "reason": reason,
    }


def compare(base, cand):
    rows = []
    for k in sorted(set(base.keys()) & set(cand.keys())):
        b = base[k]
        c = cand[k]
        b_comp = statistics.median([x["comp"] for x in b])
        c_comp = statistics.median([x["comp"] for x in c])
        b_dec = statistics.median([x["dec"] for x in b])
        c_dec = statistics.median([x["dec"] for x in c])
        comp_d = delta(c_comp, b_comp)
        dec_d = delta(c_dec, b_dec)
        avg_d = None if (comp_d is None or dec_d is None) else (comp_d + dec_d) / 2.0
        rows.append({
            "case": k,
            "roundtrip_ok_base": all(x["ok"] for x in b),
            "roundtrip_ok_cand": all(x["ok"] for x in c),
            "comp_delta_pct": comp_d,
            "dec_delta_pct": dec_d,
            "avg_delta_pct": avg_d,
        })

    srt = sorted([r for r in rows if r.get("avg_delta_pct") is not None], key=lambda x: x["avg_delta_pct"])
    return {
        "summary": summarize_rows(rows),
        "rows": rows,
        "top_regressions": srt[:10],
        "top_improvements": list(reversed(srt[-10:])),
    }


def eval_repo(repo):
    ensure_worktree(repo)
    reset_base(repo)
    build(repo["name"], repo["root_base"])
    build(repo["name"], repo["root_cand"])

    base_bin = os.path.join(repo["root_base"], repo["bin_rel"])
    cand_bin = os.path.join(repo["root_cand"], repo["bin_rel"])

    env_base = os.environ.copy()
    env_cand = os.environ.copy()
    if repo["name"] == "lzo":
        env_base["LZO_GPU_DIR"] = os.path.join(repo["root_base"], "lzo_gpu")
        env_cand["LZO_GPU_DIR"] = os.path.join(repo["root_cand"], "lzo_gpu")

    print(f"[INFO] {repo['name']} baseline")
    base_suite = run_suite(repo, base_bin, env_base)

    print(f"[INFO] {repo['name']} full candidate")
    full_suite = run_suite(repo, cand_bin, env_cand)
    out = {"full_candidate": compare(base_suite, full_suite), "groups": {}}

    for gname, rels in repo["groups"].items():
        print(f"[INFO] {repo['name']} group {gname}")
        reset_base(repo)
        copy_files(repo, rels)
        if repo["name"] == "lz4":
            rewrite_lz4_base_paths(repo["root_base"])
        build(repo["name"], repo["root_base"])
        grp_suite = run_suite(repo, base_bin, env_base)
        cmp = compare(base_suite, grp_suite)
        cmp["files"] = rels
        out["groups"][gname] = cmp
    return out


def main():
    t0 = time.time()
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "method": {
            "repeats": REPEATS,
            "threshold_pct": THRESH,
            "robust": "median + MAD isolation check",
            "isolation_rule": "|delta|>=5 and robust_z>=3.5 and sign_count<=1",
        },
        "lz4": eval_repo(LZ4),
        "lzo": eval_repo(LZO),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(OUT_JSON)
    print("elapsed_sec=", report["elapsed_sec"])


if __name__ == "__main__":
    main()
