#!/usr/bin/env bash
set -euo pipefail

# Fixed Linux code-admission gate for the development branch. It uses only
# synthetic boundary inputs and does not produce formal experiment evidence.

root=$(cd "$(dirname "$0")" && pwd)
cd "$root"

cleanup() {
    ./lz4_gpu --stop-daemon >/dev/null 2>&1 || true
    rm -f /tmp/lz4_gpu_daemon.pid /tmp/lz4_gpu_daemon.sock
    make clean >/dev/null 2>&1 || true
}
trap cleanup EXIT

make clean
make \
    CFLAGS='-O2 -Wall -Wextra -Werror -DCL_TARGET_OPENCL_VERSION=200 -I. -I../lib -I.. -pthread' \
    all interop-tools
printf 'BUILD-OK\n'

python3 -c 'from pathlib import Path; [compile(path.read_text(encoding="utf-8"), str(path), "exec") for path in map(Path, ["benchmark_real_samples.py", "verify_real_samples.py", "source_fingerprint.py", "run_formal_acceptance.py"])]'
python3 ./run_formal_acceptance.py --help >/dev/null
EVAL_CODE_SMOKE=1 bash ./verify_twophase.sh ./lz4_gpu
bash ./verify_daemon_twophase.sh ./lz4_gpu

if [[ -e /tmp/lz4_gpu_daemon.pid || -e /tmp/lz4_gpu_daemon.sock ]]; then
    printf 'Linux early gate left a daemon endpoint\n' >&2
    exit 1
fi
printf 'HETEROLZ-LINUX-EARLY-GATE-OK\n'
