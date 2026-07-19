#!/usr/bin/env bash
set -euo pipefail

# Synthetic daemon integration gate only. Formal correctness and performance use
# the registered real-sample manifests through verify_real_samples.py and
# benchmark_real_samples.py.

binary=${1:-./lz4_gpu}
root=$(cd "$(dirname "$0")" && pwd)
binary=$(cd "$(dirname "$binary")" && pwd)/$(basename "$binary")
ref_decoder="$root/tp_ref_decode"
if [[ ! -x "$binary" || ! -x "$ref_decoder" ]]; then
    printf 'daemon gate requires lz4_gpu and tp_ref_decode executables\n' >&2
    exit 2
fi
if [[ -e /tmp/lz4_gpu_daemon.pid || -e /tmp/lz4_gpu_daemon.sock ]]; then
    printf 'daemon gate requires an idle global daemon endpoint\n' >&2
    exit 2
fi

tmp=$(mktemp -d)
daemon_pid=
cleanup() {
    if [[ -n "$daemon_pid" ]] && kill -0 "$daemon_pid" 2>/dev/null; then
        kill -TERM "$daemon_pid" 2>/dev/null || true
        wait "$daemon_pid" 2>/dev/null || true
    fi
    rm -f /tmp/lz4_gpu_daemon.pid /tmp/lz4_gpu_daemon.sock
    rm -rf "$tmp"
}
trap cleanup EXIT

python3 - "$tmp/input.bin" <<'PY'
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_bytes((b"daemon-multichunk-heterolz-" * 9000)[: 3 * 65536 + 1])
PY

LZ4TP_TEST_CHUNK_BLOCKS=2 LZ4TP_PROFILE="$tmp/profile" \
    "$binary" --daemon >"$tmp/daemon.log" 2>&1 &
daemon_pid=$!
for _ in $(seq 1 100); do
    [[ -S /tmp/lz4_gpu_daemon.sock ]] && break
    if ! kill -0 "$daemon_pid" 2>/dev/null; then
        cat "$tmp/daemon.log" >&2
        exit 1
    fi
    sleep 0.05
done
if [[ ! -S /tmp/lz4_gpu_daemon.sock ]]; then
    printf 'daemon socket did not become ready\n' >&2
    cat "$tmp/daemon.log" >&2
    exit 1
fi

python3 - <<'PY'
import ctypes
import socket


class Request(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("mode", ctypes.c_int),
        ("acceleration", ctypes.c_int),
        ("block_size", ctypes.c_int),
        ("local_size", ctypes.c_int),
        ("hash_log", ctypes.c_int),
        ("flags", ctypes.c_uint32),
        ("input_size", ctypes.c_size_t),
        ("input_path", ctypes.c_char * 1024),
        ("output_path", ctypes.c_char * 1024),
    ]


request = Request()
request.magic = 0x4C5A3447
request.version = 3
request.mode = 0
request.acceleration = 1
request.block_size = 65536
request.local_size = 1
request.hash_log = 14
ctypes.memmove(ctypes.addressof(request) + Request.input_path.offset, b"x" * 1024, 1024)
ctypes.memmove(ctypes.addressof(request) + Request.output_path.offset, b"y" * 1024, 1024)

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect("/tmp/lz4_gpu_daemon.sock")
    client.sendall(bytes(request))
    status_bytes = b""
    while len(status_bytes) < ctypes.sizeof(ctypes.c_int):
        chunk = client.recv(ctypes.sizeof(ctypes.c_int) - len(status_bytes))
        if not chunk:
            raise SystemExit("daemon closed without rejecting malformed request")
        status_bytes += chunk
status = ctypes.c_int.from_buffer_copy(status_bytes).value
if status == 0:
    raise SystemExit("daemon accepted a request with unterminated paths")
PY

LZ4TP_PROFILE="$tmp/profile" "$binary" --use-daemon --twophase \
    "$tmp/input.bin" -o "$tmp/frame.lz4tp" >>"$tmp/client.log" 2>&1
"$binary" --twophase -d "$tmp/frame.lz4tp" -o "$tmp/restored.bin" >>"$tmp/client.log" 2>&1
cmp "$tmp/input.bin" "$tmp/restored.bin"
"$ref_decoder" "$tmp/frame.lz4tp" "$tmp/input.bin" >>"$tmp/client.log" 2>&1
if python3 -c \
    'import pathlib, sys; raise SystemExit(not any(pathlib.Path(sys.argv[1]).glob("*.tmp*")))' \
    "$tmp"; then
    printf 'daemon gate found a temporary output artifact\n' >&2
    exit 1
fi

"$binary" --stop-daemon >>"$tmp/client.log" 2>&1
wait "$daemon_pid"
daemon_pid=
printf 'TWOPHASE-DAEMON-EARLY-GATE-OK\n'
