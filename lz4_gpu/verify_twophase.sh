#!/usr/bin/env bash
set -euo pipefail

# Synthetic early gate only. This script checks codec boundaries, roundtrip,
# reference decoding, bridge wiring, and auto-profile plumbing. It must not be
# used to report performance, selector quality, or paper results; formal runs use
# the project-registered real samples and their recorded SHA256 values.

binary=${1:-./lz4_gpu}
root=$(cd "$(dirname "$0")" && pwd)

resolve_executable() {
    local candidate=$1
    if [[ -x "${candidate}.exe" ]]; then
        printf '%s\n' "${candidate}.exe"
    elif [[ -x "$candidate" ]]; then
        printf '%s\n' "$candidate"
    else
        printf '%s\n' "$candidate"
    fi
}

binary=$(resolve_executable "$(cd "$(dirname "$binary")" && pwd)/$(basename "$binary")")
ref_decoder=$(resolve_executable "$root/tp_ref_decode")
bridge=$(resolve_executable "$root/tp_to_lz4")

for required in "$binary" "$ref_decoder" "$bridge"; do
    if [[ ! -x "$required" ]]; then
        printf 'missing executable: %s\n' "$required" >&2
        exit 2
    fi
done

tmp=$(mktemp -d)
cleanup() {
    if [[ ${KEEP_TMP:-0} == 1 ]]; then
        printf 'preserved early-gate files: %s\n' "$tmp" >&2
    else
        rm -rf "$tmp"
    fi
}
trap cleanup EXIT

python_cmd=()
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 8))' >/dev/null 2>&1; then
        python_cmd=("$candidate")
        break
    fi
done
if [[ ${#python_cmd[@]} == 0 ]] && command -v py >/dev/null 2>&1 &&
   py -3 -c 'import sys; raise SystemExit(sys.version_info < (3, 8))' >/dev/null 2>&1; then
    python_cmd=(py -3)
fi
if [[ ${#python_cmd[@]} == 0 ]]; then
    printf 'python 3 interpreter unavailable\n' >&2
    exit 2
fi

assert_same() {
    if command -v cmp >/dev/null 2>&1; then
        cmp "$1" "$2"
    else
        "${python_cmd[@]}" -c \
            'import pathlib, sys; raise SystemExit(pathlib.Path(sys.argv[1]).read_bytes() != pathlib.Path(sys.argv[2]).read_bytes())' \
            "$1" "$2"
    fi
}

"${python_cmd[@]}" - "$tmp" <<'PY'
import pathlib
import random
import sys

root = pathlib.Path(sys.argv[1])
root.joinpath("empty.bin").write_bytes(b"")
root.joinpath("one.bin").write_bytes(b"x")
root.joinpath("block.bin").write_bytes((b"heterolz-" * 8192)[:65536])
root.joinpath("block_plus_one.bin").write_bytes((b"heterolz-" * 8193)[:65537])
rng = random.Random(20260719)
root.joinpath("random_64k.bin").write_bytes(bytes(rng.getrandbits(8) for _ in range(65536)))
root.joinpath("random_1m.bin").write_bytes(bytes(rng.getrandbits(8) for _ in range(1024 * 1024)))
root.joinpath("multichunk.bin").write_bytes((b"multichunk-heterolz-" * 12000)[: 3 * 65536 + 1])
root.joinpath("calibration_1m.bin").write_bytes((b"calibration-pattern-0123456789" * 40000)[:1024 * 1024])
root.joinpath("calibration_4m.bin").write_bytes((b"calibration-pattern-0123456789" * 150000)[:4 * 1024 * 1024])
PY

log="$tmp/early_gate.log"
report_failure() {
    local status=$?
    printf 'synthetic early gate failed; diagnostic log follows\n' >&2
    if [[ -f "$log" ]]; then
        "${python_cmd[@]}" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).read_text(errors="replace"), file=sys.stderr)' "$log"
    fi
    return "$status"
}
trap report_failure ERR

"$binary" --device-info >"$tmp/device_info.json" 2>>"$log"
"${python_cmd[@]}" - "$tmp/device_info.json" <<'PY'
import json
import pathlib
import sys

info = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert info["schema"] == "heterolz.device-info.v1"
for field in ("platform_name", "device_name", "device_vendor", "driver_version"):
    assert isinstance(info[field], str) and info[field]
for field in ("compute_units", "global_mem_bytes", "max_alloc_bytes"):
    assert isinstance(info[field], int) and info[field] > 0
PY

run_case() {
    local input=$1 n=$2
    local frame="$tmp/${input}.n${n}.lz4tp"
    local restored="$tmp/${input}.n${n}.restored"
    "$binary" --twophase -N "$n" "$tmp/$input" -o "$frame" >>"$log" 2>&1
    "$binary" --twophase -d "$frame" -o "$restored" >>"$log" 2>&1
    assert_same "$tmp/$input" "$restored"
    "$ref_decoder" "$frame" "$tmp/$input" >>"$log" 2>&1
}

for n in 1 2 4 8; do
    run_case one.bin "$n"
    run_case block_plus_one.bin "$n"
done
run_case empty.bin 1
run_case block.bin 1
run_case random_64k.bin 8

export LZ4TP_TEST_CHUNK_BLOCKS=2
metrics_frame="$tmp/multichunk.metrics.lz4tp"
metrics_restored="$tmp/multichunk.metrics.restored"
compress_metrics="$tmp/multichunk.compress.json"
decompress_metrics="$tmp/multichunk.decompress.json"
"$binary" --twophase -N 8 --metrics-json "$compress_metrics" \
    "$tmp/multichunk.bin" -o "$metrics_frame" >>"$log" 2>&1
"$binary" --twophase -d --metrics-json "$decompress_metrics" \
    "$metrics_frame" -o "$metrics_restored" >>"$log" 2>&1
unset LZ4TP_TEST_CHUNK_BLOCKS
assert_same "$tmp/multichunk.bin" "$metrics_restored"
"$ref_decoder" "$metrics_frame" "$tmp/multichunk.bin" >>"$log" 2>&1
"${python_cmd[@]}" - "$tmp/multichunk.bin" "$metrics_frame" "$compress_metrics" "$decompress_metrics" <<'PY'
import json
import pathlib
import sys

source, frame, compress_path, decompress_path = map(pathlib.Path, sys.argv[1:])
compress = json.loads(compress_path.read_text())
decompress = json.loads(decompress_path.read_text())
for metric, operation in ((compress, "compress"), (decompress, "decompress")):
    assert metric["schema"] == "heterolz.operation-metric.v1"
    assert metric["operation"] == operation
    assert metric["n"] == 8
    assert metric["block_size"] == 65536
    assert metric["hash_log"] == 14
    assert metric["chunk_blocks"] == 2
    assert metric["kernel_us"] >= 0
    assert metric["no_ocl_us"] >= 0
    assert metric["total_us"] >= metric["no_ocl_us"]
    assert metric["total_us"] >= metric["ocl_setup_us"]
assert compress["input_bytes"] == source.stat().st_size
assert compress["output_bytes"] == frame.stat().st_size
assert decompress["input_bytes"] == frame.stat().st_size
assert decompress["output_bytes"] == source.stat().st_size
PY

cp "$metrics_frame" "$tmp/trailing.lz4tp"
printf 'x' >>"$tmp/trailing.lz4tp"
if "$binary" --twophase -d "$tmp/trailing.lz4tp" -o "$tmp/trailing.restored" >>"$log" 2>&1; then
    printf 'decoder accepted a frame with trailing bytes\n' >&2
    exit 1
fi
if [[ -e "$tmp/trailing.restored" || -e "$tmp/trailing.restored.tmp" ]]; then
    printf 'failed decode left an output artifact\n' >&2
    exit 1
fi
if "${python_cmd[@]}" -c \
    'import pathlib, sys; raise SystemExit(not any(pathlib.Path(sys.argv[1]).glob("*.tmp*")))' \
    "$tmp"; then
    printf 'temporary LZ4TP1 output artifact remains\n' >&2
    exit 1
fi
if "$binary" --twophase -N 1 --metrics-json "$tmp/metric-collision.lz4tp" \
    "$tmp/one.bin" -o "$tmp/metric-collision.lz4tp" >>"$log" 2>&1; then
    printf 'metrics path was allowed to replace the frame output\n' >&2
    exit 1
fi
if [[ -e "$tmp/metric-collision.lz4tp" || -e "$tmp/metric-collision.lz4tp.tmp" ]]; then
    printf 'metrics/output collision left an artifact\n' >&2
    exit 1
fi
if "$binary" --twophase -N 1 --metrics-json "$tmp/one.bin" \
    "$tmp/one.bin" -o "$tmp/metric-input-frame.lz4tp" >>"$log" 2>&1; then
    printf 'metrics path was allowed to replace the input\n' >&2
    exit 1
fi
"${python_cmd[@]}" -c 'import pathlib, sys; assert pathlib.Path(sys.argv[1]).read_bytes() == b"x"' "$tmp/one.bin"
if [[ -e "$tmp/metric-input-frame.lz4tp" || -e "$tmp/metric-input-frame.lz4tp.tmp" ]]; then
    printf 'metrics/input collision left an output artifact\n' >&2
    exit 1
fi
if ln "$tmp/one.bin" "$tmp/one.alias.bin" 2>/dev/null; then
    if "$binary" --twophase -N 1 "$tmp/one.bin" -o "$tmp/one.alias.bin" >>"$log" 2>&1; then
        printf 'input/output hard-link alias was accepted\n' >&2
        exit 1
    fi
    "${python_cmd[@]}" -c 'import pathlib, sys; assert pathlib.Path(sys.argv[1]).read_bytes() == b"x"' "$tmp/one.bin"
fi
for invalid_case in n block hash; do
    invalid_output="$tmp/invalid-${invalid_case}.lz4tp"
    case "$invalid_case" in
        n) invalid_args=(--twophase -N 2junk "$tmp/one.bin" -o "$invalid_output") ;;
        block) invalid_args=(--twophase -N 1 -B 64Kjunk "$tmp/one.bin" -o "$invalid_output") ;;
        hash) invalid_args=(--twophase -N 1 --d-bits 14junk "$tmp/one.bin" -o "$invalid_output") ;;
    esac
    if "$binary" "${invalid_args[@]}" >>"$log" 2>&1; then
        printf 'invalid numeric CLI input was accepted: %s\n' "$invalid_case" >&2
        exit 1
    fi
    if [[ -e "$invalid_output" || -e "$invalid_output.tmp" ]]; then
        printf 'invalid CLI input left an output artifact: %s\n' "$invalid_case" >&2
        exit 1
    fi
done

if [[ ${FULL_EARLY_GATE:-0} == 1 ]]; then
    for input in empty.bin block.bin random_1m.bin; do
        for n in 1 2 4 8; do
            run_case "$input" "$n"
        done
    done
fi

standard_frame="$tmp/block.n1.lz4"
"$bridge" "$tmp/block.bin.n1.lz4tp" "$standard_frame" >>"$log" 2>&1
if command -v lz4 >/dev/null 2>&1; then
    lz4 -q -d -f "$standard_frame" "$tmp/block.stock.restored" >>"$log" 2>&1
    assert_same "$tmp/block.bin" "$tmp/block.stock.restored"
else
    printf 'stock lz4 CLI unavailable; bridge decode check skipped\n' >&2
fi

profile="$tmp/lz4tp.profile"
calibration_input="$tmp/calibration_1m.bin"
if [[ ${FULL_EARLY_GATE:-0} == 1 ]]; then calibration_input="$tmp/calibration_4m.bin"; fi
LZ4TP_PROFILE="$profile" "$binary" --calibrate "$calibration_input" >>"$log" 2>&1
LZ4TP_PROFILE="$profile" "$binary" --auto "$tmp/block_plus_one.bin" -o "$tmp/auto.lz4tp" >>"$log" 2>&1
"$binary" --twophase -d "$tmp/auto.lz4tp" -o "$tmp/auto.restored" >>"$log" 2>&1
assert_same "$tmp/block_plus_one.bin" "$tmp/auto.restored"
"$ref_decoder" "$tmp/auto.lz4tp" "$tmp/block_plus_one.bin" >>"$log" 2>&1
cp "$profile" "$tmp/profile.before"
if LZ4TP_PROFILE="$profile" "$binary" --auto "$tmp/block_plus_one.bin" -o "$profile" >>"$log" 2>&1; then
    printf 'auto output was allowed to replace its calibration profile\n' >&2
    exit 1
fi
assert_same "$tmp/profile.before" "$profile"
cp "$tmp/calibration_1m.bin" "$tmp/calibration_collision.bin"
if "$binary" --calibrate "$tmp/calibration_collision.bin" -o "$tmp/calibration_collision.bin" >>"$log" 2>&1; then
    printf 'calibration profile was allowed to replace its input\n' >&2
    exit 1
fi
assert_same "$tmp/calibration_1m.bin" "$tmp/calibration_collision.bin"
if ln "$tmp/block.bin.n1.lz4tp" "$tmp/bridge.alias.lz4tp" 2>/dev/null; then
    if "$bridge" "$tmp/block.bin.n1.lz4tp" "$tmp/bridge.alias.lz4tp" >>"$log" 2>&1; then
        printf 'bridge input/output hard-link alias was accepted\n' >&2
        exit 1
    fi
    "$ref_decoder" "$tmp/block.bin.n1.lz4tp" "$tmp/block.bin" >>"$log" 2>&1
fi

"${python_cmd[@]}" - "$tmp/device_info.json" "$tmp/overflow.profile" <<'PY'
import json
import pathlib
import sys

device = json.loads(pathlib.Path(sys.argv[1]).read_text())["device_name"]
pathlib.Path(sys.argv[2]).write_text(
    f"LZ4TP_PROFILE_V1\t65536\t14\t999999999999999999999999\t{device}\n"
)
PY
LZ4TP_PROFILE="$tmp/overflow.profile" "$binary" --auto \
    --metrics-json "$tmp/overflow.metrics.json" "$tmp/block_plus_one.bin" \
    -o "$tmp/overflow.lz4tp" >>"$log" 2>&1
"${python_cmd[@]}" - "$tmp/overflow.metrics.json" <<'PY'
import json
import pathlib
import sys

metric = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert metric["n"] == 1
PY
"$ref_decoder" "$tmp/overflow.lz4tp" "$tmp/block_plus_one.bin" >>"$log" 2>&1

if [[ ${EVAL_CODE_SMOKE:-0} == 1 ]]; then
    "${python_cmd[@]}" - "$tmp" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])

def record(name):
    path = root / name
    return {
        "relative_path": name,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

validation = record("multichunk.bin")
calibration = record("calibration_1m.bin")
(root / "validation_manifest.json").write_text(json.dumps({
    "schema": "heterolz.samples.v1",
    "selection_name": "synthetic_early_validation",
    "source_root": str(root),
    "files": [validation],
}, indent=2) + "\n")
(root / "calibration_manifest.json").write_text(json.dumps({
    "schema": "heterolz.samples.v1",
    "selection_name": "synthetic_early_calibration",
    "source_root": str(root),
    "files": [calibration],
}, indent=2) + "\n")
(root / "verification_manifest.json").write_text(json.dumps({
    "schema": "heterolz.samples.v1",
    "selection_name": "synthetic_early_correctness",
    "files": [validation, calibration],
}, indent=2) + "\n")
PY
    "${python_cmd[@]}" "$root/verify_real_samples.py" \
        --manifest "$tmp/verification_manifest.json" --sample-root "$tmp" \
        --binary "$binary" --reference-decoder "$ref_decoder" --bridge "$bridge" \
        --output "$tmp/verification_smoke.json" \
        --commands-log "$tmp/verify_smoke_commands.log" --gate-log "$tmp/verify_smoke_gate.log" \
        --work-root "$tmp" --n 1 --skip-determinism --skip-cross-venue --skip-bridge >>"$log" 2>&1
    "${python_cmd[@]}" "$root/benchmark_real_samples.py" \
        --manifest "$tmp/validation_manifest.json" \
        --calibration-manifest "$tmp/calibration_manifest.json" \
        --verification "$tmp/verification_smoke.json" \
        --sample-root "$tmp" --binary "$binary" --reference-decoder "$ref_decoder" \
        --raw-results "$tmp/eval_smoke_raw.csv" --summary "$tmp/eval_smoke_summary.csv" \
        --commands-log "$tmp/eval_smoke_commands.log" --raw-stdout "$tmp/eval_smoke.stdout" \
        --work-root "$tmp" --repetitions 1 --n 1 --skip-adaptive >>"$log" 2>&1
    "${python_cmd[@]}" - "$tmp/eval_smoke_raw.csv" "$tmp/eval_smoke_summary.csv" <<'PY'
import csv
import pathlib
import sys

raw_path, summary_path = map(pathlib.Path, sys.argv[1:])
with raw_path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
assert rows and {"chunk_blocks", "repetition", "derived"}.issubset(rows[0])
assert all(row["roundtrip_ok"] == "true" and row["reference_ok"] == "true" for row in rows)
with summary_path.open(newline="") as handle:
    summary = list(csv.DictReader(handle))
assert summary and any(row["policy"] == "calibration" for row in summary)
PY
fi

if "${python_cmd[@]}" -c \
    'import pathlib, sys; raise SystemExit(not any(pathlib.Path(sys.argv[1]).glob("*.tmp*")))' \
    "$tmp"; then
    printf 'early gate left a temporary artifact\n' >&2
    exit 1
fi

printf 'TWOPHASE-SYNTHETIC-EARLY-GATE-OK mode=%s\n' "$([[ ${FULL_EARLY_GATE:-0} == 1 ]] && printf full || printf quick)"
