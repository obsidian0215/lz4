#!/usr/bin/env bash
set -euo pipefail

# tune_block_local_global.sh
# Extended in-place: compile gpu_roundtrip_test for several BLOCK_SIZE values and sweep local sizes
# Usage: ./tune_block_local_global.sh [reps] [outdir]
#   - scans a larger grid (includes local=1 baseline)
#   - avoids creating .err files when stderr is empty

REPS=${1:-1}
OUTDIR=${2:-/root/lz4/lz4_gpu/ab_results}

# Safety: restrict OUTDIR to be inside lz4_gpu ab_results by default
ALLOWED_ROOT="/root/lz4/lz4_gpu/ab_results"
if command -v readlink >/dev/null 2>&1; then
  REAL_OUTDIR=$(readlink -f "$OUTDIR")
else
  REAL_OUTDIR="$OUTDIR"
fi
if [ "${ALLOW_OTHER_DIR:-0}" != "1" ]; then
  case "$REAL_OUTDIR" in
    "$ALLOWED_ROOT" | "$ALLOWED_ROOT"/*) ;; # ok
    *)
      echo "ERROR: OUTDIR must be under $ALLOWED_ROOT. Use ALLOW_OTHER_DIR=1 to override." >&2
      exit 1
      ;;
  esac
fi

SAMPLES=(/root/samples/sample* /root/samples/*pages-1.img)
if [ "${CREATE_DIRS:-0}" = "1" ]; then
  mkdir -p "$OUTDIR"
else
  if [ ! -d "$OUTDIR" ]; then
    echo "ERROR: OUTDIR does not exist: $OUTDIR (CREATE_DIRS=0)" >&2
    echo "Create it first (mkdir -p $OUTDIR) or rerun with CREATE_DIRS=1." >&2
    exit 1
  fi
fi

# Prepare results CSV header (overwrite previous results)
RESULT_CSV="$OUTDIR/tune_results.csv"
echo "sample,size_bytes,block_size,blocks,local,rep,compute_units,workgroups,occupancy,host_ms,comp_ms,decomp_ms,total_kernel,decomp_MBps,total_MBps,ok" > "$RESULT_CSV"

# extended block sizes (KB) and local sizes (include baseline local=1)
BLOCK_SIZES_KB=(16 32 64 128 256 512 1024)
LOCAL_SIZES=(1 8 16 32 64 128 256)

ROOTDIR="$(cd "$(dirname "$0")/.." && pwd)"
# Default clbin location produced by 'make' in the lz4_gpu folder
CLBIN=${3:-"$ROOTDIR/lz4_gpu.clbin"}
# Convert to absolute path when possible
if [ -n "$CLBIN" ]; then
  if command -v readlink >/dev/null 2>&1; then
    CLBIN=$(readlink -f "$CLBIN" || echo "$CLBIN")
  fi
fi
# If expected clbin missing, error out and suggest building it.
if [ ! -f "$CLBIN" ]; then
  echo "ERROR: CLBIN not found at $CLBIN" >&2
  echo "Please run 'make -C $ROOTDIR/lz4_gpu' to build lz4_gpu.clbin, or pass CLBIN as the 3rd argument." >&2
  exit 2
fi
SRC="$ROOTDIR/gpu_roundtrip_test.c"

if [ ${#SAMPLES[@]} -eq 0 ]; then
  echo "No samples found in /tmp/sample*" >&2
  exit 2
fi

for bs_kb in "${BLOCK_SIZES_KB[@]}"; do
  BS_BYTES=$((bs_kb * 1024))
  BIN="$OUTDIR/gpu_roundtrip_test_bs${bs_kb}k"
  echo "Compiling $BIN with BLOCK_SIZE=$BS_BYTES"
  gcc -O2 -std=c11 -DBLOCK_SIZE=$BS_BYTES -o "$BIN" "$SRC" -lOpenCL || { echo "compile failed"; exit 3; }

  # Create a small probe sample to detect device properties quickly
  PROBE="$OUTDIR/probe_${bs_kb}b.bin"
  head -c 8192 /dev/urandom > "$PROBE" || dd if=/dev/urandom of="$PROBE" bs=8192 count=1 >/dev/null 2>&1 || true
  probe_out="$OUTDIR/probe_${bs_kb}.out"
  # Instead of copying kernel binaries or source to OUTDIR, pass the absolute
  # path to the compiled CLBIN or CL source to the harness. This keeps kernel
  # files in their normal repository location while allowing the harness to
  # find them regardless of the working directory.
  CLBIN_ARG=""
  if [ -f "$CLBIN" ]; then
    CLBIN_ARG="$CLBIN"
  fi

  # run probe to grab device compute units and max work-group size (fast due to small sample)
  if [ -n "$CLBIN_ARG" ]; then
    LZ4_GPU_CLBIN="$CLBIN_ARG" "$BIN" "$PROBE" --accel 1 --local 1 > "$probe_out" 2>/dev/null || true
  else
    LZ4_GPU_CLSRC="$ROOTDIR/lz4_gpu.cl" "$BIN" "$PROBE" --accel 1 --local 1 > "$probe_out" 2>/dev/null || true
  fi
  # Robustly parse device compute units and max work-group size from probe output.
  probe_line=$(grep -m1 "Device compute units" "$probe_out" || true)
  cu=$(echo "$probe_line" | sed -n 's/.*Device compute units:[[:space:]]*\([0-9]*\).*/\1/p')
  if [ -z "$cu" ]; then cu=NA; fi
  wg_line=$(grep -m1 "max work-group size" "$probe_out" || true)
  max_wg=$(echo "$wg_line" | sed -n 's/.*max work-group size:[[:space:]]*\([0-9]*\).*/\1/p')
  if [ -z "$max_wg" ]; then max_wg=0; fi

  for l in "${LOCAL_SIZES[@]}"; do
    # skip local sizes that exceed device max (if we detected it)
    if [ "$max_wg" -ne 0 ] && [ "$l" -gt "$max_wg" ]; then
      echo "Skipping local=$l (exceeds device max $max_wg)"
      continue
    fi

    for r in $(seq 1 "$REPS"); do
      for sample in "${SAMPLES[@]}"; do
        [ -f "$sample" ] || continue
        sample_basename=$(basename "$sample")
        echo "RUN: sample=${sample_basename} bs=${bs_kb}KB local=$l rep=$r"
        out="${OUTDIR}/result_${sample_basename}_bs${bs_kb}k_local${l}_rep${r}.out"
        err_tmp="${OUTDIR}/result_${sample_basename}_bs${bs_kb}k_local${l}_rep${r}.err.tmp"

        # run; pass clbin path (if available), acceleration=1, local override
        if [ -n "$CLBIN_ARG" ]; then
          LZ4_GPU_CLBIN="$CLBIN_ARG" "$BIN" "$sample" --accel 1 --local "$l" > "$out" 2> "$err_tmp" || true
        else
          LZ4_GPU_CLSRC="$ROOTDIR/lz4_gpu.cl" "$BIN" "$sample" --accel 1 --local "$l" > "$out" 2> "$err_tmp" || true
        fi

        # if stderr empty, remove the temporary err file; otherwise rename to .err
        if [ -s "$err_tmp" ]; then
          mv "$err_tmp" "${OUTDIR}/result_${sample_basename}_bs${bs_kb}k_local${l}_rep${r}.err"
        else
          rm -f "$err_tmp"
        fi

        # Parse outputs
        size_bytes=$(stat -c%s "$sample" 2>/dev/null || echo NA)
        blocks=$(grep "Input .* => .* blocks" "$out" | awk -F'=> ' '{print $2}' | awk '{print $1}' || echo NA)
        host_ms=$(grep "Host->Device upload time" "$out" | awk '{print $4}' || echo NA)
        comp_ms=$(grep "Compress kernel time" "$out" | awk '{print $4}' || echo NA)
        decomp_ms=$(grep "Decompress kernel time" "$out" | awk '{print $4}' || echo NA)
        total_kernel=$(grep "Total kernel time" "$out" | awk '{print $4}' || echo NA)
        ok=$(grep -q "Round-trip OK" "$out" && echo OK || echo FAILED)

        # compute workgroups and occupancy if possible
        if [[ "$blocks" =~ ^[0-9]+$ ]] && [[ "$cu" =~ ^[0-9]+$ ]]; then
          workgroups=$(( (blocks + l - 1) / l ))
          if [ "$cu" -gt 0 ]; then
            occ=$(awk -v w="$workgroups" -v c="$cu" 'BEGIN{printf "%.2f", (w / c)}')
          else
            occ=NA
          fi
        else
          workgroups=NA
          occ=NA
        fi

        # compute throughput in MB/s where possible
        if [[ "$decomp_ms" =~ ^[0-9]+\.?[0-9]*$ ]] && [[ "$size_bytes" =~ ^[0-9]+$ ]] && (( $(echo "$decomp_ms > 0" | bc -l) )); then
          decomp_MBps=$(awk -v b="$size_bytes" -v ms="$decomp_ms" 'BEGIN{printf "%.2f", (b/(1024*1024))*(1000/ms)}')
        else
          decomp_MBps=NA
        fi
        if [[ "$total_kernel" =~ ^[0-9]+\.?[0-9]*$ ]] && [[ "$size_bytes" =~ ^[0-9]+$ ]] && (( $(echo "$total_kernel > 0" | bc -l) )); then
          total_MBps=$(awk -v b="$size_bytes" -v ms="$total_kernel" 'BEGIN{printf "%.2f", (b/(1024*1024))*(1000/ms)}')
        else
          total_MBps=NA
        fi

        echo "$sample_basename,$size_bytes,${bs_kb}KB,$blocks,$l,$r,$cu,$workgroups,$occ,$host_ms,$comp_ms,$decomp_ms,$total_kernel,$decomp_MBps,$total_MBps,$ok" >> "$RESULT_CSV"
      done
    done
  done
done

echo "Tuning complete; results -> $RESULT_CSV"
