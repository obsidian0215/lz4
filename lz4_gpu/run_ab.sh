#!/usr/bin/env bash
set -euo pipefail

# run_ab.sh - build vector/non-vector clbins and run A/B tests across samples
# Usage: ./run_ab.sh samples_dir

SAMPLE_FILE=""
# Support: ./run_ab.sh samples_dir  OR  ./run_ab.sh -s|--sample <file>
if [ "${1:-}" = "-s" ] || [ "${1:-}" = "--sample" ]; then
  SAMPLE_FILE="${2:-}"
  if [ -z "$SAMPLE_FILE" ]; then
    echo "Usage: $0 -s|--sample <file>   OR   $0 [samples_dir]";
    exit 1
  fi
  SAMPLES_DIR=""
else
  SAMPLES_DIR="${1:-samples}"
fi
GPU_TEST=./gpu_roundtrip_test
BUILD_BIN=./build_clbin

if [ ! -x "$GPU_TEST" ]; then
  echo "gpu_roundtrip_test not found or not executable. Attempting to compile..."
  if [ -f gpu_roundtrip_test.c ]; then
    gcc -O2 gpu_roundtrip_test.c -o gpu_roundtrip_test -lOpenCL && echo "Built gpu_roundtrip_test" || { echo "Failed to build gpu_roundtrip_test; please install OpenCL dev libs or build manually."; exit 1; }
  else
    echo "Missing gpu_roundtrip_test.c; please generate or compile the test harness.";
    exit 1
  fi
fi

echo "Ensuring helper to produce clbin exists ($BUILD_BIN)..."
if [ ! -x "$BUILD_BIN" ]; then
  echo "$BUILD_BIN not found or not executable. Attempting to compile build_clbin..."
  if [ -f build_clbin.c ]; then
    if command -v gcc >/dev/null 2>&1; then
      gcc -O2 build_clbin.c -o build_clbin -lOpenCL && echo "Built build_clbin" || { echo "Failed to build build_clbin; please install OpenCL dev libs or build manually."; exit 1; }
    else
      echo "gcc not found; please install gcc or build build_clbin manually."; exit 1
    fi
  else
    echo "Missing build_clbin.c; cannot build $BUILD_BIN"; exit 1
  fi
else
  echo "$BUILD_BIN exists"
fi

echo "Generating non-vector binary (lz4_gpu.clbin)"
"$BUILD_BIN" -o lz4_gpu.clbin ""
echo "Generating vector binary (lz4_gpu_vec.clbin)"
"$BUILD_BIN" -o lz4_gpu_vec.clbin "-DLZ4_GPU_VECTOR_IO=1"

mkdir -p ab_results
OUT_CSV=ab_results/ab_summary.csv
echo "sample,mode,host_to_device_ms,compress_ms,read_bs_ms,decompress_ms,read_decomp_ms,total_kernel_ms,roundtrip_ok,total_orig,total_comp,ratio,blocks_csv" > "$OUT_CSV"


if [ -n "$SAMPLE_FILE" ]; then
  if [ ! -f "$SAMPLE_FILE" ]; then
    echo "Sample file '$SAMPLE_FILE' not found."; exit 1
  fi
else
  # If samples dir doesn't exist or is empty, attempt to generate a test suite
  if [ ! -d "$SAMPLES_DIR" ] || [ -z "$(ls -A "$SAMPLES_DIR" 2>/dev/null)" ]; then
    echo "Samples directory '$SAMPLES_DIR' does not exist or is empty. Attempting to generate test data using generate_test_data.py ..."
    PY=""
    if command -v python3 >/dev/null 2>&1; then PY=python3; elif command -v python >/dev/null 2>&1; then PY=python; fi
    if [ -z "$PY" ]; then
      echo "Python not found; please install Python 3 or create '$SAMPLES_DIR' with test files."; exit 1
    fi
    # Run generator with --suite to create a variety of samples
    mkdir -p "$SAMPLES_DIR"
    # locate generator script in the same directory as this script (common layout)
    GEN_SCRIPT="$(cd "$(dirname "$0")" >/dev/null 2>&1 && pwd)/generate_test_data.py"
    echo "Running: $PY $GEN_SCRIPT --suite --out-dir $SAMPLES_DIR"
    $PY "$GEN_SCRIPT" --suite --out-dir "$SAMPLES_DIR" || {
      echo "generate_test_data.py failed; please create sample files manually in '$SAMPLES_DIR'"; exit 1
    }
    echo "Generated test samples in $SAMPLES_DIR"
  fi
fi

# Check there are regular files inside
shopt_enabled=0
if command -v bash >/dev/null 2>&1; then
  # enable nullglob for safe iteration if running under bash
  shopt -s nullglob 2>/dev/null || true
  shopt_enabled=1
fi

found_any=0
if [ -n "$SAMPLE_FILE" ]; then
  samples_list=("$SAMPLE_FILE")
else
  samples_list=("$SAMPLES_DIR"/*)
fi

any_fail=0
for sample in "${samples_list[@]}"; do
  [ -f "$sample" ] || continue
  found_any=1
  name=$(basename "$sample")
  for mode in "novec" "vec"; do
    binname="lz4_gpu.clbin"
    if [ "$mode" = "vec" ]; then binname="lz4_gpu_vec.clbin"; fi
    echo "Running sample $name mode $mode..."
    # run harness, capture output
    outfile=ab_results/${name}_${mode}.txt
    # allow the gpu test to fail without aborting the whole script
    set +e
    ./$GPU_TEST "$sample" "$binname" | tee "$outfile"
    rc=$?
    set -e
    # parse metrics (best-effort even if program failed)
    host_ms=$(grep "Host->Device upload time" "$outfile" | awk '{print $4}' || echo 0)
    comp_ms=$(grep "Compress kernel time" "$outfile" | awk '{print $3}' || echo 0)
    read_bs_ms=$(grep "Device->Host read blockSizes time" "$outfile" | awk '{print $6}' || echo 0)
    decomp_ms=$(grep "Decompress kernel time" "$outfile" | awk '{print $3}' || echo 0)
    read_decomp_ms=$(grep "Device->Host read decompressed time" "$outfile" | awk '{print $6}' || echo 0)
    total_kernel=$(grep "Total kernel time" "$outfile" | awk '{print $4}' || echo 0)
    # Determine OK/FAILED: prefer explicit marker in output, fallback to rc
    if grep -q "Round-trip FAILED" "$outfile" 2>/dev/null; then
      ok=FAILED
    elif grep -q "Round-trip OK" "$outfile" 2>/dev/null || grep -q "Round-trip: OK" "$outfile" 2>/dev/null; then
      ok=OK
    else
      if [ "$rc" -ne 0 ]; then ok=FAILED; else ok=UNKNOWN; fi
    fi
    # move per-block stats to per-run file to avoid overwrite
    blocks_file=ab_results/${name}_${mode}_blocks.csv
    if [ -f gpu_roundtrip_stats.csv ]; then
      mv gpu_roundtrip_stats.csv "$blocks_file"
      totline=$(grep "^TOTAL" "$blocks_file" || true)
    else
      totline=""
    fi
    if [ -n "$totline" ]; then
      total_orig=$(echo "$totline" | awk -F, '{print $2}')
      total_comp=$(echo "$totline" | awk -F, '{print $3}')
      ratio=$(echo "$totline" | awk -F, '{print $4}')
    else
      total_orig=0; total_comp=0; ratio=0
    fi
    echo "$name,$mode,$host_ms,$comp_ms,$read_bs_ms,$decomp_ms,$read_decomp_ms,$total_kernel,$ok,$total_orig,$total_comp,$ratio,$blocks_file" >> "$OUT_CSV"
    if [ "$ok" != "OK" ]; then
      any_fail=1
    fi
  done
done

if [ "$found_any" -eq 0 ]; then
  echo "No sample files found in $SAMPLES_DIR. No tests were run.";
  exit 1
fi

echo "A/B run complete. Summary: $OUT_CSV"
if [ "$any_fail" -ne 0 ]; then
  echo "Some runs failed — see ab_results/*.txt and $OUT_CSV for details."
  exit 1
fi
