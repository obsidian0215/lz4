#!/usr/bin/env bash
set -euo pipefail

# ab_scan.sh
# Parameter sweep for lz4_gpu CLI. Writes all logs/results into ab_results.
# Usage: ./ab_scan.sh [OUTDIR] [CLBIN]
# Example: ./ab_scan.sh /root/lz4/lz4_gpu/ab_results /root/lz4/lz4_gpu/lz4_gpu.clbin

# Take optional CLI flags (supporting --profile / --no-profile) and remove them from args
PROFILE=${PROFILE:-0}
NEWARGS=()
for a in "$@"; do
  case "$a" in
    --profile)
      PROFILE=1; ;;
    --no-profile)
      PROFILE=0; ;;
    *) NEWARGS+=("$a"); ;;
  esac
done
set -- "${NEWARGS[@]}"

OUTDIR=${1:-/root/lz4/lz4_gpu/ab_results}
# Optional 2nd argument remains as a convenience for specifying a clbin path that will
# be injected into the launched lz4_gpu processes via the environment variable
# LZ4_GPU_CLBIN (recommended) instead of CLI args.
CLBIN=${2:-/root/lz4/lz4_gpu/lz4_gpu.clbin}
# optional overrides: pass REPS as 3rd arg
REPS=${3:-${REPS:-3}}
# Overrides via environment variables (comma-separated or whitespace-separated)
ACCELS_OVERRIDE=${ACCELS_OVERRIDE:-}
BLOCKS_OVERRIDE=${BLOCKS_OVERRIDE:-}
LOCAL_SIZES_OVERRIDE=${LOCAL_SIZES_OVERRIDE:-}
PINNED_OPTIONS_OVERRIDE=${PINNED_OPTIONS_OVERRIDE:-}
IO_OVERLAP_OPTIONS_OVERRIDE=${IO_OVERLAP_OPTIONS_OVERRIDE:-}
SAMPLES_OVERRIDE=${SAMPLES_OVERRIDE:-}
# Daemon mode support: if DAEMON_MODE=1, start daemon server and use --use-daemon flag
DAEMON_MODE=${DAEMON_MODE:-0}
# Optional resume support: when RESUME=1, skip any run that already has a matching line
# (sample,accel,block,local,pinned,mode,run) in the results.csv file to avoid duplicates
RESUME=${RESUME:-0}
DAEMON_SOCKET=${DAEMON_SOCKET:-/tmp/lz4_gpu_daemon.sock}
DAEMON_PID=""

# New: allow running different modes as part of the same sweep via MODES_OVERRIDE
# Example: export MODES_OVERRIDE="local,daemon" (or "daemon,local") to run both
MODES_OVERRIDE=${MODES_OVERRIDE:-}
# RUN_MODE is used internally when the script re-invokes itself per-mode
RUN_MODE=${RUN_MODE:-}

# Helper: parse a whitespace or comma-separated list into a bash array variable by name
parse_list_to_array() {
  local _varname="$1"; local _value="$2"
  if [ -z "$_value" ]; then
    eval "$_varname=()"
    return
  fi
  # Replace commas with spaces, then split on whitespace
  _value="${_value//,/ }"
  local arr
  read -r -a arr <<< "$_value"
  # Safely assign parsed elements into the target array variable
  eval "$_varname=()"
  local e esc
  for e in "${arr[@]}"; do
    # escape double quotes inside the element to keep eval safe
    esc="${e//\"/\\\"}"
      eval "$_varname+=(\"$esc\")"
    done
  }
FORCE_RUNS=${FORCE_RUNS:-0}
KEEP_SUCCESS_LOGS=${KEEP_SUCCESS_LOGS:-0}
# By default, automatically create required result directories when they
# don't exist. Set CREATE_DIRS=0 in the environment if you want to require
# the target OUTDIR to be pre-created and validated.
CREATE_DIRS=${CREATE_DIRS:-1}

# Safety: restrict OUTDIR to be within the lz4_gpu/ab_results tree by default
ALLOWED_ROOT="/root/lz4/lz4_gpu/ab_results"
if command -v readlink >/dev/null 2>&1; then
  REAL_OUTDIR=$(readlink -f "$OUTDIR")
else
  REAL_OUTDIR="$OUTDIR"
fi
if [ "${ALLOW_OTHER_DIR:-0}" != "1" ]; then
  # If OUTDIR is exactly the allowed root, prefer to create a timestamped subdir
  if [ "$REAL_OUTDIR" = "$ALLOWED_ROOT" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    NEW_OUTDIR="$ALLOWED_ROOT/full_run_${TS}"
    echo "WARNING: OUTDIR set to ab_results root ($ALLOWED_ROOT). Creating subdirectory and using: $NEW_OUTDIR" >&2
    mkdir -p "$NEW_OUTDIR"
    OUTDIR="$NEW_OUTDIR"
    REAL_OUTDIR=$(readlink -f "$OUTDIR")
  fi
  case "$REAL_OUTDIR" in
    "$ALLOWED_ROOT"/*) ;; # ok, it's a subdir
    *)
      echo "ERROR: OUTDIR must be under $ALLOWED_ROOT (child directory). Use ALLOW_OTHER_DIR=1 to override." >&2
      exit 1
      ;;
  esac
fi

if [ "$CREATE_DIRS" = "1" ]; then
  if [ ! -d "$OUTDIR" ]; then
    echo "Creating OUTDIR and subdirectories: $OUTDIR"
  fi
  mkdir -p "$OUTDIR"
  mkdir -p "$OUTDIR/logs"
  mkdir -p "$OUTDIR/tmp"
  mkdir -p "$OUTDIR/failed"
else
  if [ ! -d "$OUTDIR" ]; then
    echo "ERROR: OUTDIR does not exist: $OUTDIR (CREATE_DIRS=0)" >&2
    echo "Create it first (mkdir -p $OUTDIR) or rerun with CREATE_DIRS=1." >&2
    exit 1
  fi
  if [ ! -d "$OUTDIR/logs" ] || [ ! -d "$OUTDIR/tmp" ] || [ ! -d "$OUTDIR/failed" ]; then
    echo "ERROR: OUTDIR missing required subdirectories. Expected: $OUTDIR/logs, $OUTDIR/tmp, $OUTDIR/failed" >&2
    echo "Please create them or use CREATE_DIRS=1 to create automatically." >&2
    exit 1
  fi
fi

RESULT_CSV="$OUTDIR/results.csv"
  if [ ! -d "$OUTDIR" ]; then
    echo "ERROR: OUTDIR does not exist: $OUTDIR (CREATE_DIRS=0)" >&2
    echo "Create it first (mkdir -p $OUTDIR) or rerun with CREATE_DIRS=1." >&2
    exit 1
  fi

  if [ ! -f "$RESULT_CSV" ]; then
  echo "sample,input_bytes,accel,block_size,local,pinned,io_overlap,mode,run,compressed_bytes,comp_total_ms,comp_kernel_ms,comp_h2d_ms,comp_d2h_ms,dec_total_ms,dec_kernel_ms,dec_h2d_ms,dec_d2h_ms,compression_ratio,lz4_valid_frame,ok,logfile,comp_alloc_ms,comp_event_profile_ms,comp_frame_ms,dec_alloc_ms,dec_event_profile_ms,dec_frame_ms" > "$RESULT_CSV"
fi

  # Files for monitor/restart and debug tracing
  AB_SCAN_CMD_FILE="$OUTDIR/ab_scan.cmd"
  AB_SCAN_ENV_FILE="$OUTDIR/ab_scan.env"
  AB_SCAN_PID_FILE="$OUTDIR/ab_scan.pid"
  LASTRUN_FILE="$OUTDIR/ab_scan.last_run.txt"
  TRAP_LOG="$OUTDIR/ab_scan.trap.log"

  # Write a compact invocation & environment snapshot for potential restarts
  {
    echo "# ab_scan invocation: $(date '+%Y-%m-%d %H:%M:%S')"
    printf 'CMD_LINE: '
    printf '%q ' "$0" "$OUTDIR" "$CLBIN"
    echo
  } > "$AB_SCAN_CMD_FILE"
  # Record interesting env vars (shell-safe, quoted values)
  {
    for v in FORCE_RUNS REPS ACCELS_OVERRIDE BLOCKS_OVERRIDE LOCAL_SIZES_OVERRIDE PINNED_OPTIONS_OVERRIDE IO_OVERLAP_OPTIONS_OVERRIDE MODES_OVERRIDE DAEMON_MODE LZ4_GPU_CLBIN LZ4_GPU_BIN CLSRC CREATE_DIRS KEEP_SUCCESS_LOGS PROFILE; do
      val=${!v:-}
      if [ -n "$val" ]; then
        esc=${val//\"/\\\"}
        echo "$v=\"$esc\""
      fi
    done
  } > "$AB_SCAN_ENV_FILE"

  # Write PID and create placeholder files
  echo $$ > "$AB_SCAN_PID_FILE"
  touch "$LASTRUN_FILE" "$TRAP_LOG"

  # Combined exit trap that logs the exit code and last run and also runs cleanup_daemon if defined
  combined_exit() {
    ec=${?}
    echo "$(date '+%Y-%m-%d %H:%M:%S') EXIT rc=$ec pid=$$" >> "$TRAP_LOG"
    if [ -s "$LASTRUN_FILE" ]; then
      echo "Last run: $(cat "$LASTRUN_FILE" 2>/dev/null || true)" >> "$TRAP_LOG"
    fi
    if declare -f cleanup_daemon >/dev/null 2>&1; then
      cleanup_daemon || true
    fi
  }
  trap 'combined_exit' EXIT INT TERM
  trap "echo \"$(date '+%Y-%m-%d %H:%M:%S') SIGHUP\" >> \"$TRAP_LOG\"; exit 1" HUP

RESULTS_AGG_CSV="$OUTDIR/results_agg.csv"
if [ -f "$RESULTS_AGG_CSV" ]; then rm -f "$RESULTS_AGG_CSV"; fi

if [ -n "$SAMPLES_OVERRIDE" ]; then
  parse_list_to_array TMP_SAMPLES "$SAMPLES_OVERRIDE"
  SAMPLES=( "${TMP_SAMPLES[@]}" )
else
  SAMPLES=(/root/samples/*)
fi
if [ ${#SAMPLES[@]} -eq 0 ]; then
  echo "No sample files found in /root/samples - abort" >&2
  exit 1
fi

# Parameter choices (defaults - can be overridden via ACCELS_OVERRIDE, BLOCKS_OVERRIDE, LOCAL_SIZES_OVERRIDE, PINNED_OPTIONS_OVERRIDE, SAMPLES_OVERRIDE env vars)
ACCELS=(1 2 3 4 5 6 7 8 9 10 11 12)
BLOCKS=(16k 32k 64k 128k 256k 512k)
LOCAL_SIZES=(1 2 4 8 16 32 64 128 256)
PINNED_OPTIONS=("--pinned" "--no-pinned")
IO_OVERLAP_OPTIONS=("--io-overlap" "--no-io-overlap")

# apply overrides if present (comma-separated or space-separated values)
if [ -n "$ACCELS_OVERRIDE" ]; then parse_list_to_array ACCELS "$ACCELS_OVERRIDE"; fi
if [ -n "$BLOCKS_OVERRIDE" ]; then parse_list_to_array BLOCKS "$BLOCKS_OVERRIDE"; fi
if [ -n "$LOCAL_SIZES_OVERRIDE" ]; then parse_list_to_array LOCAL_SIZES "$LOCAL_SIZES_OVERRIDE"; fi
if [ -n "$PINNED_OPTIONS_OVERRIDE" ]; then parse_list_to_array PINNED_OPTIONS "$PINNED_OPTIONS_OVERRIDE"; fi
if [ -n "$IO_OVERLAP_OPTIONS_OVERRIDE" ]; then parse_list_to_array IO_OVERLAP_OPTIONS "$IO_OVERLAP_OPTIONS_OVERRIDE"; fi

BIN_DIR="$(pwd)"
# Respect externally-set LZ4_GPU_BIN environment variable; default to repo-local lz4_gpu
LZ4_GPU_BIN="${LZ4_GPU_BIN:-${BIN_DIR}/lz4_gpu}"
echo "Using LZ4_GPU_BIN=$LZ4_GPU_BIN"
if [ ! -x "$LZ4_GPU_BIN" ]; then
  echo "ERROR: lz4_gpu binary not found in $BIN_DIR. Build it first (make -j$(nproc) lz4_gpu)" >&2
  exit 2
fi

if [ -n "$CLBIN" ] && [ -f "$CLBIN" ]; then
  echo "Using precompiled OpenCL binary via env LZ4_GPU_CLBIN=$CLBIN"
else
  CLBIN=""
fi

# Kernel source path fallback (absolute path to repo source file). Use the provided CLSRC env if one exists.
CLSRC=${CLSRC:-${BIN_DIR}/lz4_gpu.cl}
if [ -n "$CLSRC" ] && [ -f "$CLSRC" ]; then
  echo "Using kernel source via env LZ4_GPU_CLSRC=$CLSRC"
else
  CLSRC=""
fi

# Compute number of total combinations and require confirmation if too many
SAMPLES_N=${#SAMPLES[@]}
ACCELS_N=${#ACCELS[@]}
BLOCKS_N=${#BLOCKS[@]}
LOCAL_SIZES_N=${#LOCAL_SIZES[@]}
# Number of pinned / io-overlap variants
PINNED_N=${#PINNED_OPTIONS[@]}
IO_OVERLAP_N=${#IO_OVERLAP_OPTIONS[@]}
# each run uses both compress local and decompress local
COMBINATIONS=$((SAMPLES_N * ACCELS_N * BLOCKS_N * LOCAL_SIZES_N * PINNED_N * IO_OVERLAP_N * REPS))
MAX_SAFE_RUNS=5000
if [ $COMBINATIONS -gt $MAX_SAFE_RUNS ] && [ "$FORCE_RUNS" != "1" ]; then
  echo "WARNING: Computed $COMBINATIONS combinations (>$MAX_SAFE_RUNS). This may take a long time to run."
  echo "To continue, re-run with FORCE_RUNS=1 in your environment (e.g. 'FORCE_RUNS=1 ./tools/ab_scan.sh')."
  exit 1
fi

# If MODES_OVERRIDE is provided, dispatch to individual runs per mode by re-invoking this script
# Now using a single OUTDIR with 'mode' as an additional column, rather than per-mode subdirs
if [ -n "$MODES_OVERRIDE" ] && [ -z "$RUN_MODE" ]; then
  parse_list_to_array MODES "$MODES_OVERRIDE"
  if [ ${#MODES[@]} -eq 0 ]; then
    echo "ERROR: MODES_OVERRIDE set but no modes parsed" >&2
    exit 1
  fi
  # Validate and compute total runs
  for m in "${MODES[@]}"; do
    case "$m" in
      local|daemon) : ;;
      *) echo "ERROR: Unknown mode: $m" >&2; exit 1 ;;
    esac
  done

  NUM_MODES=${#MODES[@]}
  TOTAL=$((COMBINATIONS * NUM_MODES))
  if [ $TOTAL -gt $MAX_SAFE_RUNS ] && [ "$FORCE_RUNS" != "1" ]; then
    echo "WARNING: Combined runs across modes = $TOTAL. Run with FORCE_RUNS=1 to skip prompt." >&2
    echo -n "Proceed? (y/N) : "
    read -r a
    if [ "$a" != "y" ]; then
      echo "Aborting." >&2
      exit 1
    fi
  fi

  # Use the same OUTDIR for all modes (no per-mode subdirs)
  # Spawn runs for each mode sequentially by re-invoking this script with RUN_MODE set
  for m in "${MODES[@]}"; do
    MODE_LABEL="$m"
    echo "Dispatching mode '$MODE_LABEL' into shared OUTDIR: $OUTDIR"
    # Before starting local mode, ensure no daemon is running (could be leftover from previous daemon mode run)
    if [ "$MODE_LABEL" = "local" ]; then
      echo "Ensuring no daemon is running before local mode..."
      pkill -f "lz4_gpu.*--daemon" 2>/dev/null || true
      rm -f "$DAEMON_SOCKET" 2>/dev/null || true
      sleep 0.3
    fi
    # Clear MODES_OVERRIDE for the child invocation to avoid recursion; pass original OUTDIR
    RUN_MODE="$MODE_LABEL" MODES_OVERRIDE= LZ4_GPU_BIN="$LZ4_GPU_BIN" "$0" "$OUTDIR" "$CLBIN"
    # After daemon mode completes, ensure daemon is killed before next mode
    if [ "$MODE_LABEL" = "daemon" ]; then
      echo "Cleaning up daemon after daemon mode..."
      pkill -f "lz4_gpu.*--daemon" 2>/dev/null || true
      rm -f "$DAEMON_SOCKET" 2>/dev/null || true
      sleep 0.3
    fi
  done
  exit 0
fi

# When invoked with RUN_MODE set (by above dispatcher), convert into DAEMON_MODE accordingly
MODE_LABEL="local"  # default mode label
if [ -n "$RUN_MODE" ]; then
  echo "RUN_MODE: $RUN_MODE"
  case "$RUN_MODE" in
    daemon) DAEMON_MODE=1; MODE_LABEL="daemon" ;;
    local) DAEMON_MODE=0; MODE_LABEL="local" ;;
    *) echo "Unknown RUN_MODE: $RUN_MODE" >&2; exit 1 ;;
  esac
fi
# Also set MODE_LABEL from DAEMON_MODE if directly set (not via RUN_MODE)
if [ "$DAEMON_MODE" = "1" ]; then MODE_LABEL="daemon"; fi
echo "Scan configuration:"
echo "  Samples        : $SAMPLES_N"
echo "  Accelerations  : ${ACCELS[*]}"
echo "  Block sizes    : ${BLOCKS[*]}"
echo "  Local sizes    : ${LOCAL_SIZES[*]}"
echo "  Pinned options : ${PINNED_OPTIONS[*]}"
echo "  Repetitions    : $REPS"
echo "  Total runs     : $COMBINATIONS"
echo "  Daemon mode    : ${DAEMON_MODE}"

# Start daemon server if DAEMON_MODE=1
if [ "$DAEMON_MODE" = "1" ]; then
  echo "Starting daemon server on socket $DAEMON_SOCKET..."
  # Kill any existing daemon on this socket
  if [ -S "$DAEMON_SOCKET" ]; then
    echo "Removing stale socket: $DAEMON_SOCKET"
    rm -f "$DAEMON_SOCKET"
  fi
  # Start daemon in background - set LZ4_GPU_CLBIN if specified
  if [ -n "$CLBIN" ]; then
    # Provide both CLBIN and CLSRC via env var to the daemon process so it can fallback to source if needed
    LZ4_GPU_CLBIN="$CLBIN" LZ4_GPU_CLSRC="$CLSRC" "$LZ4_GPU_BIN" --daemon &
  elif [ -n "$CLSRC" ]; then
    LZ4_GPU_CLSRC="$CLSRC" "$LZ4_GPU_BIN" --daemon &
  else
    "$LZ4_GPU_BIN" --daemon &
  fi
  DAEMON_PID=$!
  echo "Daemon started with PID $DAEMON_PID"
  # Wait a moment for daemon to initialize
  sleep 1
  if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "ERROR: Daemon failed to start (PID $DAEMON_PID not running)" >&2
    exit 1
  fi
  # Setup cleanup trap to kill daemon on script exit (handles normal exit, INT, TERM)
  cleanup_daemon() {
    echo "Stopping daemon (PID $DAEMON_PID)..."
    kill "$DAEMON_PID" 2>/dev/null || true
    # Wait briefly for clean shutdown
    sleep 0.5
    # Force kill if still running
    kill -9 "$DAEMON_PID" 2>/dev/null || true
    rm -f "$DAEMON_SOCKET"
  }
  trap cleanup_daemon EXIT INT TERM
fi

for sample in "${SAMPLES[@]}"; do
  [ -f "$sample" ] || continue
  sample_basename=$(basename "$sample")
  # Ensure we get actual file size even if sample is a symlink
  if command -v readlink >/dev/null 2>&1; then
    real_sample=$(readlink -f "$sample")
  else
    real_sample="$sample"
  fi
  # Use stat if available, fallback to wc -c
  if stat -c%s "$real_sample" >/dev/null 2>&1; then
    input_bytes=$(stat -c%s "$real_sample")
  else
    input_bytes=$(wc -c < "$real_sample" 2>/dev/null || true)
  fi

  for accel in "${ACCELS[@]}"; do
    for bs in "${BLOCKS[@]}"; do
      for local in "${LOCAL_SIZES[@]}"; do
                        for pinned in "${PINNED_OPTIONS[@]}"; do
                        for io_overlap in "${IO_OVERLAP_OPTIONS[@]}"; do
                      # Debug: print pinned value and current shell environment to detect parsing issues
                      # echo "DEBUG: (pinned raw) -> [$pinned]"
                      # sanitize pinned label for filenames (none/pinned/no_pinned)
                      # pinned is expected to be "--pinned" or "--no-pinned"; compute a stable label
                      if [ "$pinned" = "--pinned" ]; then
                        pinned_label="pinned"
                      elif [ "$pinned" = "--no-pinned" ]; then
                        pinned_label="no_pinned"
                      else
                        # fallback generic sanitize
                        pinned_label=$(echo "$pinned" | sed 's/^--//g; s/-/_/g')
                      fi
                      for r in $(seq 1 "$REPS"); do
            # sanitize io overlap label for filenames
            if [ "$io_overlap" = "--io-overlap" ]; then io_overlap_label="io_overlap"; elif [ "$io_overlap" = "--no-io-overlap" ]; then io_overlap_label="no_io_overlap"; else io_overlap_label=$(echo "$io_overlap" | sed 's/^--//g; s/-/_/g'); fi
            out_lz4="$OUTDIR/tmp/${sample_basename}.acc${accel}.bs${bs}.local${local}.pin${pinned_label}.io${io_overlap_label}.r${r}.lz4"
            out_dec="$OUTDIR/tmp/${sample_basename}.acc${accel}.bs${bs}.local${local}.pin${pinned_label}.io${io_overlap_label}.r${r}.dec"
            logf="$OUTDIR/logs/${sample_basename}.acc${accel}.bs${bs}.local${local}.pin${pinned_label}.io${io_overlap_label}.r${r}.log"

            echo "RUN: sample=${sample_basename} accel=${accel} bs=${bs} local=${local} pinned=${pinned_label} io_overlap=${io_overlap_label} r=${r}"

            # If RESUME is enabled, check for an existing entry in $RESULTS_CSV and skip if present
            if [ "$RESUME" = "1" ] && [ -f "$RESULT_CSV" ]; then
              found=$(awk -F, -v s="$sample_basename" -v a="$accel" -v b="$bs" -v l="$local" -v p="$pinned_label" -v io="$io_overlap_label" -v m="$MODE_LABEL" -v r="$r" 'NR>1 && $1==s && $3==a && $4==b && $5==l && $6==p && $7==io && $8==m && $9==r {print 1; exit}' "$RESULT_CSV" || true)
              if [ "$found" = "1" ]; then
                echo "SKIP (already present): sample=${sample_basename} accel=${accel} bs=${bs} local=${local} pinned=${pinned_label} r=${r} (results.csv has an entry)"
                rm -f "$LASTRUN_FILE" 2>/dev/null || true
                continue
              fi
            fi

            # compress - build command array to avoid quoting pitfalls
            set +e
            cmd=("$LZ4_GPU_BIN" -c -v "$sample" -o "$out_lz4" --blocksize "$bs" --local "$local" -l "$accel")
            if [ "$PROFILE" = "1" ]; then
              cmd+=("-p")
            fi
            if [ -n "$pinned" ]; then cmd+=("$pinned"); fi
            if [ -n "$io_overlap" ]; then cmd+=("$io_overlap"); fi
            if [ "$DAEMON_MODE" = "1" ]; then cmd+=("--use-daemon"); fi
            ( printf 'ENV: LZ4_GPU_CLBIN=%q\n' "$CLBIN"; printf 'CMD: '; printf '%q ' "${cmd[@]}"; echo ) >> "$logf"
            if [ -n "$CLBIN" ] && [ "$DAEMON_MODE" != "1" ]; then
              LZ4_GPU_CLBIN="$CLBIN" LZ4_GPU_CLSRC="$CLSRC" "${cmd[@]}" > "$logf" 2>&1
            elif [ -n "$CLSRC" ]; then
              LZ4_GPU_CLSRC="$CLSRC" "${cmd[@]}" > "$logf" 2>&1
            else
              "${cmd[@]}" > "$logf" 2>&1
            fi
            rc=$?
            set -e
            if [ $rc -ne 0 ]; then
              echo "Compression failed for ${sample_basename} accel=${accel} bs=${bs} local=${local} (exit $rc) - see $logf"
              mv "$logf" "$OUTDIR/failed/" 2>/dev/null || true
              rm -f "$LASTRUN_FILE" 2>/dev/null || true
              continue
            fi

            # decompress - build command array
            set +e
            cmd=("$LZ4_GPU_BIN" -d -v "$out_lz4" -o "$out_dec" --local "$local")
            if [ "$PROFILE" = "1" ]; then
              cmd+=("-p")
            fi
            if [ -n "$pinned" ]; then cmd+=("$pinned"); fi
            if [ -n "$io_overlap" ]; then cmd+=("$io_overlap"); fi
            if [ "$DAEMON_MODE" = "1" ]; then cmd+=("--use-daemon"); fi
            ( printf 'ENV: LZ4_GPU_CLBIN=%q\n' "$CLBIN"; printf 'CMD: '; printf '%q ' "${cmd[@]}"; echo ) >> "$logf"
            if [ -n "$CLBIN" ] && [ "$DAEMON_MODE" != "1" ]; then LZ4_GPU_CLBIN="$CLBIN" "${cmd[@]}" >> "$logf" 2>&1; else "${cmd[@]}" >> "$logf" 2>&1; fi
            rc=$?
            set -e
            if [ $rc -ne 0 ]; then
              echo "Decompression failed for ${sample_basename} accel=${accel} bs=${bs} local=${local} (exit $rc) - see $logf"
              mv "$logf" "$OUTDIR/failed/" 2>/dev/null || true
              rm -f "$LASTRUN_FILE" 2>/dev/null || true
              continue
            fi

            # verify
              if cmp -s "$sample" "$out_dec"; then
              ok=OK
              # parse some timing info from log to include in CSV
              # Extract compression-specific metrics from the 'Compression Statistics' block
              # Support both daemon mode ("Total time (IPC + GPU): X ms") and local mode ("Total time X.XXms")
              # The printed header differs between local and daemon modes.  Local prints
              # "=== Compression Statistics ===" while daemon mode prints
              # "=== Compression Statistics (Daemon Mode) ===". Match on the short
              # keyword so both forms are accepted.
              comp_total_ms=$(awk 'BEGIN{f=0} /=== Compression Statistics/ {f=1; next} f && /Total time/ {match($0, /[0-9]+(\.[0-9]+)?/); if(RSTART) print substr($0, RSTART, RLENGTH); exit}' "$logf" || true)
              # Match both "Kernel Exec:" (local mode) and "Kernel:" (daemon mode)
              comp_kernel_ms=$(awk 'BEGIN{f=0} /=== Compression Statistics/ {f=1; next} f && /^[[:space:]]*(Kernel Exec|Kernel):/ {for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+(\.[0-9]+)?$/) {print $i; exit}}' "$logf" || true)
              comp_h2d_ms=$(awk 'BEGIN{f=0} /=== Compression Statistics/ {f=1; next} f && /Host[^A-Za-z]*Device/ {for(i=1;i<=NF;i++) if ($i ~ /[0-9]+(\.[0-9]+)?/) {print $i; exit}}' "$logf" || true)
              comp_d2h_ms=$(awk 'BEGIN{f=0} /=== Compression Statistics/ {f=1; next} f && /Device[^A-Za-z]*Host/ {for(i=1;i<=NF;i++) if ($i ~ /[0-9]+(\.[0-9]+)?/) {print $i; exit}}' "$logf" || true)
              # additional compression profile values
              comp_alloc_ms=$(awk 'BEGIN{f=0} /=== Compression Statistics/ {f=1; next} f && /Buffer Alloc:/ {match($0, /[0-9]+(\.[0-9]+)?/); if(RSTART) print substr($0, RSTART, RLENGTH); exit }' "$logf" || true)
              comp_event_profile_ms=$(awk 'BEGIN{f=0} /=== Compression Statistics/ {f=1; next} f && /event profiling/ {match($0, /[0-9]+(\.[0-9]+)?/); if(RSTART) print substr($0, RSTART, RLENGTH); exit }' "$logf" || true)
              comp_frame_ms=$(awk 'BEGIN{f=0} /=== Compression Statistics/ {f=1; next} f && /^[[:space:]]*Frame assembly:/ {match($0, /[0-9]+(\.[0-9]+)?/); if(RSTART) print substr($0, RSTART, RLENGTH); exit }' "$logf" || true)
              comp_alloc_ms=${comp_alloc_ms:-NA}
              comp_event_profile_ms=${comp_event_profile_ms:-NA}
              comp_frame_ms=${comp_frame_ms:-NA}
              # Extract decompression timings by scanning the Decompression section
              # Handle both the normal and daemon-mode variants for the header
              dec_total_ms=$(awk 'BEGIN{f=0} /=== Decompression Statistics/ {f=1; next} f && /Total time/ {match($0, /[0-9]+(\.[0-9]+)?/); if(RSTART) print substr($0, RSTART, RLENGTH); exit}' "$logf" || true)
              # Match both "Kernel Exec:" (local mode) and "Kernel:" (daemon mode)
              dec_kernel_ms=$(awk 'BEGIN{f=0} /=== Decompression Statistics/ {f=1; next} f && /^[[:space:]]*(Kernel Exec|Kernel):/ {for(i=1;i<=NF;i++){ if($i ~ /^[0-9]+(\.[0-9]+)?$/){print $i; exit}} }' "$logf" || true)
              dec_h2d_ms=$(awk 'BEGIN{f=0} /=== Decompression Statistics/ {f=1; next} f && /Host[^A-Za-z]*Device/ {for(i=1;i<=NF;i++){ if($i ~ /[0-9]+(\.[0-9]+)?/){print $i; exit}} }' "$logf" || true)
              dec_d2h_ms=$(awk 'BEGIN{f=0} /=== Decompression Statistics/ {f=1; next} f && /Device[^A-Za-z]*Host/ {for(i=1;i<=NF;i++){ if($i ~ /[0-9]+(\.[0-9]+)?/){print $i; exit}} }' "$logf" || true)
              # additional decompression profile values
              dec_alloc_ms=$(awk 'BEGIN{f=0} /=== Decompression Statistics/ {f=1; next} f && /Buffer Alloc:/ {match($0, /[0-9]+(\.[0-9]+)?/); if(RSTART) print substr($0, RSTART, RLENGTH); exit }' "$logf" || true)
              dec_event_profile_ms=$(awk 'BEGIN{f=0} /=== Decompression Statistics/ {f=1; next} f && /event profiling/ {match($0, /[0-9]+(\.[0-9]+)?/); if(RSTART) print substr($0, RSTART, RLENGTH); exit }' "$logf" || true)
              dec_frame_ms=$(awk 'BEGIN{f=0} /=== Decompression Statistics/ {f=1; next} f && /^[[:space:]]*Frame assembly:/ {match($0, /[0-9]+(\.[0-9]+)?/); if(RSTART) print substr($0, RSTART, RLENGTH); exit }' "$logf" || true)
              dec_alloc_ms=${dec_alloc_ms:-NA}
              dec_event_profile_ms=${dec_event_profile_ms:-NA}
              dec_frame_ms=${dec_frame_ms:-NA}
              dec_total_ms=${dec_total_ms:-NA}
              dec_kernel_ms=${dec_kernel_ms:-NA}
              compressed_bytes=$(stat -c%s "$out_lz4" || echo NA)
              # Check whether the generated frame is accepted by standard lz4 tool
              lz4_valid="NA"
              if command -v lz4 >/dev/null 2>&1; then
                if lz4 -t "$out_lz4" >/dev/null 2>&1; then lz4_valid=Y; else lz4_valid=N; fi
              fi
              # 'clbin_used' column removed; keep runtime log lines for debugging but do not record boolean
              # Normalize fields: compression metrics
              comp_total_ms=${comp_total_ms:-NA}
              comp_kernel_ms=${comp_kernel_ms:-NA}
              comp_h2d_ms=${comp_h2d_ms:-NA}
              comp_d2h_ms=${comp_d2h_ms:-NA}
              if [ "$compressed_bytes" = "NA" ]; then compression_ratio=NA; else compression_ratio=$(awk -v i="$input_bytes" -v o="$compressed_bytes" 'BEGIN{printf "%.3f", (i>0)?(i/o):0}'); fi

              echo "${sample_basename},${input_bytes},${accel},${bs},${local},${pinned_label},${io_overlap_label},${MODE_LABEL},${r},${compressed_bytes},${comp_total_ms},${comp_kernel_ms},${comp_h2d_ms},${comp_d2h_ms},${dec_total_ms},${dec_kernel_ms},${dec_h2d_ms},${dec_d2h_ms},${compression_ratio},${lz4_valid},${ok},${logf},${comp_alloc_ms},${comp_event_profile_ms},${comp_frame_ms},${dec_alloc_ms},${dec_event_profile_ms},${dec_frame_ms}" >> "$RESULT_CSV"

              # cleanup successful artifacts (optionally keep the log via KEEP_SUCCESS_LOGS)
              rm -f "$out_lz4" "$out_dec"
              if [ "$KEEP_SUCCESS_LOGS" != "1" ]; then
                rm -f "$logf"
              fi
            else
              ok=FAILED
              echo "Verification FAILED for ${sample_basename} accel=${accel} bs=${bs} local=${local} - see $logf"
              mv "$logf" "$OUTDIR/failed/" 2>/dev/null || true
              # keep artifacts for debugging
              rm -f "$LASTRUN_FILE" 2>/dev/null || true
            fi

            done
          done
          done
        done
      done
    done

done

echo "Scan complete. CSV results: $RESULT_CSV"

# Aggregate per-combination averages (OK runs only) into a separate CSV
# The key now includes 'mode' as part of the combination
echo "Generating aggregated averages -> $RESULTS_AGG_CSV"
awk -F"," -v OUT="$RESULTS_AGG_CSV" '
NR==1 { next }
{
  # Header: sample,input_bytes,accel,block_size,local,pinned,io_overlap,mode,run,compressed_bytes,...
  # Key includes mode and io_overlap ($7,$8) as part of the grouping
  key=$1","$3","$4","$5","$6","$7","$8
  runs[key]++
  if ($21 == "OK") {
    success[key]++
    if ($10 ~ /^[0-9]+$/) { sum_comp[key]+= $10; cnt_comp[key]++ }
    if ($11 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_total[key]+= $11; cnt_total[key]++ }
    if ($12 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_kernel[key]+= $12; cnt_kernel[key]++ }
    if ($13 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_h2d[key]+= $13; cnt_h2d[key]++ }
    if ($14 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_d2h[key]+= $14; cnt_d2h[key]++ }
    if ($15 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_dec_total[key]+= $15; cnt_dec_total[key]++ }
    if ($16 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_dec_kernel[key]+= $16; cnt_dec_kernel[key]++ }
    if ($17 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_dec_h2d[key]+= $17; cnt_dec_h2d[key]++ }
    if ($18 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_dec_d2h[key]+= $18; cnt_dec_d2h[key]++ }
    if ($19 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_ratio[key]+= $19; cnt_ratio[key]++ }
    if ($23 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_comp_alloc[key] += $23; cnt_comp_alloc[key]++ }
    if ($24 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_comp_event_profile[key] += $24; cnt_comp_event_profile[key]++ }
    if ($25 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_comp_frame[key] += $25; cnt_comp_frame[key]++ }
    if ($26 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_dec_alloc[key] += $26; cnt_dec_alloc[key]++ }
    if ($27 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_dec_event_profile[key] += $27; cnt_dec_event_profile[key]++ }
    if ($28 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_dec_frame[key] += $28; cnt_dec_frame[key]++ }
    if (lz4_map[key] == "") lz4_map[key] = $20; else if (lz4_map[key] != $20) lz4_map[key] = "MIXED"
  }
}
END {
  print "sample,accel,block_size,local,pinned,io_overlap,mode,runs,successes,avg_compressed_bytes,avg_comp_total_ms,avg_comp_kernel_ms,avg_comp_h2d_ms,avg_comp_d2h_ms,avg_decomp_total_ms,avg_decomp_kernel_ms,avg_decomp_h2d_ms,avg_decomp_d2h_ms,avg_compression_ratio,lz4_valid_frame,success_rate,avg_comp_alloc_ms,avg_comp_event_profile_ms,avg_comp_frame_ms,avg_dec_alloc_ms,avg_dec_event_profile_ms,avg_dec_frame_ms" > OUT
  for (k in runs) {
    split(k, parts, ",")
    sample = parts[1]; accel = parts[2]; bs = parts[3]; local = parts[4]; pinned = parts[5]; io = parts[6]; mode = parts[7]
    r = runs[k] + 0; s = success[k] + 0
    avg_comp = (cnt_comp[k] ? sum_comp[k]/cnt_comp[k] : "NA")
    avg_total = (cnt_total[k] ? sprintf("%.3f", sum_total[k]/cnt_total[k]) : "NA")
    avg_kernel = (cnt_kernel[k] ? sprintf("%.3f", sum_kernel[k]/cnt_kernel[k]) : "NA")
    avg_h2d = (cnt_h2d[k] ? sprintf("%.3f", sum_h2d[k]/cnt_h2d[k]) : "NA")
    avg_d2h = (cnt_d2h[k] ? sprintf("%.3f", sum_d2h[k]/cnt_d2h[k]) : "NA")
    avg_dec_total = (cnt_dec_total[k] ? sprintf("%.3f", sum_dec_total[k]/cnt_dec_total[k]) : "NA")
    avg_dec_kernel = (cnt_dec_kernel[k] ? sprintf("%.3f", sum_dec_kernel[k]/cnt_dec_kernel[k]) : "NA")
    avg_dec_h2d = (cnt_dec_h2d[k] ? sprintf("%.3f", sum_dec_h2d[k]/cnt_dec_h2d[k]) : "NA")
    avg_dec_d2h = (cnt_dec_d2h[k] ? sprintf("%.3f", sum_dec_d2h[k]/cnt_dec_d2h[k]) : "NA")
    avg_ratio = (cnt_ratio[k] ? sprintf("%.3f", sum_ratio[k]/cnt_ratio[k]) : "NA")
    avg_comp_alloc = (cnt_comp_alloc[k] ? sprintf("%.3f", sum_comp_alloc[k]/cnt_comp_alloc[k]) : "NA")
    avg_comp_event_profile = (cnt_comp_event_profile[k] ? sprintf("%.3f", sum_comp_event_profile[k]/cnt_comp_event_profile[k]) : "NA")
    avg_comp_frame = (cnt_comp_frame[k] ? sprintf("%.3f", sum_comp_frame[k]/cnt_comp_frame[k]) : "NA")
    avg_dec_alloc = (cnt_dec_alloc[k] ? sprintf("%.3f", sum_dec_alloc[k]/cnt_dec_alloc[k]) : "NA")
    avg_dec_event_profile = (cnt_dec_event_profile[k] ? sprintf("%.3f", sum_dec_event_profile[k]/cnt_dec_event_profile[k]) : "NA")
    avg_dec_frame = (cnt_dec_frame[k] ? sprintf("%.3f", sum_dec_frame[k]/cnt_dec_frame[k]) : "NA")
    lv = (lz4_map[k] ? lz4_map[k] : "NA")
    success_rate = (r > 0 ? s / r : 0)
    printf "%s,%s,%s,%s,%s,%s,%s,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.3f,%s,%s,%s,%s,%s,%s\n", sample, accel, bs, local, pinned, io, mode, r, s, avg_comp, avg_total, avg_kernel, avg_h2d, avg_d2h, avg_dec_total, avg_dec_kernel, avg_dec_h2d, avg_dec_d2h, avg_ratio, lv, success_rate, avg_comp_alloc, avg_comp_event_profile, avg_comp_frame, avg_dec_alloc, avg_dec_event_profile, avg_dec_frame >> OUT
  }
}' "$RESULT_CSV"

echo "Aggregated results written to: $RESULTS_AGG_CSV"

