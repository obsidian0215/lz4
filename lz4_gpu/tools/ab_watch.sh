#!/usr/bin/env bash
set -euo pipefail

DEFAULT_OUTDIR=/root/lz4/lz4_gpu/ab_results
# Defaults
OUTDIR="$DEFAULT_OUTDIR"
INTERVAL=60
STOP=0
DELETE_OUTDIR=0
ARGS=()
START_SCAN=no  # pass 'start' to have this script start ab_scan.sh (otherwise monitor an existing run)
QUIET=${QUIET:-0}         # default: not quiet -> prints to stdout and log
AUTO_RESTART=${AUTO_RESTART:-0}  # default: do not restart unless requested
RESTART_LIMIT=${RESTART_LIMIT:-1} # default: single restart allowed
AB_SCAN_SCRIPT="$(pwd)/tools/ab_scan.sh"
ANALYSIS_SCRIPT="$(pwd)/tools/analyze_rank_per_file.py"
MONITOR_LOG="$OUTDIR/ab_watch.log"
RESTART_LOCK_DIR="$OUTDIR/ab_watch.restart.lockdir"
AB_WATCH_PID_FILE="$OUTDIR/ab_watch.pid"

# logging helper: if QUIET is enabled, only append to MONITOR_LOG. Otherwise print+append.
log() {
  # Always append to MONITOR_LOG. Print to stdout only for non-quiet runs.
  printf "%s\n" "$*" >> "$MONITOR_LOG"
  if [ "$QUIET" != "1" ]; then
    printf "%s\n" "$*"
  fi
}

# Simple argument/flag parsing (supports: -q|--quiet, -r|--restart, --restarts <n>, -i|--interval <n>, start)
ARGS=()
usage() {
  cat <<EOF
Usage: $(basename $0) [--quiet] [--restart] [--restarts N] [--interval N] [OUTDIR] [interval] [start]
  --quiet  | -q         : Do not print progress to stdout (append only to MONITOR_LOG).
  --no-quiet | -nq      : Print progress to stdout (default).
  --restart | -r        : If ab_scan is found exited while results incomplete, automatically restart it (RESUME=1).
  --restarts N          : Allow up to N automatic restart attempts (default: 1).
  --interval N | -i N   : Interval in seconds between progress checks.
  OUTDIR                : Path for ab_results run (defaults to /root/lz4/lz4_gpu/ab_results if omitted).
  start                 : Start a new scan if none is running (equivalent to previous start arg).
  stop                  : Stop any ab_watch/ab_scan related to OUTDIR.
  --delete | delete     : When used with stop, also delete the OUTDIR afterwards.
  -h | --help           : Show this help message.
EOF
}
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage; exit 0;;
    -q|--quiet)
      QUIET=1; shift;;
    -nq|--no-quiet)
      QUIET=0; shift;;
    -r|--restart)
      AUTO_RESTART=1; shift;;
    --restarts)
      if [ -n "$2" ]; then RESTART_LIMIT="$2"; shift 2; else shift; fi;;
    --stop|stop)
      STOP=1; shift;;
    --delete|delete)
      DELETE_OUTDIR=1; shift;;
    -i|--interval)
      if [ -n "$2" ]; then INTERVAL="$2"; shift 2; else shift; fi;;
    start)
      START_SCAN=start; shift;;
    --)
      shift; break;;
    -*|--*)
      echo "Unknown option: $1" >&2; shift;;
    *)
      if [ "$OUTDIR" = "$DEFAULT_OUTDIR" ] && [ "$1" != "$DEFAULT_OUTDIR" ]; then
        OUTDIR="$1"
      else
        # If this looks numeric and no explicit interval was passed, treat as INTERVAL
        if [[ "$1" =~ ^[0-9]+$ ]]; then INTERVAL="$1"; fi
      fi
      shift;;
  esac
done

# Re-evaluate output paths in case OUTDIR was set via args
MONITOR_LOG="$OUTDIR/ab_watch.log"
RESTART_LOCK_DIR="$OUTDIR/ab_watch.restart.lockdir"
AB_WATCH_PID_FILE="$OUTDIR/ab_watch.pid"

# Only allow a single ab_watch process per OUTDIR by using a pidfile
if [ -f "$AB_WATCH_PID_FILE" ] && ps -p $(cat "$AB_WATCH_PID_FILE") >/dev/null 2>&1; then
  echo "[watch] ab_watch already running (pid $(cat "$AB_WATCH_PID_FILE")); exiting reuse mode" >&2
  exit 0
fi



stop_outdir() {
  # Kill ab_watch processes referencing this OUTDIR, but don't kill this process if it's running as stop mode
  for p in $(pgrep -af 'ab_watch.sh' | egrep "${OUTDIR//\//\\/}" | awk '{print $1}' || true); do
    if [ -n "$p" ] && [ "$p" != "$$" ]; then
      log "[stop] Terminating ab_watch pid: $p"
      kill -TERM "$p" 2>/dev/null || true
      sleep 0.5
      if ps -p "$p" >/dev/null 2>&1; then
        log "[stop] Forcing ab_watch pid: $p"
        kill -KILL "$p" 2>/dev/null || true
      fi
    fi
  done
  # Kill ab_scan processes referencing this OUTDIR
  for p in $(pgrep -af 'ab_scan.sh' | egrep "${OUTDIR//\//\\/}" | awk '{print $1}' || true); do
    if [ -n "$p" ]; then
      log "[stop] Terminating ab_scan pid: $p"
      kill -TERM "$p" 2>/dev/null || true
      sleep 0.5
      if ps -p "$p" >/dev/null 2>&1; then
        log "[stop] Forcing ab_scan pid: $p"
        kill -KILL "$p" 2>/dev/null || true
      fi
    fi
  done
  # Remove pid files and restart lock/count files
  rm -f "$OUTDIR/ab_watch.pid" "$OUTDIR/ab_scan.pid" "$OUTDIR/ab_watch.restart_count" 2>/dev/null || true
  rm -rf "$OUTDIR/ab_watch.restart.lockdir" 2>/dev/null || true
  if [ "$DELETE_OUTDIR" = "1" ]; then
    log "[stop] Deleting OUTDIR: $OUTDIR"
    rm -rf "$OUTDIR" 2>/dev/null || true
    # Do not attempt to log to the target MONITOR_LOG after deletion (it will fail). Use stdout instead.
    printf "[stop] Deleted OUTDIR: %s\n" "$OUTDIR"
  fi
  log "[stop] Done"
  exit 0
}
echo $$ > "$AB_WATCH_PID_FILE"
trap 'rm -f "$AB_WATCH_PID_FILE"; rm -rf "$RESTART_LOCK_DIR" 2>/dev/null || true' EXIT INT TERM

if [ "$STOP" = "1" ]; then
  stop_outdir
fi

function now() { date '+%Y-%m-%d %H:%M:%S'; }

if [ ! -d "$OUTDIR" ]; then
  echo "ERROR: OUTDIR not found: $OUTDIR" >&2
  exit 1
fi

RESULTS_CSV="$OUTDIR/results.csv"
RESULTS_AGG="$OUTDIR/results_agg.csv"
FULL_SCAN_LOG="$OUTDIR/full_scan.log"
AB_SCAN_PID_FILE="$OUTDIR/ab_scan.pid"

log "[watch] Starting ab_watch for OUTDIR=$OUTDIR, interval=${INTERVAL}s (start=${START_SCAN})"

if [ "$START_SCAN" = "start" ]; then
  # Start ab_scan in the background (recreate outdir if needed)
    if [ -f "$AB_SCAN_PID_FILE" ] && ps -p $(cat "$AB_SCAN_PID_FILE") > /dev/null 2>&1; then
    log "[watch] existing ab_scan running (pid $(cat \"$AB_SCAN_PID_FILE\")); not starting a new one"
  else
    TS=$(date +%Y%m%d_%H%M%S)
    OUTDIR_TS="${OUTDIR}/full_run_${TS}"
    mkdir -p "$OUTDIR_TS"/logs "$OUTDIR_TS"/tmp "$OUTDIR_TS"/failed
    log "[watch] Starting ab_scan into new outdir: $OUTDIR_TS"
    # Default full-scan environment (tunable, but following prior examples)
    export LZ4_GPU_CLBIN=${LZ4_GPU_CLBIN:-$(pwd)/lz4_gpu.clbin}
    FORCE_RUNS=1 CREATE_DIRS=1 KEEP_SUCCESS_LOGS=1 REPS=3 ACCELS_OVERRIDE="1,4,8" BLOCKS_OVERRIDE="16k,32k,64k,256k" LOCAL_SIZES_OVERRIDE="1,8,64,256" PINNED_OPTIONS_OVERRIDE="--pinned" SAMPLES_OVERRIDE="" LZ4_GPU_CLBIN="$LZ4_GPU_CLBIN" "$AB_SCAN_SCRIPT" "$OUTDIR_TS" "$LZ4_GPU_CLBIN" > "$OUTDIR_TS/full_scan.log" 2>&1 &
    echo $! > "$OUTDIR_TS/ab_scan.pid"
    log "[watch] ab_scan PID: " $(cat "$OUTDIR_TS/ab_scan.pid")
    OUTDIR="$OUTDIR_TS"
    RESULTS_CSV="$OUTDIR/results.csv"
    RESULTS_AGG="$OUTDIR/results_agg.csv"
    FULL_SCAN_LOG="$OUTDIR/full_scan.log"
  fi
fi

log "[watch] Monitoring OUTDIR=$OUTDIR"

while true; do
  log "[$(now)] Running ab_progress check"

  # Parse total runs (from full_scan.log) - fallback 0
  TOTAL=$(grep -m1 "Total runs" "$FULL_SCAN_LOG" 2>/dev/null | sed -n 's/.*Total runs *: *\([0-9]*\).*/\1/p' || true)
  TOTAL=${TOTAL:-0}

  # Count completed runs (OK or FAILED) - robust against CSV wrapping by searching for ,OK, or ,FAILED,
  if [ -f "$RESULTS_CSV" ]; then
    SUCC=$(grep -c ',OK,' "$RESULTS_CSV" || true)
    FAIL=$(grep -c ',FAILED,' "$RESULTS_CSV" || true)
    COMPLETED=$((SUCC + FAIL))
  else
    COMPLETED=0
    SUCC=0
    FAIL=0
  fi

  # Run the progress summary (inlined version of tools/ab_progress.sh) and append to monitor log for history
  # This avoids a dependency on ab_progress.sh while preserving the same output.
  {
    # ensure file exists
    if [ -f "$RESULTS_CSV" ]; then
      # compute averages and throughputs (comp & dec)
      # New CSV format: sample,input_bytes,accel,block_size,local,pinned,mode,run,compressed_bytes,comp_total_ms,...
      # Column indices: comp_total=$10, comp_kernel=$11, comp_h2d=$12, comp_d2h=$13
      #                 dec_total=$14, dec_kernel=$15, dec_h2d=$16, dec_d2h=$17, ratio=$18, lz4_valid=$19, ok=$20
      AVG_COMP_TOTAL_MS=$(awk -F, 'NR>1 && $10 ~ /^[0-9]+(\.[0-9]+)?$/ { sum+=$10; n++ } END { if(n>0) printf "%.3f", sum/n; else print "NA" }' "$RESULTS_CSV")
      AVG_COMP_KERNEL_MS=$(awk -F, 'NR>1 && $11 ~ /^[0-9]+(\.[0-9]+)?$/ { sum+=$11; n++ } END { if(n>0) printf "%.3f", sum/n; else print "NA" }' "$RESULTS_CSV")
      AVG_DEC_TOTAL_MS=$(awk -F, 'NR>1 && $14 ~ /^[0-9]+(\.[0-9]+)?$/ { sum+=$14; n++ } END { if(n>0) printf "%.3f", sum/n; else print "NA" }' "$RESULTS_CSV")
      AVG_DEC_KERNEL_MS=$(awk -F, 'NR>1 && $15 ~ /^[0-9]+(\.[0-9]+)?$/ { sum+=$15; n++ } END { if(n>0) printf "%.3f", sum/n; else print "NA" }' "$RESULTS_CSV")
      AVG_TH_COMP_TOTAL=$(awk -F, 'NR>1 && $10 ~ /^[0-9]+(\.[0-9]+)?$/ && $2 ~ /^[0-9]+$/ { th = ($2 / (1024*1024))/($10/1000); sum+=th; n++ } END { if(n>0) printf "%.2f", sum/n; else print "NA" }' "$RESULTS_CSV")
      AVG_TH_COMP_KERNEL=$(awk -F, 'NR>1 && $11 ~ /^[0-9]+(\.[0-9]+)?$/ && $2 ~ /^[0-9]+$/ { th = ($2 / (1024*1024))/($11/1000); sum+=th; n++ } END { if(n>0) printf "%.2f", sum/n; else print "NA" }' "$RESULTS_CSV")
      AVG_TH_DEC_TOTAL=$(awk -F, 'NR>1 && $14 ~ /^[0-9]+(\.[0-9]+)?$/ && $2 ~ /^[0-9]+$/ { th = ($2 / (1024*1024))/($14/1000); sum+=th; n++ } END { if(n>0) printf "%.2f", sum/n; else print "NA" }' "$RESULTS_CSV")
      AVG_TH_DEC_KERNEL=$(awk -F, 'NR>1 && $15 ~ /^[0-9]+(\.[0-9]+)?$/ && $2 ~ /^[0-9]+$/ { th = ($2 / (1024*1024))/($15/1000); sum+=th; n++ } END { if(n>0) printf "%.2f", sum/n; else print "NA" }' "$RESULTS_CSV")

      FAILURES=$(awk -F, 'NR>1 && $20 != "OK" { f++ } END { print (f+0) }' "$RESULTS_CSV")
      SUCCESS=$(awk -F, 'NR>1 && $20 == "OK" { s++ } END { print (s+0) }' "$RESULTS_CSV")
      UNIQUE_COMBOS=$(awk -F, 'NR>1 { print $1","$3","$4","$5","$6","$7 }' "$RESULTS_CSV" | sort -u | wc -l 2>/dev/null || echo 0)

      # average ms used for ETA if available
      if [ "$AVG_COMP_TOTAL_MS" = "NA" ] || [ -z "$AVG_COMP_TOTAL_MS" ]; then
        ETA_SECS="NA"
      else
        if [ "$TOTAL" -gt 0 ] && [ "$COMPLETED" -lt "$TOTAL" ]; then
          ETA_SECS=$(awk -v t="$TOTAL" -v c="$COMPLETED" -v m="$AVG_COMP_TOTAL_MS" 'BEGIN{printf "%d", int((t-c) * (m) / 1000)}')
        else
          ETA_SECS="NA"
        fi
      fi
      format_secs() { local t=$1; if [ "$t" = "NA" ]; then echo "NA"; return; fi; local h=$((t/3600)); local m=$(((t%3600)/60)); local s=$((t%60)); printf "%02d:%02d:%02d" $h $m $s; }
      ETA_HMS=$(format_secs "$ETA_SECS")

      log "AB Scan Progress Summary (OUTDIR: $OUTDIR)"
      log "----------------------------------------"
      log "Total runs planned : $TOTAL"
      log "Completed runs     : $COMPLETED"
      log "Unique combos proc : $UNIQUE_COMBOS"
      log "Successes          : $SUCCESS"
      log "Failures           : $FAILURES"
      log "Average comp total time (ms)   : ${AVG_COMP_TOTAL_MS} ms"
      log "Average comp kernel time (ms)  : ${AVG_COMP_KERNEL_MS} ms"
      log "Average comp throughput (total): ${AVG_TH_COMP_TOTAL} MB/s"
      log "Average comp throughput (kernel): ${AVG_TH_COMP_KERNEL} MB/s"
      log "Average dec total time (ms)    : ${AVG_DEC_TOTAL_MS} ms"
      log "Average dec kernel time (ms)   : ${AVG_DEC_KERNEL_MS} ms"
      log "Average dec throughput (total) : ${AVG_TH_DEC_TOTAL} MB/s"
      log "Average dec throughput (kernel): ${AVG_TH_DEC_KERNEL} MB/s"
      log "Estimated remaining ETA : ${ETA_HMS}"
      log ""
      log "Top 10 runs by compression throughput (MB/s): (throughput,sample,accel,block,local,pinned,mode,compressed_bytes,comp_total_ms,comp_kernel_ms)"
      awk -F, 'NR>1 && $10 ~ /^[0-9]+(\.[0-9]+)?$/ && $2 ~ /^[0-9]+$/ { th = ($2/ (1024*1024))/($10/1000); printf("%.3f,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", th, $1, $3, $4, $5, $6, $7, $9, $10, $11) }' "$RESULTS_CSV" | sort -rk1 -t, | head -n 10 | column -t -s, | while IFS= read -r l; do log "$l"; done || true
      log ""
      log "Top 10 COMBOS by average compression throughput (MB/s): (avg_th,sample,accel,block,local,pinned,mode,count,avg_total_ms)"
      awk -F, 'NR>1 && $10 ~ /^[0-9]+(\.[0-9]+)?$/ && $2 ~ /^[0-9]+$/ { key=$1","$3","$4","$5","$6","$7; sum_in[key]+=$2; sum_ms[key]+=$10; cnt[key]++ } END { for (k in cnt) { avg_th = (sum_in[k] * 1000.0 / sum_ms[k]) / (1024*1024); avg_ms = sum_ms[k]/cnt[k]; printf("%.3f,%s,%d,%.3f\n", avg_th, k, cnt[k], avg_ms) } }' "$RESULTS_CSV" | sort -nrk1 -t, | head -n 10 | column -t -s, | while IFS= read -r l; do log "$l"; done || true
      log ""
      log "Last 5 runs (most recent)"
      tail -n 5 "$RESULTS_CSV" | awk -F, 'NR>1 { printf("%s, mode=%s, run=%s, comp_total_ms=%s, comp_kernel_ms=%s, comp_h2d=%s, comp_d2h=%s, dec_total_ms=%s, dec_kernel_ms=%s, ratio=%s, ok=%s\n", $1, $7, $8, $10, $11, $12, $13, $14, $15, $18, $20) }' | while IFS= read -r l; do log "$l"; done
    else
      log "AB Scan Progress Summary (OUTDIR: $OUTDIR)"
      log "    Results CSV not yet available: $RESULTS_CSV"
    fi
  } 2>/dev/null || log "[watch] inline progress summary failed"

  log "[$(now)] Status: completed=${COMPLETED} total=${TOTAL} successes=${SUCC:-0} failures=${FAIL:-0}"

  # Check for ab_scan process; if missing and not completed, always restart (single-run enforced)
  AB_SCAN_RUNNING=0
  if [ -f "$AB_SCAN_PID_FILE" ]; then
    AB_SCAN_PID=$(cat "$AB_SCAN_PID_FILE" 2>/dev/null || true)
    if [ -n "$AB_SCAN_PID" ] && ps -p "$AB_SCAN_PID" > /dev/null 2>&1; then
      AB_SCAN_RUNNING=1
    fi
  fi
  # If no pidfile process, check for any ab_scan referencing the same OUTDIR
  if [ "$AB_SCAN_RUNNING" -eq 0 ]; then
    # Look for ab_scan processes that include our OUTDIR in their args
    OTHER_PIDS=$(pgrep -af "ab_scan.sh" | egrep "${OUTDIR//\//\\/}" | awk '{print $1}' || true)
    if [ -n "$OTHER_PIDS" ]; then
      AB_SCAN_RUNNING=1
      # adopt the first found pid as canonical
      FIRSTPID=$(echo "$OTHER_PIDS" | awk '{print $1}')
      echo "$FIRSTPID" > "$AB_SCAN_PID_FILE" || true
    fi
  fi
  if [ "$AUTO_RESTART" = "1" ] && [ "$AB_SCAN_RUNNING" -eq 0 ] && [ "$COMPLETED" -lt "$TOTAL" ]; then
    log "[$(now)] WARNING: ab_scan is not running and COMPLETED($COMPLETED) < TOTAL($TOTAL) - restarting"
    # Acquire an atomic lock to avoid multiple watchers racing to start a new scan
    # Check restart count allowed per run
    RESTART_COUNT_FILE="$OUTDIR/ab_watch.restart_count"
    RESTART_COUNT=$(cat "$RESTART_COUNT_FILE" 2>/dev/null || echo 0)
    if [ "$RESTART_COUNT" -ge "$RESTART_LIMIT" ]; then
      log "[$(now)] RESTART limit ($RESTART_LIMIT) reached ($RESTART_COUNT) - not restarting"
    else
      if mkdir "$RESTART_LOCK_DIR" 2>/dev/null; then
      echo "$$" > "$RESTART_LOCK_DIR/pid" 2>/dev/null || true
      AB_SCAN_CMD_FILE="$OUTDIR/ab_scan.cmd"
      AB_SCAN_ENV_FILE="$OUTDIR/ab_scan.env"
      # Import saved env if present
      if [ -f "$AB_SCAN_ENV_FILE" ]; then
        set -o allexport
        # shellcheck disable=SC1090
        source "$AB_SCAN_ENV_FILE" || true
        set +o allexport
      fi
      # Kill any lingering ab_scan processes for this OUTDIR just in case (guard)
      for p in $(pgrep -af "ab_scan.sh" | egrep "${OUTDIR//\//\\/}" | awk '{print $1}' || true); do
        if ps -p $p >/dev/null 2>&1; then
          log "[$(now)] Killing stray ab_scan pid $p before restart"
          kill -TERM $p 2>/dev/null || true; sleep 1
          if ps -p $p >/dev/null 2>&1; then kill -KILL $p 2>/dev/null || true; fi
        fi
      done
      # Start back the ab_scan script with RESUME (and be non-verbose)
      if [ -f "$AB_SCAN_CMD_FILE" ]; then
        CMD_LINE=$(grep '^CMD_LINE:' "$AB_SCAN_CMD_FILE" | sed 's/^CMD_LINE: //')
        log "[$(now)] INFO: re-invoking saved command: $CMD_LINE"
        # Use nohup to ensure it's backgrounded and avoids nesting extra shells
        nohup bash -lc "RESUME=1 $CMD_LINE" > "$FULL_SCAN_LOG" 2>&1 &
      else
        log "[$(now)] INFO: ab_scan.cmd not found - using generic restart invocation"
        nohup bash -lc "RESUME=1 FORCE_RUNS=1 CREATE_DIRS=1 KEEP_SUCCESS_LOGS=1 '${AB_SCAN_SCRIPT}' '$OUTDIR' '${LZ4_GPU_CLBIN:-}'" > "$FULL_SCAN_LOG" 2>&1 &
      fi
      NEWPID=$!
      # Write into pid file and ensure ownership
      echo "$NEWPID" > "$AB_SCAN_PID_FILE" || true
      log "[$(now)] Restarted ab_scan PID: $NEWPID"
      # cleanup lock
        rm -rf "$RESTART_LOCK_DIR" 2>/dev/null || true
        # record restart count increment
        echo $((RESTART_COUNT + 1)) > "$RESTART_COUNT_FILE" 2>/dev/null || true
        # Sleep a little to avoid restart thundering
        sleep 2
      else
        # Lock not acquired, another watcher likely starting ab_scan
        log "[$(now)] Restart lock present - skipping restart"
      fi
    fi
  fi

  # If TOTAL>0 and COMPLETED>=TOTAL, we're done
  if [ "$TOTAL" -gt 0 ] && [ "$COMPLETED" -ge "$TOTAL" ]; then
    log "[$(now)] DETECTED COMPLETION: completed=${COMPLETED} total=${TOTAL} - generating analysis"
    # If aggregated csv does not exist, generate it (re-use AWK aggregator found in ab_scan.sh)
    if [ ! -f "$RESULTS_AGG" ]; then
      log "[watch] Generating aggregated results ($RESULTS_AGG) from $RESULTS_CSV"
      # Updated AWK to handle new CSV format with mode column ($7)
      # Header: sample,input_bytes,accel,block_size,local,pinned,mode,run,compressed_bytes,comp_total_ms,...
      awk -F"," -v OUT="$RESULTS_AGG" '
NR==1 { next }
{
  key=$1","$3","$4","$5","$6","$7
  runs[key]++
  if ($20 == "OK") {
    success[key]++
    if ($9 ~ /^[0-9]+$/) { sum_comp[key]+= $9; cnt_comp[key]++ }
    if ($10 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_total[key]+= $10; cnt_total[key]++ }
    if ($11 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_kernel[key]+= $11; cnt_kernel[key]++ }
    if ($12 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_h2d[key]+= $12; cnt_h2d[key]++ }
    if ($13 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_d2h[key]+= $13; cnt_d2h[key]++ }
    if ($14 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_dec_total[key]+= $14; cnt_dec_total[key]++ }
    if ($15 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_dec_kernel[key]+= $15; cnt_dec_kernel[key]++ }
    if ($16 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_dec_h2d[key]+= $16; cnt_dec_h2d[key]++ }
    if ($17 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_dec_d2h[key]+= $17; cnt_dec_d2h[key]++ }
    if ($18 ~ /^[0-9]+(\.[0-9]+)?$/) { sum_ratio[key]+= $18; cnt_ratio[key]++ }
    if (lz4_map[key] == "") lz4_map[key] = $19; else if (lz4_map[key] != $19) lz4_map[key] = "MIXED"
  }
}
END {
  print "sample,accel,block_size,local,pinned,mode,runs,successes,avg_compressed_bytes,avg_comp_total_ms,avg_comp_kernel_ms,avg_comp_h2d_ms,avg_comp_d2h_ms,avg_decomp_total_ms,avg_decomp_kernel_ms,avg_decomp_h2d_ms,avg_decomp_d2h_ms,avg_compression_ratio,lz4_valid_frame,success_rate" > OUT
  for (k in runs) {
    split(k, parts, ",")
    sample = parts[1]; accel = parts[2]; bs = parts[3]; local = parts[4]; pinned = parts[5]; mode = parts[6]
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
    lv = (lz4_map[k] ? lz4_map[k] : "NA")
    success_rate = (r > 0 ? s / r : 0)
    printf "%s,%s,%s,%s,%s,%s,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.3f\n", sample, accel, bs, local, pinned, mode, r, s, avg_comp, avg_total, avg_kernel, avg_h2d, avg_d2h, avg_dec_total, avg_dec_kernel, avg_dec_h2d, avg_dec_d2h, avg_ratio, lv, success_rate >> OUT
  }
}' "$RESULTS_CSV"
    fi

    # Run analysis script (if available)
    if [ -f "$ANALYSIS_SCRIPT" ]; then
      log "[watch] Running analysis with $ANALYSIS_SCRIPT --results $RESULTS_AGG"
      python3 "$ANALYSIS_SCRIPT" --results "$RESULTS_AGG" --outdir "$OUTDIR" >> "$MONITOR_LOG" 2>&1 || log "[watch] analysis script failed"
    else
      log "[watch] Analysis script not found: $ANALYSIS_SCRIPT"
    fi

    log "[watch] Monitor finished - exiting"
    exit 0
  fi

  sleep "$INTERVAL"
done
