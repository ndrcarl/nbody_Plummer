#!/bin/bash
# ============================================================
#  finalize.sh — concatenates per-run logs into master_run.log
#                then runs the combined summary
#
#  Usage:
#    ./finalize.sh
# ============================================================

BASE_DIR=$(pwd)
MASTER_LOG="$BASE_DIR/master_run.log"

echo "Concatenating per-run logs..."
> "$MASTER_LOG"
for log in $(ls "$BASE_DIR"/run_*/run_*.log 2>/dev/null | sort); do
    echo "" >> "$MASTER_LOG"
    cat "$log" >> "$MASTER_LOG"
done
echo "Written to master_run.log"

echo ""
echo "=== Combined summary at $(date) ===" | tee -a "$MASTER_LOG"
python3 "$BASE_DIR/summary_runs.py" --combined >> "$MASTER_LOG" 2>&1
echo "Pipeline complete." | tee -a "$MASTER_LOG"
