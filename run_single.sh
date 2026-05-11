#!/bin/bash
# ============================================================
#  run_single.sh — runs ONE realisation of the Plummer sphere
#
#  Each run writes to its own run_NNN/run_NNN.log so multiple
#  runs can be launched in parallel without log corruption.
#
#  Usage:
#    ./run_single.sh <run_number>
# ============================================================

if [ -z "$1" ]; then
    echo "Usage: ./run_single.sh <run_number>"
    exit 1
fi

BASE_DIR=$(pwd)
RUN_NUM=$(printf "%03d" $1)
RUN_DIR="$BASE_DIR/run_${RUN_NUM}"
LOG_FILE="$RUN_DIR/run_${RUN_NUM}.log"

EPS=0.012
TSTOP=20.0
DTOUT=1/10
DTIME=1/2048
THETA=0.50

N_PARTICLES=10000
LINES_PER_SNAP=$((3 + 4 * N_PARTICLES))
N_SNAPS=200
TOTAL_LINES=$((N_SNAPS * LINES_PER_SNAP))

progress_bar() {
    local current=$1
    local total=$2
    local width=40
    local pct=$(( current * 100 / total ))
    local filled=$(( current * width / total ))
    local empty=$(( width - filled ))
    local bar=""
    for ((i=0; i<filled; i++)); do bar="${bar}#"; done
    for ((i=0; i<empty;  i++)); do bar="${bar}-"; done
    printf "\r  run %s  [%s] %3d%%  snapshot %d / %d" "$RUN_NUM" "$bar" "$pct" "$current" "$total"
}

if [ ! -x "$BASE_DIR/treecode" ]; then
    echo "Error: treecode not found or not executable in $BASE_DIR"
    exit 1
fi

mkdir -p "$RUN_DIR"

echo "=== run $1 at $(date) ===" | tee "$LOG_FILE"

cd "$RUN_DIR" || exit 1

echo "  Step 1: sampling_plummer.py" | tee -a "$LOG_FILE"
python3 "$BASE_DIR/sampling_plummer.py" >> "$LOG_FILE" 2>&1

echo "  Step 2: treecode" | tee -a "$LOG_FILE"

"$BASE_DIR/treecode"    \
    in=plummer.txt      \
    out=plummer.out     \
    dtime=$DTIME        \
    eps=$EPS            \
    theta=$THETA        \
    usequad=false       \
    tstop=$TSTOP        \
    dtout=$DTOUT        \
    options=out-phi     \
    >> "$LOG_FILE" 2>&1 &

TREECODE_PID=$!

echo ""
while kill -0 $TREECODE_PID 2>/dev/null; do
    if [ -f "plummer.out" ]; then
        current_lines=$(wc -l < "plummer.out")
        current_snaps=$(( current_lines / LINES_PER_SNAP ))
        [ $current_snaps -gt $N_SNAPS ] && current_snaps=$N_SNAPS
        progress_bar $current_snaps $N_SNAPS
    else
        printf "\r  run %s  waiting for first snapshot..." "$RUN_NUM"
    fi
    sleep 2
done

progress_bar $N_SNAPS $N_SNAPS
echo ""
echo "  treecode done." | tee -a "$LOG_FILE"

echo "  Step 3: analysis" | tee -a "$LOG_FILE"
python3 "$BASE_DIR/raggio.py"                     >> "$LOG_FILE" 2>&1
python3 "$BASE_DIR/virial_stability.py"           >> "$LOG_FILE" 2>&1
python3 "$BASE_DIR/plot_density.py"   plummer.out >> "$LOG_FILE" 2>&1
python3 "$BASE_DIR/jeans.py"          plummer.out >> "$LOG_FILE" 2>&1
python3 "$BASE_DIR/mass_vcirc.py"     plummer.out >> "$LOG_FILE" 2>&1
python3 "$BASE_DIR/vel_dist.py"       plummer.out >> "$LOG_FILE" 2>&1
python3 "$BASE_DIR/orbits.py"         plummer.out >> "$LOG_FILE" 2>&1
python3 "$BASE_DIR/plot_snapshots.py" plummer.out >> "$LOG_FILE" 2>&1
python3 "$BASE_DIR/plot_analysis.py"              >> "$LOG_FILE" 2>&1

echo "  Step 4: summary_runs.py" | tee -a "$LOG_FILE"
python3 "$BASE_DIR/summary_runs.py" "$RUN_DIR" >> "$LOG_FILE" 2>&1

echo "=== run $1 completed at $(date) ===" | tee -a "$LOG_FILE"

cd "$BASE_DIR" || exit 1
