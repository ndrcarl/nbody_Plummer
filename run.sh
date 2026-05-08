#!/bin/bash
# ============================================================
#  run.sh — Equilibrium Plummer Sphere Pipeline
# ============================================================

NUM_RUNS=1
BASE_DIR=$(pwd)
EPS=0.012   # The optimal softening calculated for N=10k
TSTOP=10.0  # Run for 10 crossing times to prove stability

if [ ! -x "$BASE_DIR/treecode" ]; then
    echo "Error: treecode not found or not executable"
    exit 1
fi

LOG_FILE="$BASE_DIR/master_run.log"
> "$LOG_FILE"

echo "########## PLUMMER SPHERE STABILITY TEST ##########" | tee -a "$LOG_FILE"
echo "Softening: eps = $EPS" | tee -a "$LOG_FILE"

for i in $(seq 1 $NUM_RUNS); do
    RUN_NUM=$(printf "%03d" $i)
    RUN_DIR="$BASE_DIR/run_${RUN_NUM}"
    mkdir -p "$RUN_DIR"
    cd "$RUN_DIR" || exit 1

    echo "=== run $i at $(date) ===" | tee -a "$LOG_FILE"

    echo "  Generating ICs..." | tee -a "$LOG_FILE"
    python3 "$BASE_DIR/sampling_plummer.py" >> "$LOG_FILE" 2>&1

    echo "  Running treecode..." | tee -a "$LOG_FILE"
    "$BASE_DIR/treecode" \
        in=plummer.txt  \
        out=plummer.out \
        dtime=1/2048    \
        eps=$EPS        \
        theta=0.75      \
        usequad=false   \
        tstop=$TSTOP    \
        dtout=1/10      \
        options=out-phi \
        >> "$LOG_FILE" 2>&1

    echo "  Running Analysis..." | tee -a "$LOG_FILE"
    python3 "$BASE_DIR/raggio.py" >> "$LOG_FILE" 2>&1
    python3 "$BASE_DIR/virial_stability.py" >> "$LOG_FILE" 2>&1
    python3 "$BASE_DIR/plot_analysis.py" >> "$LOG_FILE" 2>&1
    python3 "$BASE_DIR/plot_snapshots.py" plummer.out >> "$LOG_FILE" 2>&1

    cd "$BASE_DIR" || exit 1
done

echo "Running final summary..." | tee -a "$LOG_FILE"
python3 "$BASE_DIR/summary_runs.py" >> "$LOG_FILE" 2>&1
echo "Pipeline complete!" | tee -a "$LOG_FILE"
