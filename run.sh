#!/bin/bash
# ============================================================
#  run.sh — runs all NUM_RUNS realisations sequentially
#           then calls finalize.sh
#
#  For parallel runs, call run_single.sh manually in separate
#  terminals, then call finalize.sh when all are done.
#
#  Usage:
#    ./run.sh
# ============================================================

NUM_RUNS=5
BASE_DIR=$(pwd)

if [ ! -x "$BASE_DIR/treecode" ]; then
    echo "Error: treecode not found or not executable in $BASE_DIR"
    exit 1
fi

echo "======================================"
echo "  PLUMMER SPHERE STABILITY TEST"
echo "  eps   = 0.012  (d_mean/10, fixed)"
echo "  tstop = 20.0   (~13 t_dyn)"
echo "  runs  = $NUM_RUNS"
echo "======================================"

for i in $(seq 1 $NUM_RUNS); do
    bash "$BASE_DIR/run_single.sh" $i
done

bash "$BASE_DIR/finalize.sh"
