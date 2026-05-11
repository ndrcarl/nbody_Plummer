#!/bin/bash
# ============================================================
#  setup.sh — checks the environment before running anything
#
#  Verifies:
#    - treecode binary exists and is executable
#    - all required python scripts are present
#    - python3 and required packages are available
#
#  Usage:
#    bash setup.sh
# ============================================================

BASE_DIR=$(pwd)
ALL_OK=true

echo "======================================"
echo "  PLUMMER SPHERE — ENVIRONMENT CHECK"
echo "======================================"

# ---- treecode ----
echo ""
echo "Checking treecode..."
if [ -x "$BASE_DIR/treecode" ]; then
    echo "  [OK] treecode found and executable"
else
    echo "  [FAIL] treecode not found or not executable"
    echo "         run: make && chmod +x treecode"
    ALL_OK=false
fi

# ---- python scripts ----
echo ""
echo "Checking python scripts..."
SCRIPTS=(
    sampling_plummer.py
    raggio.py
    virial_stability.py
    plot_density.py
    jeans.py
    mass_vcirc.py
    vel_dist.py
    orbits.py
    plot_snapshots.py
    plot_analysis.py
    summary_runs.py
)

for s in "${SCRIPTS[@]}"; do
    if [ -f "$BASE_DIR/$s" ]; then
        echo "  [OK] $s"
    else
        echo "  [FAIL] $s not found"
        ALL_OK=false
    fi
done

# ---- python packages ----
echo ""
echo "Checking python packages..."
python3 -c "import numpy" 2>/dev/null && echo "  [OK] numpy" || { echo "  [FAIL] numpy"; ALL_OK=false; }
python3 -c "import matplotlib" 2>/dev/null && echo "  [OK] matplotlib" || { echo "  [FAIL] matplotlib"; ALL_OK=false; }

# ---- log file ----
echo ""
echo "Initialising master_run.log..."
> "$BASE_DIR/master_run.log"
echo "  [OK] master_run.log cleared"

# ---- result ----
echo ""
echo "======================================"
if [ "$ALL_OK" = true ]; then
    echo "  All checks passed. Ready to run."
    echo "  bash run.sh"
else
    echo "  Some checks FAILED. Fix before running."
fi
echo "======================================"
