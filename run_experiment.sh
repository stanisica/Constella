#!/usr/bin/env bash
set -euo pipefail

SCENARIO="scenarios/scenario_constella.json"

echo "============================================"
echo "  Step 1/5: Running OCRI solver"
echo "============================================"
bash run_simultaneous.sh

echo ""
echo "============================================"
echo "  Step 2/5: Evaluating OCRI baselines"
echo "============================================"
python3 evaluate_baselines.py "$SCENARIO"

echo ""
echo "============================================"
echo "  Step 3/5: Evaluating LIA (v1)"
echo "============================================"
python3 evaluate_lia.py "$SCENARIO"

echo ""
echo "============================================"
echo "  Step 4/5: Evaluating LIA (v2)"
echo "============================================"
python3 evaluate_lia_v2.py "$SCENARIO"

echo ""
echo "============================================"
echo "  Step 5/5: Plotting results"
echo "============================================"
python3 plot_results.py
python3 plot_lia.py
python3 plot_lia_v2.py

echo ""
echo "============================================"
echo "  All done!"
echo "============================================"
