#!/usr/bin/env bash
set -euo pipefail

echo "Running evaluation..."
python3 constella-evaluation/evaluate_constella.py

echo ""
echo "Running benchmark..."
python3 constella-evaluation/benchmark_timing.py

echo ""
echo "Plotting results..."
cd constella-evaluation && python3 plot_constella.py

echo "Done."
