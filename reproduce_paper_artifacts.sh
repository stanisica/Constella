#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/artifact-output/paper-results}"
CACHE_DIR="${TMPDIR:-/tmp}/constella-artifact-cache"
MODEL_LAYERS_DIR="$OUTPUT_DIR/model-layers"

mkdir -p "$OUTPUT_DIR" "$MODEL_LAYERS_DIR" "$CACHE_DIR/matplotlib" "$CACHE_DIR/fontconfig"

export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export XDG_CACHE_HOME="$CACHE_DIR"
export FONTCONFIG_PATH="${FONTCONFIG_PATH:-}"
export CONSTELLA_MODEL_LAYERS_DIR="$MODEL_LAYERS_DIR"
export CONSTELLA_RESULTS_DIR="$OUTPUT_DIR"

echo "Reproducing Constella paper artifacts"
echo "Repository root: $ROOT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

echo "[1/5] Generating model-layer profiles"
python3 "$ROOT_DIR/constella-evaluation/generate_model_layers.py" \
  --models all \
  --output-dir "$MODEL_LAYERS_DIR"

echo "[2/5] Running evaluation"
python3 "$ROOT_DIR/constella-evaluation/evaluate_constella.py"

echo "[3/5] Running timing benchmark (50 iterations)"
python3 "$ROOT_DIR/constella-evaluation/benchmark_timing.py" --iterations 50

echo "[4/5] Rendering paper figures"
(
  cd "$ROOT_DIR/constella-evaluation"
  python3 plot_constella.py
)

echo "[5/5] Exporting artifact bundle"
python3 "$ROOT_DIR/constella-evaluation/export_artifact_bundle.py" \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "Artifact bundle generated in: $OUTPUT_DIR"
