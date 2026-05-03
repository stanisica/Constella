#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$ROOT_DIR/.constella-venv"
PYTHON_BIN="$VENV_DIR/bin/python"
CACHE_DIR="${TMPDIR:-/tmp}/constella-artifact-cache"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Expected environment not found at $VENV_DIR"
  echo "Create it first with scripts/create_env.sh"
  exit 1
fi

mkdir -p "$CACHE_DIR/matplotlib" "$CACHE_DIR/fontconfig"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export XDG_CACHE_HOME="$CACHE_DIR"

echo "Checking Python imports"
"$PYTHON_BIN" -c "import mip, numpy, matplotlib, torch, torchvision, torchinfo; print('OK')"

echo "Checking model-layer regeneration"
"$PYTHON_BIN" "$ROOT_DIR/constella-evaluation/generate_model_layers.py" --models all --check

echo "Environment check completed successfully"
