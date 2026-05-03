#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$ROOT_DIR/.constella-venv"
PYTHON_BIN="$VENV_DIR/bin/python"
RESULTS_DIR="${1:-$ROOT_DIR/artifact-output/paper-results}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Expected environment not found at $VENV_DIR"
  echo "Create it first with scripts/create_env.sh"
  exit 1
fi

"$PYTHON_BIN" "$ROOT_DIR/constella-evaluation/validate_results.py" "$RESULTS_DIR"
