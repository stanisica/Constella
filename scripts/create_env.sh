#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$ROOT_DIR/.constella-venv"
PYTHON_BIN="python3.12"

if [[ -d "$VENV_DIR" ]]; then
  echo "Environment already exists at $VENV_DIR"
  echo "Remove it first with scripts/remove_env.sh"
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Required interpreter '$PYTHON_BIN' not found on PATH"
  echo "Install or provision Python 3.12, then rerun this script"
  exit 1
fi

echo "Creating Constella environment with $PYTHON_BIN"
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"

echo "Environment created at $VENV_DIR"
