#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$ROOT_DIR/.constella-venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Environment not found at $VENV_DIR"
  exit 1
fi

rm -rf "$VENV_DIR"
echo "Removed $VENV_DIR"
