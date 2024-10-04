#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
pip install -q -q -r "$REQUIREMENTS_FILE"

PYTHON_SCRIPT="$SCRIPT_DIR/main.py"
python "$PYTHON_SCRIPT"
