$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_DIR = "$SCRIPT_DIR\venv"
if (-Not (Test-Path -Path $VENV_DIR)) {
    python -m venv $VENV_DIR
}
$VENV_ACTIVATION = "$VENV_DIR\Scripts\Activate.ps1"
& $VENV_ACTIVATION

$REQUIREMENTS_FILE = "$SCRIPT_DIR\requirements.txt"
pip install -q -q -r $REQUIREMENTS_FILE

$PYTHON_SCRIPT = "$SCRIPT_DIR\main.py"
python $PYTHON_SCRIPT
