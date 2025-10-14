#!/bin/bash
#source venv/bin/activate

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Kronos Project Setup Script ---"

# Force the use of Python 3.11 for compatibility
PYTHON_EXEC="python3.11"
if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo "Error: python3.11 command not found. Please install Python 3.11 to continue."
    exit 1
fi

echo "Found Python interpreter: $PYTHON_EXEC"

# Check the version of the selected python
PYTHON_VERSION=$($PYTHON_EXEC -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" > "3.12" ]]; then
    echo "Warning: Python $PYTHON_VERSION is detected. This version may not be fully compatible with all dependencies (like PyTorch)." >&2
    echo "It is recommended to use Python 3.10, 3.11, or 3.12 for best results." >&2
fi


echo "[1/4] Creating Python virtual environment in './venv'..."
$PYTHON_EXEC -m venv venv

# Activate the virtual environment
source venv/bin/activate
echo "[2/4] Virtual environment activated."

echo "[3/4] Installing dependencies..."
# Upgrade pip and build tools to the latest versions
pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install dependencies from the root requirements.txt
echo "Installing main dependencies from requirements.txt..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install dependencies from the webui/requirements.txt
echo "Installing Web UI dependencies from webui/requirements.txt..."
pip install -r webui/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "All dependencies installed successfully."

echo "[4/4] Setup complete!"
echo ""
echo "--- Next Steps ---"
echo "The virtual environment 'venv' is now active."
echo ""
echo "You can run the project using one of the following commands:"
echo ""
echo "1. To start the Web UI (recommended):"
echo "   cd webui"
echo "   python app.py"
echo ""
echo "2. To run the command-line prediction example:"
echo "   python examples/prediction_example.py"
echo ""
echo "To deactivate the virtual environment later, simply run the command: deactivate"
