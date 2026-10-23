#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 22.04/24.04 x86_64 VM: compiler, venv, Python packages.
#
#   bench/vm_setup.sh            # creates .venv/ in this directory
#   . .venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v g++ >/dev/null || ! python3 -m venv --help >/dev/null 2>&1; then
    sudo apt-get update -q
    sudo apt-get install -y -q build-essential python3-venv numactl
fi
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q "numpy>=1.24" "scipy>=1.10" "pytest>=7"
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
python -c "import numpy, scipy, torch; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__)"
g++ --version | head -1
