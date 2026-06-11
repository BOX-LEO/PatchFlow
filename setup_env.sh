#!/usr/bin/env bash
#
# Create a fresh conda environment for PatchFlow and install all dependencies.
#
# Usage:
#   bash setup_env.sh
#
# Override defaults with environment variables:
#   ENV_NAME=myenv PY_VERSION=3.11 bash setup_env.sh
#
set -e

ENV_NAME="${ENV_NAME:-patchflow}"
PY_VERSION="${PY_VERSION:-3.10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure conda is available and enable `conda activate` in this non-interactive shell.
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found on PATH. Install Miniconda/Anaconda first." >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

echo ">>> Creating conda env '${ENV_NAME}' with Python ${PY_VERSION}..."
conda create -y -n "${ENV_NAME}" python="${PY_VERSION}"

echo ">>> Activating '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

echo ">>> Upgrading pip..."
pip install --upgrade pip

echo ">>> Installing dependencies from requirements.txt..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo ">>> Done. Activate the environment with:"
echo "    conda activate ${ENV_NAME}"
