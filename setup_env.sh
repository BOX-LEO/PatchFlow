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
# CUDA wheel tag for torch. The system GPU driver supports up to CUDA 12.2,
# so cu121 wheels are used by default. Set CUDA_TAG=cpu for a CPU-only build,
# or e.g. CUDA_TAG=cu118 to match an older driver.
CUDA_TAG="${CUDA_TAG:-cu121}"

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

echo ">>> Installing torch + torchvision (${CUDA_TAG}) to match the GPU driver..."
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo ">>> Installing remaining dependencies from requirements.txt..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo ">>> Verifying CUDA availability..."
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo ""
echo ">>> Done. Activate the environment with:"
echo "    conda activate ${ENV_NAME}"
