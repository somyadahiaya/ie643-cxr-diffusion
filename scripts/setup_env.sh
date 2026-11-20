#!/usr/bin/env bash
# Create the .venv311 environment that every script in this repo expects (Python 3.11 + CUDA 12.6 torch).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
uv venv --python 3.11 .venv311
uv pip install --python .venv311/bin/python -r requirements-lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
.venv311/bin/python -c "import torch, diffusers, peft, torchxrayvision; print('ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
