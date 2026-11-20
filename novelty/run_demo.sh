#!/usr/bin/env bash
# Launch the demo (demo/app.py) from a full clone, using the local files instead of downloading from Hugging Face.
#   bash novelty/run_demo.sh [GPU] [PORT]          -> http://127.0.0.1:PORT
# Needs the base weights: bash scripts/download_weights.sh. From your laptop: ssh -L 7860:127.0.0.1:7860 <server>
cd "$(dirname "$0")/.."
GPU=${1:-0}; PORT=${2:-7860}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 .venv311/bin/python demo/app.py --local_root . --data_dir data --port "$PORT" "${@:3}"
