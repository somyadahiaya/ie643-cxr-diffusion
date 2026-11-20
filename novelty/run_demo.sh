#!/usr/bin/env bash
# Launch the generation + classification demo (Gradio) on one GPU.
#   bash novelty/run_demo.sh [GPU] [PORT]          -> http://127.0.0.1:PORT
# From your laptop:  ssh -L 7860:127.0.0.1:7860 <user>@<server>   then open http://localhost:7860
cd "$(dirname "$0")/.."
GPU=${1:-0}; PORT=${2:-7860}
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 .venv311/bin/python novelty/demo_app.py --port "$PORT" "${@:3}"
