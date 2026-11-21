#!/usr/bin/env bash
# One-time upload of the staged large files (../hf_release, built next to this repo) to the (public) HF model repo.
#   .venv311/bin/hf auth login            # once, with a write token from https://huggingface.co/settings/tokens
#   bash scripts/upload_weights_to_hf.sh <hf_user>/ie643-cxr-diffusion-weights
# Re-run the same command to resume if the upload is interrupted.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${1:?usage: upload_weights_to_hf.sh <hf_user>/<hf_repo>}
STAGE=${STAGE:-../hf_release}
HF=${HF:-../.venv311/bin/hf}; [ -x "$HF" ] || HF=.venv311/bin/hf
"$HF" repos create "$REPO" --exist-ok
"$HF" upload "$REPO" "$STAGE" . --commit-message "IE643 CXR diffusion: weights, all LoRA steps, synthetic sets"
echo "done: https://huggingface.co/$REPO"
