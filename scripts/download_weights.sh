#!/usr/bin/env bash
# Fetch the large files that are not in git (see CHECKPOINTS.md) from the Hugging Face model repo.
#   bash scripts/download_weights.sh <hf_user>/<hf_repo>                 # base model weights needed by all code + demo (~2 GB)
#   bash scripts/download_weights.sh <hf_user>/<hf_repo> --all           # + PRISM, phase-1 accelerate checkpoints, all LoRA steps (~7 GB)
#   bash scripts/download_weights.sh <hf_user>/<hf_repo> --synthetic     # + the 15 generated image sets (~7 GB),
#                                                                         #   unpacked into data/CheXpert-v1.0-small/
# Private repo: run `.venv311/bin/hf auth login` first.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${1:?usage: download_weights.sh <hf_user>/<hf_repo> [--all] [--synthetic]}; shift || true
HF=${HF:-.venv311/bin/hf}; [ -x "$HF" ] || HF=hf
ALL=0; SYN=0
for a in "$@"; do case $a in --all) ALL=1;; --synthetic) SYN=1;; esac; done

# always: base SD-1.5 CXR weights (UNet/VAE/text encoder) + checksums
"$HF" download "$REPO" --local-dir . --include "novelty/prajiv_base/*" --include "SHA256SUMS"
if [ $ALL = 1 ]; then
  "$HF" download "$REPO" --local-dir . --include "prism/*" --include "cxr_diffusion_checkpoints/*" \
      --include "ie643_code/ie643/cxr_diffusion_checkpoints/*" --include "novelty/runs/*"
fi
if [ $SYN = 1 ]; then
  "$HF" download "$REPO" --local-dir . --include "synthetic_sets/*"
  mkdir -p data/CheXpert-v1.0-small
  for t in synthetic_sets/*.tar; do echo "unpacking $t"; tar -xf "$t" -C data/CheXpert-v1.0-small; done
fi
echo "verifying checksums of everything downloaded ..."
sha256sum -c SHA256SUMS --ignore-missing --quiet && echo "checksums OK"
