#!/usr/bin/env bash
# usage: novelty/pipeline_after_train.sh <run_name|base_prajiv> <gpu>
#   run_name = folder in novelty/runs (uses best.json to pick the checkpoint)
#   base_prajiv = the team's existing P-Rajiv unet_clip/epoch_2 LoRA (current setup), regenerated with the same sampler
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=$1; GPU=$2
PY=.venv311/bin/python
if [ "$RUN" = "base_prajiv" ]; then
  LORA=ie643_code/ie643/cxr_diffusion_checkpoints/prajiv/unet_clip/epoch_2
else
  STEP=$($PY -c "import json;print(json.load(open('novelty/runs/$RUN/best.json'))['best_step'])")
  LORA=novelty/runs/$RUN/step_$STEP
fi
echo "[$RUN] using LoRA $LORA"
CUDA_VISIBLE_DEVICES=$GPU $PY novelty/generate.py --lora_dir "$LORA" --name gen_$RUN --bs 8
CUDA_VISIBLE_DEVICES=$GPU $PY novelty/evaluate_synth.py --root data/CheXpert-v1.0-small/gen_$RUN --name $RUN
for s in 42 43 44; do
  CUDA_VISIBLE_DEVICES=$GPU $PY novelty/downstream_train.py --train_csv novelty/csv/classifier_train_mixed_gen_$RUN.csv \
      --name $RUN --seed $s > novelty/logs/ds_${RUN}_$s.log 2>&1 &
done
wait
echo "[$RUN] PIPELINE DONE"
