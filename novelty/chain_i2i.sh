#!/usr/bin/env bash
# DA-Fusion-style img2img augmentation with the CE+Adv LoRA at the sweep-selected guidance / LoRA scale (GPU 0)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv311/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ -f novelty/results/sweep_ce_adv_selected.json ]; do sleep 30; done
CFG=$($PY -c "import json;print(json.load(open('novelty/results/sweep_ce_adv_selected.json'))['cfg'])")
SC=$($PY -c "import json;print(json.load(open('novelty/results/sweep_ce_adv_selected.json'))['lora_scale'])")
echo "[i2i] cfg=$CFG lora_scale=$SC"
CUDA_VISIBLE_DEVICES=0 $PY novelty/img2img_augment.py --cfg $CFG --lora_scale $SC
CUDA_VISIBLE_DEVICES=0 $PY novelty/evaluate_synth.py --root data/CheXpert-v1.0-small/gen_ce_adv_i2i --name ce_adv_i2i
for s in 42 43 44; do
  CUDA_VISIBLE_DEVICES=0 $PY novelty/downstream_train.py --train_csv novelty/csv/classifier_train_mixed_gen_ce_adv_i2i.csv \
      --name ce_adv_i2i --seed $s > novelty/logs/ds_ce_adv_i2i_$s.log 2>&1 &
done
wait
echo "[i2i] CHAIN DONE"
