#!/usr/bin/env bash
# Fix #1 (CheXpert-style synthetic files + synthetic No Finding) and Fix #2 (guidance / LoRA-scale sweep)
# usage: novelty/chain_fix12.sh A   -> "#1 only"  (existing CE+Adv images, cfg 7.5)              GPU 1
#        novelty/chain_fix12.sh B   -> "#2 only" and "#1+#2" (selected cfg / LoRA scale)          GPU 0
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv311/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LORA=novelty/runs/ce_adv/step_1000

downstream() {  # $1=name $2=csv $3=gpu
  for s in 42 43 44; do
    CUDA_VISIBLE_DEVICES=$3 $PY novelty/downstream_train.py --train_csv "$2" --name "$1" --seed $s \
        > novelty/logs/ds_$1_$s.log 2>&1 &
  done
  wait
}

if [ "$1" = "A" ]; then
  until grep -q "wrote 500 images" novelty/logs/gen_ce_adv_nofinding.log; do sleep 30; done
  $PY novelty/chexpert_style.py --sources gen_ce_adv,gen_ce_adv_nofinding --name gen_ce_adv_jpg
  CUDA_VISIBLE_DEVICES=1 $PY novelty/evaluate_synth.py --root data/CheXpert-v1.0-small/gen_ce_adv_jpg --name ce_adv_jpg
  downstream ce_adv_jpg novelty/csv/classifier_train_mixed_gen_ce_adv_jpg.csv 1
  echo "[A] CHAIN DONE"
fi

if [ "$1" = "B" ]; then
  until [ -f novelty/results/sweep_ce_adv_selected.json ]; do sleep 30; done
  CFG=$($PY -c "import json;print(json.load(open('novelty/results/sweep_ce_adv_selected.json'))['cfg'])")
  SC=$($PY -c "import json;print(json.load(open('novelty/results/sweep_ce_adv_selected.json'))['lora_scale'])")
  echo "[B] selected cfg=$CFG lora_scale=$SC"
  if [ "$CFG" = "7.5" ] && [ "$SC" = "1.0" ]; then
    echo "[B] sweep kept the original setting -> #2 is a no-op; only the #1 result applies"; exit 0
  fi
  CUDA_VISIBLE_DEVICES=0 $PY novelty/generate.py --lora_dir $LORA --name gen_ce_adv_swept --cfg $CFG --lora_scale $SC --bs 8
  CUDA_VISIBLE_DEVICES=0 $PY novelty/evaluate_synth.py --root data/CheXpert-v1.0-small/gen_ce_adv_swept --name ce_adv_swept
  CUDA_VISIBLE_DEVICES=0 $PY novelty/generate.py --lora_dir $LORA --name gen_ce_adv_swept_nofinding --cfg $CFG \
      --lora_scale $SC --targets "No Finding:500" --bs 8
  $PY novelty/chexpert_style.py --sources gen_ce_adv_swept,gen_ce_adv_swept_nofinding --name gen_ce_adv_swept_jpg
  # "#2 only" and "#1+#2" downstream, 3 seeds each, in parallel
  downstream ce_adv_swept novelty/csv/classifier_train_mixed_gen_ce_adv_swept.csv 0 &
  downstream ce_adv_swept_jpg novelty/csv/classifier_train_mixed_gen_ce_adv_swept_jpg.csv 0 &
  wait
  echo "[B] CHAIN DONE"
fi
