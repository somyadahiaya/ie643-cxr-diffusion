#!/usr/bin/env bash
# recipe (real-count class weights + synthetic weight 0.5 + last 3 epochs real-only) on Margin+Adv and "#2 only"
cd "$(dirname "$0")/.."
PY=.venv311/bin/python
RECIPE="--class_weight_from real --syn_weight 0.5 --real_ft_epochs 3"
until grep -q 'PIPELINE DONE' novelty/logs/pipe_margin_adv.log && grep -q 'CHAIN DONE' novelty/logs/chain_swept.log; do sleep 60; done
for s in 42 43 44; do
  CUDA_VISIBLE_DEVICES=1 $PY novelty/downstream_train.py --train_csv novelty/csv/classifier_train_mixed_gen_margin_adv.csv \
      --name margin_adv_recipe --seed $s $RECIPE > novelty/logs/ds_margin_adv_recipe_$s.log 2>&1 &
  CUDA_VISIBLE_DEVICES=0 $PY novelty/downstream_train.py --train_csv novelty/csv/classifier_train_mixed_gen_ce_adv_swept.csv \
      --name ce_adv_swept_recipe --seed $s $RECIPE > novelty/logs/ds_ce_adv_swept_recipe_$s.log 2>&1 &
done
wait
echo "[recipe2] CHAIN DONE"
