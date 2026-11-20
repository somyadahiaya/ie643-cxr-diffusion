#!/usr/bin/env bash
# Downstream "synthetic-data recipe" on the two best synthetic sets:
#   R1 = class weights from real counts + synthetic loss weight 0.5 + last 3 epochs real-only
#   R2 = R1 + filtering (reward-head margin >= real median, anatomy pass)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv311/bin/python
RECIPE="--class_weight_from real --syn_weight 0.5 --real_ft_epochs 3"
CUDA_VISIBLE_DEVICES=1 $PY novelty/filter_synth.py --mixed_csv novelty/csv/classifier_train_mixed_gen_ce_adv.csv --name gen_ce_adv
CUDA_VISIBLE_DEVICES=1 $PY novelty/filter_synth.py --mixed_csv novelty/csv/classifier_train_mixed_gen_ce_adv_i2i.csv --name gen_ce_adv_i2i
run() {  # name csv gpu
  for s in 42 43 44; do
    CUDA_VISIBLE_DEVICES=$3 $PY novelty/downstream_train.py --train_csv "$2" --name "$1" --seed $s $RECIPE \
        > novelty/logs/ds_$1_$s.log 2>&1 &
  done
}
run ce_adv_recipe      novelty/csv/classifier_train_mixed_gen_ce_adv.csv          1
run ce_adv_i2i_recipe  novelty/csv/classifier_train_mixed_gen_ce_adv_i2i.csv      0
wait
run ce_adv_recipe_filt     novelty/csv/classifier_train_mixed_gen_ce_adv_filt.csv     1
run ce_adv_i2i_recipe_filt novelty/csv/classifier_train_mixed_gen_ce_adv_i2i_filt.csv 0
wait
echo "[recipe] CHAIN DONE"
