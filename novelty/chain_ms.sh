#!/usr/bin/env bash
# Margin+Adv with the multi-scale Projected-GAN discriminator: train -> generate/eval/downstream -> recipe downstream
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv311/bin/python
GPU=${1:-1}
RUN=margin_adv_ms
CUDA_VISIBLE_DEVICES=$GPU $PY novelty/train_adv_ce.py --out novelty/runs/$RUN --cls_loss margin --d_multiscale \
    > novelty/logs/train_$RUN.log 2>&1
bash novelty/pipeline_after_train.sh $RUN $GPU > novelty/logs/pipe_$RUN.log 2>&1
RECIPE="--class_weight_from real --syn_weight 0.5 --real_ft_epochs 3"
for s in 42 43 44; do
  CUDA_VISIBLE_DEVICES=$GPU $PY novelty/downstream_train.py --train_csv novelty/csv/classifier_train_mixed_gen_$RUN.csv \
      --name ${RUN}_recipe --seed $s $RECIPE > novelty/logs/ds_${RUN}_recipe_$s.log 2>&1 &
done
wait
echo "[$RUN] CHAIN DONE"
