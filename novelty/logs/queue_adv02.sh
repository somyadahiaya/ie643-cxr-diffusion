#!/usr/bin/env bash
# start lambda_adv=0.2 sensitivity run on GPU 1 once all control/ce_only DenseNet seeds are finished
cd /home/surajracha/pedal_for_healthcare/cs6101/Project_IE643
done_count() { grep -l 'SELECTED' novelty/logs/ds_control_4?.log novelty/logs/ds_ce_only_4?.log 2>/dev/null | wc -l; }
until [ "$(done_count)" -ge 6 ]; do sleep 60; done
CUDA_VISIBLE_DEVICES=1 .venv311/bin/python novelty/train_adv_ce.py --out novelty/runs/adv_only_l02 --lam_cls 0 --lam_adv 0.2 > novelty/logs/train_adv_only_l02.log 2>&1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True novelty/pipeline_after_train.sh adv_only_l02 1
