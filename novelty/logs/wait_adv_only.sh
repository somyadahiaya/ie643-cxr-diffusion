#!/usr/bin/env bash
# when adv_only training finishes, run its pipeline on GPU 0
cd /home/surajracha/pedal_for_healthcare/cs6101/Project_IE643
until [ -f novelty/runs/adv_only/best.json ]; do sleep 30; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True novelty/pipeline_after_train.sh adv_only 0
