#!/usr/bin/env bash
cd /home/surajracha/pedal_for_healthcare/cs6101/Project_IE643
until [ -f novelty/runs/margin_adv/best.json ]; do sleep 30; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True novelty/pipeline_after_train.sh margin_adv 1
