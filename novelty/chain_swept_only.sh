#!/usr/bin/env bash
# "#2 only": wait for the running swept generation (pid 1817458) to finish, then evaluate + 3 downstream seeds.
# ("#1 + #2" dropped: #1 alone showed no downstream benefit.)
cd "$(dirname "$0")/.."
PY=.venv311/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 1817458 2>/dev/null; do sleep 30; done
n=$(find data/CheXpert-v1.0-small/gen_ce_adv_swept -name '*.png' | wc -l)
echo "[swept] generation finished with $n images"
[ "$n" -lt 2173 ] && CUDA_VISIBLE_DEVICES=0 $PY novelty/generate.py --lora_dir novelty/runs/ce_adv/step_1000 \
    --name gen_ce_adv_swept --cfg 4.0 --lora_scale 0.75 --bs 8      # resumes if it was interrupted
CUDA_VISIBLE_DEVICES=0 $PY novelty/evaluate_synth.py --root data/CheXpert-v1.0-small/gen_ce_adv_swept --name ce_adv_swept
for s in 42 43 44; do
  CUDA_VISIBLE_DEVICES=0 $PY novelty/downstream_train.py --train_csv novelty/csv/classifier_train_mixed_gen_ce_adv_swept.csv \
      --name ce_adv_swept --seed $s > novelty/logs/ds_ce_adv_swept_$s.log 2>&1 &
done
wait
echo "[swept] CHAIN DONE"
