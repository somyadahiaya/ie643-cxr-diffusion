# Reproducing every result

All commands run from the repository root. Hardware used: 2× RTX A6000 (48 GB); every step fits on one GPU.
`PY=.venv311/bin/python`. The `novelty/chain_*.sh` scripts are the exact chains that produced the results. Some
contain `until … sleep` waits on the original machine's logs or PIDs; you can run the commands inside them directly.

## 0. Environment, weights, data
```bash
bash scripts/setup_env.sh
bash scripts/download_weights.sh SillySom/ie643-cxr-diffusion-weights            # add --all for PRISM/phase-1/all LoRA steps
# CheXpert-v1.0-small -> data/CheXpert-v1.0-small/{train,valid}  (data/README.md)
PY=.venv311/bin/python
```
Splits (frontal, single-label, no uncertain labels, patient-disjoint) are committed CSVs in `ie643_code/ie643/`:
| file | rows | use |
|---|---|---|
| `diffusion_train_single_frontal_clean.csv` | 4,401 | LoRA training (6 classes) |
| `classifier_train_single_frontal_clean_real1500.csv` | 16,624 | DenseNet training (10% of patients held out for validation inside `downstream_train.py`) |
| `global_test_single_frontal_clean.csv` | 700 | test set (50 per class), only for final scores |

## 1. Phase 1 (team pipeline, run on Google Colab)
See `ie643_code/ie643/readme.md`. `train_combined.py` trained the LoRAs in
`ie643_code/ie643/cxr_diffusion_checkpoints/` (P-Rajiv `unet/epoch_7` and `unet_clip/epoch_2`; PRISM `unet/epoch_3`
and `unet_clip/epoch_8`). The phase-1 synthetic sets came from `ie643_code/ie643/generate_synthetic_dataset.py` and
`generate_synthetic_dataset.py`, and the classifier from `classifier_train.py`. Phase 2 starts from
**P-Rajiv `unet_clip/epoch_2`**, the "baseline" everywhere below.

## 2. Frozen heads and held-out judges (real images only)
```bash
CUDA_VISIBLE_DEVICES=0 $PY novelty/train_heads.py      # -> novelty/heads/ (committed; re-running reproduces them)
```
Reward head on TorchXRayVision `densenet121-res224-all`; held-out judge probe on `densenet121-res224-mimic_ch`
(29.7% on real test images); Mahalanobis OOD stats; PSPNet anatomy thresholds.

## 3. LoRA variants (each continues from the baseline LoRA; 15–60 min on one A6000)
```bash
T="$PY novelty/train_adv_ce.py"
CUDA_VISIBLE_DEVICES=0 $T --out novelty/runs/control      --lam_cls 0   --lam_adv 0
CUDA_VISIBLE_DEVICES=0 $T --out novelty/runs/ce_only      --lam_cls 0.1 --lam_adv 0   # stopped by hand after step 1000 (reward hacking)
CUDA_VISIBLE_DEVICES=0 $T --out novelty/runs/adv_only     --lam_cls 0   --lam_adv 0.05
CUDA_VISIBLE_DEVICES=0 $T --out novelty/runs/adv_only_l02 --lam_cls 0   --lam_adv 0.2
CUDA_VISIBLE_DEVICES=0 $T --out novelty/runs/ce_adv       --lam_cls 0.1 --lam_adv 0.05
CUDA_VISIBLE_DEVICES=0 $T --out novelty/runs/margin_adv   --cls_loss margin
CUDA_VISIBLE_DEVICES=0 $T --out novelty/runs/margin_adv_ms --cls_loss margin --d_multiscale
```
Every run saves `step_{0,500,…,2000}`, scores a fixed seed grid with the held-out judges every 500 steps
(`eval_log.json`), and writes the selected step to `best.json`. Selected steps: control 2000, ce_only 1000,
adv_only 2000, adv_only_l02 1500, ce_adv 1000, margin_adv 1500, margin_adv_ms 1000. Full hyper-parameters are in each
`novelty/runs/<run>/config.json`.

## 4. Generate → evaluate → downstream (per generator)
```bash
for RUN in base_prajiv control ce_only adv_only adv_only_l02 ce_adv margin_adv margin_adv_ms; do
  bash novelty/pipeline_after_train.sh $RUN 0
done
```
Each call:
1. Generates 2,173 images (500 each EC/LL/Pn/PO, 173 Fracture; 50 DDIM steps, CFG 7.5; seed = 42 + 10⁶·class + i)
   into `data/CheXpert-v1.0-small/gen_<run>/`.
2. Writes `novelty/results/eval_<run>.{json,csv}`: requested-finding rate, defects, KID, diversity, memorisation.
3. Trains the DenseNet on real + synthetic with seeds 42/43/44 → `novelty/downstream/<run>_seed*/log.json`.

Real-only reference:
```bash
for s in 42 43 44; do $PY novelty/downstream_train.py --train_csv ie643_code/ie643/classifier_train_single_frontal_clean_real1500.csv --name real_only --seed $s; done
```

## 5. Generation-side variants and the training recipe
```bash
CUDA_VISIBLE_DEVICES=0 $PY novelty/sweep_cfg.py                     # guidance / LoRA-scale sweep -> results/sweep_ce_adv_selected.json
bash novelty/chain_fix12.sh A                                       # CheXpert-style JPEG + synthetic No Finding  (ce_adv_jpg)
bash novelty/chain_fix12.sh B; bash novelty/chain_swept_only.sh     # tuned guidance cfg 4.0, LoRA scale 0.75   (ce_adv_swept)
bash novelty/chain_i2i.sh                                           # DA-Fusion img2img from real images         (ce_adv_i2i)
bash novelty/chain_recipe.sh                                        # [recipe] and [recipe + filter] for ce_adv and ce_adv_i2i
bash novelty/chain_recipe2.sh                                       # [recipe] for margin_adv and ce_adv_swept
# multi-scale D: train + pipeline + recipe in one chain (repeats step 3/4 for margin_adv_ms)
bash novelty/chain_ms.sh 0
```
Recipe = `--class_weight_from real --syn_weight 0.5 --real_ft_epochs 3`.
Filter = `filter_synth.py`, which keeps images with reward-head margin ≥ real median that also pass the anatomy check.

## 6. Tables, figures, report, poster
```bash
$PY novelty/make_report.py                        # -> novelty/report/tables.md, *.csv, fig_*.png (only complete runs)
cp novelty/report/fig_{generation_per_pathology,samples_same_seeds,checkpoint_selection,downstream_delta_key}.png report/figs/
cd report && tectonic IE643_Project_Report.tex    # 19-page report
cd ../poster && tectonic Gradient_Mavericks_IE643_Poster.tex
```
Where the report's numbers come from:
- Generation tables: `novelty/report/tables.md` → *requested_finding_rate / defect_rate / kid_x1e3* (MEAN rows).
- Downstream tables: `novelty/report/downstream_summary.csv`.
- Per pathology: `downstream_delta_vs_real.csv`.

## 7. Demo
```bash
bash novelty/run_demo.sh 0 7860          # ssh -L 7860:127.0.0.1:7860 <server> to view remotely
```
The two demo classifiers were trained with:
```bash
$PY novelty/downstream_train.py --train_csv ie643_code/ie643/classifier_train_single_frontal_clean_real1500.csv --name demo_real_only --seed 42 --save_model
$PY novelty/downstream_train.py --train_csv novelty/csv/classifier_train_mixed_gen_ce_adv.csv --name demo_ce_adv_recipe --seed 42 \
    --class_weight_from real --syn_weight 0.5 --real_ft_epochs 3 --save_model
```
