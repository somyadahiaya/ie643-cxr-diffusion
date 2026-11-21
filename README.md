# Text-Conditioned Synthetic Chest X-ray Generation Using Diffusion Models for Data Augmentation

**IE643 Course Project, Team Gradient Mavericks**: Saksham Jain, Somya Dahiaya

We LoRA-fine-tune a Stable Diffusion 1.5 chest X-ray generator (P-Rajiv) to synthesise rare CheXpert findings
(Enlarged Cardiomediastinum, Lung Lesion, Pneumonia, Pleural Other, Fracture). We add two image-level training
signals to the denoising loss:
- a **frozen-classifier loss** (CE, or our bounded *matched-confidence margin*) that rewards showing the requested finding;
- a **conditional projection-discriminator (adversarial) loss** on frozen CXR features that rewards looking real.

Both are computed on a differentiable one-step image estimate (ReFL-style). Every judgement uses
**held-out** models that were never used in training.

| Generator | Requested finding ↑ | Structural defects ↓ | KID×10³ ↓ | Downstream macro-F1 (plain / [recipe]) |
|---|---|---|---|---|
| Real images only | 29.7% (judge on real) | n/a | n/a | **0.285 ± 0.012** |
| P-Rajiv LoRA (phase-1 baseline) | 15.7% | 32.7% | 12.8 | 0.256 / n/a |
| CE only | 35.3% | 78.5% (reward hacking) | 29.9 | 0.258 / n/a |
| Adversarial only | 25.9% | 9.8% | 2.9 | 0.270 / n/a |
| CE + adversarial | 45.9% | 24.6% | 8.2 | 0.269 / **0.283** |
| **Margin + adversarial** | **51.2%** | 18.2% | 7.8 | 0.278 / 0.279 |
| Margin + adv, multi-scale D | 43.8% | 16.5% | 8.7 | 0.264 / 0.281 |

The losses triple the requested-finding rate and cut defects. Downstream (14-class DenseNet-121, 3 seeds, 700 real test
images), augmentation reaches **parity** with real-only training but does not beat it. This matches the
literature for full-data, in-distribution settings; see `report/IE643_Project_Report.pdf`, Sec. 7.3.

---

## Repository layout
```
README.md  REPRODUCE.md  CHECKPOINTS.md        start here / step-by-step reproduction / every checkpoint and where it lives
requirements.txt  requirements-lock.txt       direct deps / exact environment used for all results (Python 3.11, CUDA 12.6)
scripts/                                      setup_env.sh, download_weights.sh, upload_weights_to_hf.sh
ie643_code/ie643/                             PHASE 1 (team code): LoRA training, generation, DenseNet classifier, data splits
  cxr_diffusion_checkpoints/                  phase-1 LoRAs (P-Rajiv unet/unet_clip, PRISM unet/unet_clip) + logs
cxr_diffusion_checkpoints/epoch_8/            phase-1 P-Rajiv LoRA epoch 8 (accelerate checkpoint; big files on HF)
generate_synthetic_dataset.py, generation.log phase-1 PRISM synthetic-set generation
novelty/                                      PHASE 2 (novelty): all code, results and logs
  cxr_common.py sd_utils.py                   shared: classes/prompts, frozen backbones, heads, D, anatomy checker, sampler
  train_heads.py                              frozen reward head, held-out judge, OOD + anatomy calibration (real images only)
  train_adv_ce.py                             classifier- + adversarially-guided LoRA fine-tuning (CE / margin, single / multi-scale D)
  generate.py evaluate_synth.py               2,173-image synthetic sets / per-pathology held-out evaluation
  downstream_train.py                         14-class DenseNet-121 with/without synthetic data (validation-based epoch selection)
  sweep_cfg.py img2img_augment.py chexpert_style.py filter_synth.py   generation-side variants
  pipeline_after_train.sh chain_*.sh          the exact command chains that produced every result
  make_report.py                              rebuilds novelty/report/ (tables.md, csv, figures) from results/ + downstream/
  run_demo.sh                                 runs demo/app.py on this clone's local files
  heads/                                      frozen heads + calibration stats (needed by training, evaluation, demo)
  runs/<run>/                                 configs, train/eval logs, the selected LoRA checkpoint (all steps on HF)
  results/  downstream/  logs/  csv/          evaluation json/csv, per-run DenseNet logs (+2 demo weights), raw logs, train CSVs
  report/                                     generated tables and figures used in the report and poster
  prajiv_base/                                base SD-1.5 CXR pipeline configs/tokenizer (weights on HF)
demo/                                         Gradio app and self-contained Kaggle GPU notebook; downloads models from HF
report/                                       LaTeX report (IE643_Project_Report.tex/.pdf, refs.bib, figs); novelty_2page/ = 2-page novelty report
poster/                                       A0 poster (Gradient_Mavericks_IE643_Poster.tex/.pdf)
data/                                         put CheXpert-v1.0-small here (see data/README.md)
```

## Demo (no clone needed)

**Demo video:** [Google Drive](https://drive.google.com/drive/folders/1G2jfzdX-hmI5Eav6i108QYIP172fx8tv?usp=sharing)

### Run on Kaggle GPU

Download **[demo/cxr_demo_kaggle.ipynb](demo/cxr_demo_kaggle.ipynb)**, create a
[Kaggle notebook](https://www.kaggle.com/code), and upload it with **File > Import Notebook**.
Enable **GPU** and **Internet** under **Settings / Session options**, then run the three code cells in order.
Open the printed `gradio.live` link, use username **`cxr`** and the generated password, and keep the final cell
and Kaggle session running. The password and link change on each launch. The weights are public; no Hugging Face
token is required. Tested on a Kaggle Tesla T4 GPU.

GPU access may require Kaggle phone verification and available GPU quota.
Setup details, troubleshooting, and shutdown instructions: **[demo/README.md](demo/README.md#run-on-kaggle-gpu)**.

### Run on your own NVIDIA GPU

`demo/` is a self-contained Gradio app: prompt → synthetic X-ray from any of the 7 report generators → held-out judge,
anatomy/OOD defect check and two DenseNet-121 classifiers. It downloads only the models it needs (~2.2 GB) from
[`SillySom/ie643-cxr-diffusion-weights`](https://huggingface.co/SillySom/ie643-cxr-diffusion-weights):
```bash
pip install -r demo/requirements.txt
python demo/app.py                # http://127.0.0.1:7860
```
Full instructions, options and GPU notes: **[demo/README.md](demo/README.md)**.

## Quick start (full project)
```bash
bash scripts/setup_env.sh                                     # .venv311 with the exact package versions
bash scripts/download_weights.sh                              # base model weights (~2 GB); see CHECKPOINTS.md
# CheXpert-v1.0-small: download from Stanford and unzip into data/   (see data/README.md)
bash novelty/run_demo.sh 0 7860                               # demo on the local files
```
Full reproduction of every number, table and figure: **[REPRODUCE.md](REPRODUCE.md)**.

## Large files
Git holds the code and everything under ~30 MB per file: the selected LoRA of every run, the judge heads, and the two
demo DenseNets. The base UNet/VAE/text encoder, PRISM, the phase-1 epoch_8 checkpoint, every intermediate LoRA step and
the 15 generated image sets are in a Hugging Face model repo. **[CHECKPOINTS.md](CHECKPOINTS.md)** lists every file,
its size, its location and the script that uses it. `SHA256SUMS` verifies the downloads.

## Notes
- **Data licence:** CheXpert may not be redistributed, so it is not included. Generated images can be regenerated
  exactly from the checkpoints and seeds; we verified this to < 2/255 mean pixel difference.
- **Determinism:** LoRA training, generation and evaluation are seeded. DenseNet training on GPU is not bit-exact,
  so re-runs land within the reported seed spread (e.g. real-only seed 42: 0.272 in the report, 0.276 on re-run).
- **Demo classifiers:** the report's downstream runs saved only logs. `novelty/downstream/demo_*_seed42/best_model.pt`
  are re-trainings with the same settings (`downstream_train.py --save_model`).
- Run configs record the absolute paths of the machine they ran on; the code itself uses repo-relative paths.
