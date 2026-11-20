# Demo: prompt → synthetic chest X-ray → classification

A single-file Gradio app (`app.py`). It downloads the report's models from the Hugging Face repo
[`SillySom/ie643-cxr-diffusion-weights`](https://huggingface.co/SillySom/ie643-cxr-diffusion-weights)
(about 2.2 GB, cached after the first run), so **you do not need to clone the full project**.

**Generate & classify tab**
1. Type a prompt (or pick one of the 6 training prompts).
2. Choose any of the 7 generators from the report. They all use the same seeds, so the comparison is fair.
3. For each image you get:
   - the held-out judge's verdict and whether the image shows the requested finding;
   - the anatomy check (lung/heart segmentation) and the out-of-distribution check, which together give a structural-defect flag;
   - the top predictions of two 14-class DenseNet-121 classifiers (real-only, and real + synthetic).

**Classify an X-ray tab:** the same analysis for any X-ray you upload.

## Requirements
- An NVIDIA GPU with ≥ 8 GB memory (tested on an RTX A6000), CUDA 12.x drivers, Python 3.11.
- Access to the weights repo. It is private, so you need a Hugging Face **read** token from an account with
  access: https://huggingface.co/settings/tokens

## Run it (without cloning the project)
```bash
mkdir cxr-demo && cd cxr-demo
# get the three demo files: either from the GitHub repo's demo/ folder ...
#   app.py  requirements.txt  README.md
# ... or from the weights repo:
pip install "huggingface-hub==2.3.0"
hf auth login                                   # paste your read token (or: export HF_TOKEN=hf_...)
hf download SillySom/ie643-cxr-diffusion-weights --include "demo/*" --local-dir .
cd demo

python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

python app.py                                   # first run downloads ~2.2 GB of models, then:
                                                #   * Running on local URL:  http://127.0.0.1:7860
```
Open http://127.0.0.1:7860. On a remote server, forward the port from your laptop first:
`ssh -L 7860:127.0.0.1:7860 <user>@<server>`, then open http://localhost:7860.

### Options
| flag | default | meaning |
|---|---|---|
| `--port 7860` / `--host 127.0.0.1` | | where to serve the app |
| `--cache_dir DIR` | `~/.cache/huggingface` | where the downloaded models are stored |
| `--data_dir DIR` | none | folder that contains `CheXpert-v1.0-small/`; enables "Load random real test image" |
| `--local_root DIR` | none | use a full clone of the project (with weights from `scripts/download_weights.sh`) instead of downloading |
| `--hf_repo ID` | `SillySom/ie643-cxr-diffusion-weights` | a different copy of the weights |
| `--share` | off | public gradio.live link (routes traffic through Gradio's servers) |

Pick the GPU with `CUDA_VISIBLE_DEVICES=0 python app.py`. From a full clone of the project you can also run
`bash novelty/run_demo.sh [GPU] [PORT]`, which uses the local files.

## What is downloaded
| HF path | size | what |
|---|---|---|
| `novelty/prajiv_base/` | 2.0 GB | P-Rajiv CXR Stable Diffusion 1.5 (UNet, VAE, text encoder, tokenizer, scheduler) |
| `ie643_code/.../prajiv/unet_clip/epoch_2/{unet_lora,text_encoder_lora}` | 17 MB | phase-1 baseline LoRA |
| `novelty/runs/<run>/step_<selected>/` ×6 | 17 MB each | report-selected LoRA of control, CE only, adversarial only, CE + adv, margin + adv, multi-scale D |
| `novelty/heads/{eval_head.pt, ood_stats.pt, anatomy.json, heads_report.json}` | 4 MB | held-out judge probe, OOD detector, anatomy thresholds |
| `novelty/downstream/demo_*_seed42/best_model.pt` | 2 × 27 MB | the two 14-class DenseNet-121 classifiers |

TorchXRayVision then fetches its own `densenet121-res224-mimic_ch` backbone and PSPNet segmenter (~100 MB) on first use.

## Reading the results
- The **held-out judge** was never used during training. On *real* test images it is only 29.7% accurate (chance
  16.7%), so a single image's verdict is noisy. The report's rates average over 2,173 images per generator.
- **Structural defect** = the anatomy check fails (exactly one left and one right lung, plausible lung/heart areas,
  correct sides) **or** the image is out of distribution in the judge's feature space.
- Images are reproducible: the same generator, prompt, seed, steps and CFG give the same image as the report's
  synthetic sets.
- Research/coursework only, not for clinical use.
