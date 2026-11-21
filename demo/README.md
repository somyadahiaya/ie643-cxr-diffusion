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

## Run on Kaggle GPU

Use **[cxr_demo_kaggle.ipynb](cxr_demo_kaggle.ipynb)** to run the demo without a local NVIDIA GPU.
The notebook is self-contained: you do not need to upload `app.py`, clone the repository, or use a Hugging Face token.
The model weights are public.

1. Download `cxr_demo_kaggle.ipynb` from this repository. On GitHub, open the file and select **Download raw file**.
2. Create a notebook at [Kaggle Code](https://www.kaggle.com/code), then select **File > Import Notebook**
   and upload the downloaded `.ipynb`. Keep the notebook private.
3. Under **Settings / Session options**, select an available **GPU** (for example, T4) and enable **Internet**.
   If the GPU option is disabled, complete Kaggle's phone verification in your account settings, reload the editor,
   and check your remaining GPU quota. Changing Python code cannot enable a GPU blocked by your account settings.
4. Run the three code cells in order, or select **Run All**:
   - **Check GPU, Internet, and public weights:** verifies CUDA and access to the required model files.
   - **Install the demo:** creates the demo environment using Kaggle's existing CUDA PyTorch installation.
   - **Launch the demo:** downloads the weights, loads all seven generators and the classifiers, and starts Gradio.
5. Wait for the printed `https://...gradio.live` link. Open it and sign in with username **`cxr`**
   and the password printed in the launch cell. The password is randomly generated for each launch;
   the link also changes. No credentials are stored in the committed notebook.
6. Leave the final cell and Kaggle session running while using the demo. To finish, interrupt the final cell
   and select **Stop session** in Kaggle to release the GPU.

The first run installs dependencies and downloads about 2.2 GB of model assets, plus the TorchXRayVision models.
Downloads are cached under `/kaggle/working/cxr-demo/model-cache` while the session's files remain available.
The app uses one GPU even if Kaggle assigns two. It automatically selects fp16 on T4/P100 GPUs and bf16 on
supported newer GPUs; fp16 results can differ from the report's bf16 results. Start with one image per generator.

This notebook was validated on Kaggle with a Tesla T4, Python 3.13, PyTorch `2.11.0+cu130`, and CUDA 13.0.
Both baseline and Margin + Adversarial generation, anatomy/OOD checks, and downstream classification completed
with seed 42, 50 DDIM steps, and CFG 7.5. These checks confirm execution, not that every image shows the requested finding.

### Kaggle troubleshooting

- **No CUDA GPU:** enable a GPU before running the notebook; restart the session if Kaggle requests it.
- **Download fails:** check that Internet is enabled and the public weights repository is reachable.
- **App exits:** read the exception printed by the final cell, which includes the end of
  `/kaggle/working/cxr-demo/demo.log`.
- **Out of GPU memory:** use one image per generator and stop other GPU workloads in the session.
- **Demo link stops working:** check that the Kaggle session and launch cell are still running.
  After restarting the launch cell, use its new link and password.

The link routes through Gradio. Do not upload sensitive patient data or publish notebook outputs containing the
generated session password. Clear outputs before sharing a notebook copy.

## Requirements for the standalone Python app
- An NVIDIA GPU with ≥ 8 GB memory (tested on an RTX A6000), CUDA 12.x drivers, Python 3.11.
- Internet access to the public weights repository. No Hugging Face token is required.

## Run it (without cloning the project)
```bash
mkdir cxr-demo && cd cxr-demo
# get the three demo files: either from the GitHub repo's demo/ folder ...
#   app.py  requirements.txt  README.md
# ... or from the weights repo:
pip install "huggingface-hub==2.3.0"
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
