# Checkpoints and artifacts

**git** = committed in this repository. **HF** = Hugging Face model repo (fetched by `scripts/download_weights.sh`,
verified against `SHA256SUMS`). HF paths mirror repo paths, so downloads land where the code expects them.

## Generators used in the report (phase 2)
All novelty LoRAs are rank-16 adapters on the UNet attention layers (`to_q/k/v/out.0`) plus the text encoder. Each
continues from the baseline LoRA. "Selected" = the step chosen by held-out checkpoint selection (`best.json`), which is
used for every reported number and by the demo.

| Generator | Selected checkpoint (git, ~17 MB) | Other steps (HF) |
|---|---|---|
| P-Rajiv LoRA baseline (phase 1) | `ie643_code/ie643/cxr_diffusion_checkpoints/prajiv/unet_clip/epoch_2/{unet_lora,text_encoder_lora}` | n/a |
| Control | `novelty/runs/control/step_2000` | step_0…1500 |
| CE only | `novelty/runs/ce_only/step_1000` | step_0, 500 |
| Adversarial only | `novelty/runs/adv_only/step_2000` | step_0…1500 |
| Adversarial only, λ=0.2 | `novelty/runs/adv_only_l02/step_1500` | step_0…2000 |
| CE + adversarial | `novelty/runs/ce_adv/step_1000` | step_0…2000 |
| Margin + adversarial (best) | `novelty/runs/margin_adv/step_1500` | step_0…2000 |
| Margin + adv, multi-scale D | `novelty/runs/margin_adv_ms/step_1000` | step_0…2000 |

Base model (required by every generator): `novelty/prajiv_base/`. The configs and tokenizer are in git; the weights are on HF:
| file | size | note |
|---|---|---|
| `novelty/prajiv_base/unet/diffusion_pytorch_model.safetensors` | 1.6 GB | P-Rajiv CXR SD-1.5 UNet, recovered from the team's epoch_2 accelerate checkpoint |
| `novelty/prajiv_base/vae/diffusion_pytorch_model.safetensors` | 160 MB | from `P-RAJIV/cxr_stable_diffusion_v1_5` |
| `novelty/prajiv_base/text_encoder/model.safetensors` | 235 MB | from `P-RAJIV/cxr_stable_diffusion_v1_5` |

## Judges, heads and classifiers (git)
| path | used by |
|---|---|
| `novelty/heads/reward_head.pt` | training: frozen CE / margin reward head on XRV `densenet121-res224-all` |
| `novelty/heads/eval_head.pt` | evaluation + demo: held-out 6-class judge probe on XRV `densenet121-res224-mimic_ch` |
| `novelty/heads/ood_stats.pt`, `anatomy.json`, `feat_stats.pt` | Mahalanobis OOD detector, PSPNet anatomy thresholds, reward-backbone feature mean/std for the discriminator |
| `novelty/heads/real_{train,test}_feats.pt`, `real_val_split.csv`, `heads_report.json` | KID / diversity references, split, calibration report |
| `novelty/runs/margin_*/real_margins.json` | per-class real-image margin targets m_y |
| `novelty/downstream/demo_real_only_seed42/best_model.pt` | demo: 14-class DenseNet-121, real only (test macro-F1 0.276) |
| `novelty/downstream/demo_ce_adv_recipe_seed42/best_model.pt` | demo: real + CE+Adv synthetic [recipe] (0.273) |

The TorchXRayVision backbones and the PSPNet segmenter are downloaded automatically by `torchxrayvision` on first use.
For the report's downstream runs only logs were kept (`novelty/downstream/<setting>_seed{42,43,44}/log.json`).

## Phase-1 artifacts
| path | where | note |
|---|---|---|
| `ie643_code/ie643/cxr_diffusion_checkpoints/prajiv/unet/epoch_7`, `prism/unet/epoch_3`, `prism/unet_clip/epoch_8` | git | phase-1 LoRAs (P-Rajiv UNet-only; PRISM UNet-only / UNet+CLIP) |
| `ie643_code/ie643/cxr_diffusion_checkpoints/prajiv/unet_clip/epoch_2/model.safetensors` | HF (1.6 GB) | full accelerate checkpoint of the baseline (the LoRA dirs next to it are in git) |
| `cxr_diffusion_checkpoints/epoch_8/` | git (LoRAs, optimizer, scheduler) + HF (`model.safetensors` 1.6 GB, `model_1.safetensors` 239 MB) | P-Rajiv unet_clip epoch 8, full resume state |
| `prism/prism_model.safetensors` | HF (3.3 GB) | PRISM UNet weights for SD-1.5 (`--model_type prism --prism_weights …`) |

## Generated image sets (HF `synthetic_sets/*.tar`, ~6.3 GB; or regenerate)
`gen_base_prajiv, gen_control, gen_ce_only, gen_adv_only, gen_adv_only_l02, gen_ce_adv, gen_margin_adv, gen_margin_adv_ms`
(2,173 images each), `gen_ce_adv_swept` (cfg 4.0, LoRA scale 0.75), `gen_ce_adv_i2i` (DA-Fusion), `gen_ce_adv_nofinding`,
`gen_ce_adv_jpg` (CheXpert-style), and the phase-1 sets `prajiv_synthetic_unet_clip_7_5`, `prism_synthetic_unet_clip_7_5`.

## Not included
- **CheXpert-v1.0-small** (licence). See `data/README.md`.
- Per-run copies of the training split (`novelty/downstream/*/train*.csv`, `val.csv`). `downstream_train.py` recreates them deterministically.
