# Novelty stage: classifier-guided + adversarial LoRA fine-tuning

## Idea
The original LoRA fine-tuning used only the diffusion (noise-MSE) loss. Nothing in it checks whether the
generated X-ray (a) looks like a real chest X-ray or (b) shows the prompted pathology. We add two
image-level signals, computed on the model's own samples during training:

```
L_G = L_diff                                          original ε-MSE on real CheXpert images (kept as anchor)
    + λ_cls · CE( h_cls(φ(x̂0)), y_prompt )            frozen 6-class classifier  -> "shows requested finding"
    + λ_adv · ( −D(φ(x̂0), y_prompt) )                 discriminator              -> "looks like a real CXR of class y"

L_D = max(0, 1 − D(φ(x̂0_real,t), y_real)) + max(0, 1 + D(φ(x̂0), y_prompt))         hinge GAN loss
```

* **φ**: frozen TorchXRayVision DenseNet-121 (`densenet121-res224-all`), with 1024-d pooled features.
* **h_cls**: a linear head on φ, trained on real images only, then frozen. It gives the CE signal the instructor asked for.
* **D**: a small projection-discriminator head (Miyato & Koyama 2018) on the frozen φ, with spectral norm. It is
  trained alternately with the LoRA. The frozen backbone with small heads follows ADT (arXiv 2504.11423), and it keeps D
  stable with only 4.4k training images.
* **x̂0 (ReFL-style, Xu et al. 2023)**: sample from pure noise with 20 DDIM steps and CFG 7.5, with no gradient, down
  to a random timestep t ∈ [50, 400]. Then take ONE UNet step with gradient, form x̂0 = (z_t − √(1−ᾱ_t)·ε̂)/√ᾱ_t, and
  decode it with the VAE. The gradient flows into both the UNet LoRA and the text-encoder LoRA.
* **t-matched real reference**: the real images D sees are also one-step x̂0 estimates, made from real latents
  noised to the same t. D therefore cannot win just by spotting one-step blur.
* **Also new**: 10% caption dropout, so classifier-free guidance has an unconditional branch to contrast against.

## Evaluation protocol: held-out judges only
Nothing used as a training signal is used to judge the results.
* **Requested-finding rate**: a 6-class linear probe on **TorchXRayVision `mimic_ch`** features. That backbone was trained
  on MIMIC only, so it has no CheXpert leakage and is not the reward backbone. The probe gets 29.7% on real test images
  (chance is 16.7%), so we also report the rate **normalised** by the probe's per-class recall on REAL test images.
* **Structural defect**: an image counts as defective if it fails the PSPNet (ChestX-Det) anatomy check OR is
  out-of-distribution (Mahalanobis distance in `mimic_ch` feature space, above the 95th percentile of real validation
  images).
  * The anatomy check requires exactly one dominant left-lung and right-lung region, lung and heart areas within the
    1st–99th percentile of real images, and lungs and heart on the correct sides.
  * Real test images pass the anatomy check 95.7% of the time and are flagged OOD 1.7% of the time.
* **Other per-class metrics**: KID against real images of the same class, within-class diversity, and
  nearest-neighbour distance to the training set (memorisation).
* **Checkpoint selection**: every 500 steps, a FIXED seed grid is scored (24 images per class), and the checkpoint
  with the best (requested-finding − defect) on the 5 target classes is selected. This replaces picking a checkpoint
  from the noisy loss curve.
* **Downstream**: the team's DenseNet-121 14-class setup is retrained on real-only and on real + 2173 synthetic images
  (the same per-class counts as before), with 3 seeds each.
  * The epoch is selected on a patient-level **validation split** of the real training rows. The original script used
    the test set for this.
  * Results are reported on the same held-out real test set (50 images per class), per pathology.

## Runs
| run | λ_cls | λ_adv | purpose |
|---|---|---|---|
| base_prajiv | – | – | the team's current P-Rajiv unet_clip epoch_2 LoRA |
| control | 0 | 0 | same extra 2000 steps and caption dropout, no new loss (isolates the effect of the losses) |
| ce_only | 0.1 | 0 | classifier (CE) signal only |
| ce_adv | 0.1 | 0.05 | CE + adversarial |

## Commands
```bash
python novelty/train_heads.py                                   # frozen heads + calibration (real images only)
python novelty/train_adv_ce.py --out novelty/runs/ce_adv --lam_cls 0.1 --lam_adv 0.05
novelty/pipeline_after_train.sh ce_adv 0                        # generate -> evaluate -> downstream x3 seeds
python novelty/make_report.py                                   # tables + figures in novelty/report/
```
