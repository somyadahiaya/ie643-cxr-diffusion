# Data

## CheXpert-v1.0-small (not included: the licence forbids redistribution)
Request access and download it from the Stanford AIMI site: https://stanfordaimi.azurewebsites.net/datasets/
(the CheXpert page). Then unzip so that this layout exists:
```
data/CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
data/CheXpert-v1.0-small/valid/...
```
All split CSVs use paths relative to `data/`, e.g. `CheXpert-v1.0-small/train/patient27339/study1/view1_frontal.jpg`.

## Generated image sets
Each `data/CheXpert-v1.0-small/gen_<run>/` holds the 2,173-image set of one generator plus `synthetic_labels.csv`.
There are two ways to get them:
- regenerate them exactly with `novelty/generate.py` (see REPRODUCE.md step 4);
- `bash scripts/download_weights.sh SillySom/ie643-cxr-diffusion-weights --synthetic`, which downloads the 15 tar archives
  (13 novelty sets + the phase-1 P-Rajiv and PRISM sets, ~7 GB) and unpacks them here.

`phase1_synthetic_labels/` keeps the label lists of the two phase-1 synthetic sets.
