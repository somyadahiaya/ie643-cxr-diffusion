"""Batched, reproducible synthetic-set generation (replaces generate_synthetic_dataset.py for the novelty runs).

Fixes vs the original script: seeds no longer use Python's per-process-randomised hash(); batching; no CPU offload.
Writes  data/CheXpert-v1.0-small/<name>/<Class>/*.png,  synthetic_labels.csv,
and     novelty/csv/classifier_train_mixed_<name>.csv  (real1500 + synthetic, same format as before).
"""
import os, sys, argparse
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cxr_common import *  # noqa
from sd_utils import load_sd, attach_lora, sample

TARGETS = {"Lung Lesion": 500, "Pleural Other": 500, "Pneumonia": 500,
           "Enlarged Cardiomediastinum": 500, "Fracture": 173}          # same as the original run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", default="prajiv", choices=["prajiv", "prism"])
    p.add_argument("--prism_weights", default=None)
    p.add_argument("--lora_dir", required=True)
    p.add_argument("--lora_mode", default="unet_clip")
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--n_scale", type=float, default=1.0, help="multiply TARGETS (e.g. 0.5 for a quick set)")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--targets", default=None, help='override, e.g. "No Finding:500"')
    a = p.parse_args()

    pipe, ddim, _ = load_sd(a.model_type, a.prism_weights)
    pipe = attach_lora(pipe, a.lora_dir, a.lora_mode, trainable=False)
    pipe.unet.eval(); pipe.text_encoder.eval()
    if a.lora_scale != 1.0:
        from sd_utils import set_lora_scale
        print("scaled LoRA layers:", set_lora_scale(pipe, a.lora_scale), "x", a.lora_scale)
    targets = TARGETS if a.targets is None else \
        {t.split(":")[0].strip(): int(t.split(":")[1]) for t in a.targets.split(",")}

    root = os.path.join(DATA_DIR, "CheXpert-v1.0-small", a.name)
    rows = []
    for c, n in targets.items():
        n = int(round(n * a.n_scale))
        ci = CLASS_TO_IDX[c]
        d = os.path.join(root, c.replace(" ", "_")); os.makedirs(d, exist_ok=True)
        fname = lambda i: f"synthetic_{c.replace(' ', '_')}_{i:04d}.png"
        todo = [i for i in range(n) if not os.path.exists(os.path.join(d, fname(i)))]   # resumable
        for s in tqdm(range(0, len(todo), a.bs), desc=f"{c} ({len(todo)} to do)"):
            idx = todo[s:s + a.bs]
            seeds = [a.base_seed + 1_000_000 * ci + i for i in idx]
            imgs = sample(pipe, ddim, [prompt_for(c)] * len(idx), seeds, a.steps, a.cfg)
            for i, im in zip(idx, imgs):
                im.save(os.path.join(d, fname(i)))
        rows += [{"Path": f"{c.replace(' ', '_')}/{fname(i)}", "Label": c, "Is_Synthetic": 1} for i in range(n)]
    pd.DataFrame(rows).to_csv(os.path.join(root, "synthetic_labels.csv"), index=False)

    # mixed CSV for classifier_train.py (paths relative to DATA_DIR)
    real = pd.read_csv(CSV_CLS)
    syn = []
    all14 = [c for c in real.columns[5:19]]
    for r in rows:
        row = {col: 0.0 for col in all14}
        row.update({"Path": f"CheXpert-v1.0-small/{a.name}/{r['Path']}", "Sex": "Unknown", "Age": 0,
                    "Frontal/Lateral": "Frontal", "AP/PA": "AP", r["Label"]: 1.0, "Patient_ID": "synthetic"})
        syn.append(row)
    os.makedirs(os.path.join(PROJECT, "novelty", "csv"), exist_ok=True)
    out = os.path.join(PROJECT, "novelty", "csv", f"classifier_train_mixed_{a.name}.csv")
    pd.concat([real, pd.DataFrame(syn)[real.columns]], ignore_index=True).to_csv(out, index=False)
    print("wrote", len(rows), "images to", root, "and", out)


if __name__ == "__main__":
    main()
