"""Recipe step 3: filter synthetic images WITHOUT using the held-out evaluator.

Keep a synthetic image of class y only if
  (a) the TRAINING-time reward head (TorchXRayVision 'all' backbone + frozen linear head) gives it a logit margin
      z_y - max_{k!=y} z_k >= m_y, where m_y = median margin of correctly-classified REAL images of class y
      ("at least as recognisable as a typical real image"), and
  (b) it passes the lung/heart anatomy check (thresholds calibrated on real images).
The evaluation backbone (mimic_ch) is NOT used, and the downstream test set is untouched.
Writes novelty/csv/classifier_train_mixed_<name>_filt.csv and a json with keep-rates per class.
"""
import os, sys, json, argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cxr_common import *  # noqa


class PathDS(Dataset):
    def __init__(self, paths):
        self.p = paths

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return load_image_tensor(os.path.join(DATA_DIR, self.p[i]))


@torch.no_grad()
def margins(paths, labels, phi, head, dev):
    out = []
    for x in DataLoader(PathDS(paths), batch_size=64, num_workers=8):
        out.append(head(backbone_features(phi, to_xrv(x.to(dev)))).cpu())
    z = torch.cat(out); y = torch.tensor(labels)
    other = z.clone(); other[torch.arange(len(y)), y] = -1e9
    return z[torch.arange(len(y)), y] - other.max(1).values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixed_csv", required=True)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    dev = "cuda"
    phi = load_backbone(REWARD_BACKBONE, dev)
    head = load_linear_head(os.path.join(HEADS_DIR, "reward_head.pt"), dev)
    anat = AnatomyChecker(dev, json.load(open(os.path.join(HEADS_DIR, "anatomy.json"))))

    # thresholds m_y from REAL diffusion-training images (same definition as the margin loss)
    real = pd.read_csv(CSV_DIFF); real["label"] = real.apply(label_of_row, axis=1)
    real = real.groupby("label").head(150)
    mr = margins(real.Path.tolist(), real.label.map(CLASS_TO_IDX).tolist(), phi, head, dev)
    m_y = {c: float(mr[(real.label == c).values][mr[(real.label == c).values] > 0].median()) for c in CLASSES}

    df = pd.read_csv(a.mixed_csv)
    syn = df[df.Path.str.contains("synthetic")].copy()
    syn["label"] = syn.apply(label_of_row, axis=1)
    ms = margins(syn.Path.tolist(), syn.label.map(CLASS_TO_IDX).tolist(), phi, head, dev).numpy()
    ok_anat = []
    for x in DataLoader(PathDS(syn.Path.tolist()), batch_size=64, num_workers=8):
        ok_anat.append(anat.passes(anat.measure(x)))
    import numpy as np
    ok_anat = np.concatenate(ok_anat)
    keep = (ms >= syn.label.map(m_y).values) & ok_anat
    syn["keep"] = keep
    rep = {"m_y": m_y, "kept_total": int(keep.sum()), "of": int(len(keep)),
           "per_class": {c: [int(g.keep.sum()), int(len(g))] for c, g in syn.groupby("label")}}
    out = os.path.join(PROJECT, "novelty", "csv", f"classifier_train_mixed_{a.name}_filt.csv")
    pd.concat([df[~df.Path.str.contains("synthetic")], df.loc[syn.index[keep]]]).to_csv(out, index=False)
    json.dump(rep, open(os.path.join(PROJECT, "novelty", "results", f"filter_{a.name}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), "\nCSV:", out)


if __name__ == "__main__":
    main()
