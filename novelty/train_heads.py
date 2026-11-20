"""Step 1: build the frozen classifier heads + evaluation calibration from REAL images only.

Outputs (novelty/heads/):
  reward_head.pt      LinearHead on REWARD_BACKBONE features  -> CE signal used during LoRA fine-tuning
  eval_head.pt        LinearHead on EVAL_BACKBONE features    -> held-out judge of "shows requested finding"
  feat_stats.pt       feature mean/std (reward backbone) for the discriminator head
  ood_stats.pt        Mahalanobis stats on eval-backbone features + 95th-pct threshold from real val
  anatomy.json        anatomy-check thresholds calibrated on real val images
  real_test_feats.pt  eval-backbone features of real test images (for KID / normalised conditioning)
  heads_report.json   accuracy of the heads on real val/test (needed to normalise conditioning accuracy)

Patient-level train/val split carved out of classifier_train_*.csv, test set never used for fitting.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cxr_common import *  # noqa


class RealDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return load_image_tensor(os.path.join(DATA_DIR, r["Path"])), CLASS_TO_IDX[r["label"]]


def six_class_df(csv):
    df = pd.read_csv(csv)
    df["label"] = df.apply(label_of_row, axis=1)
    df = df[df["label"].notna()].copy()
    df["pid"] = df["Path"].str.split("/").str[2]
    return df


@torch.no_grad()
def extract(df, backbones, device, anat=None, bs=64):
    dl = DataLoader(RealDS(df), batch_size=bs, num_workers=8)
    feats = {k: [] for k in backbones}
    meas = []
    for x, _ in dl:
        x = x.to(device)
        xx = to_xrv(x)
        for k, m in backbones.items():
            feats[k].append(backbone_features(m, xx).cpu())
        if anat is not None:
            meas.append(anat.measure(x))
    feats = {k: torch.cat(v).numpy() for k, v in feats.items()}
    if anat is not None:
        meas = {k: np.concatenate([m[k] for m in meas]) for k in meas[0]}
    return feats, meas


def fit_head(Xtr, ytr, Xva, yva):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    best = None
    for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit((Xtr - mu) / sd, ytr)
        acc = balanced_accuracy_score(yva, clf.predict((Xva - mu) / sd))
        print(f"   C={C}: val balanced acc {acc:.3f}")
        if best is None or acc > best[0]:
            best = (acc, C, clf)
    acc, C, clf = best
    head = LinearHead()
    head.mu.copy_(torch.tensor(mu)); head.sd.copy_(torch.tensor(sd))
    head.fc.weight.data.copy_(torch.tensor(clf.coef_, dtype=torch.float32))
    head.fc.bias.data.copy_(torch.tensor(clf.intercept_, dtype=torch.float32))
    return head, clf, (mu, sd), C


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(HEADS_DIR, exist_ok=True)
    rng = np.random.RandomState(0)

    df = six_class_df(CSV_CLS)
    pids = df["pid"].unique()
    val_p = set(rng.choice(pids, int(0.15 * len(pids)), replace=False))
    tr, va = df[~df.pid.isin(val_p)], df[df.pid.isin(val_p)]
    te = six_class_df(CSV_TEST)
    print("train/val/test sizes:", len(tr), len(va), len(te))
    print(tr.label.value_counts().to_dict())
    va[["Path", "label"]].to_csv(os.path.join(HEADS_DIR, "real_val_split.csv"), index=False)

    bbs = {"reward": load_backbone(REWARD_BACKBONE, device), "eval": load_backbone(EVAL_BACKBONE, device)}
    anat = AnatomyChecker(device)

    print("extracting train features ..."); Ftr, _ = extract(tr, bbs, device)
    print("extracting val features ...");   Fva, Mva = extract(va, bbs, device, anat)
    print("extracting test features ...");  Fte, Mte = extract(te, bbs, device, anat)
    ytr = tr.label.map(CLASS_TO_IDX).values; yva = va.label.map(CLASS_TO_IDX).values
    yte = te.label.map(CLASS_TO_IDX).values

    report = {}
    for k in ("reward", "eval"):
        print(f"\n== {k} head ({REWARD_BACKBONE if k == 'reward' else EVAL_BACKBONE})")
        head, clf, (mu, sd), C = fit_head(Ftr[k], ytr, Fva[k], yva)
        torch.save(head.state_dict(), os.path.join(HEADS_DIR, f"{k}_head.pt"))
        pte = clf.predict((Fte[k] - mu) / sd)
        rep = classification_report(yte, pte, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                    output_dict=True, zero_division=0)
        report[k] = {"C": C, "test_acc": accuracy_score(yte, pte),
                     "test_balanced_acc": balanced_accuracy_score(yte, pte),
                     "test_recall_per_class": {c: rep[c]["recall"] for c in CLASSES}}
        print(json.dumps(report[k], indent=1))
        if k == "reward":
            torch.save({"mu": torch.tensor(mu), "sd": torch.tensor(sd)}, os.path.join(HEADS_DIR, "feat_stats.pt"))

    # ---- OOD (Mahalanobis) on eval-backbone features
    lw = LedoitWolf().fit(Ftr["eval"])
    prec, mean = lw.precision_, lw.location_
    def maha(F): d = F - mean; return np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d))
    thr = float(np.percentile(maha(Fva["eval"]), 95))
    report["ood"] = {"threshold_95pct_real_val": thr, "real_test_ood_rate": float((maha(Fte["eval"]) > thr).mean())}
    torch.save({"mean": torch.tensor(mean), "prec": torch.tensor(prec), "thr": thr}, os.path.join(HEADS_DIR, "ood_stats.pt"))

    # ---- anatomy thresholds from real val (1st-99th percentile)
    th = {k: [float(np.nanpercentile(Mva[k], 1)), float(np.nanpercentile(Mva[k], 99))]
          for k in ("areaL", "areaR", "heart_area")}
    json.dump(th, open(os.path.join(HEADS_DIR, "anatomy.json"), "w"), indent=1)
    anat.th = th
    report["anatomy"] = {"thresholds": th,
                         "real_val_pass_rate": float(anat.passes(Mva).mean()),
                         "real_test_pass_rate": float(anat.passes(Mte).mean()),
                         "real_test_pass_per_class": {c: float(anat.passes(Mte)[yte == i].mean())
                                                      for i, c in enumerate(CLASSES)}}
    print(json.dumps(report["ood"], indent=1)); print(json.dumps(report["anatomy"], indent=1))

    torch.save({"feats": torch.tensor(Fte["eval"]), "y": torch.tensor(yte),
                "paths": te.Path.tolist()}, os.path.join(HEADS_DIR, "real_test_feats.pt"))
    # real TRAIN eval-feats per class (for nearest-neighbour memorisation check)
    torch.save({"feats": torch.tensor(Ftr["eval"]), "y": torch.tensor(ytr)}, os.path.join(HEADS_DIR, "real_train_feats.pt"))
    json.dump(report, open(os.path.join(HEADS_DIR, "heads_report.json"), "w"), indent=1)
    print("\nsaved to", HEADS_DIR)


if __name__ == "__main__":
    main()
