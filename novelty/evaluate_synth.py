"""Per-pathology evaluation of a synthetic set (folder with synthetic_labels.csv).

All judgements use the HELD-OUT evaluator (TorchXRayVision mimic_ch backbone: never saw CheXpert, and is not
the backbone used as reward during training) + the PSPNet anatomy checker.

Per class:
  requested_finding_rate  eval-head predicts the prompted class (6-way)
  normalised_cond         requested_finding_rate / eval-head recall on REAL test images of that class
  cond_auroc              one-vs-rest AUROC of eval-head prob for class c (generated c vs generated others)
  xrv_prob_gen / _real    mimic_ch's own calibrated probability of the prompted pathology (4 covered classes)
  anatomy_pass, ood_rate, defect_rate (= anatomy fail OR OOD)
  kid (x1e3)              KID vs real train images of the same class (eval-backbone features)
  diversity               mean pairwise cosine distance (gen) | diversity_real for the same class
  nn_dist_median          nearest-real-train-neighbour cosine distance (memorisation if ~0)
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cxr_common import *  # noqa

XRV_NAME = {"Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum", "Lung Lesion": "Lung Lesion",
            "Pneumonia": "Pneumonia", "Fracture": "Fracture"}       # Pleural Other / No Finding not covered


class SynDS(Dataset):
    def __init__(self, root, df):
        self.root, self.df = root, df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return load_image_tensor(os.path.join(self.root, self.df.Path[i]))


def kid(x, y, n_sub=50, sub=200, rng=np.random.RandomState(0)):
    d = x.shape[1]; m = min(sub, len(x), len(y)); vals = []
    for _ in range(n_sub):
        a = x[rng.choice(len(x), m, replace=False)]; b = y[rng.choice(len(y), m, replace=False)]
        kxx = (a @ a.T / d + 1) ** 3; kyy = (b @ b.T / d + 1) ** 3; kxy = (a @ b.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def cos_np(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="folder containing synthetic_labels.csv")
    p.add_argument("--name", required=True)
    p.add_argument("--max_per_class", type=int, default=500)
    p.add_argument("--from_folders", action="store_true", help="derive labels from <root>/<Class_Name>/*.png")
    a = p.parse_args()
    dev = "cuda"
    out_dir = os.path.join(PROJECT, "novelty", "results"); os.makedirs(out_dir, exist_ok=True)

    if a.from_folders:
        rows = [{"Path": f"{d}/{f}", "Label": d.replace("_", " ")} for d in sorted(os.listdir(a.root))
                if d.replace("_", " ") in CLASS_TO_IDX for f in sorted(os.listdir(os.path.join(a.root, d)))
                if f.endswith(".png")]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(os.path.join(a.root, "synthetic_labels.csv"))
    df = df.groupby("Label").head(a.max_per_class).reset_index(drop=True)
    y = df.Label.map(CLASS_TO_IDX).values

    import torchxrayvision as xrv
    bb = xrv.models.DenseNet(weights=EVAL_BACKBONE).to(dev).eval()
    head = load_linear_head(os.path.join(HEADS_DIR, "eval_head.pt"), dev)
    o = torch.load(os.path.join(HEADS_DIR, "ood_stats.pt"))
    mean, prec, thr = o["mean"].float().to(dev), o["prec"].float().to(dev), o["thr"]
    anat = AnatomyChecker(dev, json.load(open(os.path.join(HEADS_DIR, "anatomy.json"))))
    rep = json.load(open(os.path.join(HEADS_DIR, "heads_report.json")))

    feats, probs, xrvp, ood, ok = [], [], [], [], []
    with torch.no_grad():
        for x in DataLoader(SynDS(a.root, df), batch_size=64, num_workers=8):
            x = x.to(dev); xx = to_xrv(x)
            f = backbone_features(bb, xx)
            feats.append(f.cpu()); probs.append(torch.softmax(head(f), 1).cpu())
            xrvp.append(bb(xx).cpu())
            d = f - mean; ood.append((torch.sqrt(((d @ prec) * d).sum(1)) > thr).cpu())
            ok.append(anat.passes(anat.measure(x)))
    F_ = torch.cat(feats).numpy(); P = torch.cat(probs).numpy(); XP = torch.cat(xrvp).numpy()
    OOD = torch.cat(ood).numpy(); OK = np.concatenate(ok)

    # real references
    rt = torch.load(os.path.join(HEADS_DIR, "real_test_feats.pt"))
    rtr = torch.load(os.path.join(HEADS_DIR, "real_train_feats.pt"))
    Ftr, ytr = rtr["feats"].numpy(), rtr["y"].numpy()
    # mimic_ch pathology probability on real test images (computed from cached features)
    with torch.no_grad():
        real_test_xrv = bb.classifier(rt["feats"].to(dev))
        real_test_xrv = xrv.models.op_norm(torch.sigmoid(real_test_xrv), bb.op_threshs).cpu().numpy()
    yte = rt["y"].numpy()

    res = {}
    for c in sorted(set(df.Label), key=CLASSES.index):
        ci = CLASS_TO_IDX[c]; m = y == ci
        r = {"n": int(m.sum())}
        r["requested_finding_rate"] = float((P[m].argmax(1) == ci).mean())
        real_rec = rep["eval"]["test_recall_per_class"][c]
        r["eval_real_recall"] = real_rec
        r["normalised_cond"] = r["requested_finding_rate"] / real_rec if real_rec > 0 else None
        if len(set(y)) > 1:
            r["cond_auroc"] = float(roc_auc_score(m.astype(int), P[:, ci]))
        if c in XRV_NAME:
            j = bb.pathologies.index(XRV_NAME[c])
            r["xrv_prob_gen"] = float(XP[m, j].mean())
            r["xrv_prob_real"] = float(real_test_xrv[yte == ci, j].mean())
            r["xrv_prob_real_other"] = float(real_test_xrv[yte != ci, j].mean())
        r["anatomy_pass"] = float(OK[m].mean()); r["ood_rate"] = float(OOD[m].mean())
        r["defect_rate"] = float((~OK[m] | OOD[m]).mean())
        real_c = Ftr[ytr == ci]
        k, ks = kid(F_[m], real_c); r["kid_x1e3"] = k * 1e3; r["kid_std_x1e3"] = ks * 1e3
        g = cos_np(F_[m]); rr = cos_np(real_c)
        sel = np.random.RandomState(0).choice(len(rr), min(500, len(rr)), replace=False)
        r["diversity"] = float(1 - (g @ g.T)[np.triu_indices(len(g), 1)].mean())
        r["diversity_real"] = float(1 - (rr[sel] @ rr[sel].T)[np.triu_indices(len(sel), 1)].mean())
        r["nn_dist_median"] = float(np.median(1 - (g @ rr.T).max(1)))
        res[c] = r
    tg = [res[c] for c in TARGET_CLASSES if c in res]
    res["MEAN(targets)"] = {k: float(np.mean([t[k] for t in tg if t.get(k) is not None]))
                            for k in tg[0] if k != "n" and all(isinstance(t.get(k), (int, float)) for t in tg)}
    res["confusion_matrix"] = {"rows_prompted": CLASSES,
                               "matrix": confusion_matrix(y, P.argmax(1), labels=list(range(len(CLASSES)))).tolist()}

    json.dump(res, open(os.path.join(out_dir, f"eval_{a.name}.json"), "w"), indent=1)
    cols = ["n", "requested_finding_rate", "normalised_cond", "cond_auroc", "xrv_prob_gen", "xrv_prob_real",
            "anatomy_pass", "ood_rate", "defect_rate", "kid_x1e3", "diversity", "diversity_real", "nn_dist_median"]
    tab = pd.DataFrame({c: res[c] for c in res if c != "confusion_matrix"}).T.reindex(columns=cols)
    tab.to_csv(os.path.join(out_dir, f"eval_{a.name}.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(f"\n=== {a.name} ===\n", tab)


if __name__ == "__main__":
    main()
