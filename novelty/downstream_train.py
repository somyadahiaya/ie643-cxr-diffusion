"""Downstream DenseNet121 (14-class) with / without synthetic augmentation.

Re-uses the team's classifier_train.py (same dataset class, model, augmentation, loss, optimiser) with two changes:
  1. model selection uses a patient-level VALIDATION split of the REAL training rows, never the test set
     (the original script stepped the LR scheduler and picked epochs on the test set);
  2. more dataloader workers.
The test set (global_test_single_frontal_clean.csv, 50 real images / class) is only evaluated, and the
reported numbers are from the epoch with the best validation macro-F1.
"""
import os, sys, json, argparse, tempfile
# Private temp dir: a /tmp cleaner on this shared server deleted the DataLoader's /tmp/pymp-* sockets mid-run
# ("Pin memory thread exited unexpectedly"). Keep multiprocessing temp files inside the project instead.
_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")
os.makedirs(_TMP, exist_ok=True)
os.environ["TMPDIR"] = _TMP
tempfile.tempdir = _TMP
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cxr_common import CODE_DIR, CSV_TEST, DATA_DIR, PROJECT  # noqa
sys.path.insert(0, CODE_DIR)
import classifier_train as ct  # the team's script


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=12)
    # synthetic-data recipe (defaults = original behaviour)
    p.add_argument("--class_weight_from", default="all", choices=["all", "real"],
                   help="real: class weights from REAL counts only, so synthetic images don't dilute real rare-class weight")
    p.add_argument("--syn_weight", type=float, default=1.0, help="per-sample loss weight of synthetic images")
    p.add_argument("--real_ft_epochs", type=int, default=0,
                   help="two-stage: the LAST k of --epochs are trained on real images only")
    p.add_argument("--save_model", action="store_true",
                   help="save the weights of the validation-selected epoch to <out>/best_model.pt (used by the demo)")
    a = p.parse_args()
    ct.set_seed(a.seed)
    dev = "cuda"
    out = os.path.join(PROJECT, "novelty", "downstream", f"{a.name}_seed{a.seed}")
    os.makedirs(out, exist_ok=True)

    df = pd.read_csv(a.train_csv)
    syn = df.Path.str.contains("synthetic")
    real = df[~syn].copy(); real["pid"] = real.Path.str.split("/").str[2]
    rng = np.random.RandomState(0)                       # same val split for every run / seed
    pids = np.sort(real.pid.unique())
    val_p = set(rng.choice(pids, int(0.1 * len(pids)), replace=False))
    tr = pd.concat([real[~real.pid.isin(val_p)], df[syn]]).drop(columns="pid")
    va = real[real.pid.isin(val_p)].drop(columns="pid")
    tr_csv, va_csv = os.path.join(out, "train.csv"), os.path.join(out, "val.csv")
    tr.to_csv(tr_csv, index=False); va.to_csv(va_csv, index=False)
    print(f"train {len(tr)} (synthetic {int(syn.sum())}) | val {len(va)}")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    t_tr = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                               transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), norm])
    t_te = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    class SynFlagDS(ct.CheXpertClassifierDataset):          # also returns 1 for synthetic rows
        def __getitem__(self, i):
            x, y = super().__getitem__(i)
            return x, y, float("synthetic" in self.df.Path.iloc[i])

    ds_tr = SynFlagDS(tr_csv, DATA_DIR, t_tr)
    real_tr_csv = os.path.join(out, "train_real_only.csv")
    tr[~tr.Path.str.contains("synthetic")].to_csv(real_tr_csv, index=False)
    mk = lambda ds, sh: DataLoader(ds, batch_size=a.batch_size, shuffle=sh, num_workers=a.workers, pin_memory=True)
    dl_tr = mk(ds_tr, True)
    dl_tr_real = mk(SynFlagDS(real_tr_csv, DATA_DIR, t_tr), True)
    dl_va = mk(ct.CheXpertClassifierDataset(va_csv, DATA_DIR, t_te), False)
    dl_te = mk(ct.CheXpertClassifierDataset(CSV_TEST, DATA_DIR, t_te), False)

    model = ct.get_densenet121_classifier(14).to(dev)
    cw_df = ds_tr.df if a.class_weight_from == "all" else ds_tr.df[~ds_tr.df.Path.str.contains("synthetic")]
    counts = cw_df[ds_tr.target_classes].sum(0).values
    w = torch.tensor(np.sqrt(counts.sum() / (14 * (counts + 1e-5))), dtype=torch.float, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)                         # for evaluation
    crit_each = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1, reduction="none")  # for training
    opt = optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2, factor=0.5)
    json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=1)

    hist, best = [], None
    for ep in range(a.epochs):
        model.train(); tl = 0
        real_phase = ep >= a.epochs - a.real_ft_epochs
        loader = dl_tr_real if real_phase else dl_tr
        for x, y, s in tqdm(loader, desc=f"ep{ep + 1}{' [real-only]' if real_phase else ''}", mininterval=30):
            x, y, s = x.to(dev), y.to(dev), s.to(dev).float()
            sw = 1.0 - s * (1.0 - a.syn_weight)                      # 1 for real, syn_weight for synthetic
            opt.zero_grad()
            l = crit_each(model(x), y)
            loss = (l * sw).sum() / (w[y] * sw).sum()               # weighted mean, same scale as the default
            loss.backward(); opt.step(); tl += loss.item()
        v = ct.evaluate(model, dl_va, crit, dev)
        t = ct.evaluate(model, dl_te, crit, dev)
        sched.step(v[0])
        rec = {"epoch": ep + 1, "train_loss": tl / len(loader), "real_only_epoch": real_phase, "val_loss": v[0], "val_f1": v[4], "val_acc": v[1],
               "test_loss": t[0], "test_acc": t[1], "test_f1": t[4], "test_auroc": t[5],
               "test_per_class": {c: t[6][c] for c in ds_tr.target_classes}, "test_confusion": t[7].tolist()}
        hist.append(rec)
        print(f"ep{ep + 1}: val_f1={v[4]:.3f} test_f1={t[4]:.3f} test_auroc={t[5]:.3f}", flush=True)
        if best is None or v[4] > best["val_f1"]:
            best = rec
            if a.save_model:
                torch.save({"state_dict": model.state_dict(), "classes": list(ds_tr.target_classes),
                            "epoch": ep + 1, "val_f1": v[4], "test_f1": t[4], "config": vars(a)},
                           os.path.join(out, "best_model.pt"))
        json.dump({"history": hist, "selected_by_val": best}, open(os.path.join(out, "log.json"), "w"), indent=1)
    print("SELECTED (by val macro-F1): epoch", best["epoch"], "test_f1", round(best["test_f1"], 4))


if __name__ == "__main__":
    main()
