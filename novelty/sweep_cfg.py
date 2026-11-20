"""Fix #2: guidance-scale x LoRA-scale sweep for a trained LoRA (reduce 'textbook' exaggeration, raise diversity).

Fixed seed grid, 50 DDIM steps (same as the final generation), N images per target class, held-out evaluator.
Selection rule (fixed BEFORE looking at results):
    lowest mean per-class KID vs real images, among configs with
    requested_finding_rate >= 0.30 (about 2x the baseline's 0.157) and defect_rate <= 0.30.
"""
import os, sys, json, argparse, itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cxr_common import *  # noqa
from sd_utils import load_sd, attach_lora, sample, set_lora_scale
from evaluate_synth import kid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_dir", default=os.path.join(PROJECT, "novelty/runs/ce_adv/step_1000"))
    p.add_argument("--name", default="ce_adv")
    p.add_argument("--cfgs", default="7.5,5,4")
    p.add_argument("--scales", default="1.0,0.75")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--bs", type=int, default=8)
    a = p.parse_args()
    dev = "cuda"

    pipe, ddim, _ = load_sd("prajiv")
    pipe = attach_lora(pipe, a.lora_dir, "unet_clip", trainable=False)
    pipe.unet.eval(); pipe.text_encoder.eval()

    bb = load_backbone(EVAL_BACKBONE, dev)
    head = load_linear_head(os.path.join(HEADS_DIR, "eval_head.pt"), dev)
    o = torch.load(os.path.join(HEADS_DIR, "ood_stats.pt"))
    mean, prec, thr = o["mean"].float().to(dev), o["prec"].float().to(dev), o["thr"]
    anat = AnatomyChecker(dev, json.load(open(os.path.join(HEADS_DIR, "anatomy.json"))))
    rtr = torch.load(os.path.join(HEADS_DIR, "real_train_feats.pt"))
    Ftr, ytr = rtr["feats"].numpy(), rtr["y"].numpy()

    out_f = os.path.join(PROJECT, "novelty", "results", f"sweep_{a.name}.json")
    results = json.load(open(out_f)) if os.path.exists(out_f) else {}
    for cfg, sc in itertools.product([float(x) for x in a.cfgs.split(",")], [float(x) for x in a.scales.split(",")]):
        key = f"cfg{cfg}_lora{sc}"
        if key in results:
            continue
        set_lora_scale(pipe, sc)
        per = {}
        for c in TARGET_CLASSES:
            ci = CLASS_TO_IDX[c]
            seeds = [7_000_000 + 1000 * ci + i for i in range(a.n)]     # same seeds for every config
            imgs = torch.cat([sample(pipe, ddim, [prompt_for(c)] * len(seeds[i:i + a.bs]), seeds[i:i + a.bs],
                                     50, cfg, return_tensor=True) for i in range(0, a.n, a.bs)])
            with torch.no_grad():
                f = backbone_features(bb, to_xrv(imgs))
                pred = head(f).argmax(1)
                d = f - mean
                ood = (torch.sqrt(((d @ prec) * d).sum(1)) > thr).cpu().numpy()
            ok = anat.passes(anat.measure(imgs))
            F = f.cpu().numpy(); g = F / np.linalg.norm(F, axis=1, keepdims=True)
            per[c] = {"requested_finding_rate": pred.eq(ci).float().mean().item(),
                      "defect_rate": float((~ok | ood).mean()),
                      "kid_x1e3": kid(F, Ftr[ytr == ci], sub=a.n)[0] * 1e3,
                      "diversity": float(1 - (g @ g.T)[np.triu_indices(len(g), 1)].mean())}
        m = {k: float(np.mean([per[c][k] for c in TARGET_CLASSES])) for k in per[TARGET_CLASSES[0]]}
        results[key] = {"cfg": cfg, "lora_scale": sc, "mean": m, "per_class": per}
        json.dump(results, open(out_f, "w"), indent=1)
        print(f"{key}: find={m['requested_finding_rate']:.3f} defect={m['defect_rate']:.3f} "
              f"kid={m['kid_x1e3']:.2f} div={m['diversity']:.3f}", flush=True)

    ok = {k: v for k, v in results.items()
          if v["mean"]["requested_finding_rate"] >= 0.30 and v["mean"]["defect_rate"] <= 0.30}
    pool = ok if ok else results
    best = min(pool, key=lambda k: pool[k]["mean"]["kid_x1e3"])
    sel = {"selected": best, "cfg": results[best]["cfg"], "lora_scale": results[best]["lora_scale"],
           "constraints_met": bool(ok), "rule": "min KID s.t. find>=0.30 and defect<=0.30"}
    json.dump(sel, open(os.path.join(PROJECT, "novelty", "results", f"sweep_{a.name}_selected.json"), "w"), indent=1)
    print("SELECTED", sel)


if __name__ == "__main__":
    main()
