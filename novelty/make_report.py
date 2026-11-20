"""Collect everything into poster/report tables + figures (novelty/report/)."""
import os, sys, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cxr_common import PROJECT, CLASSES, TARGET_CLASSES, CSV_TEST, DATA_DIR, prompt_for  # noqa

N = os.path.join(PROJECT, "novelty")
OUT = os.path.join(N, "report"); os.makedirs(OUT, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker, matplotlib.lines  # noqa

# one fixed colour per generator (order of GEN), reused in every figure
PALETTE = ["#7f7f7f", "#bcbd22", "#ff7f0e", "#17becf", "#9467bd", "#d62728", "#e377c2", "#1f77b4", "#8c564b", "#2ca02c"]
plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})


def wrap(c):
    return {"Enlarged Cardiomediastinum": "Enlarged\nCardiomediastinum", "Pleural Other": "Pleural\nOther",
            "Lung Lesion": "Lung\nLesion"}.get(c, c)


def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# name in results/  ->  label on the poster
GEN = {"base_prajiv": "P-Rajiv LoRA (current)", "control": "+ extra training (control)",
       "ce_only": "+ CE loss", "adv_only": "+ adversarial loss", "adv_only_l02": "+ adversarial (λ=0.2)",
       "ce_adv": "+ CE + adversarial", "ce_adv_swept": "+ CE + adv (tuned guidance)",
       "margin_adv": "+ matched-confidence margin + adv", "margin_adv_ms": "+ margin + multi-scale adv", "ce_adv_i2i": "+ CE + adv, img2img from real (DA-Fusion)"}
DS = {"real_only": "Real only", "base_prajiv": "Real + P-Rajiv (current)", "control": "Real + control",
      "ce_only": "Real + CE", "adv_only": "Real + Adv", "adv_only_l02": "Real + Adv (λ=0.2)",
      "ce_adv": "Real + CE+Adv", "ce_adv_jpg": "Real + CE+Adv [#1 CheXpert-style + NF]",
      "ce_adv_swept": "Real + CE+Adv [#2 tuned guidance]",
      "ce_adv_swept_jpg": "Real + CE+Adv [#1 + #2]",
      "margin_adv": "Real + Margin+Adv", "margin_adv_ms": "Real + Margin+Adv multi-scale D", "ce_adv_i2i": "Real + CE+Adv img2img (DA-Fusion)",
      "ce_adv_recipe": "Real + CE+Adv [recipe]", "ce_adv_recipe_filt": "Real + CE+Adv [recipe + filter]",
      "ce_adv_i2i_recipe": "Real + DA-Fusion [recipe]", "ce_adv_i2i_recipe_filt": "Real + DA-Fusion [recipe + filter]",
      "margin_adv_recipe": "Real + Margin+Adv [recipe]", "ce_adv_swept_recipe": "Real + CE+Adv tuned guidance [recipe]",
      "margin_adv_ms_recipe": "Real + Margin+Adv multi-scale D [recipe]"}


def gen_tables():
    rows = []
    for k, lab in GEN.items():
        f = os.path.join(N, "results", f"eval_{k}.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        for c in TARGET_CLASSES + ["MEAN(targets)"]:
            if c in r:
                rows.append({"model": lab, "pathology": c, **{m: r[c].get(m) for m in
                             ["requested_finding_rate", "normalised_cond", "cond_auroc", "defect_rate",
                              "anatomy_pass", "ood_rate", "kid_x1e3", "diversity", "nn_dist_median"]}})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "generation_per_pathology.csv"), index=False)
    for m in ["requested_finding_rate", "defect_rate", "kid_x1e3", "normalised_cond"]:
        piv = df.pivot(index="pathology", columns="model", values=m).reindex(TARGET_CLASSES + ["MEAN(targets)"])
        piv = piv[[v for v in GEN.values() if v in piv.columns]]
        piv.to_csv(os.path.join(OUT, f"gen_{m}.csv"))
        with open(os.path.join(OUT, "tables.md"), "a") as fh:
            fh.write(f"\n### {m}\n\n" + piv.round(3).to_markdown() + "\n")
    # bar chart: requested finding vs defect per pathology (stacked panels, one shared legend BELOW the plots)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    for i, m in enumerate(["requested_finding_rate", "defect_rate"]):
        piv = df.pivot(index="pathology", columns="model", values=m).reindex(TARGET_CLASSES)
        piv = piv[[v for v in GEN.values() if v in piv.columns]]
        piv.index = [wrap(c) for c in piv.index]
        piv.plot.bar(ax=ax[i], rot=0, width=0.85, legend=False, color=PALETTE[:piv.shape[1]], edgecolor="white", lw=0.3)
        ax[i].set_title({"requested_finding_rate": "Shows the requested finding (higher is better)",
                         "defect_rate": "Structural defect rate (lower is better)"}[m], fontsize=12, loc="left")
        ax[i].set_xlabel(""); ax[i].set_ylim(0, 1); ax[i].grid(axis="y", alpha=0.3); ax[i].set_axisbelow(True)
        ax[i].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        style(ax[i])
    ax[1].tick_params(axis="x", labelsize=11)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_generation_per_pathology.png"), dpi=200); plt.close(fig)


def ckpt_curves():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    gkeys = list(GEN)
    for k in ["control", "ce_only", "adv_only", "adv_only_l02", "ce_adv", "margin_adv", "margin_adv_ms"]:
        f = os.path.join(N, "runs", k, "eval_log.json")
        if not os.path.exists(f):
            continue
        L = json.load(open(f))
        s = [r["step"] for r in L]
        col = PALETTE[gkeys.index(k)]          # same colour per model as in the generation figure
        ax[0].plot(s, [r["mean_target"]["cond_acc"] for r in L], "o-", color=col, label=GEN[k], lw=2)
        ax[1].plot(s, [r["mean_target"]["defect_rate"] for r in L], "o-", color=col, label=GEN[k], lw=2)
        b = os.path.join(N, "runs", k, "best.json")
        if os.path.exists(b):                  # mark the selected checkpoint
            st = json.load(open(b)).get("best_step")
            for i, key in enumerate(["cond_acc", "defect_rate"]):
                y = [r["mean_target"][key] for r in L if r["step"] == st]
                if y:
                    ax[i].scatter([st], y, s=160, facecolors="none", edgecolors=col, lw=2, zorder=5)
    ax[0].set_title("Held-out requested-finding rate (higher is better)", fontsize=11, loc="left")
    ax[1].set_title("Held-out defect rate (lower is better)", fontsize=11, loc="left")
    for a in ax:
        a.set_xlabel("LoRA training step"); a.set_ylim(0, 1); a.grid(alpha=0.3); style(a)
        a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    h, l = ax[0].get_legend_handles_labels()
    h.append(matplotlib.lines.Line2D([], [], marker="o", ls="", mfc="none", mec="k", ms=11, mew=1.5))
    l.append("selected checkpoint")
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_checkpoint_selection.png"), dpi=200); plt.close(fig)


def downstream():
    rows = []
    for k, lab in DS.items():
        for f in sorted(glob.glob(os.path.join(N, "downstream", f"{k}_seed*", "log.json"))):
            L = json.load(open(f))
            if len(L["history"]) < 10:          # only FINISHED runs (log.json is rewritten every epoch)
                print("skipping unfinished run:", f)
                continue
            b = L["selected_by_val"]
            rec = {"setting": lab, "seed": f.split("seed")[-1].split("/")[0], "macro_f1": b["test_f1"],
                   "acc": b["test_acc"], "auroc": b["test_auroc"]}
            for c, v in b["test_per_class"].items():
                rec[c] = v["f1-score"]
            rows.append(rec)
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "downstream_all_seeds.csv"), index=False)
    cols = ["macro_f1", "acc", "auroc"] + [c for c in df.columns if c in CLASSES or c in
                                           ["Lung Opacity", "Support Devices", "Pleural Effusion", "Edema",
                                            "Pneumothorax", "Atelectasis", "Cardiomegaly", "Consolidation"]]
    g = df.groupby("setting")[cols]
    mu, sd = g.mean().round(3), g.std().round(3)
    summ = pd.DataFrame({c: [f"{m:.3f} ± {s:.3f}" for m, s in zip(mu[c], sd[c])] for c in cols}, index=mu.index)
    summ["n_seeds"] = g.size()
    summ = summ.T[[v for v in DS.values() if v in summ.index]]
    summ.to_csv(os.path.join(OUT, "downstream_summary.csv"))
    mean = g.mean().T[[v for v in DS.values() if v in g.mean().index]]
    if "Real only" in mean.columns:
        delta = mean.drop(columns="Real only").sub(mean["Real only"], axis=0).round(3)
        delta.to_csv(os.path.join(OUT, "downstream_delta_vs_real.csv"))
        with open(os.path.join(OUT, "tables.md"), "a") as fh:
            fh.write("\n### Downstream test F1 (mean ± std over seeds)\n\n" + summ.to_markdown() +
                     "\n\n### Δ vs real-only (mean)\n\n" + delta.to_markdown() + "\n")
        delta_heatmap(delta, mean)
        delta_bars_key(delta)


def delta_heatmap(delta, mean):
    """All settings x (5 synthesised classes + macro-F1): annotated ΔF1 heat-map -- no legend needed."""
    cols = TARGET_CLASSES + ["macro_f1"]
    H = delta.reindex(cols).T                                   # rows = settings
    H = H.loc[H["macro_f1"].sort_values(ascending=False).index]  # best macro-F1 on top
    lim = float(np.nanmax(np.abs(H.values)))
    fig, ax = plt.subplots(figsize=(12.5, 0.42 * len(H) + 2.4))
    im = ax.imshow(H.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            v = H.values[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.6 * lim else "black")
    ax.set_xticks(range(len(cols)))
    ro = mean["Real only"]
    ax.set_xticklabels([f"{wrap(c)}\n(real-only {ro[c]:.2f})" for c in TARGET_CLASSES]
                       + [f"Macro-F1\n(14 classes,\nreal-only {ro['macro_f1']:.3f})"], fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(H))); ax.set_yticklabels([s.replace("Real + ", "+ ") for s in H.index], fontsize=10)
    ax.axvline(len(cols) - 1.5, color="k", lw=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("ΔF1 vs real-only")
    ax.set_title("Test F1 change vs real-only training (mean of 3 seeds; blue = better, red = worse)",
                 fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_downstream_delta.png"), dpi=200); plt.close(fig)


def delta_bars_key(delta):
    """Poster version: only the headline settings, grouped bars, legend below."""
    key = ["Real + P-Rajiv (current)", "Real + CE+Adv", "Real + CE+Adv [recipe]",
           "Real + Margin+Adv [recipe]", "Real + DA-Fusion [recipe + filter]"]
    key = [k for k in key if k in delta.columns]
    D = delta.reindex(TARGET_CLASSES + ["macro_f1"])[key]
    D.index = [wrap(c) for c in TARGET_CLASSES] + ["Macro-F1\n(14 classes)"]
    fig, ax = plt.subplots(figsize=(11, 5))
    D.plot.bar(ax=ax, rot=0, width=0.8, legend=False, color=["#9e9e9e", "#8c564b", "#d62728", "#1f77b4", "#2ca02c"],
               edgecolor="white", lw=0.3)
    ax.axhline(0, color="k", lw=1); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True); style(ax)
    ax.axvline(len(D) - 1.5, color="grey", lw=1, ls="--")
    ax.set_ylabel("ΔF1 vs real-only (mean of 3 seeds)")
    ax.set_title("Downstream DenseNet: per-pathology F1 change from adding synthetic images", fontsize=11, loc="left")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_downstream_delta_key.png"), dpi=200); plt.close(fig)


SAMPLE_ROWS = [("base_prajiv", "P-Rajiv LoRA\n(current)"), ("control", "+ extra training\n(control)"),
               ("ce_only", "+ CE loss"), ("adv_only", "+ adversarial\nloss"), ("ce_adv", "+ CE +\nadversarial"),
               ("margin_adv", "+ margin +\nadversarial"), ("ce_adv_i2i", "DA-Fusion\nimg2img*"), ("REAL", "Real X-rays\n(test set)")]


def _sample_axes(axrow, files, label, fs=10):
    from PIL import Image
    for a, f in zip(axrow, files):
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        if f and os.path.exists(f):
            a.imshow(Image.open(f).convert("L").resize((256, 256)), cmap="gray", vmin=0, vmax=255)
    axrow[0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=fs, labelpad=8)


def _files(k, c, n):
    root = os.path.join(PROJECT, "data", "CheXpert-v1.0-small")
    if k == "REAL":
        t = pd.read_csv(CSV_TEST)
        return [os.path.join(DATA_DIR, p) for p in t[t[c] == 1].Path.head(n)]
    d = os.path.join(root, f"gen_{k}", c.replace(" ", "_"))
    return [os.path.join(d, f"synthetic_{c.replace(' ', '_')}_{j:04d}.png") for j in range(n)]


def sample_grid(n=6):
    """One labelled figure per pathology (rows = generators, same seeds) + one overview figure."""
    rows = [(k, l) for k, l in SAMPLE_ROWS
            if k == "REAL" or os.path.isdir(os.path.join(PROJECT, "data", "CheXpert-v1.0-small", f"gen_{k}"))]
    note = ("Columns = the same 6 noise seeds for every text-to-image model.  "
            "*DA-Fusion starts from a real training X-ray (not the same seeds).")
    for c in TARGET_CLASSES:
        fig, ax = plt.subplots(len(rows), n, figsize=(1.75 * n + 1.8, 1.75 * len(rows) + 0.9))
        for r, (k, l) in enumerate(rows):
            _sample_axes(ax[r], _files(k, c, n), l)
        fig.suptitle(f"Prompt: \"{prompt_for(c)}\"", fontsize=12, x=0.02, ha="left")
        fig.text(0.02, 0.005, note, fontsize=8, color="0.3")
        fig.subplots_adjust(left=0.15, right=0.995, top=0.95, bottom=0.03, wspace=0.03, hspace=0.05)
        fig.savefig(os.path.join(OUT, f"fig_samples_{c.replace(' ', '_')}.png"), dpi=150); plt.close(fig)
    # overview: 2 images per pathology, all pathologies side by side
    m = 2
    fig, ax = plt.subplots(len(rows), m * len(TARGET_CLASSES), figsize=(1.45 * m * len(TARGET_CLASSES) + 1.8,
                                                                       1.45 * len(rows) + 1.0))
    for r, (k, l) in enumerate(rows):
        files = sum((_files(k, c, m) for c in TARGET_CLASSES), [])
        _sample_axes(ax[r], files, l, fs=9)
    for ci, c in enumerate(TARGET_CLASSES):
        ax[0, ci * m].set_title(wrap(c), fontsize=10, loc="left")
    fig.text(0.02, 0.005, note.replace("6 noise seeds", "2 noise seeds per pathology"), fontsize=8, color="0.3")
    fig.subplots_adjust(left=0.11, right=0.995, top=0.93, bottom=0.03, wspace=0.03, hspace=0.05)
    fig.savefig(os.path.join(OUT, "fig_samples_same_seeds.png"), dpi=150); plt.close(fig)
    if os.path.exists(os.path.join(OUT, "fig_samples_same_seeds.txt")):
        os.remove(os.path.join(OUT, "fig_samples_same_seeds.txt"))


if __name__ == "__main__":
    if os.path.exists(os.path.join(OUT, "tables.md")):
        os.remove(os.path.join(OUT, "tables.md"))
    gen_tables(); ckpt_curves(); downstream(); sample_grid()
    print(open(os.path.join(OUT, "tables.md")).read() if os.path.exists(os.path.join(OUT, "tables.md")) else "no tables yet")
