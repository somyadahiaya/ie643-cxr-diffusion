"""Interactive demo: text prompt -> synthetic chest X-ray -> classification + structural-defect check.

Uses exactly the checkpoints evaluated in the report:
  * generators: the phase-1 P-Rajiv LoRA (epoch_2) and every novelty LoRA at the step chosen by the held-out
    checkpoint selection (novelty/runs/<run>/best.json -> step_<best_step>), same as pipeline_after_train.sh;
    sampler = 50 DDIM steps, CFG 7.5 (report settings, adjustable).
  * held-out judge: TorchXRayVision densenet121-res224-mimic_ch + the 6-class linear probe (heads/eval_head.pt),
    Mahalanobis OOD detector (heads/ood_stats.pt), PSPNet anatomy check (heads/anatomy.json).
  * downstream 14-class DenseNet-121: real-only and real + CE+Adv synthetic [recipe], seed 42, epoch chosen on
    validation (novelty/downstream/demo_*_seed42/best_model.pt, written by downstream_train.py --save_model).

Run:  CUDA_VISIBLE_DEVICES=0 .venv311/bin/python novelty/demo_app.py  [--port 7860] [--share]
"""
import os, sys, json, argparse, threading
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cxr_common import (PROJECT, DATA_DIR, CODE_DIR, CSV_TEST, HEADS_DIR, CLASSES, EVAL_BACKBONE,  # noqa
                        prompt_for, to_xrv, backbone_features, load_linear_head, AnatomyChecker)
from sd_utils import load_sd, sample

NOVELTY = os.path.join(PROJECT, "novelty")
RUNS = os.path.join(NOVELTY, "runs")
BASELINE_LORA = os.path.join(CODE_DIR, "cxr_diffusion_checkpoints", "prajiv", "unet_clip", "epoch_2")


def _best(run):
    return json.load(open(os.path.join(RUNS, run, "best.json")))["best_step"]


# key, UI label, run name (None = phase-1 baseline), report numbers: requested finding %, defects %, KID x1e3,
# downstream macro-F1 (plain augmentation, mean of 3 seeds)
GENERATORS = [
    ("baseline", "P-Rajiv LoRA (baseline, phase 1)", None, 15.7, 32.7, 12.8, 0.256),
    ("control", "Control (+2k steps, no new loss)", "control", 23.6, 51.8, 10.3, 0.275),
    ("ce_only", "CE only (reward-hacks)", "ce_only", 35.3, 78.5, 29.9, 0.258),
    ("adv_only", "Adversarial only", "adv_only", 25.9, 9.8, 2.9, 0.270),
    ("ce_adv", "CE + adversarial", "ce_adv", 45.9, 24.6, 8.2, 0.269),
    ("margin_adv", "Margin + adversarial (best)", "margin_adv", 51.2, 18.2, 7.8, 0.278),
    ("margin_adv_ms", "Margin + adv, multi-scale D", "margin_adv_ms", 43.8, 16.5, 8.7, 0.264),
]
LABEL2KEY = {g[1]: g[0] for g in GENERATORS}
KEY2LABEL = {g[0]: g[1] for g in GENERATORS}


def lora_path(run):
    return BASELINE_LORA if run is None else os.path.join(RUNS, run, f"step_{_best(run)}")


CLASSIFIERS = [  # key, UI label, folder in novelty/downstream
    ("real", "DenseNet-121, real only", "demo_real_only_seed42"),
    ("syn", "DenseNet-121, real + CE+Adv synthetic [recipe]", "demo_ce_adv_recipe_seed42"),
]
DS_CLASSES = ["No Finding", "Lung Opacity", "Support Devices", "Pleural Effusion", "Edema", "Pneumothorax",
              "Atelectasis", "Cardiomegaly", "Fracture", "Consolidation", "Enlarged Cardiomediastinum",
              "Lung Lesion", "Pneumonia", "Pleural Other"]


def requested_class(prompt):
    p = prompt.lower()
    for c in CLASSES[1:]:
        if c.lower() in p:
            return c
    if "no finding" in p or "normal" in p:
        return "No Finding"
    return None


class Engine:
    def __init__(self, device="cuda"):
        from peft import PeftModel
        import torchxrayvision as xrv
        from torchvision import transforms, models
        self.dev = device
        self.lock = threading.Lock()
        print("[demo] loading P-Rajiv SD-1.5 + LoRA adapters ...", flush=True)
        self.pipe, self.ddim, _ = load_sd("prajiv", device=device)
        for i, (key, _, run, *_r) in enumerate(GENERATORS):
            p = lora_path(run)
            print(f"[demo]   {key:14s} <- {os.path.relpath(p, PROJECT)}", flush=True)
            u, t = os.path.join(p, "unet_lora"), os.path.join(p, "text_encoder_lora")
            if i == 0:
                self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, u, adapter_name=key)
                self.pipe.text_encoder = PeftModel.from_pretrained(self.pipe.text_encoder, t, adapter_name=key)
            else:
                self.pipe.unet.load_adapter(u, adapter_name=key)
                self.pipe.text_encoder.load_adapter(t, adapter_name=key)
        for m in (self.pipe.unet, self.pipe.text_encoder):
            m.eval()
            for n, prm in m.named_parameters():
                if "lora_" in n:
                    prm.data = prm.data.float()
                prm.requires_grad_(False)
        self.pipe.to(device)

        print("[demo] loading held-out judge, OOD detector, anatomy checker ...", flush=True)
        self.bb = xrv.models.DenseNet(weights=EVAL_BACKBONE).to(device).eval()
        self.head = load_linear_head(os.path.join(HEADS_DIR, "eval_head.pt"), device)
        o = torch.load(os.path.join(HEADS_DIR, "ood_stats.pt"))
        self.ood_mean, self.ood_prec, self.ood_thr = o["mean"].float().to(device), o["prec"].float().to(device), float(o["thr"])
        self.anat = AnatomyChecker(device, json.load(open(os.path.join(HEADS_DIR, "anatomy.json"))))
        rep = json.load(open(os.path.join(HEADS_DIR, "heads_report.json")))
        self.judge_real_acc = rep["eval"]["test_acc"]

        self.ds_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self._models = models
        self.ds = {}            # loaded lazily once training has finished
        print("[demo] ready.", flush=True)

    # ------------------------------------------------------------------ downstream classifiers
    def classifier_status(self, key):
        folder = dict((k, f) for k, _, f in CLASSIFIERS)[key]
        d = os.path.join(NOVELTY, "downstream", folder)
        ck, log = os.path.join(d, "best_model.pt"), os.path.join(d, "log.json")
        if not (os.path.exists(ck) and os.path.exists(log)):
            return None, "not trained yet"
        lg = json.load(open(log))
        n_ep = len(lg["history"])
        if n_ep < 10:
            return None, f"still training (epoch {n_ep}/10)"
        return ck, f"epoch {lg['selected_by_val']['epoch']} selected on validation, test macro-F1 {lg['selected_by_val']['test_f1']:.3f}"

    def get_classifier(self, key):
        if key in self.ds:
            return self.ds[key]
        ck, _ = self.classifier_status(key)
        if ck is None:
            return None
        import torch.nn as nn
        m = self._models.densenet121(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(m.classifier.in_features, 14))
        sd = torch.load(ck, map_location="cpu")
        m.load_state_dict(sd["state_dict"])
        self.ds[key] = m.to(self.dev).eval()
        return self.ds[key]

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, key, prompt, seeds, steps, cfg):
        for net in (self.pipe.unet, self.pipe.text_encoder):
            net.set_adapter(key)
        return sample(self.pipe, self.ddim, [prompt] * len(seeds), seeds, int(steps), float(cfg), return_tensor=True)

    # ------------------------------------------------------------------ analysis
    @torch.no_grad()
    def analyse(self, x, pil_images, requested=None):
        """x: (B,3,512,512) in [-1,1]; pil_images: matching grayscale PIL images. Returns list of dicts."""
        x = x.to(self.dev).float()
        xx = to_xrv(x)
        f = backbone_features(self.bb, xx)
        probs = torch.softmax(self.head(f), 1).cpu().numpy()
        d = f - self.ood_mean
        maha = torch.sqrt(((d @ self.ood_prec) * d).sum(1)).cpu().numpy()
        meas = self.anat.measure(x)
        ok = self.anat.passes(meas)
        masks = (torch.sigmoid(self.anat.seg(to_xrv(x, 512))) > 0.5).cpu().numpy()
        ds_out = {}
        for key, _, _ in CLASSIFIERS:
            m = self.get_classifier(key)
            if m is not None:
                inp = torch.stack([self.ds_tf(im.convert("RGB")) for im in pil_images]).to(self.dev)
                ds_out[key] = torch.softmax(m(inp), 1).cpu().numpy()
        out = []
        for b in range(len(x)):
            r = {"judge": {c: float(probs[b, i]) for i, c in enumerate(CLASSES)},
                 "maha": float(maha[b]), "ood": bool(maha[b] > self.ood_thr), "anatomy_ok": bool(ok[b]),
                 "meas": {k: float(v[b]) for k, v in meas.items()}}
            r["defect"] = (not r["anatomy_ok"]) or r["ood"]
            top = max(r["judge"], key=r["judge"].get)
            r["judge_top"] = top
            r["requested"] = requested
            r["shows_requested"] = None if requested is None else (top == requested)
            for key in ds_out:
                r["ds_" + key] = {c: float(ds_out[key][b, i]) for i, c in enumerate(DS_CLASSES)}
            r["overlay"] = self.overlay(pil_images[b], masks[b])
            out.append(r)
        return out

    def overlay(self, pil, m):
        base = np.asarray(pil.convert("L").resize((512, 512)), dtype=np.float32)
        rgb = np.stack([base] * 3, -1)
        t = self.anat.seg.targets
        for name, col in (("Left Lung", (60, 130, 255)), ("Right Lung", (40, 200, 120)), ("Heart", (240, 70, 70))):
            mk = m[t.index(name)].astype(bool)
            rgb[mk] = 0.6 * rgb[mk] + 0.4 * np.array(col, dtype=np.float32)
        return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def load_real_tensor(pil):
    """Same preprocessing as cxr_common.load_image_tensor (gray -> 512 bicubic -> [-1,1], 3 channels)."""
    g = np.asarray(pil.convert("L").resize((512, 512), Image.BICUBIC), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(g)[None].repeat(3, 1, 1)


# ====================================================================== UI
def build_ui(E):
    import gradio as gr

    test_df = pd.read_csv(CSV_TEST)
    presets = [prompt_for(c) for c in CLASSES]

    def fmt_p(d, k=None):
        c = max(d, key=d.get) if k is None else k
        return f"{c} ({100 * d[c]:.0f}%)"

    def summary_rows(results):
        rows = []
        for i, r in enumerate(results):
            row = {"#": i + 1}
            if "gen" in r:
                row.update({"Generator": r["gen"], "Seed": r["seed"]})
            row["Held-out judge: top class"] = fmt_p(r["judge"])
            if r["requested"]:
                row[f"p(requested)"] = f"{100 * r['judge'][r['requested']]:.0f}%"
                row["Shows requested finding?"] = "yes" if r["shows_requested"] else "no"
            row["Anatomy check"] = "pass" if r["anatomy_ok"] else "FAIL"
            row["OOD"] = "yes" if r["ood"] else "no"
            row["Structural defect?"] = "YES" if r["defect"] else "no"
            for key, lab, _ in CLASSIFIERS:
                row[lab] = fmt_p(r["ds_" + key]) if "ds_" + key in r else "(not available)"
            rows.append(row)
        return pd.DataFrame(rows)

    def detail(r):
        lines = [f"**Requested finding:** {r['requested'] or 'not one of the 6 trained classes'}"
                 + ("" if r["requested"] is None else f" — held-out judge {'agrees' if r['shows_requested'] else 'disagrees'}")]
        lines.append(f"**Anatomy check:** {'pass' if r['anatomy_ok'] else 'FAIL'} — lung components L/R = "
                     f"{int(r['meas']['compL'])}/{int(r['meas']['compR'])} (need 1/1), lung area L/R = "
                     f"{100 * r['meas']['areaL']:.1f}% / {100 * r['meas']['areaR']:.1f}%, heart area "
                     f"{100 * r['meas']['heart_area']:.1f}%")
        lines.append(f"**OOD (Mahalanobis, held-out features):** {r['maha']:.1f} vs threshold {E.ood_thr:.1f} → "
                     f"{'out of distribution' if r['ood'] else 'in distribution'}")
        lines.append(f"**Structural defect:** {'YES' if r['defect'] else 'no'}  (= anatomy fail OR OOD)")
        ds = [gr.update(value=r.get("ds_" + k), label=lab + ("" if "ds_" + k in r else
              f" — {E.classifier_status(k)[1]}")) for k, lab, _ in CLASSIFIERS]
        return [r["overlay"], r["judge"], "\n\n".join(lines)] + ds

    def classifier_md():
        return "\n".join(f"- **{lab}**: {E.classifier_status(k)[1]}" for k, lab, _ in CLASSIFIERS)

    # ---------------------------------------------------------------- callbacks
    def on_preset(p):
        return p

    def run_generate(prompt, gens, n, seed, steps, cfg, progress=gr.Progress()):
        if not prompt.strip():
            raise gr.Error("Enter a prompt.")
        if not gens:
            raise gr.Error("Pick at least one generator.")
        req = requested_class(prompt)
        seeds = [int(seed) + i for i in range(int(n))]
        results, gallery = [], []
        with E.lock:
            for gi, lab in enumerate(gens):
                progress((gi, len(gens)), desc=f"Generating with {lab}")
                x = E.generate(LABEL2KEY[lab], prompt, seeds, steps, cfg)
                pil = [Image.fromarray(((im.mean(0) + 1) * 127.5).round().clamp(0, 255).byte().cpu().numpy(), "L")
                       for im in x]
                for r, s, im in zip(E.analyse(x, pil, req), seeds, pil):
                    r.update(gen=lab, seed=s, image=im)
                    results.append(r)
                    tag = "defect" if r["defect"] else "ok"
                    shows = "" if req is None else (" · shows finding" if r["shows_requested"] else " · finding not detected")
                    gallery.append((im, f"#{len(results)} {lab} · seed {s} · {tag}{shows}"))
        progress((len(gens), len(gens)))
        first = detail(results[0])
        return [gallery, summary_rows(results), results] + first

    def on_select(results, evt: gr.SelectData):
        if not results:
            return [gr.update()] * (3 + len(CLASSIFIERS))
        return detail(results[evt.index])

    def random_test(cls):
        rows = test_df[test_df[cls] == 1.0]
        row = rows.sample(1).iloc[0]
        im = Image.open(os.path.join(DATA_DIR, row.Path)).convert("L")
        return im, cls

    def run_classify(img, true_label):
        if img is None:
            raise gr.Error("Upload an image or load a real test image first.")
        pil = img.convert("L")
        with E.lock:
            r = E.analyse(load_real_tensor(pil)[None], [pil])[0]
        r["requested"] = None
        df = summary_rows([r])
        if true_label:
            df.insert(1, "True label (CheXpert)", true_label)
        return [df] + detail(r)

    # ---------------------------------------------------------------- layout
    gen_rows = "\n".join(f"| {lab} | `{os.path.relpath(lora_path(run), PROJECT)}` | {rf}% | {df_}% | {kid} | {f1:.3f} |"
                         for _, lab, run, rf, df_, kid, f1 in GENERATORS)
    about = f"""
### Checkpoints used (same as the report's test set)
| Generator | LoRA checkpoint | Requested finding ↑ | Defects ↓ | KID×10³ ↓ | Downstream macro-F1 |
|---|---|---|---|---|---|
{gen_rows}
| *Real images (reference)* | — | 29.7% (judge accuracy) | — | — | **0.285** (real only) |

Report numbers: mean over the 5 rare pathologies, 2,173 images per generator, 50 DDIM steps, CFG 7.5.
Downstream: 14-class DenseNet-121, mean of 3 seeds on 700 real test images.

### How images are judged
- **Held-out judge**: TorchXRayVision DenseNet-121 trained on MIMIC only (`mimic_ch`), with a 6-class linear probe.
  It was never used during training. On *real* test images it reaches **{100 * E.judge_real_acc:.1f}%** accuracy (chance 16.7%),
  so one image's verdict is noisy. The rates in the report are averaged over hundreds of images.
- **Structural defect** = anatomy check fails (PSPNet lung/heart segmentation: exactly one left and one right lung,
  plausible areas, correct sides) **or** Mahalanobis out-of-distribution in the judge's feature space.
- **Downstream classifiers**: the project's 14-class DenseNet-121 (ImageNet init), seed 42, epoch chosen on a
  patient-level validation split.

The generators were trained on the 6 template prompts (`Chest X-ray showing <finding>`, and `{prompt_for('No Finding')}`).
Free-text prompts work but are outside the training distribution.
"""

    with gr.Blocks(title="CXR Diffusion Demo") as demo:
        gr.Markdown("# Text-Conditioned Synthetic Chest X-ray Generation\n"
                    "Gradient Mavericks · IE643. Type a prompt, generate chest X-rays with the LoRA checkpoints from "
                    "the report, and see what the held-out judge, the anatomy/OOD defect check and the downstream "
                    "DenseNet classifiers say about each image.")
        results_state = gr.State([])
        with gr.Tab("Generate & classify"):
            with gr.Row():
                with gr.Column(scale=1):
                    preset = gr.Dropdown(presets, value=prompt_for("Pneumonia"), label="Prompt preset (training prompts)")
                    prompt = gr.Textbox(value=prompt_for("Pneumonia"), label="Prompt", lines=2)
                    gens = gr.CheckboxGroup([g[1] for g in GENERATORS],
                                            value=[KEY2LABEL["baseline"], KEY2LABEL["margin_adv"]],
                                            label="Generators (same seeds for each → fair comparison)")
                    with gr.Row():
                        n = gr.Slider(1, 4, value=2, step=1, label="Images per generator")
                        seed = gr.Number(value=42, precision=0, label="Seed")
                    with gr.Accordion("Sampler (report: 50 steps, CFG 7.5)", open=False):
                        steps = gr.Slider(10, 50, value=50, step=1, label="DDIM steps")
                        cfg = gr.Slider(1.0, 12.0, value=7.5, step=0.5, label="Classifier-free guidance")
                    go = gr.Button("Generate and classify", variant="primary")
                    cls_md = gr.Markdown(classifier_md())
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="Generated X-rays (click one for details)", columns=4, height=520,
                                         object_fit="contain")
            table = gr.Dataframe(label="Per-image verdicts", interactive=False, wrap=True)
            with gr.Row():
                ov = gr.Image(label="Anatomy segmentation (blue/green = lungs, red = heart)", height=380)
                judge = gr.Label(label="Held-out judge (6 classes)", num_top_classes=6)
                with gr.Column():
                    ds_labels = [gr.Label(label=lab, num_top_classes=5) for _, lab, _ in CLASSIFIERS]
            info = gr.Markdown()
            preset.change(on_preset, preset, prompt)
            go.click(run_generate, [prompt, gens, n, seed, steps, cfg],
                     [gallery, table, results_state, ov, judge, info] + ds_labels).then(classifier_md, None, cls_md)
            gallery.select(on_select, results_state, [ov, judge, info] + ds_labels)

        with gr.Tab("Classify an X-ray"):
            gr.Markdown("Upload any frontal chest X-ray, or load a random **real** image from the held-out test set, "
                        "and run the same judge, defect check and downstream classifiers.")
            with gr.Row():
                with gr.Column(scale=1):
                    up = gr.Image(type="pil", label="Chest X-ray", height=380)
                    with gr.Row():
                        cls_pick = gr.Dropdown(DS_CLASSES, value="Pneumonia", label="Test-set class")
                        rnd = gr.Button("Load random real test image")
                    true_lab = gr.Textbox(label="True label (only for test-set images)", interactive=False)
                    cbtn = gr.Button("Classify", variant="primary")
                with gr.Column(scale=2):
                    ctable = gr.Dataframe(label="Verdict", interactive=False, wrap=True)
                    with gr.Row():
                        cov = gr.Image(label="Anatomy segmentation", height=360)
                        cjudge = gr.Label(label="Held-out judge (6 classes)", num_top_classes=6)
                    cds = [gr.Label(label=lab, num_top_classes=5) for _, lab, _ in CLASSIFIERS]
                    cinfo = gr.Markdown()
            up.upload(lambda: "", None, true_lab)
            rnd.click(random_test, cls_pick, [up, true_lab])
            cbtn.click(run_classify, [up, true_lab], [ctable, cov, cjudge, cinfo] + cds)

        with gr.Tab("About the models"):
            gr.Markdown(about)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link (sends traffic via Gradio's servers)")
    a = ap.parse_args()
    E = Engine("cuda")
    demo = build_ui(E)
    demo.queue(default_concurrency_limit=1).launch(server_name=a.host, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
