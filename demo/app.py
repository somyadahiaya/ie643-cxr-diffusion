"""Standalone demo: text prompt -> synthetic chest X-ray -> classification + structural-defect check.

Self-contained: needs only this file and the packages in requirements.txt. The models (about 2.2 GB) are
downloaded from the Hugging Face repo on first run and cached (~/.cache/huggingface, or --cache_dir).

  export HF_TOKEN=hf_...                     # the weights repo is private
  python app.py                              # -> http://127.0.0.1:7860
  python app.py --local_root /path/to/ie643-cxr-diffusion   # use a full clone instead of downloading

Models (exactly the checkpoints evaluated in the report):
  * generators: phase-1 P-Rajiv LoRA (epoch_2) and every novelty LoRA at the step chosen by held-out checkpoint
    selection; sampler = DDIM 50 steps, CFG 7.5 (report settings, adjustable in the UI)
  * held-out judge: TorchXRayVision densenet121-res224-mimic_ch + 6-class linear probe; Mahalanobis OOD detector;
    PSPNet lung/heart anatomy check
  * downstream 14-class DenseNet-121: real-only and real + CE+Adv synthetic [recipe] (seed 42, validation-selected)
"""
import os, json, argparse, threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HF_REPO = "SillySom/ie643-cxr-diffusion-weights"

# ---------------------------------------------------------------------------------------------- what to load
CLASSES = ["No Finding", "Enlarged Cardiomediastinum", "Lung Lesion", "Pneumonia", "Pleural Other", "Fracture"]
DS_CLASSES = ["No Finding", "Lung Opacity", "Support Devices", "Pleural Effusion", "Edema", "Pneumothorax",
              "Atelectasis", "Cardiomegaly", "Fracture", "Consolidation", "Enlarged Cardiomediastinum",
              "Lung Lesion", "Pneumonia", "Pleural Other"]
EVAL_BACKBONE = "densenet121-res224-mimic_ch"      # held-out judge backbone (MIMIC only, never used in training)

BASELINE_LORA = "ie643_code/ie643/cxr_diffusion_checkpoints/prajiv/unet_clip/epoch_2"
# key, UI label, LoRA dir (selected step = novelty/runs/<run>/best.json), report numbers:
# requested finding %, defects %, KID x1e3, downstream macro-F1 (plain augmentation, mean of 3 seeds)
GENERATORS = [
    ("baseline", "P-Rajiv LoRA (baseline, phase 1)", BASELINE_LORA, 15.7, 32.7, 12.8, 0.256),
    ("control", "Control (+2k steps, no new loss)", "novelty/runs/control/step_2000", 23.6, 51.8, 10.3, 0.275),
    ("ce_only", "CE only (reward-hacks)", "novelty/runs/ce_only/step_1000", 35.3, 78.5, 29.9, 0.258),
    ("adv_only", "Adversarial only", "novelty/runs/adv_only/step_2000", 25.9, 9.8, 2.9, 0.270),
    ("ce_adv", "CE + adversarial", "novelty/runs/ce_adv/step_1000", 45.9, 24.6, 8.2, 0.269),
    ("margin_adv", "Margin + adversarial (best)", "novelty/runs/margin_adv/step_1500", 51.2, 18.2, 7.8, 0.278),
    ("margin_adv_ms", "Margin + adv, multi-scale D", "novelty/runs/margin_adv_ms/step_1000", 43.8, 16.5, 8.7, 0.264),
]
LABEL2KEY = {g[1]: g[0] for g in GENERATORS}
KEY2LABEL = {g[0]: g[1] for g in GENERATORS}
CLASSIFIERS = [  # key, UI label, folder
    ("real", "DenseNet-121, real only", "novelty/downstream/demo_real_only_seed42"),
    ("syn", "DenseNet-121, real + CE+Adv synthetic [recipe]", "novelty/downstream/demo_ce_adv_recipe_seed42"),
]
HEADS = "novelty/heads"
BASE = "novelty/prajiv_base"
TEST_CSV = "ie643_code/ie643/global_test_single_frontal_clean.csv"

ALLOW = ([f"{BASE}/*", f"{HEADS}/eval_head.pt", f"{HEADS}/ood_stats.pt", f"{HEADS}/anatomy.json",
          f"{HEADS}/heads_report.json", TEST_CSV]
         + [f"{g[2]}/unet_lora/*" for g in GENERATORS] + [f"{g[2]}/text_encoder_lora/*" for g in GENERATORS]
         + [f"{c[2]}/best_model.pt" for c in CLASSIFIERS] + [f"{c[2]}/log.json" for c in CLASSIFIERS])


def fetch_assets(repo, cache_dir=None, token=None):
    from huggingface_hub import snapshot_download
    print(f"[demo] fetching models from https://huggingface.co/{repo} (first run downloads ~2.2 GB) ...", flush=True)
    return snapshot_download(repo, allow_patterns=ALLOW, cache_dir=cache_dir, token=token)


def prompt_for(cls):
    return "Normal chest X-ray with no significant findings" if cls == "No Finding" else "Chest X-ray showing " + cls


def requested_class(prompt):
    p = prompt.lower()
    for c in CLASSES[1:]:
        if c.lower() in p:
            return c
    if "no finding" in p or "normal" in p:
        return "No Finding"
    return None


# ---------------------------------------------------------------------------------------------- model pieces
# (identical to novelty/cxr_common.py and novelty/sd_utils.py, so outputs match the report)
def to_xrv(img, size=224):
    """(B,3|1,H,W) in [-1,1] -> (B,1,size,size) in [-1024,1024] (TorchXRayVision input range)."""
    if img.shape[1] == 3:
        img = img.mean(1, keepdim=True)
    if img.shape[-1] != size:
        img = F.interpolate(img.float(), size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return img.float().clamp(-1, 1) * 1024.0


def backbone_features(m, x_xrv):
    return F.adaptive_avg_pool2d(F.relu(m.features(x_xrv)), 1).flatten(1)


class LinearHead(nn.Module):
    def __init__(self, dim=1024, n=len(CLASSES)):
        super().__init__()
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.fc = nn.Linear(dim, n)

    def forward(self, f):
        return self.fc((f - self.mu) / self.sd)


class AnatomyChecker:
    """TorchXRayVision PSPNet lungs/heart; passes iff one dominant left and right lung, plausible areas, correct sides."""
    def __init__(self, device, th):
        import torchxrayvision as xrv
        self.seg = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
        t = self.seg.targets
        self.iL, self.iR, self.iH = t.index("Left Lung"), t.index("Right Lung"), t.index("Heart")
        self.device, self.th = device, th

    @torch.no_grad()
    def masks(self, img):
        return (torch.sigmoid(self.seg(to_xrv(img.to(self.device), 512))) > 0.5).cpu().numpy()

    def measure(self, m):
        from scipy import ndimage
        out = {k: [] for k in ["areaL", "areaR", "compL", "compR", "heart_area", "cxL", "cxR", "cxH"]}
        for b in range(m.shape[0]):
            for key, idx in (("L", self.iL), ("R", self.iR)):
                mask = m[b, idx]
                lab, n = ndimage.label(mask)
                sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
                out["area" + key].append(mask.mean())
                out["comp" + key].append(int((sizes > 0.02 * mask.size).sum()))
                xs = np.nonzero(mask)[1]
                out["cx" + key].append(xs.mean() / mask.shape[1] if len(xs) else np.nan)
            h = m[b, self.iH]
            out["heart_area"].append(h.mean())
            xs = np.nonzero(h)[1]
            out["cxH"].append(xs.mean() / h.shape[1] if len(xs) else np.nan)
        return {k: np.array(v, dtype=float) for k, v in out.items()}

    def passes(self, meas):
        th, ok = self.th, np.ones(len(meas["areaL"]), dtype=bool)
        for k in ("areaL", "areaR", "heart_area"):
            ok &= (meas[k] >= th[k][0]) & (meas[k] <= th[k][1])
        ok &= (meas["compL"] == 1) & (meas["compR"] == 1)
        ok &= np.nan_to_num(meas["cxR"], nan=1.0) < np.nan_to_num(meas["cxL"], nan=0.0)   # patient's right = image left
        ok &= (np.nan_to_num(meas["cxH"], nan=-1) > np.nan_to_num(meas["cxR"], nan=2)) & \
              (np.nan_to_num(meas["cxH"], nan=2) < np.nan_to_num(meas["cxL"], nan=-1))
        return ok


def encode_prompts(pipe, prompts, device):
    ids = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(device)
    return pipe.text_encoder(ids)[0]


@torch.no_grad()
def sample(pipe, ddim, prompts, seeds, steps, cfg, device):
    """Deterministic DDIM + classifier-free guidance; one CPU generator per seed (batch-independent noise)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cond, uncond = encode_prompts(pipe, prompts, device), encode_prompts(pipe, [""] * len(prompts), device)
        z = torch.stack([torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(int(s))) for s in seeds]
                        ).to(device) * ddim.init_noise_sigma
        ddim.set_timesteps(steps, device=device)
        for t in ddim.timesteps:
            out = pipe.unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([uncond, cond])).sample
            e_u, e_c = out.chunk(2)
            z = ddim.step((e_u + cfg * (e_c - e_u)).float(), t, z.float()).prev_sample
        z = z / pipe.vae.config.scaling_factor
        return torch.cat([pipe.vae.decode(z[i:i + 4]).sample.float().clamp(-1, 1) for i in range(0, len(z), 4)])


# ---------------------------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, root, device="cuda", data_dir=None):
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        from peft import PeftModel
        import torchxrayvision as xrv
        from torchvision import transforms
        P = lambda *a: os.path.join(root, *a)
        self.root, self.dev, self.data_dir = root, device, data_dir
        self.lock = threading.Lock()

        print("[demo] loading P-Rajiv SD-1.5 + 7 LoRA adapters ...", flush=True)
        self.pipe = StableDiffusionPipeline.from_pretrained(P(BASE), torch_dtype=torch.bfloat16, safety_checker=None,
                                                            requires_safety_checker=False)
        self.ddim = DDIMScheduler.from_config(self.pipe.scheduler.config)
        for i, (key, _, d, *_r) in enumerate(GENERATORS):
            u, t = P(d, "unet_lora"), P(d, "text_encoder_lora")
            if i == 0:
                self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, u, adapter_name=key)
                self.pipe.text_encoder = PeftModel.from_pretrained(self.pipe.text_encoder, t, adapter_name=key)
            else:
                self.pipe.unet.load_adapter(u, adapter_name=key)
                self.pipe.text_encoder.load_adapter(t, adapter_name=key)
        for m in (self.pipe.unet, self.pipe.text_encoder, self.pipe.vae):
            m.eval()
            for n, prm in m.named_parameters():
                if "lora_" in n:
                    prm.data = prm.data.float()
                prm.requires_grad_(False)
        self.pipe.to(device)

        print("[demo] loading held-out judge, OOD detector, anatomy checker ...", flush=True)
        self.bb = xrv.models.DenseNet(weights=EVAL_BACKBONE).to(device).eval()
        self.head = LinearHead().to(device)
        self.head.load_state_dict(torch.load(P(HEADS, "eval_head.pt"), map_location=device))
        self.head.eval()
        o = torch.load(P(HEADS, "ood_stats.pt"), map_location="cpu")
        self.ood_mean, self.ood_prec = o["mean"].float().to(device), o["prec"].float().to(device)
        self.ood_thr = float(o["thr"])
        self.anat = AnatomyChecker(device, json.load(open(P(HEADS, "anatomy.json"))))
        self.judge_real_acc = json.load(open(P(HEADS, "heads_report.json")))["eval"]["test_acc"]

        print("[demo] loading downstream DenseNet-121 classifiers ...", flush=True)
        from torchvision import models
        self.ds, self.ds_info = {}, {}
        for key, _, d in CLASSIFIERS:
            ck = P(d, "best_model.pt")
            if not os.path.exists(ck):
                self.ds_info[key] = "not available"
                continue
            m = models.densenet121(weights=None)
            m.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(m.classifier.in_features, 14))
            sd = torch.load(ck, map_location="cpu")
            m.load_state_dict(sd["state_dict"])
            self.ds[key] = m.to(device).eval()
            self.ds_info[key] = f"epoch {sd['epoch']} selected on validation, test macro-F1 {sd['test_f1']:.3f}"
        self.ds_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.test_df = None
        if data_dir and os.path.isdir(os.path.join(data_dir, "CheXpert-v1.0-small")) and os.path.exists(P(TEST_CSV)):
            self.test_df = pd.read_csv(P(TEST_CSV))
        print("[demo] ready.", flush=True)

    @torch.no_grad()
    def generate(self, key, prompt, seeds, steps, cfg):
        for net in (self.pipe.unet, self.pipe.text_encoder):
            net.set_adapter(key)
        return sample(self.pipe, self.ddim, [prompt] * len(seeds), seeds, int(steps), float(cfg), self.dev)

    @torch.no_grad()
    def analyse(self, x, pil_images, requested=None):
        x = x.to(self.dev).float()
        f = backbone_features(self.bb, to_xrv(x))
        probs = torch.softmax(self.head(f), 1).cpu().numpy()
        d = f - self.ood_mean
        maha = torch.sqrt(((d @ self.ood_prec) * d).sum(1)).cpu().numpy()
        masks = self.anat.masks(x)
        meas = self.anat.measure(masks)
        ok = self.anat.passes(meas)
        ds_out = {}
        if self.ds:
            inp = torch.stack([self.ds_tf(im.convert("RGB")) for im in pil_images]).to(self.dev)
            ds_out = {k: torch.softmax(m(inp), 1).cpu().numpy() for k, m in self.ds.items()}
        out = []
        for b in range(len(x)):
            r = {"judge": {c: float(probs[b, i]) for i, c in enumerate(CLASSES)}, "maha": float(maha[b]),
                 "ood": bool(maha[b] > self.ood_thr), "anatomy_ok": bool(ok[b]),
                 "meas": {k: float(v[b]) for k, v in meas.items()}, "requested": requested}
            r["defect"] = (not r["anatomy_ok"]) or r["ood"]
            r["judge_top"] = max(r["judge"], key=r["judge"].get)
            r["shows_requested"] = None if requested is None else (r["judge_top"] == requested)
            for k in ds_out:
                r["ds_" + k] = {c: float(ds_out[k][b, i]) for i, c in enumerate(DS_CLASSES)}
            r["overlay"] = self.overlay(pil_images[b], masks[b])
            out.append(r)
        return out

    def overlay(self, pil, m):
        rgb = np.stack([np.asarray(pil.convert("L").resize((512, 512)), dtype=np.float32)] * 3, -1)
        t = self.anat.seg.targets
        for name, col in (("Left Lung", (60, 130, 255)), ("Right Lung", (40, 200, 120)), ("Heart", (240, 70, 70))):
            mk = m[t.index(name)].astype(bool)
            rgb[mk] = 0.6 * rgb[mk] + 0.4 * np.array(col, dtype=np.float32)
        return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def to_tensor_512(pil):
    """Same preprocessing as training/evaluation: gray -> 512 bicubic -> [-1,1], 3 channels."""
    g = np.asarray(pil.convert("L").resize((512, 512), Image.BICUBIC), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(g)[None].repeat(3, 1, 1)


def to_pil(x):
    return [Image.fromarray(((im.mean(0) + 1) * 127.5).round().clamp(0, 255).byte().cpu().numpy(), "L") for im in x]


# ---------------------------------------------------------------------------------------------- UI
def build_ui(E):
    import gradio as gr

    def fmt_p(d):
        c = max(d, key=d.get)
        return f"{c} ({100 * d[c]:.0f}%)"

    def summary_rows(results):
        rows = []
        for i, r in enumerate(results):
            row = {"#": i + 1}
            if "gen" in r:
                row.update({"Generator": r["gen"], "Seed": r["seed"]})
            row["Held-out judge: top class"] = fmt_p(r["judge"])
            if r["requested"]:
                row["p(requested)"] = f"{100 * r['judge'][r['requested']]:.0f}%"
                row["Shows requested finding?"] = "yes" if r["shows_requested"] else "no"
            row["Anatomy check"] = "pass" if r["anatomy_ok"] else "FAIL"
            row["OOD"] = "yes" if r["ood"] else "no"
            row["Structural defect?"] = "YES" if r["defect"] else "no"
            for k, lab, _ in CLASSIFIERS:
                row[lab] = fmt_p(r["ds_" + k]) if "ds_" + k in r else "(not available)"
            rows.append(row)
        return pd.DataFrame(rows)

    def detail(r):
        req = r["requested"]
        lines = [f"**Requested finding:** {req or 'not one of the 6 trained classes'}"
                 + ("" if req is None else f" — held-out judge {'agrees' if r['shows_requested'] else 'disagrees'}"),
                 f"**Anatomy check:** {'pass' if r['anatomy_ok'] else 'FAIL'} — lung components L/R = "
                 f"{int(r['meas']['compL'])}/{int(r['meas']['compR'])} (need 1/1), lung area L/R = "
                 f"{100 * r['meas']['areaL']:.1f}% / {100 * r['meas']['areaR']:.1f}%, heart area "
                 f"{100 * r['meas']['heart_area']:.1f}%",
                 f"**OOD (Mahalanobis, held-out features):** {r['maha']:.1f} vs threshold {E.ood_thr:.1f} → "
                 f"{'out of distribution' if r['ood'] else 'in distribution'}",
                 f"**Structural defect:** {'YES' if r['defect'] else 'no'}  (= anatomy fail OR OOD)"]
        return [r["overlay"], r["judge"], "\n\n".join(lines)] + [r.get("ds_" + k) for k, _, _ in CLASSIFIERS]

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
                pil = to_pil(x)
                for r, s, im in zip(E.analyse(x, pil, req), seeds, pil):
                    r.update(gen=lab, seed=s)
                    results.append(r)
                    shows = "" if req is None else (" · shows finding" if r["shows_requested"] else " · finding not detected")
                    gallery.append((im, f"#{len(results)} {lab} · seed {s} · {'defect' if r['defect'] else 'ok'}{shows}"))
        return [gallery, summary_rows(results), results] + detail(results[0])

    def on_select(results, evt: gr.SelectData):
        return detail(results[evt.index]) if results else [gr.update()] * (3 + len(CLASSIFIERS))

    def random_test(cls):
        rows = E.test_df[E.test_df[cls] == 1.0]
        row = rows.sample(1).iloc[0]
        return Image.open(os.path.join(E.data_dir, row.Path)).convert("L"), cls

    def run_classify(img, true_label):
        if img is None:
            raise gr.Error("Upload an image first.")
        pil = img.convert("L")
        with E.lock:
            r = E.analyse(to_tensor_512(pil)[None], [pil])[0]
        df = summary_rows([r])
        if true_label:
            df.insert(1, "True label (CheXpert)", true_label)
        return [df] + detail(r)

    gen_rows = "\n".join(f"| {lab} | `{d}` | {rf}% | {df_}% | {kid} | {f1:.3f} |"
                         for _, lab, d, rf, df_, kid, f1 in GENERATORS)
    cls_rows = "\n".join(f"- **{lab}**: {E.ds_info[k]}" for k, lab, _ in CLASSIFIERS)
    about = f"""
### Checkpoints (the ones evaluated in the report)
| Generator | LoRA checkpoint | Requested finding ↑ | Defects ↓ | KID×10³ ↓ | Downstream macro-F1 |
|---|---|---|---|---|---|
{gen_rows}
| *Real images (reference)* | n/a | 29.7% (judge accuracy) | n/a | n/a | **0.285** (real only) |

Report numbers: mean over the 5 rare pathologies, 2,173 images per generator, 50 DDIM steps, CFG 7.5.
Downstream: 14-class DenseNet-121, mean of 3 seeds, 700 real test images.

### Classifiers in this demo
{cls_rows}

### How images are judged
- **Held-out judge**: TorchXRayVision DenseNet-121 trained on MIMIC only (`mimic_ch`) + 6-class linear probe; never
  used in training. On *real* test images it is **{100 * E.judge_real_acc:.1f}%** accurate (chance 16.7%), so a single
  verdict is noisy. The report averages over hundreds of images.
- **Structural defect** = anatomy check fails (PSPNet: exactly one left and one right lung, plausible areas, correct
  sides) **or** Mahalanobis out-of-distribution in the judge's feature space.
- Generators were trained on the 6 template prompts; free text works but is outside the training distribution.

Models: https://huggingface.co/{HF_REPO}
"""

    with gr.Blocks(title="CXR Diffusion Demo") as demo:
        gr.Markdown("# Text-Conditioned Synthetic Chest X-ray Generation\n"
                    "Gradient Mavericks · IE643. Generate chest X-rays from a prompt with the report's LoRA "
                    "checkpoints, then see what the held-out judge, the anatomy/OOD defect check and the downstream "
                    "DenseNet classifiers say about each image.")
        results_state = gr.State([])
        with gr.Tab("Generate & classify"):
            with gr.Row():
                with gr.Column(scale=1):
                    preset = gr.Dropdown([prompt_for(c) for c in CLASSES], value=prompt_for("Pneumonia"),
                                         label="Prompt preset (training prompts)")
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
            preset.change(lambda p: p, preset, prompt)
            go.click(run_generate, [prompt, gens, n, seed, steps, cfg],
                     [gallery, table, results_state, ov, judge, info] + ds_labels)
            gallery.select(on_select, results_state, [ov, judge, info] + ds_labels)

        with gr.Tab("Classify an X-ray"):
            gr.Markdown("Upload any frontal chest X-ray and run the same judge, defect check and classifiers."
                        + ("" if E.test_df is not None else
                           " *(Start the app with `--data_dir` pointing at a folder containing CheXpert-v1.0-small "
                           "to also load random real test images.)*"))
            with gr.Row():
                with gr.Column(scale=1):
                    up = gr.Image(type="pil", label="Chest X-ray", height=380)
                    with gr.Row(visible=E.test_df is not None):
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
            if E.test_df is not None:
                rnd.click(random_test, cls_pick, [up, true_lab])
            cbtn.click(run_classify, [up, true_lab], [ctable, cov, cjudge, cinfo] + cds)

        with gr.Tab("About the models"):
            gr.Markdown(about)
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf_repo", default=HF_REPO, help="Hugging Face repo with the weights")
    ap.add_argument("--local_root", default=None, help="use a local clone of the code repo (+ downloaded weights) instead")
    ap.add_argument("--cache_dir", default=None, help="where to cache the downloaded models")
    ap.add_argument("--data_dir", default=None, help="folder containing CheXpert-v1.0-small/ (optional: random test images)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio.live link (traffic goes through Gradio's servers)")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required (Stable Diffusion sampling in bf16).")
    root = a.local_root or fetch_assets(a.hf_repo, a.cache_dir, os.environ.get("HF_TOKEN"))
    E = Engine(root, "cuda", a.data_dir)
    build_ui(E).queue(default_concurrency_limit=1).launch(server_name=a.host, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
