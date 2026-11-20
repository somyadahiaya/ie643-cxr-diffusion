"""DA-Fusion-style augmentation (Trabucco et al., ICLR 2024, arXiv 2302.07944) with our fine-tuned LoRA.

Instead of sampling from pure noise, each synthetic image starts from a REAL training image of the same class:
encode it with the VAE, noise it to strength s ~ U(0.4, 0.6) (fraction of the 50-step DDIM schedule), and
denoise with the CE+Adv LoRA conditioned on the class prompt. This keeps real anatomy, real diversity and the
real label, while the generator adds variation.

Source images: real rows of the classifier training CSV, EXCLUDING the patients that downstream_train.py
uses for validation (same fixed split), so no validation information leaks into training.
Same per-class counts as generate.py. Writes PNGs + mixed CSV in the same format as generate.py.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cxr_common import *  # noqa
from sd_utils import load_sd, attach_lora, set_lora_scale, encode_prompts, init_latents, cfg_eps
from generate import TARGETS


def val_patients(real):
    pids = np.sort(real.Path.str.split("/").str[2].unique())
    return set(np.random.RandomState(0).choice(pids, int(0.1 * len(pids)), replace=False))   # = downstream_train.py


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_dir", default=os.path.join(PROJECT, "novelty/runs/ce_adv/step_1000"))
    p.add_argument("--name", default="gen_ce_adv_i2i")
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--smin", type=float, default=0.4)
    p.add_argument("--smax", type=float, default=0.6)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--bs", type=int, default=8)
    a = p.parse_args()
    dev = "cuda"

    pipe, ddim, _ = load_sd("prajiv")
    pipe = attach_lora(pipe, a.lora_dir, "unet_clip", trainable=False)
    pipe.unet.eval(); pipe.text_encoder.eval()
    set_lora_scale(pipe, a.lora_scale)
    sf = pipe.vae.config.scaling_factor

    real = pd.read_csv(CSV_CLS)
    real["label"] = real.apply(label_of_row, axis=1)
    vp = val_patients(real)
    pool_df = real[~real.Path.str.split("/").str[2].isin(vp)]
    root = os.path.join(DATA_DIR, "CheXpert-v1.0-small", a.name)
    rows, rng = [], np.random.RandomState(123)
    for c, n in TARGETS.items():
        ci = CLASS_TO_IDX[c]
        pool = pool_df[pool_df.label == c].Path.tolist()
        order = rng.permutation(len(pool))
        srcs = [pool[order[i % len(pool)]] for i in range(n)]   # cycle through real images of the class
        d = os.path.join(root, c.replace(" ", "_")); os.makedirs(d, exist_ok=True)
        print(f"{c}: {n} images from {len(pool)} real sources", flush=True)
        for s0 in tqdm(range(0, n, a.bs), desc=c):
            idx = list(range(s0, min(s0 + a.bs, n)))
            strength = float(rng.uniform(a.smin, a.smax))
            fns = [os.path.join(d, f"synthetic_{c.replace(' ', '_')}_{i:04d}.png") for i in idx]
            if all(os.path.exists(f) for f in fns):
                rows += [{"Path": os.path.relpath(f, root), "Label": c, "Is_Synthetic": 1, "Source": srcs[i],
                          "Strength": strength} for f, i in zip(fns, idx)]
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = torch.stack([load_image_tensor(os.path.join(DATA_DIR, srcs[i])) for i in idx]).to(dev)
                z0 = pipe.vae.encode(x.to(pipe.vae.dtype)).latent_dist.mean.float() * sf
                ddim.set_timesteps(a.steps, device=dev)
                k0 = int(round(a.steps * (1 - strength)))
                ts = ddim.timesteps[k0:]
                noise = init_latents([9_000_000 + 1000 * ci + i for i in idx], dev)
                z = ddim.add_noise(z0, noise, ts[:1].repeat(len(idx)))
                cond = encode_prompts(pipe, [prompt_for(c)] * len(idx), dev)
                uncond = encode_prompts(pipe, [""] * len(idx), dev)
                for t in ts:
                    z = ddim.step(cfg_eps(pipe.unet, z, t, cond, uncond, a.cfg).float(), t, z.float()).prev_sample
                z = z / sf
                img = torch.cat([pipe.vae.decode(z[j:j + 4]).sample.float().clamp(-1, 1) for j in range(0, len(z), 4)])
            arr = ((img.mean(1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            from PIL import Image
            for f, i, im in zip(fns, idx, arr):
                Image.fromarray(im, mode="L").save(f)
                rows.append({"Path": os.path.relpath(f, root), "Label": c, "Is_Synthetic": 1, "Source": srcs[i],
                             "Strength": strength})
    pd.DataFrame(rows).to_csv(os.path.join(root, "synthetic_labels.csv"), index=False)

    syn = []
    for r in rows:
        row = {col: 0.0 for col in real.columns[5:19]}
        row.update({"Path": f"CheXpert-v1.0-small/{a.name}/{r['Path']}", "Sex": "Unknown", "Age": 0,
                    "Frontal/Lateral": "Frontal", "AP/PA": "AP", r["Label"]: 1.0, "Patient_ID": "synthetic"})
        syn.append(row)
    cols = [c for c in real.columns if c != "label"]
    out = os.path.join(PROJECT, "novelty", "csv", f"classifier_train_mixed_{a.name}.csv")
    pd.concat([real[cols], pd.DataFrame(syn)[cols]], ignore_index=True).to_csv(out, index=False)
    print("wrote", len(rows), "img2img images to", root, "and", out)


if __name__ == "__main__":
    main()
