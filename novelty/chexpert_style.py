"""Fix #1: remove the trivial 'this image is synthetic' cue before downstream training.

Real CheXpert-small files are 8-bit grayscale JPEGs (quality ~95, measured from the quantisation tables),
mostly 390x320 px. Generated files are 512x512 RGB PNGs. A classifier can learn 'synthetic -> rare class'
from that alone. This script re-encodes synthetic images like real ones (grayscale, a W x H sampled from the
real size distribution, JPEG q=95) and writes the mixed training CSV.

Several source folders can be merged, e.g. the 5 rare classes + synthetic 'No Finding' images, so that
'synthetic' is no longer confined to the rare classes.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cxr_common import *  # noqa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", required=True, help="comma-separated folders under CheXpert-v1.0-small/")
    p.add_argument("--name", required=True)
    p.add_argument("--quality", type=int, default=95)
    a = p.parse_args()
    base = os.path.join(DATA_DIR, "CheXpert-v1.0-small")
    real = pd.read_csv(CSV_CLS)
    rng = np.random.RandomState(0)
    sizes = [Image.open(os.path.join(DATA_DIR, pth)).size for pth in real.Path.sample(1000, random_state=0)]

    out_root = os.path.join(base, a.name)
    rows = []
    for src in a.sources.split(","):
        for d in sorted(os.listdir(os.path.join(base, src))):
            cls = d.replace("_", " ")
            if cls not in CLASS_TO_IDX:
                continue
            os.makedirs(os.path.join(out_root, d), exist_ok=True)
            for f in sorted(os.listdir(os.path.join(base, src, d))):
                if not f.endswith(".png"):
                    continue
                W, H = sizes[rng.randint(len(sizes))]
                im = Image.open(os.path.join(base, src, d, f)).convert("L").resize((W, H), Image.LANCZOS)
                fn = f.replace(".png", ".jpg")
                im.save(os.path.join(out_root, d, fn), "JPEG", quality=a.quality)
                rows.append({"Path": f"{d}/{fn}", "Label": cls, "Is_Synthetic": 1})
    pd.DataFrame(rows).to_csv(os.path.join(out_root, "synthetic_labels.csv"), index=False)

    syn = []
    for r in rows:
        row = {c: 0.0 for c in real.columns[5:19]}
        row.update({"Path": f"CheXpert-v1.0-small/{a.name}/{r['Path']}", "Sex": "Unknown", "Age": 0,
                    "Frontal/Lateral": "Frontal", "AP/PA": "AP", r["Label"]: 1.0, "Patient_ID": "synthetic"})
        syn.append(row)
    out = os.path.join(PROJECT, "novelty", "csv", f"classifier_train_mixed_{a.name}.csv")
    pd.concat([real, pd.DataFrame(syn)[real.columns]], ignore_index=True).to_csv(out, index=False)
    print(f"wrote {len(rows)} CheXpert-style JPEGs to {out_root}; per class:",
          pd.Series([r['Label'] for r in rows]).value_counts().to_dict(), "\nCSV:", out)


if __name__ == "__main__":
    main()
