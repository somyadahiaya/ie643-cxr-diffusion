"""Shared pieces for the novelty-stage experiments.

- class list / prompts (identical to the original dataloader.py captions)
- differentiable conversion from diffusion images ([-1,1], 3x512x512) to TorchXRayVision input
- frozen-backbone feature extractor + linear class head + adversarial (projection) head
- anatomy check with the TorchXRayVision PSPNet lung/heart segmenter
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT, "data")                       # contains CheXpert-v1.0-small/
CODE_DIR = os.path.join(PROJECT, "ie643_code", "ie643")
CSV_DIFF = os.path.join(CODE_DIR, "diffusion_train_single_frontal_clean.csv")
CSV_CLS = os.path.join(CODE_DIR, "classifier_train_single_frontal_clean_real1500.csv")
CSV_TEST = os.path.join(CODE_DIR, "global_test_single_frontal_clean.csv")
HEADS_DIR = os.path.join(PROJECT, "novelty", "heads")

# The 6 classes the diffusion model is trained on (same order as dataloader.py)
CLASSES = ["No Finding", "Enlarged Cardiomediastinum", "Lung Lesion",
           "Pneumonia", "Pleural Other", "Fracture"]
TARGET_CLASSES = CLASSES[1:]          # the 5 rare pathologies we synthesise
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

REWARD_BACKBONE = "densenet121-res224-all"       # used INSIDE training (reward / discriminator)
EVAL_BACKBONE = "densenet121-res224-mimic_ch"    # held-out judge: trained on MIMIC only (no CheXpert)


def prompt_for(cls):
    if cls == "No Finding":
        return "Normal chest X-ray with no significant findings"
    return "Chest X-ray showing " + cls


def label_of_row(row):
    hits = [c for c in CLASSES if float(row[c]) == 1.0]
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------------------
# image conversion
# ---------------------------------------------------------------------------
def to_xrv(img, size=224):
    """img: (B,3,H,W) or (B,1,H,W) in [-1,1]  ->  (B,1,size,size) in [-1024,1024]. Differentiable."""
    if img.shape[1] == 3:
        img = img.mean(1, keepdim=True)
    if img.shape[-1] != size:
        img = F.interpolate(img.float(), size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return img.float().clamp(-1, 1) * 1024.0


def load_backbone(name, device):
    import torchxrayvision as xrv
    m = xrv.models.DenseNet(weights=name).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def backbone_features(m, x_xrv):
    """1024-d pooled DenseNet features (non-inplace relu so autograd is safe)."""
    f = m.features(x_xrv)
    return F.adaptive_avg_pool2d(F.relu(f), 1).flatten(1)


class LinearHead(nn.Module):
    """Standardise features then linear layer (weights exported from sklearn LogisticRegression)."""
    def __init__(self, dim=1024, n=len(CLASSES)):
        super().__init__()
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.fc = nn.Linear(dim, n)

    def forward(self, f):
        return self.fc((f - self.mu) / self.sd)


class ProjectionDiscriminatorHead(nn.Module):
    """Small trainable head on frozen backbone features.
    D(x, y) = w^T h(x) + <embed(y), h(x)>   (projection discriminator, Miyato & Koyama 2018)
    so it scores 'does this look like a REAL image of class y'."""
    def __init__(self, dim=1024, hidden=256, n=len(CLASSES)):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        self.h = nn.Sequential(sn(nn.Linear(dim, hidden)), nn.LeakyReLU(0.2),
                               sn(nn.Linear(hidden, hidden)), nn.LeakyReLU(0.2))
        self.out = sn(nn.Linear(hidden, 1))
        self.embed = sn(nn.Embedding(n, hidden))

    def forward(self, f, y):
        h = self.h((f - self.mu) / self.sd)
        return self.out(h).squeeze(1) + (self.embed(y) * h).sum(1)


MS_TAPS = {"b2": ("denseblock2", 512), "b3": ("denseblock3", 1024), "b4": ("norm5", 1024)}  # 28x28, 14x14, 7x7


def backbone_multiscale(m, x_xrv):
    """Multi-scale DenseNet features for the Projected-GAN discriminator.
    Returns {"pool": (B,1024) -- numerically identical to backbone_features --,
             "b2": (B,512,28,28), "b3": (B,1024,14,14), "b4": (B,1024,7,7)}."""
    want = {v[0]: k for k, v in MS_TAPS.items()}
    out, h = {}, x_xrv
    for name, mod in m.features.named_children():
        h = mod(h)
        if name in want:
            out[want[name]] = h
    out["pool"] = F.adaptive_avg_pool2d(F.relu(out["b4"]), 1).flatten(1)
    return out


class SpatialProjHead(nn.Module):
    """Projected-GAN style patch discriminator on ONE frozen feature map (Sauer et al., NeurIPS 2021).
    standardise -> FIXED random 1x1 channel projection (CCM, not trained) -> 2x SN conv3x3 -> per-location
    projection-discriminator logit  D(x,y)[i,j] = w^T h_ij + <embed(y), h_ij>."""
    def __init__(self, cin, n=len(CLASSES), proj=128, hidden=128, seed=0):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        self.register_buffer("mu", torch.zeros(cin))
        self.register_buffer("sd", torch.ones(cin))
        g = torch.Generator().manual_seed(1234 + seed)
        self.register_buffer("P", torch.randn(proj, cin, generator=g) / cin ** 0.5)
        self.body = nn.Sequential(sn(nn.Conv2d(proj, hidden, 3, padding=1)), nn.LeakyReLU(0.2),
                                  sn(nn.Conv2d(hidden, hidden, 3, padding=1)), nn.LeakyReLU(0.2))
        self.out = sn(nn.Conv2d(hidden, 1, 1))
        self.embed = sn(nn.Embedding(n, hidden))

    def forward(self, f, y):
        f = (f - self.mu[None, :, None, None]) / self.sd[None, :, None, None]
        h = self.body(torch.einsum("oc,bchw->bohw", self.P, f))
        return self.out(h).squeeze(1) + (self.embed(y)[:, :, None, None] * h).sum(1)   # (B,H,W) patch logits


class MultiScaleProjD(nn.Module):
    """Global pooled projection head (the original D) + patch heads on 3 DenseNet stages.
    forward returns {scale: logits}; losses are averaged over scales so lam_adv stays comparable."""
    def __init__(self):
        super().__init__()
        self.glob = ProjectionDiscriminatorHead()
        self.heads = nn.ModuleDict({k: SpatialProjHead(c, seed=i) for i, (k, (_, c)) in enumerate(MS_TAPS.items())})

    @torch.no_grad()
    def init_stats(self, feats, glob_mu, glob_sd):
        self.glob.mu.copy_(glob_mu); self.glob.sd.copy_(glob_sd)
        for k, hd in self.heads.items():
            f = feats[k].float()
            hd.mu.copy_(f.mean((0, 2, 3))); hd.sd.copy_(f.std((0, 2, 3)).clamp_min(1e-3))

    def forward(self, feats, y):
        return {"pool": self.glob(feats["pool"].float(), y),
                **{k: hd(feats[k].float(), y) for k, hd in self.heads.items()}}


def load_linear_head(path, device):
    head = LinearHead().to(device)
    head.load_state_dict(torch.load(path, map_location=device))
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


# ---------------------------------------------------------------------------
# anatomy / structural-defect check
# ---------------------------------------------------------------------------
class AnatomyChecker:
    """Uses TorchXRayVision PSPNet (ChestX-Det) to segment lungs + heart.
    An image PASSES if it has exactly one dominant left-lung and right-lung region,
    each with a plausible area, lungs on the correct sides, and a heart between them.
    Thresholds are calibrated on REAL test images (see train_heads.py) and stored in a json."""
    def __init__(self, device, thresholds=None):
        import torchxrayvision as xrv
        self.seg = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
        t = self.seg.targets
        self.iL, self.iR, self.iH = t.index("Left Lung"), t.index("Right Lung"), t.index("Heart")
        self.device = device
        self.th = thresholds

    @torch.no_grad()
    def measure(self, img):
        """img: (B,C,H,W) in [-1,1]. Returns dict of per-image numpy arrays."""
        from scipy import ndimage
        x = to_xrv(img.to(self.device), 512)
        m = (torch.sigmoid(self.seg(x)) > 0.5).cpu().numpy()
        out = {k: [] for k in ["areaL", "areaR", "compL", "compR", "fracL", "fracR", "heart_area",
                               "cxL", "cxR", "cxH"]}
        for b in range(m.shape[0]):
            for key, idx in (("L", self.iL), ("R", self.iR)):
                mask = m[b, idx]
                lab, n = ndimage.label(mask)
                sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
                out["area" + key].append(mask.mean())
                # number of "large" components (>2% of the image) -> duplicated chests give 2+
                out["comp" + key].append(int((sizes > 0.02 * mask.size).sum()))
                out["frac" + key].append(sizes.max() / max(sizes.sum(), 1))
                ys, xs = np.nonzero(mask)
                out["cx" + key].append(xs.mean() / mask.shape[1] if len(xs) else np.nan)
            h = m[b, self.iH]
            out["heart_area"].append(h.mean())
            ys, xs = np.nonzero(h)
            out["cxH"].append(xs.mean() / h.shape[1] if len(xs) else np.nan)
        return {k: np.array(v, dtype=float) for k, v in out.items()}

    def passes(self, meas):
        th = self.th
        ok = np.ones(len(meas["areaL"]), dtype=bool)
        for k in ("areaL", "areaR", "heart_area"):
            ok &= (meas[k] >= th[k][0]) & (meas[k] <= th[k][1])
        ok &= (meas["compL"] == 1) & (meas["compR"] == 1)
        # radiological convention: patient's RIGHT lung is on the image LEFT
        ok &= np.nan_to_num(meas["cxR"], nan=1.0) < np.nan_to_num(meas["cxL"], nan=0.0)
        ok &= (np.nan_to_num(meas["cxH"], nan=-1) > np.nan_to_num(meas["cxR"], nan=2)) & \
              (np.nan_to_num(meas["cxH"], nan=2) < np.nan_to_num(meas["cxL"], nan=-1))
        return ok


def load_image_tensor(path, size=512):
    """Same preprocessing as the original dataloader.py: gray -> direct resize -> 3ch -> [-1,1]."""
    import cv2
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(im)[None].repeat(3, 1, 1)
