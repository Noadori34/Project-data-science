"""Shared pieces for Phase 2 kidney segmentation (plan.md section 3).

- Letterbox: pad to keep the aspect ratio, resize to 512 x 512 (plan 3.3), and the exact inverse.
- Transforms: MONAI [S21] percentile intensity scaling, and the same augmentation for image and
  mask (rotation, gamma, translation, zoom as in [24]; Gaussian noise [S21]).
- Models: U-Net [S18] with ResNet-34 [S17] and DeepLabV3+ via segmentation_models_pytorch [S19].
- Post-processing: keep the largest connected region and fill holes (plan 3.3).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ================================================================ letterbox
@dataclass
class Letterbox:
    orig_h: int
    orig_w: int
    side: int  # square side before resizing
    top: int
    left: int
    size: int  # network input size


def letterbox(img: np.ndarray, size: int = 512, interp=cv2.INTER_LINEAR):
    """Zero-pad to a centred square (keeps the aspect ratio), then resize to size x size."""
    h, w = img.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    sq = np.zeros((side, side) + img.shape[2:], dtype=img.dtype)
    sq[top:top + h, left:left + w] = img
    out = cv2.resize(sq, (size, size), interpolation=interp)
    return out, Letterbox(h, w, side, top, left, size)


def unletterbox(arr: np.ndarray, lb: Letterbox, interp=cv2.INTER_LINEAR) -> np.ndarray:
    """Inverse of `letterbox` for a (size, size) probability map or mask."""
    sq = cv2.resize(arr.astype(np.float32), (lb.side, lb.side), interpolation=interp)
    return sq[lb.top:lb.top + lb.orig_h, lb.left:lb.left + lb.orig_w]


# ================================================================ post-processing
def postprocess(mask: np.ndarray) -> np.ndarray:
    """Largest connected component + hole filling (plan 3.3). The dataset A ground-truth masks
    are one solid region (see prepare_masks.py check)."""
    m = mask.astype(bool)
    if not m.any():
        return m.astype(np.uint8)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(m).astype(np.uint8)


# ================================================================ MONAI transforms [S21]
def build_transforms(train: bool, size: int = 512):
    """Dict transforms on channel-first arrays: image (1, S, S) float, mask (1, S, S).
    `track_meta=False` avoids the MetaTensor error the team hit earlier (plan 2.5)."""
    from monai.transforms import (
        Compose,
        EnsureTyped,
        RandAdjustContrastd,
        RandAffined,
        RandGaussianNoised,
        ScaleIntensityRangePercentilesd,
    )

    t = [ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True)]
    if train:
        t += [
            RandAffined(
                keys=["image", "mask"], prob=0.8,
                rotate_range=np.deg2rad(15),            # rotation [24]
                translate_range=(0.08 * size, 0.08 * size),  # translation [24]
                scale_range=(0.15, 0.15),               # zoom [24]
                mode=("bilinear", "nearest"), padding_mode="zeros",
            ),
            RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.7, 1.5)),  # gamma [24]
            RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.02),  # noise [S21]
        ]
    t += [EnsureTyped(keys=["image", "mask"], track_meta=False)]
    return Compose(t)


def to_network_input(img01):
    """(B,1,H,W) tensor in [0,1] -> 3-channel ImageNet-normalised tensor for smp encoders."""
    import torch

    x = img01.repeat(1, 3, 1, 1)
    mean = torch.as_tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ================================================================ dataset
class KidneySegDataset:
    """Dataset A images + kidney masks, letterboxed to `size`. Images are cached in memory
    (about 1 GB at 512 px for all 1,985 images)."""

    def __init__(self, ids, size=512, train=False, loader=None):
        from .data import load_image_a, load_mask_a

        self.ids = list(ids)
        self.size = size
        self.tf = build_transforms(train, size)
        self.load_img = loader[0] if loader else load_image_a
        self.load_mask = loader[1] if loader else load_mask_a
        self._cache = {}

    def __len__(self):
        return len(self.ids)

    def _get(self, i):
        if i not in self._cache:
            iid = self.ids[i]
            img, lb = letterbox(self.load_img(iid), self.size)
            msk, _ = letterbox(self.load_mask(iid), self.size, interp=cv2.INTER_NEAREST)
            self._cache[i] = (img, msk, lb)
        return self._cache[i]

    def __getitem__(self, i):
        img, msk, _ = self._get(i)
        d = self.tf({"image": img[None].astype(np.float32), "mask": msk[None].astype(np.float32)})
        return d["image"].float(), (d["mask"] > 0.5).float(), i


# ================================================================ models [S19]
MODEL_SPECS = {
    # S-A: U-Net [S18] with ResNet-34 [S17] encoder, ImageNet
    "unet_r34": dict(arch="Unet", encoder="resnet34"),
    # S-B: DeepLabV3+ as compared in [24] Table 5; ResNet-50 encoder, same depth as the URI-CADS backbone
    "deeplabv3p_r50": dict(arch="DeepLabV3Plus", encoder="resnet50"),
}


def build_model(name: str, pretrained: bool = True):
    import segmentation_models_pytorch as smp

    spec = MODEL_SPECS[name]
    cls = getattr(smp, spec["arch"])
    return cls(encoder_name=spec["encoder"], encoder_weights="imagenet" if pretrained else None,
               in_channels=3, classes=1)


def predict_probs(model, img_gray: np.ndarray, size: int, device) -> np.ndarray:
    """Probability map at the original resolution for one grayscale image."""
    import torch

    tf = build_transforms(train=False, size=size)
    lbimg, lb = letterbox(img_gray, size)
    d = tf({"image": lbimg[None].astype(np.float32), "mask": np.zeros((1, size, size), np.float32)})
    x = to_network_input(d["image"][None].to(device))
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    return np.clip(unletterbox(p, lb), 0, 1)
