"""Data loading for dataset A (URI-CADS [24]) and dataset B (KidneyUS [18]). Plan.md 1.1."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

from . import config as C


# ---------------------------------------------------------------- dataset A
@dataclass
class LabelA:
    image_id: str  # file stem, e.g. "17"
    diagnosis: int  # Global diagnosis: 0 healthy, 1 pathological
    pcd: int  # Global pathologies, first flag
    hc: int  # Global pathologies, second flag
    view: int  # 0 transverse, 1 longitudinal
    boxes: list = field(default_factory=list)  # (x, y, w, h, cls)


def parse_label(path: Path) -> LabelA:
    """Parse one bbdd_kidney/labels/*.txt file. Box format is `x y w h class` (checked: 99% of box
    centres fall inside the kidney mask with this convention)."""
    d, pcd, hc, view, boxes = 0, 0, 0, 0, []
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, val = [s.strip() for s in line.split(":", 1)]
        if key == "Global diagnosis":
            d = int(val)
        elif key == "Global pathologies":
            pcd, hc = (int(v) for v in val.split()[:2])
        elif key == "View":
            view = int(val)
        elif key.startswith("Bounding box"):
            x, y, w, h, c = (int(float(v)) for v in val.split()[:5])
            boxes.append((x, y, w, h, c))
    return LabelA(Path(path).stem, d, pcd, hc, view, boxes)


def list_dataset_a() -> pd.DataFrame:
    rows = []
    for p in sorted(C.A_LABELS.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        lab = parse_label(p)
        rows.append(dict(image_id=lab.image_id, diagnosis=lab.diagnosis, pcd=lab.pcd, hc=lab.hc,
                         view=lab.view, n_boxes=len(lab.boxes)))
    return pd.DataFrame(rows)


def read_gray(path: Path) -> np.ndarray:
    """Read an image as uint8 grayscale (H, W). RGB files are converted."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_image_a(image_id: str) -> np.ndarray:
    return read_gray(C.A_IMAGES / f"{image_id}.jpg")


def load_mask_a(image_id: str) -> np.ndarray:
    """Kidney mask (H, W) uint8 {0,1}. Reads the PNG cache if present, else the `.mat` key `mask`."""
    png = C.MASKS_PNG / f"{image_id}.png"
    if png.exists():
        return (cv2.imread(str(png), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    m = sio.loadmat(str(C.A_MASKS_MAT / f"{image_id}.mat"))["mask"]
    return (m > 0).astype(np.uint8)


def load_label_a(image_id: str) -> LabelA:
    return parse_label(C.A_LABELS / f"{image_id}.txt")


# ---------------------------------------------------------------- dataset B
def list_dataset_b() -> pd.DataFrame:
    """KidneyUS [18] images with a non-empty capsule mask (sonographer 1), plus view, quality
    and native / transplant (taken from the comments field, as the README describes)."""
    df = pd.read_csv(C.DATA_B_CSV)
    meta = df.drop_duplicates("filename")[["filename", "file_attributes"]].copy()
    attrs = meta.file_attributes.map(json.loads)
    meta["view"] = attrs.map(lambda a: a.get("View", "n.r."))
    meta["quality"] = attrs.map(lambda a: a.get("Quality", "n.r."))
    meta["kidney_type"] = attrs.map(lambda a: "transplant" if "transpl" in str(a.get("Comments", "")).lower() else "native")
    meta = meta.drop(columns="file_attributes")
    keep = []
    for f in meta.filename:
        mp = C.DATA_B_CAPSULE / f
        ip = C.DATA_B_IMAGES / f
        ok = mp.exists() and ip.exists()
        if ok:
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            ok = m is not None and m.max() > 0
        keep.append(ok)
    return meta[np.array(keep)].reset_index(drop=True)


def load_image_b(filename: str) -> np.ndarray:
    return read_gray(C.DATA_B_IMAGES / filename)


def load_capsule_b(filename: str) -> np.ndarray:
    return (cv2.imread(str(C.DATA_B_CAPSULE / filename), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)


# ---------------------------------------------------------------- geometry helpers
def mask_to_box(mask: np.ndarray, margin: int = 0):
    """Tight (x0, y0, x1, y1) box of a binary mask, optionally with a margin. None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    return (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(w - 1, int(xs.max()) + margin), min(h - 1, int(ys.max()) + margin))
