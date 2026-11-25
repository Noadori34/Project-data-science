"""Phase 2, step 3.1: convert dataset A `.mat` masks to PNG and check them.

Outputs
- results/masks_png/{id}.png                kidney masks (0/255)
- results/segmentation/mask_check.csv       per image: size match, n components, holes, box coverage
- results/segmentation/overlays/*.png       sample overlays: image + mask outline + lesion boxes

Run: python -m kidney.prepare_masks [--n-overlays 40]
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import ndimage

from . import config as C
from .data import list_dataset_a, load_image_a, load_label_a

BOX_COLORS = {1: (0, 165, 255), 2: (255, 255, 0), 3: (255, 0, 255), 4: (0, 255, 255), 5: (0, 0, 255),
              6: (128, 0, 255), 7: (255, 0, 0), 8: (0, 128, 255), 9: (128, 128, 0)}


def overlay(img: np.ndarray, mask: np.ndarray, boxes=(), pred: np.ndarray | None = None, text: str = "") -> np.ndarray:
    """Green = reference mask outline, red = predicted outline, boxes coloured by class."""
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cs, -1, (0, 255, 0), 2)
    if pred is not None:
        cs, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cs, -1, (0, 0, 255), 2)
    for x, y, w, h, c in boxes:
        cv2.rectangle(rgb, (x, y), (x + w, y + h), BOX_COLORS.get(c, (255, 255, 255)), 1)
        cv2.putText(rgb, C.BOX_CLASSES.get(c, str(c)), (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    BOX_COLORS.get(c, (255, 255, 255)), 1)
    if text:
        cv2.putText(rgb, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return rgb


def main(n_overlays: int = 40, limit: int | None = None):
    C.MASKS_PNG.mkdir(parents=True, exist_ok=True)
    (C.SEG_DIR / "overlays").mkdir(parents=True, exist_ok=True)
    df = list_dataset_a()
    if limit:
        df = df.head(limit)
    rng = np.random.default_rng(C.SEED)
    show = set(rng.choice(df.image_id, size=min(n_overlays, len(df)), replace=False))
    rows = []
    for iid in df.image_id:
        m = (sio.loadmat(str(C.A_MASKS_MAT / f"{iid}.mat"))["mask"] > 0).astype(np.uint8)
        img = load_image_a(iid)
        cv2.imwrite(str(C.MASKS_PNG / f"{iid}.png"), m * 255)
        lab, n = ndimage.label(m)
        holes = int(ndimage.binary_fill_holes(m).sum() - m.sum())
        L = load_label_a(iid)
        inside = [m[min(y + h // 2, m.shape[0] - 1), min(x + w // 2, m.shape[1] - 1)] > 0 for x, y, w, h, _ in L.boxes]
        rows.append(dict(image_id=iid, h=m.shape[0], w=m.shape[1], size_match=img.shape == m.shape,
                         area_frac=float(m.mean()), n_components=int(n), hole_px=holes,
                         n_boxes=len(L.boxes), boxes_centre_in_mask=int(sum(inside)), diagnosis=L.diagnosis))
        if iid in show:
            cv2.imwrite(str(C.SEG_DIR / "overlays" / f"{iid}.png"),
                        overlay(img, m, L.boxes, text=f"#{iid} dx={L.diagnosis}"))
    chk = pd.DataFrame(rows)
    chk.to_csv(C.SEG_DIR / "mask_check.csv", index=False)
    print(f"{len(chk)} masks converted")
    print("size mismatch:", int((~chk.size_match).sum()))
    print("empty masks:", int((chk.area_frac == 0).sum()))
    print("masks with >1 component:", int((chk.n_components > 1).sum()), "| with holes:", int((chk.hole_px > 0).sum()))
    nb = chk.n_boxes.sum()
    print(f"lesion box centres inside the kidney mask: {chk.boxes_centre_in_mask.sum()}/{nb}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-overlays", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    a = ap.parse_args()
    main(a.n_overlays, a.limit)
