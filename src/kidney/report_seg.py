"""Phase 2 reporting: results table S-A to S-D, kidney_boxes.csv and failure-case overlays.

- Table (plan.md 6): per model, mean +- SD over the 5 test folds of the fold-mean Dice / IoU / HD95,
  plus a 95% bootstrap CI [S27] on the pooled out-of-fold per-image Dice; compared with
  URI-CADS-Segmentation [24] (Dice 91.23, IoU 84.99) and its DeepLabV3+ (89.34, 81.80).
- kidney_boxes.csv: kidney box per image from the chosen model's out-of-fold masks, used by
  Phase 3 (kidney crop) and by the automatic SAM prompts.
- Failure overlays: the lowest-Dice test images per model (green = reference, red = prediction).

Run: python -m kidney.report_seg [--boxes-from unet_r34] [--n-fail 16]
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from . import config as C
from .data import load_image_a, load_label_a, load_mask_a, mask_to_box
from .metrics import bootstrap_ci, mean_sd
from .prepare_masks import overlay
from .splits import read_folds

BASELINES = [
    dict(model="URI-CADS-Segmentation [24]", dice="91.23", iou="84.99", hd95="n.r.", dice_ci="", vs_24="baseline"),
    dict(model="DeepLabV3+ in [24]", dice="89.34", iou="81.80", hd95="n.r.", dice_ci="", vs_24=""),
]
TARGET_DICE = 91.23


def model_dirs():
    return sorted(p.parent.name for p in C.SEG_DIR.glob("*/per_image.csv"))


def results_table() -> pd.DataFrame:
    rows = []
    for m in model_dirs():
        d = pd.read_csv(C.SEG_DIR / m / "per_image.csv", dtype={"image_id": str})
        fm = d.groupby("fold")[["dice", "iou", "hd95"]].mean()
        md, sdd = mean_sd(fm.dice * 100)
        mi, sdi = mean_sd(fm.iou * 100)
        mh, sdh = mean_sd(fm.hd95)
        lo, hi = bootstrap_ci(d.dice.values * 100)
        rows.append(dict(model=m, n_folds=d.fold.nunique(), n_images=len(d),
                         dice=f"{md:.2f} ± {sdd:.2f}", iou=f"{mi:.2f} ± {sdi:.2f}", hd95=f"{mh:.1f} ± {sdh:.1f}",
                         dice_ci=f"[{lo:.2f}, {hi:.2f}]",
                         vs_24=("beats" if md > TARGET_DICE else "below") + f" ({md - TARGET_DICE:+.2f})"))
    return pd.concat([pd.DataFrame(BASELINES), pd.DataFrame(rows)], ignore_index=True)


def kidney_boxes(model: str, margin: int = 0) -> pd.DataFrame:
    folds = read_folds()
    rows = []
    for iid, f in zip(folds.image_id, folds.fold):
        p = C.PRED_MASKS / model / f"{iid}.png"
        pred = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
        b = mask_to_box(pred > 0, margin) if pred is not None else None
        g = mask_to_box(load_mask_a(iid))
        rows.append(dict(image_id=iid, fold=f, model=model,
                         x0=b[0] if b else None, y0=b[1] if b else None, x1=b[2] if b else None, y1=b[3] if b else None,
                         gt_x0=g[0], gt_y0=g[1], gt_x1=g[2], gt_y1=g[3],
                         pred_area=int((pred > 0).sum()) if pred is not None else None))
    return pd.DataFrame(rows)


def failure_overlays(model: str, n: int = 16):
    d = pd.read_csv(C.SEG_DIR / model / "per_image.csv", dtype={"image_id": str}).nsmallest(n, "dice")
    out = C.SEG_DIR / "failures" / model
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for _, r in d.iterrows():
        p = C.PRED_MASKS / model / f"{r.image_id}.png"
        if not p.exists():
            continue
        img, gt = load_image_a(r.image_id), load_mask_a(r.image_id)
        pred = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0
        ov = overlay(img, gt, load_label_a(r.image_id).boxes, pred, f"#{r.image_id} Dice {r.dice:.2f}")
        cv2.imwrite(str(out / f"{r.image_id}.png"), ov)
        tiles.append(cv2.resize(ov, (320, 320)))
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(C.SEG_DIR / "failures" / f"{model}_grid.png"), grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes-from", default="unet_r34")
    ap.add_argument("--n-fail", type=int, default=16)
    a = ap.parse_args()
    t = results_table()
    t.to_csv(C.SEG_DIR / "summary.csv", index=False)
    try:
        md = t.to_markdown(index=False)
    except ImportError:  # tabulate not installed
        md = t.to_string(index=False)
    (C.SEG_DIR / "summary.md").write_text(md)
    print(t.to_string(index=False))
    if (C.PRED_MASKS / a.boxes_from).exists():
        kb = kidney_boxes(a.boxes_from)
        kb.to_csv(C.RESULTS / "kidney_boxes.csv", index=False)
        print("saved", C.RESULTS / "kidney_boxes.csv", f"({kb.x0.notna().sum()} boxes)")
    for m in model_dirs():
        failure_overlays(m, a.n_fail)
    print("failure overlays in", C.SEG_DIR / "failures")


if __name__ == "__main__":
    main()
