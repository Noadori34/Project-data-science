"""Phase 2, plan.md 3.5: external kidney segmentation test on KidneyUS [18] (dataset B).

KidneyUS is never used for training or model selection. For every model:
- Dice / IoU [24] and HD95 [S29] against the capsule outline (sonographer 1, reviewed masks);
- the same against an ellipse fitted to the capsule, to separate model error from the difference
  in outline style (dataset A masks are smooth polygons [24], KidneyUS capsules follow the border [18]);
- results split by view, quality and native vs transplant.

CNNs (S-A, S-B): each of the 5 fold models predicts, scores are given per fold model (mean +- SD)
and for the 5-model average probability ("ens").
SAM models (S-C, S-D): `gt` box from the capsule, `auto` box from the S-A ensemble prediction.

Outputs: results/segmentation/external/{model}_per_image.csv and external_summary.csv
Run:
    python -m kidney.eval_external --models unet_r34 deeplabv3p_r50 medsam ultrasam medsam_ft
    python -m kidney.eval_external --models unet_r34 --smoke
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from . import config as C
from .data import list_dataset_b, load_capsule_b, load_image_b, mask_to_box
from .metrics import bootstrap_ci, mean_sd, seg_scores
from .seg_common import build_model, postprocess, predict_probs

OUT = C.SEG_DIR / "external"


def fit_ellipse_mask(mask: np.ndarray) -> np.ndarray:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return mask.copy()
    c = max(cs, key=cv2.contourArea)
    if len(c) < 5:
        return mask.copy()
    e = np.zeros_like(mask, dtype=np.uint8)
    cv2.ellipse(e, cv2.fitEllipse(c), 1, thickness=-1)
    return e


def score_row(pred, cap, ell):
    r = {}
    r.update({f"{k}": v for k, v in seg_scores(pred, cap).items()})
    r.update({f"{k}_ellipse": v for k, v in seg_scores(pred, ell).items()})
    return r


def cnn_models(name, smoke):
    import torch

    dev = C.device()
    models = []
    for f in range(C.N_FOLDS):
        ck = C.CHECKPOINTS / f"{name}_fold{f}.pt"
        if not ck.exists():
            if smoke:
                continue
            raise FileNotFoundError(f"{ck}: train {name} first")
        m = build_model(name, pretrained=False).to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev))
        models.append((f, m.eval()))
    if not models:  # smoke without checkpoints: random weights
        models = [(0, build_model(name, pretrained=False).to(dev).eval())]
    return models, dev


def run_cnn(name, meta, size, smoke):
    models, dev = cnn_models(name, smoke)
    rows, ens_masks = [], {}
    for _, r in meta.iterrows():
        img, cap = load_image_b(r.filename), load_capsule_b(r.filename)
        ell = fit_ellipse_mask(cap)
        probs = []
        for f, m in models:
            p = predict_probs(m, img, size, dev)
            probs.append(p)
            rows.append(dict(filename=r.filename, member=f"fold{f}", **score_row(postprocess(p > 0.5), cap, ell)))
        ens = postprocess(np.mean(probs, 0) > 0.5)
        ens_masks[r.filename] = ens
        rows.append(dict(filename=r.filename, member="ens", **score_row(ens, cap, ell)))
    return pd.DataFrame(rows), ens_masks


def run_sam(name, meta, prompts, smoke, auto_masks):
    import torch
    import torch.nn.functional as F

    from .finetune_medsam import decode, preprocess, to1024
    from .sam_prompt import load_medsam, load_ultrasam, medsam_predict, sam_predict

    dev = C.device()
    decoders = None
    if name == "ultrasam":
        from segment_anything import SamPredictor

        sam = load_ultrasam(random_init=smoke).to(dev).eval()
        pred_fn = SamPredictor(sam)
    else:
        sam = load_medsam(random_init=smoke).to(dev).eval()
        if name == "medsam_ft":
            decoders = [torch.load(C.CHECKPOINTS / f"medsam_ft_decoder_fold{f}.pt", map_location=dev)
                        for f in range(C.N_FOLDS) if (C.CHECKPOINTS / f"medsam_ft_decoder_fold{f}.pt").exists()]
            if not decoders and not smoke:
                raise FileNotFoundError("no fine-tuned MedSAM decoders; run kidney.finetune_medsam first")
    rows = []
    for _, r in meta.iterrows():
        img, cap = load_image_b(r.filename), load_capsule_b(r.filename)
        ell = fit_ellipse_mask(cap)
        for src in prompts:
            box = mask_to_box(cap) if src == "gt" else mask_to_box(auto_masks.get(r.filename, np.zeros(1)))
            if box is None:
                pred = np.zeros_like(cap)
            elif name == "ultrasam":
                pred = postprocess(sam_predict(pred_fn, img, box) > 0.5)
            elif decoders:  # average the 5 fine-tuned decoders on one shared embedding
                with torch.no_grad():
                    emb = sam.image_encoder(preprocess(img).to(dev))
                    ps = []
                    for sd in decoders:
                        sam.mask_decoder.load_state_dict(sd)
                        low = decode(sam, emb, to1024(box, *cap.shape))
                        ps.append(torch.sigmoid(F.interpolate(low, size=cap.shape, mode="bilinear", align_corners=False))[0, 0].cpu().numpy())
                pred = postprocess(np.mean(ps, 0) > 0.5)
            else:
                pred = postprocess(medsam_predict(sam, img, box, dev) > 0.5)
            rows.append(dict(filename=r.filename, member=src, box_missing=box is None, **score_row(pred, cap, ell)))
    return pd.DataFrame(rows)


def summarise(df, meta, model):
    d = df.merge(meta, on="filename")
    out = []
    for member, g in d.groupby("member"):
        for col, val in [("all", "all")] + [(c, v) for c in ("view", "quality", "kidney_type") for v in sorted(d[c].unique())]:
            gg = g if col == "all" else g[g[col] == val]
            if len(gg) == 0:
                continue
            lo, hi = bootstrap_ci(gg.dice.values)
            out.append(dict(model=model, member=member, subset="all" if col == "all" else f"{col}={val}", n=len(gg),
                            dice=gg.dice.mean(), dice_ci=f"[{lo:.3f}, {hi:.3f}]", iou=gg.iou.mean(),
                            hd95_median=gg.hd95.median(), dice_ellipse=gg.dice_ellipse.mean()))
    s = pd.DataFrame(out)
    folds = s[(s.subset == "all") & s.member.str.startswith("fold")]
    if len(folds) > 1:
        m, sd = mean_sd(folds.dice)
        print(f"[{model}] fold models on KidneyUS: Dice {m:.4f} +- {sd:.4f}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["unet_r34", "deeplabv3p_r50", "medsam", "ultrasam"])
    ap.add_argument("--prompt", nargs="+", default=["gt", "auto"], choices=["gt", "auto"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.size = 128
    OUT.mkdir(parents=True, exist_ok=True)
    meta = list_dataset_b()
    if a.smoke:
        meta = meta.head(3)
    print(f"KidneyUS images with a capsule mask: {len(meta)}")

    auto_masks, summaries = {}, []
    order = sorted(a.models, key=lambda m: m != "unet_r34")  # S-A first: its masks give the auto boxes
    for name in order:
        if name in ("unet_r34", "deeplabv3p_r50"):
            df, ens = run_cnn(name, meta, a.size, a.smoke)
            if name == "unet_r34":
                auto_masks = ens
        else:
            prompts = a.prompt if auto_masks else [p for p in a.prompt if p == "gt"]
            df = run_sam(name, meta, prompts, a.smoke, auto_masks)
        df.to_csv(OUT / f"{name}_per_image.csv", index=False)
        s = summarise(df, meta, name)
        summaries.append(s)
        print(s[s.subset == "all"].to_string(index=False))
    pd.concat(summaries).to_csv(OUT / "external_summary.csv", index=False)
    print("saved", OUT / "external_summary.csv")


if __name__ == "__main__":
    main()
