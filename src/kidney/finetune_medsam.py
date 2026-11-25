"""Phase 2 (optional, plan.md 3.4): fine-tune the MedSAM mask decoder on the training folds [S3].

The image and prompt encoders are frozen, so each image embedding is computed once and cached
(results/cache/medsam_emb, ~2 MB per image in fp16, git-ignored). Training follows the MedSAM
recipe [S3]: box prompt from the true mask with a random shift of up to 20 px (1024 scale),
Dice loss [S24] + BCE on the 256 x 256 low-resolution logits, AdamW [S30]. Early stopping on
validation-fold Dice; the test fold is scored once with `gt` and `auto` boxes, like sam_prompt.py.

Run:
    python -m kidney.finetune_medsam --folds 0 1 2 3 4
    python -m kidney.finetune_medsam --folds 0 --smoke     # CPU, random weights, few images
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from . import config as C
from .data import load_image_a, load_mask_a, mask_to_box
from .metrics import seg_scores
from .sam_prompt import get_box, load_medsam
from .seg_common import postprocess
from .splits import read_folds, split_ids

CACHE = C.RESULTS / "cache" / "medsam_emb"


def preprocess(img_gray):
    import torch

    x = cv2.resize(np.repeat(img_gray[..., None], 3, axis=2), (1024, 1024), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    x = (x - x.min()) / max(x.max() - x.min(), 1e-8)
    return torch.from_numpy(x).permute(2, 0, 1)[None]


def embedding(sam, iid, dev, cache=True):
    import torch

    f = CACHE / f"{iid}.npy"
    if cache and f.exists():
        return torch.from_numpy(np.load(f).astype(np.float32))[None].to(dev)
    with torch.no_grad():
        e = sam.image_encoder(preprocess(load_image_a(iid)).to(dev))
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(f, e[0].cpu().numpy().astype(np.float16))
    return e


def decode(sam, emb, box1024):
    import torch

    b = torch.as_tensor(np.asarray(box1024, np.float32), device=emb.device)[None, None]
    with torch.no_grad():
        sparse, dense = sam.prompt_encoder(points=None, boxes=b, masks=None)
    low, _ = sam.mask_decoder(image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
                              sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
    return low  # (1,1,256,256) logits


def to1024(box, h, w):
    return np.array(box, np.float32) / np.array([w, h, w, h]) * 1024


def jitter(box, rng, max_shift=20):
    b = box + rng.integers(-max_shift, max_shift + 1, size=4)
    return np.clip(b, 0, 1023)


def train_fold(args, fold, folds, sam_init_state):
    import segmentation_models_pytorch as smp
    import torch
    import torch.nn.functional as F

    from .train_seg import append_run

    dev = C.device()
    C.seed_everything(C.SEED + fold)
    rng = np.random.default_rng(C.SEED + fold)
    sam = load_medsam(random_init=args.smoke)
    sam.load_state_dict(sam_init_state)
    sam.to(dev)
    for p in list(sam.image_encoder.parameters()) + list(sam.prompt_encoder.parameters()):
        p.requires_grad = False
    sam.mask_decoder.train()

    ids = split_ids(fold, folds)
    if args.smoke:
        ids = {k: v[: args.smoke_n] for k, v in ids.items()}
    gts = {i: load_mask_a(i) for i in ids["train"] + ids["val"]}

    def target256(iid):
        return torch.from_numpy(cv2.resize(gts[iid], (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.float32))[None, None].to(dev)

    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=args.lr, weight_decay=0.01)
    ckpt = C.CHECKPOINTS / f"medsam_ft_decoder_fold{fold}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best, wait, best_ep = -1.0, 0, -1
    t0 = time.time()
    for ep in range(args.epochs):
        sam.mask_decoder.train()
        order = rng.permutation(ids["train"])
        losses = []
        for k in range(0, len(order), args.batch_size):
            opt.zero_grad(set_to_none=True)
            loss = 0
            batch = order[k:k + args.batch_size]
            for iid in batch:
                g = gts[iid]
                box = jitter(to1024(mask_to_box(g), *g.shape), rng)
                low = decode(sam, embedding(sam, iid, dev, cache=not args.smoke), box)
                t = target256(iid)
                loss = loss + dice_loss(low, t) + bce(low, t)
            (loss / len(batch)).backward()
            opt.step()
            losses.append(loss.item() / len(batch))
        sam.mask_decoder.eval()
        vd = []
        with torch.no_grad():
            for iid in ids["val"]:
                g = gts[iid]
                low = decode(sam, embedding(sam, iid, dev, cache=not args.smoke), to1024(mask_to_box(g), *g.shape))
                p = F.interpolate(low, size=g.shape, mode="bilinear", align_corners=False)[0, 0].cpu().numpy() > 0
                vd.append(seg_scores(p, g)["dice"])
        v = float(np.mean(vd))
        print(f"[medsam_ft f{fold}] ep {ep} loss {np.mean(losses):.4f} val Dice {v:.4f}")
        if v > best:
            best, best_ep, wait = v, ep, 0
            torch.save(sam.mask_decoder.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= args.patience:
                break

    sam.mask_decoder.load_state_dict(torch.load(ckpt, map_location=dev))
    sam.mask_decoder.eval()
    out = []
    for src in args.prompt:
        key = f"medsam_ft_{src}"
        (C.PRED_MASKS / key).mkdir(parents=True, exist_ok=True)
        for iid in ids["test"]:
            g = load_mask_a(iid)
            box = get_box(iid, src, args.auto_from, g)
            if box is None:
                pp = np.zeros_like(g)
            else:
                with torch.no_grad():
                    low = decode(sam, embedding(sam, iid, dev, cache=not args.smoke), to1024(box, *g.shape))
                    p = F.interpolate(low, size=g.shape, mode="bilinear", align_corners=False)[0, 0].cpu().numpy() > 0
                pp = postprocess(p)
            cv2.imwrite(str(C.PRED_MASKS / key / f"{iid}.png"), pp * 255)
            r = {"image_id": iid, "fold": fold, "prompt": src, "box_missing": box is None}
            r.update(seg_scores(pp, g))
            out.append(r)
        g = pd.DataFrame([r for r in out if r["prompt"] == src])
        append_run(dict(timestamp=datetime.now().isoformat(timespec="seconds"), phase="seg", model=key,
                        pretraining="medical (MedSAM), decoder fine-tuned", fold=fold, best_epoch=best_ep, val_dice=best,
                        test_dice=g.dice.mean(), test_iou=g.iou.mean(), test_hd95=g.hd95.mean(), test_dice_raw=None,
                        n_test=len(g), minutes=(time.time() - t0) / 60, smoke=args.smoke,
                        config=json.dumps({k: v for k, v in vars(args).items() if k != "folds"}),
                        versions=json.dumps(C.library_versions())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(C.N_FOLDS)))
    ap.add_argument("--prompt", nargs="+", default=["gt", "auto"], choices=["gt", "auto"])
    ap.add_argument("--auto-from", default="unet_r34")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=3)
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.prompt = 1, ["gt"]
    init = load_medsam(random_init=args.smoke).state_dict()
    folds = read_folds()
    res = pd.concat([train_fold(args, f, folds, init) for f in args.folds])
    for src, g in res.groupby("prompt"):
        d = C.SEG_DIR / f"medsam_ft_{src}"
        d.mkdir(parents=True, exist_ok=True)
        f = d / "per_image.csv"
        if f.exists() and not args.smoke:
            old = pd.read_csv(f, dtype={"image_id": str})
            g = pd.concat([old[~old.fold.isin(args.folds)], g])
        g.to_csv(f, index=False)
        print(f"[medsam_ft_{src}] Dice {g.dice.mean():.4f}")


if __name__ == "__main__":
    main()
