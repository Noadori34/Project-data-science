"""Phase 2: train S-A (U-Net ResNet-34) and S-B (DeepLabV3+) under the shared 5-fold protocol.

Plan.md 3.3: letterbox to 512, same augmentation for image and mask, Dice loss [S24] + BCE,
AdamW [S30] with cosine schedule, early stopping on validation-fold Dice, largest-region +
hole-filling post-processing. The test fold is used once, after selection on the validation fold
(plan 1.2). Metrics: Dice, IoU [24], HD95 [S29] at the original image resolution.

Outputs per model
- results/checkpoints/{model}_fold{k}.pt                       best-epoch weights (git-ignored)
- results/pred_masks/{model}/{id}.png                          out-of-fold predicted masks (0/255)
- results/segmentation/{model}/per_image.csv                   per-image metrics, raw and post-processed
- results/segmentation/{model}/history_fold{k}.csv             loss / val Dice per epoch
- results/runs.csv                                             one row per (model, fold)

Run on the GPU:
    python -m kidney.train_seg --model unet_r34 --folds 0 1 2 3 4
    python -m kidney.train_seg --model deeplabv3p_r50
CPU smoke test (a few images, 1 epoch, small input, no download):
    python -m kidney.train_seg --model unet_r34 --folds 0 --smoke
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
from .data import load_image_a, load_mask_a
from .metrics import seg_scores
from .seg_common import KidneySegDataset, build_model, postprocess, predict_probs, to_network_input
from .splits import fold_roles, read_folds, split_ids


def append_run(row: dict) -> None:
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(C.RUNS_CSV, mode="a", header=not C.RUNS_CSV.exists(), index=False)


def batch_dice(logits, target, eps=1e-6) -> float:
    import torch

    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * target).sum((1, 2, 3))
    s = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return ((2 * inter + eps) / (s + eps)).mean().item()


def train_fold(args, fold: int, folds: pd.DataFrame) -> pd.DataFrame:
    import segmentation_models_pytorch as smp
    import torch
    from torch.utils.data import DataLoader

    C.seed_everything(C.SEED + fold)
    dev = C.device()
    ids = split_ids(fold, folds)
    if args.smoke:
        ids = {k: v[: args.smoke_n] for k, v in ids.items()}

    tr = KidneySegDataset(ids["train"], size=args.size, train=True)
    va = KidneySegDataset(ids["val"], size=args.size, train=False)
    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                       drop_last=len(tr) > args.batch_size, persistent_workers=args.workers > 0)
    dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(args.model, pretrained=not args.no_pretrained).to(dev)
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)  # [S24]
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)  # [S30]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt = C.CHECKPOINTS / f"{args.model}_fold{fold}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best, best_ep, wait, hist = -1.0, -1, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        losses = []
        for x, y, _ in dl_tr:
            x, y = to_network_input(x.to(dev)), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                out = model(x)
                loss = dice_loss(out, y) + bce(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()

        model.eval()
        vd = []
        with torch.no_grad():
            for x, y, _ in dl_va:
                with torch.autocast(device_type=dev.type, enabled=use_amp):
                    out = model(to_network_input(x.to(dev)))
                vd.append(batch_dice(out.float(), y.to(dev)) * len(x))
        val_dice = float(np.sum(vd) / max(len(va), 1))
        hist.append(dict(epoch=ep, train_loss=float(np.mean(losses)), val_dice=val_dice, lr=sched.get_last_lr()[0]))
        print(f"[{args.model} f{fold}] ep {ep:3d} loss {hist[-1]['train_loss']:.4f} val Dice {val_dice:.4f}")
        if val_dice > best:  # early stopping on validation-fold Dice, keep best epoch
            best, best_ep, wait = val_dice, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"early stop at epoch {ep}, best epoch {best_ep}")
                break

    out_dir = C.SEG_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(out_dir / f"history_fold{fold}.csv", index=False)

    # ---- test fold, used once, at original resolution
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    pred_dir = C.PRED_MASKS / args.model
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for iid in ids["test"]:
        img, gt = load_image_a(iid), load_mask_a(iid)
        prob = predict_probs(model, img, args.size, dev)
        raw = (prob > 0.5).astype(np.uint8)
        pp = postprocess(raw)
        cv2.imwrite(str(pred_dir / f"{iid}.png"), pp * 255)
        r = {"image_id": iid, "fold": fold}
        r.update({f"{k}_raw": v for k, v in seg_scores(raw, gt).items()})
        r.update(seg_scores(pp, gt))
        rows.append(r)
    res = pd.DataFrame(rows)
    print(f"[{args.model} f{fold}] TEST Dice {res.dice.mean():.4f} IoU {res.iou.mean():.4f} HD95 {res.hd95.median():.1f}px")

    append_run(dict(
        timestamp=datetime.now().isoformat(timespec="seconds"), phase="seg", model=args.model,
        pretraining="none" if args.no_pretrained else "ImageNet", fold=fold, best_epoch=best_ep, val_dice=best,
        test_dice=res.dice.mean(), test_iou=res.iou.mean(), test_hd95=res.hd95.mean(),
        test_dice_raw=res.dice_raw.mean(), n_test=len(res), minutes=(time.time() - t0) / 60,
        smoke=args.smoke, config=json.dumps({k: v for k, v in vars(args).items() if k != "folds"}),
        versions=json.dumps(C.library_versions()),
    ))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["unet_r34", "deeplabv3p_r50"])
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(C.N_FOLDS)))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="CPU check: few images, 1 epoch, 128 px, no download")
    ap.add_argument("--smoke-n", type=int, default=6)
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.size, args.batch_size, args.workers, args.no_pretrained = 1, 128, 2, 0, True

    folds = read_folds()
    print("fold roles:", {f: fold_roles(f) for f in args.folds})
    all_res = []
    for f in args.folds:
        all_res.append(train_fold(args, f, folds))
    res = pd.concat(all_res)
    out = C.SEG_DIR / args.model / "per_image.csv"
    if out.exists() and not args.smoke:  # keep rows of folds trained in earlier calls
        old = pd.read_csv(out, dtype={"image_id": str})
        res = pd.concat([old[~old.fold.isin(args.folds)], res])
    res.sort_values(["fold", "image_id"]).to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
