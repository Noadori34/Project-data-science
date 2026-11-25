"""Phase 2: S-C MedSAM [S3] and S-D UltraSam [S1] with box prompts (plan.md 3.4).

Two prompt sources for each image of dataset A:
- `gt`   box from the true kidney mask: the upper bound.
- `auto` box from the S-A U-Net out-of-fold prediction (results/pred_masks/unet_r34): fully
         automatic. Because those masks are out-of-fold, no test image is seen in training.

Both models are SAM ViT-B networks loaded with `segment_anything`.
- MedSAM: official preprocessing (resize to 1024 x 1024, min-max to [0,1]), box scaled to 1024,
  logits upsampled to the original size (MedSAM_Inference.py in the MedSAM repo [S3]).
- UltraSam: released as an MMDetection checkpoint [S1]. `load_ultrasam` maps its keys onto the
  SAM layout and refuses to run if fewer than 95% of SAM weights are found, so a wrong conversion
  cannot silently produce numbers. Standard SAM preprocessing (longest side 1024, pad, pixel
  mean/std) via `SamPredictor`.

Weights (download once, not committed), into weights/:
- MedSAM:   medsam_vit_b.pth   https://zenodo.org/records/10689643
- UltraSam: UltraSam.pth       https://github.com/CAMMA-public/UltraSam (README, "checkpoint")

Run:
    python -m kidney.sam_prompt --model medsam   --prompt gt auto
    python -m kidney.sam_prompt --model ultrasam --prompt gt auto
    python -m kidney.sam_prompt --model medsam --prompt gt --smoke   # CPU, random weights, 3 images
Metrics are reported per fold from folds.csv so they line up with S-A / S-B.
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
from .seg_common import postprocess
from .splits import read_folds

MEDSAM_CKPT = C.WEIGHTS / "medsam_vit_b.pth"
ULTRASAM_CKPT = C.WEIGHTS / "UltraSam.pth"


# ================================================================ loading
def load_medsam(ckpt=MEDSAM_CKPT, random_init: bool = False):
    from segment_anything import sam_model_registry

    return sam_model_registry["vit_b"](checkpoint=None if random_init else str(ckpt))


_ATTN = ["layers.0.self_attn", "layers.0.cross_attn_token_to_image", "layers.0.cross_attn_image_to_token",
         "layers.1.self_attn", "layers.1.cross_attn_token_to_image", "layers.1.cross_attn_image_to_token",
         "final_attn_token_to_image"]


def mmdet_to_sam(sd: dict) -> dict:
    """Invert UltraSam's SAM -> MMDetection weight conversion (weights/convert_sam_bb_to_mmdet.py and
    weights/convert_sam_to_mmdet.py in the UltraSam repo [S1]) to get a `segment_anything` state dict.
    Checked by round trip in tests/test_ultrasam_mapping.py."""
    import torch

    out = {}
    for k, v in sd.items():
        for pre in ("module.", "model."):
            if k.startswith(pre):
                k = k[len(pre):]
        if k.startswith("backbone."):
            n = k[len("backbone."):]
            n = n.replace("ffn.layers.0.0", "mlp.lin1").replace("ffn.layers.1", "mlp.lin2")
            n = n.replace("channel_reduction", "neck").replace("ln", "norm")
            n = n.replace("patch_embed.projection", "patch_embed.proj").replace("layers.", "blocks.")
            out["image_encoder." + n] = v
        elif k.startswith("decoder."):
            if k.endswith(".attn.in_proj_bias"):
                q, kk, vv = torch.chunk(v, 3)
                base = k[: -len(".in_proj_bias")]
                for name, t in (("q_proj", q), ("k_proj", kk), ("v_proj", vv)):
                    out[_dec(f"{base}.{name}_bias")] = t
                continue
            out[_dec(k)] = v
        elif k.startswith("bbox_head."):
            out["mask_decoder." + k[len("bbox_head."):]] = v
        elif k == "prompt_encoder.label_encoder.label_embedding.weight":
            rows = v.split([1, 1, 1, 1, 1, 1, 4, 1])
            names = ["prompt_encoder.no_mask_embed.weight"] + [f"prompt_encoder.point_embeddings.{i}.weight" for i in range(4)] + [
                "prompt_encoder.not_a_point_embed.weight", "mask_decoder.mask_tokens.weight", "mask_decoder.iou_token.weight"]
            out.update(dict(zip(names, rows)))
        elif k.startswith("prompt_encoder."):
            out[k] = v
    return out


def _dec(k: str) -> str:
    """mmdet decoder key -> SAM mask_decoder.transformer key."""
    n = k[len("decoder."):]
    n = n.replace("mlp.layers.0.0", "mlp.lin1").replace("mlp.layers.1", "mlp.lin2").replace("post_norm", "norm_final_attn")
    for a in ("cross_attn_image_to_token", "cross_attn_token_to_image", "final_attn_token_to_image", "self_attn"):
        n = n.replace(f"{a}.attn", a)
    for p in ("q", "k", "v"):
        n = n.replace(f"{p}_proj_weight", f"{p}_proj.weight").replace(f"{p}_proj_bias", f"{p}_proj.bias")
    return "mask_decoder.transformer." + n


def load_ultrasam(ckpt=ULTRASAM_CKPT, random_init: bool = False, min_match: float = 0.95):
    import torch
    from segment_anything import sam_model_registry

    sam = sam_model_registry["vit_b"](checkpoint=None)
    if random_init:
        return sam
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd.get("model", sd))
    mapped = mmdet_to_sam(sd)
    target = sam.state_dict()
    ok = {k: v for k, v in mapped.items() if k in target and target[k].shape == v.shape}
    frac = len(ok) / len(target)
    print(f"UltraSam: matched {len(ok)}/{len(target)} SAM tensors ({frac:.1%})")
    if frac < min_match:
        missing = [k for k in target if k not in ok][:15]
        raise RuntimeError(
            f"UltraSam checkpoint maps to only {frac:.1%} of SAM weights. First unmatched SAM keys: {missing}. "
            f"Example checkpoint keys: {list(sd)[:15]}. Adjust mmdet_to_sam, or run the official "
            "MMDetection inference from the UltraSam repo and save masks to results/pred_masks/ultrasam_<prompt>/.")
    sam.load_state_dict(ok, strict=False)
    return sam


# ================================================================ inference
def medsam_predict(sam, img_gray: np.ndarray, box_xyxy, dev) -> np.ndarray:
    """MedSAM inference for one image and one box (MedSAM_Inference.py [S3]). Returns prob map (H, W)."""
    import torch
    import torch.nn.functional as F

    H, W = img_gray.shape
    img3 = np.repeat(img_gray[..., None], 3, axis=2)
    x = cv2.resize(img3, (1024, 1024), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    x = (x - x.min()) / max(x.max() - x.min(), 1e-8)
    x = torch.from_numpy(x).permute(2, 0, 1)[None].to(dev)
    box = np.array(box_xyxy, dtype=np.float32) / np.array([W, H, W, H]) * 1024
    with torch.no_grad():
        emb = sam.image_encoder(x)
        b = torch.as_tensor(box, device=dev)[None, None]
        sparse, dense = sam.prompt_encoder(points=None, boxes=b, masks=None)
        low, _ = sam.mask_decoder(image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
                                  sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                  multimask_output=False)
        p = torch.sigmoid(F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False))
    return p[0, 0].cpu().numpy()


def sam_predict(predictor, img_gray: np.ndarray, box_xyxy) -> np.ndarray:
    """Standard SAM preprocessing via SamPredictor (used for UltraSam)."""
    import torch

    predictor.set_image(np.repeat(img_gray[..., None], 3, axis=2))
    with torch.no_grad():
        logits, _, _ = predictor.predict(box=np.array(box_xyxy), multimask_output=False, return_logits=True)
    return 1 / (1 + np.exp(-logits[0]))


def get_box(iid: str, source: str, auto_from: str = "unet_r34", gt=None):
    if source == "gt":
        return mask_to_box(gt if gt is not None else load_mask_a(iid))
    p = C.PRED_MASKS / auto_from / f"{iid}.png"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: train S-A first (python -m kidney.train_seg --model unet_r34)")
    return mask_to_box(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0)


def run(model_name: str, prompts, smoke: bool = False, auto_from: str = "unet_r34", sam=None, tag: str | None = None):
    from .train_seg import append_run

    dev = C.device()
    if sam is None:
        loader = load_medsam if model_name == "medsam" else load_ultrasam
        sam = loader(random_init=smoke)
    sam = sam.to(dev).eval()
    predictor = None
    if model_name != "medsam":
        from segment_anything import SamPredictor

        predictor = SamPredictor(sam)

    folds = read_folds()
    if smoke:
        folds = folds.groupby("fold").head(1).head(3)
    name = tag or model_name
    for src in prompts:
        key = f"{name}_{src}"
        out_dir = C.PRED_MASKS / key
        out_dir.mkdir(parents=True, exist_ok=True)
        rows, t0 = [], time.time()
        for iid, fold in zip(folds.image_id, folds.fold):
            img, gt = load_image_a(iid), load_mask_a(iid)
            box = get_box(iid, src, auto_from, gt)
            if box is None:  # automatic box failed (empty U-Net mask): count as a miss
                pp = np.zeros_like(gt)
            else:
                prob = medsam_predict(sam, img, box, dev) if predictor is None else sam_predict(predictor, img, box)
                pp = postprocess(prob > 0.5)
            cv2.imwrite(str(out_dir / f"{iid}.png"), pp * 255)
            r = {"image_id": iid, "fold": int(fold), "prompt": src, "box_missing": box is None}
            r.update(seg_scores(pp, gt))
            rows.append(r)
        res = pd.DataFrame(rows)
        d = C.SEG_DIR / key
        d.mkdir(parents=True, exist_ok=True)
        res.to_csv(d / "per_image.csv", index=False)
        print(f"[{key}] Dice {res.dice.mean():.4f} IoU {res.iou.mean():.4f} ({len(res)} images)")
        for f, g in res.groupby("fold"):
            append_run(dict(timestamp=datetime.now().isoformat(timespec="seconds"), phase="seg", model=key,
                            pretraining="medical (MedSAM)" if model_name == "medsam" else "ultrasound (UltraSam)",
                            fold=int(f), best_epoch=None, val_dice=None, test_dice=g.dice.mean(), test_iou=g.iou.mean(),
                            test_hd95=g.hd95.mean(), test_dice_raw=None, n_test=len(g),
                            minutes=(time.time() - t0) / 60, smoke=smoke,
                            config=json.dumps({"prompt": src, "auto_from": auto_from}),
                            versions=json.dumps(C.library_versions())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["medsam", "ultrasam"])
    ap.add_argument("--prompt", nargs="+", default=["gt", "auto"], choices=["gt", "auto"])
    ap.add_argument("--auto-from", default="unet_r34", help="model whose OOF masks give the automatic box")
    ap.add_argument("--smoke", action="store_true", help="random weights, 3 images, CPU")
    a = ap.parse_args()
    run(a.model, a.prompt, a.smoke, a.auto_from)


if __name__ == "__main__":
    main()
