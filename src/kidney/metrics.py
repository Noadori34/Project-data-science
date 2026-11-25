"""Metric functions shared by all phases (plan.md 1.3).

Segmentation: Dice and IoU [24], HD95 [S29].
Classification: AUC and SP@95 [24], sensitivity / specificity / F1 at a threshold [S25].
Uncertainty: mean +- SD over folds, 95% bootstrap CI on pooled out-of-fold values [S27].
Model comparison: DeLong test for two correlated AUCs [S26].
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, stats


# ================================================================ segmentation
def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    s = p.sum() + g.sum()
    if s == 0:
        return 1.0
    return float(2 * np.logical_and(p, g).sum() / s)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    u = np.logical_or(p, g).sum()
    if u == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / u)


def _surface(m: np.ndarray) -> np.ndarray:
    return m ^ ndimage.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)


def hd95(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0)) -> float:
    """95th percentile symmetric Hausdorff distance between mask borders, in pixels unless a
    spacing is given [S29]. NaN if one mask is empty and the other is not."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float("nan")
    sp, sg = _surface(p), _surface(g)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([dt_g[sp], dt_p[sg]])
    return float(np.percentile(d, 95))


def seg_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {"dice": dice(pred, gt), "iou": iou(pred, gt), "hd95": hd95(pred, gt)}


# ================================================================ classification
def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def spec_at_sens(y: np.ndarray, s: np.ndarray, sens: float = 0.95) -> tuple[float, float]:
    """Specificity at the first ROC point with sensitivity >= `sens` (SP@95 in [24]).
    Returns (specificity, threshold)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.argmax(tpr >= sens))
    return float(1 - fpr[i]), float(thr[i])


def threshold_metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    from sklearn.metrics import f1_score

    y = np.asarray(y).astype(int)
    p = (np.asarray(s) >= thr).astype(int)
    tp, tn = int(((p == 1) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    fp, fn = int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
    return {"sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1), "f1": float(f1_score(y, p, zero_division=0))}


def cls_scores(y, s, thr: float | None = None) -> dict:
    """AUC, SP@95 and threshold metrics. If no threshold is given, the SP@95 operating point is used."""
    y, s = np.asarray(y), np.asarray(s)
    sp95, t95 = spec_at_sens(y, s)
    out = {"auc": auc(y, s), "sp95": sp95, "thr": t95 if thr is None else thr}
    out.update(threshold_metrics(y, s, out["thr"]))
    return out


# ================================================================ uncertainty
def mean_sd(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def bootstrap_ci(values=None, *, y=None, s=None, fn=None, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42):
    """95% percentile bootstrap CI [S27].

    - Per-image metrics (Dice, IoU, HD95): pass `values`; the CI is for their mean.
    - Pooled metrics (AUC, SP@95): pass `y`, `s` and `fn(y, s)`; cases are resampled.
    """
    rng = np.random.default_rng(seed)
    stats_ = []
    if values is not None:
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        for _ in range(n_boot):
            stats_.append(v[rng.integers(0, len(v), len(v))].mean())
    else:
        y, s = np.asarray(y), np.asarray(s)
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) < 2:
                continue
            stats_.append(fn(y[i], s[i]))
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ================================================================ DeLong [S26]
def _midrank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def delong_test(y, s1, s2) -> dict:
    """Two-sided DeLong test for two correlated ROC AUCs on the same cases [S26]
    (fast structural-components form)."""
    y = np.asarray(y).astype(int)
    s = np.vstack([np.asarray(s1, float), np.asarray(s2, float)])
    pos, neg = s[:, y == 1], s[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    aucs, v10, v01 = [], [], []
    for k in range(2):
        tx, ty, tz = _midrank(pos[k]), _midrank(neg[k]), _midrank(np.concatenate([pos[k], neg[k]]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n)
        v01.append(1 - (tz[m:] - ty) / m)
    v10, v01 = np.array(v10), np.array(v01)
    cov = np.cov(v10) / m + np.cov(v01) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "z": float(z), "p": float(p)}
