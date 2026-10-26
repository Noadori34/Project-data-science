"""Sanity checks for kidney.metrics and the letterbox inverse.
Run: PYTHONPATH=src python -m pytest tests -q"""
import numpy as np

from kidney.metrics import bootstrap_ci, cls_scores, delong_test, dice, hd95, iou, spec_at_sens
from kidney.seg_common import letterbox, postprocess, unletterbox


def square(h, w, y0, x0, s):
    m = np.zeros((h, w), np.uint8)
    m[y0:y0 + s, x0:x0 + s] = 1
    return m


def test_dice_iou():
    a, b = square(50, 50, 10, 10, 10), square(50, 50, 10, 15, 10)  # half overlap
    assert abs(dice(a, b) - 0.5) < 1e-6
    assert abs(iou(a, b) - 1 / 3) < 1e-6
    assert dice(a, a) == 1.0 and dice(np.zeros_like(a), np.zeros_like(a)) == 1.0


def test_hd95_shift():
    a, b = square(60, 60, 10, 10, 20), square(60, 60, 10, 13, 20)  # 3 px shift
    assert 2.9 <= hd95(a, b) <= 3.01
    assert hd95(a, a) == 0.0
    assert np.isnan(hd95(a, np.zeros_like(a)))


def test_sp95_and_delong():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    s_good = y + rng.normal(0, 0.5, 400)
    s_bad = y + rng.normal(0, 2.0, 400)
    sp, _ = spec_at_sens(y, s_good)
    assert 0 < sp < 1
    r = cls_scores(y, s_good)
    assert r["sens"] >= 0.95
    d = delong_test(y, s_good, s_bad)
    assert d["auc1"] > d["auc2"] and d["p"] < 0.001
    assert delong_test(y, s_good, s_good)["p"] == 1.0


def test_bootstrap():
    lo, hi = bootstrap_ci(np.full(100, 0.9))
    assert abs(lo - 0.9) < 1e-9 and abs(hi - 0.9) < 1e-9


def test_letterbox_roundtrip():
    m = square(300, 450, 50, 100, 120)
    lb_img, lb = letterbox(m.astype(np.float32), 512)
    back = unletterbox(lb_img, lb) > 0.5
    assert back.shape == m.shape and dice(back, m) > 0.98


def test_postprocess():
    m = square(100, 100, 10, 10, 40) | square(100, 100, 80, 80, 5)
    m[20:25, 20:25] = 0  # hole
    pp = postprocess(m)
    assert pp[85, 85] == 0 and pp[22, 22] == 1
