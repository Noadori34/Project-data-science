"""Figures for the official Phase 2 report (Documents/official_report.md).

Reads only files written by the pipeline (results/...), so it can be re-run after any new experiment:
    PYTHONPATH=src python scripts/make_figures.py [--out Documents/figures]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from kidney import config as C
from kidney.data import load_image_a, load_mask_a

# Reference palette (dataviz skill, light mode): fixed slot order, text in ink colours, baseline in grey.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, SURFACE, BASE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb", "#9a9994"
TARGET = 91.23  # URI-CADS-Segmentation Dice [24]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})

SEG = C.SEG_DIR


def per_image(model):
    return pd.read_csv(SEG / model / "per_image.csv", dtype={"image_id": str})


def labels_a() -> pd.DataFrame:
    rows = []
    for p in sorted(C.A_LABELS.glob("*.txt")):
        t = p.read_text()
        g = re.search(r"Global pathologies:\s*(\d)\s+(\d)", t)
        v = re.search(r"View:\s*(\d+)", t)
        cls = {int(l.split()[-1]) for l in re.findall(r"Bounding box \d+:.*", t)}
        rows.append(dict(image_id=p.stem, PCD=int(g.group(1)), HC=int(g.group(2)),  # flag 1 = PCD, flag 2 = HC [24]
                         view=int(v.group(1)) if v else -1, cls=cls))
    return pd.DataFrame(rows)


def save(fig, out, name):
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", out / name)


# ------------------------------------------------------------------ figure 1: pipeline
def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_axis_off(); ax.set_xlim(0, 11); ax.set_ylim(0, 3.6)

    def box(x, y, w, h, title, body, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=edge, lw=1.6))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.68, body, ha="center", va="top", fontsize=8.5, color=INK2, linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))

    box(0.1, 1.0, 2.1, 2.3, "Dataset A", "URI-CADS [24]\n1,985 images\nkidney masks\n5 leak-free folds\n(3 train / 1 val / 1 test)", BLUE)
    box(2.8, 2.0, 2.5, 1.45, "CNNs (trained)", "U-Net ResNet-34\nDeepLabV3+ ResNet-50", ORANGE)
    box(2.8, 0.15, 2.5, 1.55, "SAM models (box prompt)", "MedSAM [S3], UltraSam [S1]\nzero-shot + MedSAM\ndecoder fine-tuned", VIOLET)
    box(6.0, 1.0, 2.2, 2.3, "Evaluation", "out-of-fold test\nDice, IoU, HD95\n95 % bootstrap CI\nvs URI-CADS [24]", AQUA)
    box(8.8, 1.0, 2.1, 2.3, "External test", "KidneyUS [18]\n487 images\nother hospital,\nnever trained on", GREEN)
    arrow(2.2, 2.5, 2.8, 2.7); arrow(2.2, 1.8, 2.8, 1.0)
    arrow(4.05, 2.0, 4.05, 1.7)
    ax.text(4.15, 1.85, "auto box", fontsize=7.5, color=MUTED, va="center")
    arrow(5.3, 2.7, 6.0, 2.4); arrow(5.3, 0.9, 6.0, 1.6); arrow(8.2, 2.15, 8.8, 2.15)
    save(fig, out, "fig1_pipeline.png")


# ------------------------------------------------------------------ figure 2: data overview
def fig_data(out, lab):
    mc = pd.read_csv(SEG / "mask_check.csv", dtype={"image_id": str})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    ax = axes[0]
    n = mc.diagnosis.value_counts().sort_index()
    ax.bar(["Healthy", "Pathological"], n.values, color=[BLUE, ORANGE], width=0.6)
    for i, v in enumerate(n.values):
        ax.text(i, v + 25, f"{v:,} ({v / n.sum():.0%})", ha="center", fontsize=9, color=INK)
    ax.set_title("A. Diagnosis"); ax.set_ylabel("Images"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 1750)
    ax = axes[1]
    vc = lab.view.value_counts().sort_index()
    ax.bar(["View 0\n(transverse)", "View 1\n(longitudinal)"], vc.values, color=[BLUE, ORANGE], width=0.6)
    for i, v in enumerate(vc.values):
        ax.text(i, v + 25, f"{v:,} ({v / vc.sum():.0%})", ha="center", fontsize=9, color=INK)
    ax.set_title("B. View (as reported in [24])"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 2100)
    ax = axes[2]
    ax.hist(mc.area_frac * 100, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    med = mc.area_frac.median() * 100
    ax.axvline(med, color=INK2, lw=1.2, ls="--")
    ax.text(med + 6, ax.get_ylim()[1] * 0.85, f"median {med:.1f} %\n(dashed line)", color=INK2, fontsize=9)
    ax.set_title("C. Kidney size (mask area, % of image)"); ax.set_xlabel("% of image"); ax.set_ylabel("Images")
    fig.tight_layout()
    save(fig, out, "fig2_data_overview.png")


# ------------------------------------------------------------------ figure 3: main comparison with [24]
def fig_comparison(out):
    s = pd.read_csv(SEG / "summary.csv")
    name = {"unet_r34": "U-Net ResNet-34", "deeplabv3p_r50": "DeepLabV3+ ResNet-50",
            "medsam_auto": "MedSAM zero-shot", "ultrasam_auto": "UltraSam zero-shot", "medsam_ft_auto": "MedSAM fine-tuned",
            "medsam_gt": "MedSAM zero-shot", "ultrasam_gt": "UltraSam zero-shot", "medsam_ft_gt": "MedSAM fine-tuned"}
    rows = [("TN-SCUI2020 winner [S13]", 87.23, None, "base"), ("DeepLabV3+ in [24]", 89.34, None, "base"),
            ("URI-CADS full [24]", 89.38, None, "base"), ("URI-CADS-Segmentation [24]", 91.23, None, "base")]
    for m, grp in [("medsam_auto", "auto"), ("ultrasam_auto", "auto"), ("medsam_ft_auto", "auto"),
                   ("unet_r34", "cnn"), ("deeplabv3p_r50", "cnn"),
                   ("medsam_gt", "gt"), ("ultrasam_gt", "gt"), ("medsam_ft_gt", "gt")]:
        r = s[s.model == m].iloc[0]
        lo, hi = map(float, r.dice_ci.strip("[]").split(","))
        rows.append((name[m] + (" (true box)" if grp == "gt" else " (auto box)" if grp == "auto" else ""),
                     float(r.dice.split("±")[0]), (lo, hi), grp))
    col = {"base": BASE, "cnn": ORANGE, "auto": BLUE, "gt": AQUA}
    fig, ax = plt.subplots(figsize=(9, 5.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, ci, g) in zip(y, rows):
        ax.barh(yi, v - 80, left=80, height=0.62, color=col[g])
        if ci:
            ax.errorbar(v, yi, xerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        ax.text(max(v, ci[1] if ci else v) + 0.15, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.axvline(TARGET, color=INK2, ls="--", lw=1.2)
    ax.text(TARGET + 0.15, len(rows) - 1.5, "URI-CADS-\nSegmentation\n91.23", color=INK2, fontsize=8.5, va="center")
    ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlim(80, 97.5)
    ax.set_xlabel("Dice (%) on dataset A, 1,985 images, 5-fold out-of-fold (bars: 95 % bootstrap CI)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BASE, label="Published in [24]"), Patch(color=ORANGE, label="Ours: CNN (fully automatic)"),
                       Patch(color=BLUE, label="Ours: SAM with automatic box"),
                       Patch(color=AQUA, label="Ours: SAM with true box (upper bound)")],
              loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=2, fontsize=8.5)
    ax.set_title("Kidney segmentation on dataset A compared with the URI-CADS baseline [24]", loc="left")
    save(fig, out, "fig3_comparison_uricads.png")


# ------------------------------------------------------------------ figure 4: per fold + training curves
def fig_folds_training(out):
    runs = pd.read_csv(C.RUNS_CSV)
    runs = runs[runs.smoke.astype(str) == "False"] if "smoke" in runs else runs
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axes[0]
    for i, (m, c, lab) in enumerate([("unet_r34", BLUE, "U-Net"), ("deeplabv3p_r50", ORANGE, "DeepLabV3+")]):
        r = runs[runs.model == m].sort_values("fold")
        ax.plot(r.fold + (i - 0.5) * 0.12, r.test_dice * 100, "o", color=c, ms=8, label=lab,
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axhline(TARGET, color=INK2, ls="--", lw=1.2)
    ax.text(4.3, TARGET - 0.12, "[24] 91.23", color=INK2, fontsize=8.5, va="top", ha="right")
    ax.set_xticks(range(5), [f"Fold {k}" for k in range(5)]); ax.set_ylabel("Test Dice (%)")
    ax.set_ylim(90.5, 93.5); ax.legend(loc="upper left", fontsize=9); ax.grid(axis="x", visible=False)
    ax.set_title("A. Test Dice of every fold", loc="left")
    ax = axes[1]
    for m, c, lab in [("unet_r34", BLUE, "U-Net"), ("deeplabv3p_r50", ORANGE, "DeepLabV3+")]:
        for k in range(5):
            h = pd.read_csv(SEG / m / f"history_fold{k}.csv")
            ax.plot(h.epoch, h.val_dice * 100, color=c, lw=1.4, alpha=0.85, label=lab if k == 0 else None)
    ax.set_ylim(87.5, 94); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Dice (%)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("B. Validation Dice during training (5 folds each, early stopping)", loc="left")
    fig.tight_layout()
    save(fig, out, "fig4_folds_training.png")


# ------------------------------------------------------------------ figure 5: subgroups
def fig_subgroups(out, lab):
    groups = [("All images", lambda d: d.index == d.index), ("Healthy", lambda d: d.diagnosis == 0),
              ("Pathological", lambda d: d.diagnosis == 1), ("PCD", lambda d: d.PCD == 1), ("HC", lambda d: d.HC == 1),
              ("Simple cyst", lambda d: d.cls.map(lambda s: 2 in s)), ("Complicated cyst", lambda d: d.cls.map(lambda s: 3 in s)),
              ("Hydronephrosis", lambda d: d.cls.map(lambda s: 7 in s)),
              ("View 0 (transverse)", lambda d: d.view == 0), ("View 1 (longitudinal)", lambda d: d.view == 1)]
    folds = pd.read_csv(C.FOLDS_CSV, dtype={"image_id": str})[["image_id", "diagnosis"]]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(groups))[::-1]
    for i, (m, c, labm) in enumerate([("unet_r34", BLUE, "U-Net"), ("deeplabv3p_r50", ORANGE, "DeepLabV3+"),
                                      ("medsam_ft_gt", AQUA, "MedSAM fine-tuned, true box")]):
        d = per_image(m).merge(lab, on="image_id").merge(folds, on="image_id")
        vals = [d[f(d)].dice.mean() * 100 for _, f in groups]
        ax.plot(vals, y + (1 - i) * 0.22, "o", color=c, ms=8, label=labm, markeredgecolor=SURFACE, markeredgewidth=2)
    ns = []
    d = per_image("unet_r34").merge(lab, on="image_id").merge(folds, on="image_id")
    for g, f in groups:
        ns.append(f"{g}  (n={int(f(d).sum()):,})")
    ax.axvline(TARGET, color=INK2, ls="--", lw=1.2)
    ax.set_yticks(y, ns); ax.set_xlabel("Mean Dice (%)"); ax.set_xlim(86, 96.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, fontsize=8.5); ax.grid(axis="y", visible=False)
    ax.set_title("Dice by subgroup (dataset A). Dashed line: URI-CADS-Segmentation 91.23 [24]", loc="left")
    save(fig, out, "fig5_subgroups.png")


# ------------------------------------------------------------------ figure 7: box quality vs Dice
def fig_box(out):
    b = pd.read_csv(C.RESULTS / "kidney_boxes.csv", dtype={"image_id": str})
    ix = (np.minimum(b.x1, b.gt_x1) - np.maximum(b.x0, b.gt_x0)).clip(lower=0)
    iy = (np.minimum(b.y1, b.gt_y1) - np.maximum(b.y0, b.gt_y0)).clip(lower=0)
    inter = ix * iy
    b["box_iou"] = inter / ((b.x1 - b.x0) * (b.y1 - b.y0) + (b.gt_x1 - b.gt_x0) * (b.gt_y1 - b.gt_y0) - inter)
    d = (b.merge(per_image("unet_r34")[["image_id", "dice"]].rename(columns={"dice": "unet"}))
         .merge(per_image("medsam_ft_gt")[["image_id", "dice"]].rename(columns={"dice": "ftgt"})))
    bins = [0, .5, .7, .8, .9, 1.0]
    labs = ["≤ 0.5", "0.5–0.7", "0.7–0.8", "0.8–0.9", "> 0.9"]
    d["bin"] = pd.cut(d.box_iou, bins, labels=labs)
    g = d.groupby("bin", observed=True).agg(n=("unet", "size"), unet=("unet", "mean"), ftgt=("ftgt", "mean"),
                                            loss=("unet", lambda s: (1 - s).sum()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    x = np.arange(len(g))
    ax.plot(x, g.unet * 100, "o-", color=BLUE, lw=2, ms=8, label="U-Net (its own box)", markeredgecolor=SURFACE, markeredgewidth=2)
    ax.plot(x, g.ftgt * 100, "o-", color=AQUA, lw=2, ms=8, label="MedSAM fine-tuned, true box", markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xticks(x, [f"{l}\n(n={n})" for l, n in zip(g.index, g.n)])
    ax.set_xlabel("IoU between the U-Net kidney box and the reference box"); ax.set_ylabel("Mean Dice (%)")
    ax.set_ylim(55, 100); ax.legend(loc="lower right", fontsize=9)
    ax.set_title("A. Dice falls only when the kidney is mislocated", loc="left")
    ax = axes[1]
    share_img = g.n / g.n.sum() * 100
    share_loss = g.loss / g.loss.sum() * 100
    w = 0.38
    ax.bar(x - w / 2 - 0.01, share_img, w, color=BLUE, label="Share of images")
    ax.bar(x + w / 2 + 0.01, share_loss, w, color=ORANGE, label="Share of lost Dice")
    for xi, a, l in zip(x, share_img, share_loss):
        ax.text(xi - w / 2, a + 1, f"{a:.0f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi + w / 2, l + 1, f"{l:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, g.index); ax.set_xlabel("Box IoU"); ax.set_ylabel("%"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("B. 18 % of images (IoU < 0.8) cause 43 % of the error", loc="left")
    fig.tight_layout()
    save(fig, out, "fig7_box_quality.png")


# ------------------------------------------------------------------ figure 6: external test
def fig_external(out):
    e = pd.read_csv(SEG / "external" / "external_summary.csv")
    sel = [("unet_r34", "ens", "U-Net (5-model ensemble)", BLUE), ("deeplabv3p_r50", "ens", "DeepLabV3+ (ensemble)", ORANGE),
           ("medsam_ft", "auto", "MedSAM fine-tuned, auto box", AQUA), ("ultrasam", "auto", "UltraSam, auto box", YELLOW),
           ("ultrasam", "gt", "UltraSam, true box (upper bound)", MAGENTA)]
    subsets = [("all", "All (487)"), ("view=Longitudinal", "Longitudinal (349)"), ("view=Transverse", "Transverse (133)"),
               ("kidney_type=native", "Native (338)"), ("kidney_type=transplant", "Transplant (149)")]
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(subsets)); w = 0.16
    for i, (m, mem, lab, c) in enumerate(sel):
        vals = [e[(e.model == m) & (e.member == mem) & (e.subset == s)].dice.iloc[0] * 100 for s, _ in subsets]
        ax.bar(x + (i - 2) * (w + 0.01), vals, w, color=c, label=lab)
    ax.set_xticks(x, [s[1] for s in subsets]); ax.set_ylim(80, 97); ax.set_ylabel("Dice (%)")
    ax.grid(axis="x", visible=False); ax.legend(ncol=3, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("External test on KidneyUS [18] (never used for training): transverse views are the weak spot", loc="left")
    save(fig, out, "fig6_external_kidneyus.png")


# ------------------------------------------------------------------ figure 8: example predictions
def _panel(iid, model, title):
    img = load_image_a(iid)
    gt = load_mask_a(iid)
    pred = cv2.imread(str(C.PRED_MASKS / model / f"{iid}.png"), cv2.IMREAD_GRAYSCALE) > 0
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    hexrgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    for m, col in [(gt, BLUE), (pred, ORANGE)]:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cs, -1, hexrgb(col), 3)
    return rgb, title


def fig_examples(out):
    d = per_image("unet_r34").sort_values("dice").reset_index(drop=True)
    good = d.iloc[[int(len(d) * q) for q in (0.97, 0.75, 0.5)]]
    bad = d[d.image_id.isin(["32", "987", "211"])]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for ax, (_, r) in zip(axes[0], good.iterrows()):
        im, t = _panel(r.image_id, "unet_r34", f"Typical result: #{r.image_id}, Dice {r.dice:.2f}")
        ax.imshow(im); ax.set_title(t, fontsize=10); ax.set_axis_off()
    why = {"32": "spills into a neighbouring organ", "987": "only part of a large kidney", "211": "includes a large cyst"}
    for ax, (_, r) in zip(axes[1], bad.iterrows()):
        im, t = _panel(r.image_id, "unet_r34", f"Failure: #{r.image_id}, Dice {r.dice:.2f}\n{why[r.image_id]}")
        ax.imshow(im); ax.set_title(t, fontsize=10); ax.set_axis_off()
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=3, label="Reference (nephrologists)"),
                        Line2D([], [], color=ORANGE, lw=3, label="U-Net prediction")],
               loc="lower center", ncol=2, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=3.5)
    save(fig, out, "fig8_examples.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.ROOT / "Documents" / "figures"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    lab = labels_a()
    fig_pipeline(out); fig_data(out, lab); fig_comparison(out); fig_folds_training(out)
    fig_subgroups(out, lab); fig_box(out); fig_external(out); fig_examples(out)


if __name__ == "__main__":
    main()
