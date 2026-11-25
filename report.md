# Phase 2 – Kidney segmentation: investigation report

Branch `segmentation`. Code: `src/kidney/` (see `README_segmentation.md` for what each module does).
Full pipeline: `scripts/run_segmentation.sh`, run on 25–26 Oct 2026 (16:11 → 00:20, about 8 h on one laptop GPU).
Full log: `logs/full_run.log` (not committed). Numbers in square brackets refer to the [References](#references).

> Citation keys follow the team's `references.md` and literature review: [1]–[23] review papers, [24] the URI-CADS baseline,
> [S1]–[S13] pretrained models and prior work. Method references that are not in that list (architectures, losses, metrics)
> are numbered [M1]–[M13] at the end.

## Summary

- **Baseline:** URI-CADS [24], from the same hospital and the same 1,985-image dataset (dataset A). Its segmentation-only
  model reports **Dice 91.23 / IoU 84.99**; its full multi-task system reports 89.38 / 81.41.
- **Both of our supervised CNNs beat the baseline**, using a leak-free 5-fold protocol of the same shape (3 train / 1 validation / 1 test folds):
  **DeepLabV3+ ResNet-50 [M2, M3] 92.35 ± 0.48** (+1.12) and **U-Net ResNet-34 [M1, M3] 92.20 ± 0.52** (+0.97).
  The baseline value lies outside both of our 95 % CIs. The DeepLabV3+ that URI-CADS trained as a reference scored
  89.34, so our DeepLabV3+ is **+3.0 Dice** higher on the same data.
- **Foundation models without training are close, but need a good box.** With the true kidney box, UltraSam [S1]
  reaches 93.38 and MedSAM [S3] 90.27. With an automatic box from our U-Net they drop to 90.13 and 87.60.
- **Fine-tuning only the MedSAM mask decoder** (a few minutes per fold) gives the best fully automatic SAM (91.79)
  and the best upper bound (95.01 with the true box).
- **External test (KidneyUS [18], 487 images, never trained on):** ensemble Dice 0.90–0.91, and UltraSam-auto 0.911.
  URI-CADS reports no external test, so this result has no baseline to compare with.
- **Same weak spots as the baseline:** like URI-CADS, our errors concentrate on low-contrast kidneys (hyperechoic cortex,
  HC, and poor corticomedullary distinction, PCD) and on images where neighbouring structures look like kidney. Our models also score lower on the
  minority view (7 % of dataset A).

## 1. Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB |
| Python env | `~/ds_env` (Python 3.12) |
| torch / torchvision [M13] | 2.11.0+cu130 / 0.26.0+cu130 |
| Extra packages installed | segmentation-models-pytorch [M11], timm, monai, segment-anything [M4], opencv-python-headless, scikit-image, tabulate, **mmengine** |
| CNN training | 512 px letterbox, Dice [M6] + BCE loss, AdamW [M7] lr 3e-4, cosine schedule [M8], batch 8, AMP, ≤ 100 epochs, early stopping (patience 15) on validation-fold Dice, ImageNet-pretrained encoders [M12] |
| MedSAM fine-tune | mask decoder only (encoders frozen), Dice + BCE, ≤ 30 epochs, patience 6, recipe of [S3] |
| Post-processing | largest connected component + hole filling |
| Metrics | Dice, IoU (as in [24]), HD95 [M9], 95 % percentile bootstrap CI [M10] |
| Seed | 42 (+ fold index for each training run) |
| Foundation-model weights | `weights/medsam_vit_b.pth` (MedSAM [S3], Zenodo 10689643), `weights/UltraSam.pth` (UltraSam [S1]) |

## 2. Data checks (dataset A, URI-CADS [24])

Source: `results/segmentation/mask_check.csv` (`kidney.prepare_masks`) and the label files.

- **1,985 images** (one per clinical case, Toshiba Xario 660a, 2009–2018 [24]). Every `.mat` mask has the same size as its
  image. Most images are 561 × 800 px (min 375 × 382).
- **Classes:** 450 normal / 1,535 pathological (77.3 %), as in [24].
- **View field:** View 0 = 1,842 images (92.8 %), View 1 = 143 (7.2 %). This matches the "93 % transversal / 7 %
  longitudinal" that [24] reports, so View 0 is most likely the paper's "transversal". The label files do not document the coding.
- **Global pathologies:** 595 images with poor corticomedullary distinction (PCD, flag 1), 509 with hyperechoic cortex (HC, flag 2) [24].
- **Kidney size:** the kidney mask covers a median **7.1 %** of the image (IQR 5.2–9.2 %, range 1.1–27.3 %).
- **Mask shape:** 6 masks have more than one component and 2 have holes. In [24] the masks are polygons drawn over an
  ellipse placed by the nephrologists, so they are smoother than a true border tracing.
- **Lesion boxes:** 2,478 boxes on 1,323 pathological images, none on normal images. Read as `x y w h class`,
  **2,445 / 2,478 (98.7 %)** box centres fall inside the kidney mask. Lesions are almost always inside the kidney, so a kidney
  crop (Phase 3) loses almost no lesion.

### Split

- The team's leak-free split (`results/eda/folds.csv`) is the canonical `results/splits/folds.csv`. The copy shipped with
  the code was checked image by image and is identical.
- 1,893 groups, of which **87 near-duplicate groups** are kept inside one fold. [24] does not mention near-duplicates, so their
  split may place copies of one image in both training and test.
- Fold sizes 396–399; pathological share 77.3–77.4 % in every fold.
- Repetition *k*: test = fold *k*, validation = fold *k*+1, train = the other three (the same 3 / 1 / 1 scheme as [24]).
  Every image is tested exactly once, at its original resolution.

### External set (dataset B, KidneyUS [18])

- 534 capsule masks from sonographer 1 (`reviewed_masks_1/capsule`), of which **487** images have a usable capsule outline.
  349 longitudinal / 133 transverse / 5 other views; 338 native / 149 transplant kidneys; 485 of 487 rated "Fair" quality.
- KidneyUS is never used for training or model selection.

## 3. Pipeline checks before the full run

- **Unit tests:** 7 / 7 pass (`tests/`: metrics and the UltraSam key mapping).
- **Smoke test** (`run_segmentation.sh --smoke`, random weights, a few images): every step 0–9 ran end to end. Smoke outputs
  were deleted before the full run.
- **UltraSam checkpoint [S1]:** the README warned that the MMDetection → SAM conversion had never been tried on the real checkpoint.
  Loading it needed `mmengine`. After installing it, `load_ultrasam` matched **314 / 314 SAM tensors (100 %)**.

## 4. Comparison with the baseline study: URI-CADS [24]

### 4.1 Protocol

| | URI-CADS [24] | This work |
|---|---|---|
| Data | Dataset A, 1,985 images | same images and masks |
| Cross-validation | 5-fold; 3 train / 1 val / 1 test per repetition | same 3 / 1 / 1 scheme, test = fold *k*, val = fold *k*+1 |
| Split construction | not described (no mention of near-duplicates) | group-aware, 87 near-duplicate groups kept in one fold |
| Hyper-parameter tuning | on the first fold, then reused | fixed in advance; early stopping on each validation fold |
| Segmentation model | Mask R-CNN [M5] branch on ResNet-50 [M3] + FPN (detect kidney, then segment) | U-Net / DeepLabV3+ (dense segmentation), SAM-type box-prompted models |
| Pretraining | pre-trained PyTorch Faster / Mask R-CNN with ResNet-50 backbone | ImageNet encoders; MedSAM / UltraSam foundation weights |
| Augmentation | rotation, gamma, translation, zoom (checked by nephrologists) | the same four (rotation ±15°, translation ±8 %, zoom ±15 %, gamma 0.7–1.5) + Gaussian noise; same geometric transform for image and mask |
| Metrics | Dice, IoU (dataset mean) | Dice, IoU, HD95, fold mean ± SD, bootstrap 95 % CI |
| External test | none | KidneyUS [18], 487 images |
| Task | segmentation + global / local diagnosis (multi-task) | segmentation only (classification is Phase 3) |

### 4.2 Results on dataset A

Dice / IoU in %. Rows marked [24] are copied from Table 5 of [24]. Our rows: mean ± SD over the 5 test folds, and the 95 %
bootstrap CI of the pooled per-image Dice. Δ = difference from URI-CADS-Segmentation.

| Model | Source | Dice | IoU | Dice 95 % CI | Δ Dice | Δ IoU |
|---|---|---|---|---|---|---|
| TN-SCUI2020 winner [S13] | [24] | 87.23 | 79.31 | – | −4.00 | −5.68 |
| DeepLabV3+ [M2] | [24] | 89.34 | 81.80 | – | −1.89 | −3.19 |
| URI-CADS (full multi-task) | [24] | 89.38 | 81.41 | – | −1.85 | −3.58 |
| **URI-CADS-Segmentation** | [24] | **91.23** | **84.99** | – | baseline | baseline |
| U-Net ResNet-34 | ours | 92.20 ± 0.52 | 86.16 ± 0.79 | [91.89, 92.47] | **+0.97** | **+1.17** |
| DeepLabV3+ ResNet-50 | ours | **92.35 ± 0.48** | **86.39 ± 0.64** | [92.07, 92.63] | **+1.12** | **+1.40** |
| MedSAM, zero-shot, auto box | ours | 87.60 ± 0.35 | 78.63 ± 0.49 | [87.28, 87.93] | −3.63 | −6.36 |
| UltraSam, zero-shot, auto box | ours | 90.13 ± 0.42 | 82.64 ± 0.62 | [89.83, 90.44] | −1.10 | −2.35 |
| MedSAM decoder fine-tuned, auto box | ours | 91.79 ± 0.40 | 85.47 ± 0.59 | [91.49, 92.07] | +0.56 | +0.48 |
| *Upper bounds (box from the reference mask, not automatic):* | | | | | | |
| MedSAM, zero-shot, true box | ours | 90.27 ± 0.12 | 82.51 ± 0.18 | [90.10, 90.44] | −0.96 | −2.48 |
| UltraSam, zero-shot, true box | ours | 93.38 ± 0.14 | 87.68 ± 0.24 | [93.27, 93.49] | +2.15 | +2.69 |
| MedSAM decoder fine-tuned, true box | ours | 95.01 ± 0.08 | 90.57 ± 0.14 | [94.92, 95.10] | +3.78 | +5.58 |

What this comparison shows:

1. **Both fully automatic CNNs beat URI-CADS-Segmentation** in Dice and IoU. 91.23 lies below the lower end of both CIs
   (91.89 and 92.07), and every one of the 10 CNN test folds scores ≥ 91.43. Since only the summary numbers of [24] are available,
   a paired test (e.g. on per-image Dice) is not possible. The comparison is of our CI against their point value.
2. **The biggest difference is for the same architecture.** [24] trained DeepLabV3+ as a reference and got 89.34; ours gets 92.35
   (+3.0 Dice, +4.6 IoU) on the same images. The difference must come from training choices (ResNet-50 encoder with ImageNet
   weights, 512 px input, Dice + BCE loss, early stopping on validation Dice), not from the architecture. [24] concluded that
   "detect-then-segment" (Mask R-CNN) is better than dense segmentation. Our results do not support that conclusion once the
   dense model is trained well.
3. **Our split is at least as hard as theirs.** We keep near-duplicates out of the test fold, and [24] does not report doing so.
   A leaky split would make *their* numbers optimistic, not ours, so the gap is unlikely to come from the split.
4. **A zero-shot foundation model with automatic boxes does not beat the baseline** (UltraSam-auto −1.10, MedSAM-auto −3.63).
   Fine-tuning the MedSAM decoder for a few minutes per fold brings it above the baseline (+0.56).
5. **Not comparable:** [24] reports no HD95, no SD across folds, no CI and no external test. These columns of our results have no
   baseline.

### 4.3 Error analysis, compared with [24]

[24] attributes its worst segmentation errors to low kidney/background contrast (hyperechoic cortex, poor corticomedullary
distinction) and to other organs that look like kidney. It also notes that its DeepLabV3+ "struggles" with PCD and polycystic
kidneys. Our Dice by subgroup (dataset A, out-of-fold):

| Subgroup | n | U-Net | DeepLabV3+ | UltraSam auto | MedSAM-FT auto | UltraSam true box |
|---|---|---|---|---|---|---|
| No HC and no PCD | 1,276 | 92.71 | 92.83 | 90.62 | 92.26 | 93.48 |
| Poor corticomedullary distinction (PCD) | 595 | 91.24 | 91.55 | 89.24 | 90.96 | 93.25 |
| Hyperechoic cortex (HC) | 509 | **90.48** | **90.67** | 88.45 | 90.11 | 92.89 |
| Complicated cyst (CCY) | 79 | **87.88** | **87.60** | 85.43 | 87.77 | 92.36 |
| Simple cyst (SCY) | 595 | 90.77 | 91.13 | 88.52 | 90.36 | 92.82 |
| Hydronephrosis (HYD) | 469 | 92.45 | 92.63 | 90.70 | 91.96 | 93.67 |
| View 0 (≈ "transversal" in [24]) | 1,842 | 92.55 | 92.62 | 90.36 | 92.16 | 93.31 |
| View 1 (≈ "longitudinal" in [24]) | 143 | **87.75** | **88.88** | 87.18 | 87.07 | 94.18 |

- **HC and PCD lower Dice, confirming [24]**, but by only 1.3–2.2 points for our CNNs. HC is the harder of the two (90.5–90.7),
  not PCD as [24] reports for its DeepLabV3+. Even on HC kidneys our CNNs
  score above the *overall* Dice of the DeepLabV3+ in [24] (89.34).
- **Complicated cysts** are the hardest local pathology (about 87.7). The model must decide whether a large cyst is part of the
  kidney, and the reference outline is not consistent about this (see section 6).
- **The minority view** (View 1, 7 % of the data) is about 4–5 points worse for every automatic model, but not with a true box
  (94.18). The models outline these kidneys well once they find them; the difficulty is *locating* them. [24] does not
  report results by view.

## 5. External test on KidneyUS [18] (487 images)

Dice against the sonographer-1 capsule outline. CNNs: the mean ± SD of the 5 fold models, and the average-probability ensemble.
The ellipse column scores against an ellipse fitted to the capsule, because the dataset A masks are ellipse-based polygons [24]
while KidneyUS capsules follow the border [18]. Source: `results/segmentation/external/external_summary.csv`.

| Model | Dice (all) | 95 % CI | Dice vs ellipse | HD95 median (px) | Longitudinal (349) | Transverse (133) | Native (338) | Transplant (149) |
|---|---|---|---|---|---|---|---|---|
| U-Net, 5 fold models | 0.892 ± 0.004 | | | | | | | |
| U-Net, ensemble | 0.902 | [0.890, 0.912] | 0.896 | 29.4 | 0.919 | 0.859 | 0.903 | 0.899 |
| DeepLabV3+, 5 fold models | 0.894 ± 0.006 | | | | | | | |
| DeepLabV3+, ensemble | 0.905 | [0.895, 0.914] | 0.900 | 29.1 | 0.917 | 0.875 | 0.910 | 0.894 |
| MedSAM, auto box | 0.895 | [0.884, 0.905] | 0.888 | 35.4 | 0.910 | 0.857 | 0.896 | 0.894 |
| UltraSam, auto box | **0.911** | [0.900, 0.921] | 0.892 | 25.1 | 0.928 | 0.869 | 0.912 | 0.909 |
| MedSAM-FT, auto box | 0.896 | [0.884, 0.906] | 0.895 | 31.0 | 0.913 | 0.852 | 0.897 | 0.895 |
| MedSAM, true box | 0.933 | [0.927, 0.937] | 0.920 | 23.6 | 0.934 | 0.930 | 0.935 | 0.928 |
| UltraSam, true box | **0.952** | [0.947, 0.955] | 0.926 | 16.5 | 0.954 | 0.947 | 0.955 | 0.944 |
| MedSAM-FT, true box | 0.940 | [0.935, 0.944] | 0.936 | 21.1 | 0.941 | 0.937 | 0.943 | 0.935 |

Findings:

- **Generalisation is good.** On a different hospital, scanner and annotation style, Dice falls by only about 2 points
  (0.92 → 0.90 for the CNNs). Ensembling the 5 fold models adds about 1 point over a single model.
- **Transverse views are the main external weakness** of every automatic pipeline (0.85–0.88 vs 0.91–0.93 longitudinal).
  The gap disappears with the true box (0.93–0.95), so the problem is again *finding* the kidney, not outlining it.
  *Open question:* if View 0 in dataset A really is "transversal" as [24] says, the models were trained mostly on transverse
  images and still do worse on KidneyUS transverse views. In that case the two datasets probably use the view names differently,
  or their transverse images look different (KidneyUS includes transplants and other scanners). The dataset A view coding should be
  checked with the dataset authors.
- **Transplant kidneys** score about the same Dice as native ones, but have a higher HD95 (about 38 vs 25 px for the CNNs).
- **UltraSam generalises best of the zero-shot models**, which fits its pretraining on ultrasound only [S1]. The MedSAM decoder
  fine-tuned on dataset A gains +4.7 points in-domain (true box) but only +0.7 on KidneyUS, so part of that gain is specific
  to the dataset A annotation style.
- The ellipse scores are slightly *lower* than the true-capsule scores for most models, so the difference in outline style
  does not explain the external drop.
- One "Poor"-quality image fails for every model (Dice 0).

## 6. Failure cases

Overlays: `results/segmentation/failures/{model}_grid.png` (16 lowest-Dice test images per model; green = reference, red =
prediction, boxes = lesion labels). The worst U-Net cases (Dice 0.30–0.60) fall into three groups:

1. **Over-segmentation into neighbouring structures:** the prediction spreads into a large cyst, the liver/spleen or bowel
   (e.g. #32, #1600, #211, #726, #1525, #1247 – a transplant "INJERTO"). This is the "other organs that resemble the kidney"
   error that [24] also reports.
2. **Under-segmentation of large or zoomed kidneys:** only the central sinus or part of the kidney is found when the kidney
   fills most of the image (e.g. #987, #463, #629, #988).
3. **Debatable references:** in several cases (#211, #1525) the reference outline leaves out a large exophytic cyst that the lesion
   box marks as a kidney finding, and the model includes it. This is partly a question of annotation convention.

Unlike [24], where the detector missed only one kidney, our dense models never "miss" a kidney entirely, but 4–5 images per CNN
have Dice < 0.5 (U-Net 4, DeepLabV3+ 5). U-Net and DeepLabV3+ share only 6 of their 16 worst images, which suggests a CNN ensemble
could help on dataset A too.

## 7. Conclusions

1. **The baseline is beaten.** Both fully automatic CNNs exceed URI-CADS-Segmentation [24] (91.23 Dice / 84.99 IoU):
   U-Net 92.20 / 86.16 and DeepLabV3+ 92.35 / 86.39, with small fold-to-fold variation (SD ≈ 0.5) and a split that is at
   least as strict as the original.
2. **A well-trained dense model is enough.** The same architecture that [24] reported at 89.34 reaches 92.35 here, which
   contradicts the paper's conclusion that detect-then-segment is needed. U-Net ResNet-34 matches DeepLabV3+ within the CI
   while training about 30 % faster, and is the recommended kidney model for Phase 3.
3. **SAM-type models depend on the prompt.** Given a correct box they are the most accurate (UltraSam 93.4 zero-shot,
   MedSAM-FT 95.0). With a U-Net box they are no better than the U-Net. UltraSam-auto is the best external model (0.911), so a
   refinement stage may still help across hospitals.
4. **External robustness is good** (Dice ≈ 0.90 on KidneyUS [18] without adaptation). [24] gives no external result to compare
   with. The main gap, in both datasets, is the minority / transverse view, where the kidney is not *found* reliably.
5. **For Phase 3 (classification on kidney crops):** `results/kidney_boxes.csv` gives a U-Net box for all 1,985 images. Since
   98.7 % of lesion centres lie inside the kidney mask, cropping to the predicted box (with a small margin) should keep nearly
   all lesions.

### Limitations / next steps

- The comparison with [24] uses their published summary numbers only. Their folds and per-image results are not available, so no
  paired statistical test (e.g. Wilcoxon on per-image Dice) is possible.
- One run per configuration (seed 42). The fold SD measures variation across data splits, not across seeds.
- HD95 is in pixels of the original image, and pixel spacing differs between scanners and datasets.
- The dataset A view coding (View 0 / 1) is inferred from the proportions in [24], not documented.
- Not tried: CNN ensembles on dataset A, test-time augmentation, fine-tuning UltraSam, and training with KidneyUS transverse views.

## 8. What to improve, based on the baseline [24]

### 8.1 Checks on the existing predictions (no training needed)

**Shape regularisation does not help.** [24] says its detect-then-segment output is "more regularized" than DeepLabV3+, and its
reference masks are polygons drawn over an ellipse (a perfect ellipse fit reaches Dice 96.2 against them, a convex hull 99.4).
Applying the same shape prior to our predictions:

| Post-processing (out-of-fold, 1,985 images) | U-Net | DeepLabV3+ | MedSAM-FT auto |
|---|---|---|---|
| none (current) | 92.20 | 92.35 | 91.79 |
| convex hull | 92.20 | 92.35 | 91.85 |
| fitted ellipse | 91.12 | 91.25 | 90.96 |

Our masks are already convex, and forcing an ellipse costs about 1 point. Combining the U-Net and DeepLabV3+ masks does not help
either (intersection 92.30, union 92.25). **The outline shape is not the problem.**

**Errors come from locating the kidney.** Images grouped by the IoU between the U-Net kidney box and the reference box:

| Box IoU | Images | U-Net Dice | MedSAM-FT, U-Net box | MedSAM-FT, true box |
|---|---|---|---|---|
| > 0.9 | 959 | 95.86 | 95.33 | 95.75 |
| 0.8–0.9 | 668 | 92.68 | 92.46 | 94.68 |
| 0.7–0.8 | 196 | 87.01 | 86.69 | 93.89 |
| 0.5–0.7 | 130 | 78.67 | 78.29 | 93.46 |
| ≤ 0.5 | 32 | 59.17 | 57.94 | 92.95 |

- When the box is right, the U-Net outline is already as good as the best upper bound (95.9 vs 95.8).
- The 358 images (18 %) with box IoU < 0.8 cause **43 % of all lost Dice**. On those images a correct box still gives about 93.
- MedSAM-FT with the U-Net box only copies the U-Net's mistakes, because the box comes from the same model.

So the gain must come from **finding the kidney and its extent**, which is the central idea of [24]. Its kidney detector
"successfully detects all but one kidney in the database", and the paper concludes that "the kidney detection approach (with
subsequent segmentation through Mask R-CNN) is more effective" than dense segmentation.

### 8.2 Recommended next steps

Same rules as before: the same 5 folds and seed, one change per experiment, a paired per-image comparison with the U-Net, and
KidneyUS [18] as a pure external test.

| # | Change | Idea from [24] | Expected effect |
|---|---|---|---|
| R1 | **Train a dedicated kidney detector** (Faster R-CNN ResNet-50-FPN [M5], ImageNet / COCO weights, as in [24]) and use its box instead of the U-Net box | Detect first, then segment | Better box IoU on the 18 % of hard images; measured directly as box IoU against the reference box |
| R2 | **Detector box → fine-tuned SAM decoder** (MedSAM-FT, already trained) | Segment inside the detected kidney region | Somewhere between 92.2 (now) and 95.0 (true-box upper bound), depending on R1's box quality |
| R3 | **Fine-tune the UltraSam [S1] decoder**, as already done for MedSAM | Fixes a limitation of [24]: ImageNet only, no ultrasound pretraining | Zero-shot UltraSam beats zero-shot MedSAM (93.4 vs 90.3 with the true box) and generalises best to KidneyUS, so a fine-tuned UltraSam should be the strongest second stage |
| R4 | **Augmentation and oversampling for the detector**: the four [24] augmentations plus flip and wider zoom; oversample View 1 (7 % of the data) | [24] augmentation; view imbalance (93 % transverse) | Fewer misses on the rare view and on zoomed kidneys |

Not recommended, based on [24] and our checks:
- **Multi-task heads in the segmentation model.** In [24], adding the diagnosis branches *lowered* segmentation Dice from 91.23 to
  89.38. Multi-task belongs in the Phase 3 classifier, not here.
- **Shape priors / ellipse fitting** (section 8.1) and **speckle filtering or pixel-imbalance losses** (the Dice loss [M6] already
  handles the 7 % kidney area; the team decided not to pursue these).

Cost: about 2 h of GPU time for the detector (5 folds), and under 1 h each for R2 and R3 (decoder only, cached image embeddings).

## 9. Future plan

### 9.1 Phase 2, improvement round (next)

| Step | Work | Output | Success criterion |
|---|---|---|---|
| 1 | R1: train the Faster R-CNN kidney detector on the 5 folds | `results/det_boxes.csv`, box IoU per image | median box IoU > U-Net box (0.897); fewer images < 0.8 (now 18 %) |
| 2 | R2: detector box → MedSAM-FT, on dataset A and KidneyUS | `pred_masks/medsam_ft_det/` | Dice > U-Net 92.20 with a paired test p < 0.05; gains on View 1 and HC |
| 3 | R3: fine-tune the UltraSam decoder; run with gt, U-Net and detector boxes | `pred_masks/ultrasam_ft_*` | true-box Dice > MedSAM-FT 95.01; KidneyUS > 0.911 |
| 4 | R4: detector augmentation + View 1 oversampling (only if R1 still misses the minority view) | | View 1 box IoU close to View 0 |
| 5 | Update `report.md`, the official report and `results/kidney_boxes.csv` with the best pipeline | | |

### 9.2 Phase 3, classification (main goal of the project)

The kidney model from Phase 2 feeds the classifier: crop each image to the predicted kidney box (with a margin), since 98.7 % of
lesion centres lie inside the kidney.

| Task | Baseline to beat [24] | Notes |
|---|---|---|
| Healthy vs pathological | AUC 87.41, SP@95 60.59 | 450 healthy vs 1,535 pathological: class weights or a balanced sampler; report calibration and CIs ([24] reports neither) |
| Multi-class (HC, PCD, C, PYR, HYD, O) | average AUC 81.90, SP@95 37.76 | "Others" is the weakest category in [24] (AUC 69.32); multi-task heads belong here (they helped classification in [24]) |
| External check | – | KidneyUS [18] has no pathology labels, so external validation of the classifier needs another dataset |

Data issues to settle before Phase 3: 31 of the 87 near-duplicate groups have inconsistent labels between copies (e.g. #4 pathological
and #296 healthy are the same image), and some images show the diagnosis as on-screen text ("QUISTE", "HIDRONEFROSIS"), which a
classifier could read instead of the anatomy.

## Output files

| File | Content |
|---|---|
| `results/segmentation/summary.{csv,md}` | Main results table |
| `results/segmentation/{model}/per_image.csv` | Per-image Dice / IoU / HD95 (raw and post-processed) |
| `results/segmentation/{model}/history_fold{k}.csv` | Training curves (CNNs) |
| `results/runs.csv` | One row per model and fold, with config and library versions |
| `results/pred_masks/{model}/{id}.png` | Out-of-fold predicted masks |
| `results/kidney_boxes.csv` | Predicted kidney box per image (U-Net), for Phase 3 |
| `results/segmentation/external/` | KidneyUS per-image scores and `external_summary.csv` |
| `results/segmentation/failures/` | Lowest-Dice overlays per model |
| `results/checkpoints/` | Model weights (git-ignored) |

## References

Numbering follows the team's `references.md` (copied from `~/Downloads/references.md`, last updated 2026-10-24) and the literature
review `Documents/kidney_ultrasound_DL_review.docx`. Only the entries cited in this report are listed.

### Baseline

- **[24]** Molina-Moreno M., González-Díaz I., Rivera Gorrín M., Burguera Vion V., Díaz-de-María F. URI-CADS: A fully
  automated computer-aided diagnosis system for ultrasound renal imaging. *Journal of Imaging Informatics in Medicine*, 2024.
  https://doi.org/10.1007/s10278-024-01055-4. Code and data: https://github.com/miguel55/URI-CADS. (Dataset A; local copy
  `Documents/s10278-024-01055-4.pdf`.)

### Dataset B

- **[18]** Singla R, Ringstrom C, Hu G, Lessoway V, Reid J, Nguan C, Rohling R. The Open Kidney Ultrasound Data Set. *arXiv*
  2206.06657, 2022; in *Advances in Simplifying Medical Ultrasound (ASMUS 2023)*, LNCS, pp. 155–164, Springer. (Dataset B, KidneyUS.)

### Pretrained models and prior work

- **[S1]** Meyer A., Murali A., Zarin F., Mutter D., Padoy N. UltraSam: A foundation model for ultrasound using large
  open-access segmentation datasets. arXiv:2411.16222, 2024. https://github.com/CAMMA-public/UltraSam
- **[S3]** Ma J., He Y., Li F., Han L., You C., Wang B. Segment anything in medical images (MedSAM). *Nature Communications*
  15:654, 2024. https://doi.org/10.1038/s41467-024-44824-z. Weights: https://zenodo.org/records/10689643
- **[S13]** Wang M. 1st place solution for the Thyroid Nodule Segmentation and Classification in Ultrasound Images Grand
  Challenge (TN-SCUI 2020). https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st

### Methods (not in `references.md`)

- **[M1]** Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*,
  LNCS 9351, pp. 234–241.
- **[M2]** Chen L-C, Zhu Y, Papandreou G, Schroff F, Adam H. Encoder-decoder with atrous separable convolution for semantic
  image segmentation (DeepLabV3+). *ECCV 2018*, pp. 801–818.
- **[M3]** He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. *CVPR 2016*, pp. 770–778.
- **[M4]** Kirillov A, Mintun E, Ravi N, et al. Segment Anything. *ICCV 2023*, pp. 4015–4026.
- **[M5]** He K, Gkioxari G, Dollár P, Girshick R. Mask R-CNN. *ICCV 2017*, pp. 2961–2969.
- **[M6]** Milletari F, Navab N, Ahmadi S-A. V-Net: Fully convolutional neural networks for volumetric medical image
  segmentation (Dice loss). *3DV 2016*, pp. 565–571.
- **[M7]** Loshchilov I, Hutter F. Decoupled weight decay regularization (AdamW). *ICLR 2019*.
- **[M8]** Loshchilov I, Hutter F. SGDR: Stochastic gradient descent with warm restarts (cosine schedule). *ICLR 2017*.
- **[M9]** Huttenlocher DP, Klanderman GA, Rucklidge WJ. Comparing images using the Hausdorff distance. *IEEE TPAMI*
  15(9):850–863, 1993.
- **[M10]** Efron B, Tibshirani RJ. *An Introduction to the Bootstrap*. Chapman & Hall, 1993.
- **[M11]** Iakubovskii P. Segmentation Models PyTorch. https://github.com/qubvel/segmentation_models.pytorch, 2019.
- **[M12]** Deng J, Dong W, Socher R, Li L-J, Li K, Fei-Fei L. ImageNet: A large-scale hierarchical image database. *CVPR 2009*.
- **[M13]** Paszke A, Gross S, Massa F, et al. PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 2019*.
