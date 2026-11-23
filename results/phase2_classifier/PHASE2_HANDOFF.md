# Phase 2 handoff: automatic-crop classifier

All code is in `notebooks/07_automatic_crop_classifier.ipynb`. Run it from top to bottom.

## Model for Phase 3

**`nb06crops_weighted/imagenet/`**: ImageNet EfficientNet-B0, class-weighted cross-entropy, trained on the **Phase 1 crops as given** (`oof_kidney_crops/`, notebook 06) with Phase 1's `folds_master.csv` outer folds.

| File | Content |
|---|---|
| `fold_0.pt` … `fold_4.pt` | Checkpoint at best inner-validation ROC-AUC: `model_state_dict`, `threshold` (chosen operating point), `threshold_youden`, `threshold_rule`, `config` |
| `oof_predictions.csv` | 1,985 held-out predictions: `probability`, `threshold`, `predicted_label`, `outcome`, `tp/tn/fp/fn`, `fold`, `image_id` |
| `operating_points.csv` | Held-out FP/FN for each threshold rule |
| `summary.json` | Pooled metrics and `chosen_operating_point` (including `ensemble_threshold`) |
| `fold_metrics.csv`, `inner_validation_predictions_fold_k.csv`, `training_history_fold_k.csv` | Per-fold details (fold metrics use the Youden threshold) |

### Input pipeline (must match exactly)

1. Image `oof_kidney_crops/<image_id>.jpg` (Phase 1 crop). Its position in the original image is `crop_x0, crop_y0, crop_x1, crop_y1` in `results/eda/oof_crop_iou_stats.csv`; use these to map Grad-CAM back to the full image.
2. RGB → per-image 1–99 percentile intensity scaling → resize 224×224 → `ToTensor()` in [0, 1]. **No ImageNet mean/std normalization.**
3. `softmax(logits)[:, 1]` = probability of pathological (label 1); predicted pathological if `probability >= threshold`.

The classifier head is replaced, so build the model as in notebook 07 (`make_model`, `EVAL_TF`):

```python
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 2))
ckpt = torch.load(".../fold_0.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
```

Grad-CAM target layer: `model.conv_head` (or `model.blocks[-1]`). Section 6 of notebook 07 reloads every checkpoint and reproduces `oof_predictions.csv` exactly.

For OOF explanations, use each image's held-out model (`fold` column). The 5-model ensemble has trained on 4/5 of the images.

## Results (internal 5-fold nested CV, 1,985 images)

Notebook 05 (doctor-drawn mask crops, same nested splits) is the fair oracle reference.

| Model | Pooled OOF AUC | Mean fold AUC |
|---|---|---|
| Notebook 05, oracle crops | 0.8209 | 0.8278 |
| Notebook 07 first run (`baseline`) | 0.7893 | 0.7897 |
| Notebook 07 `robust_augmentation` | 0.8115 | 0.8135 |
| Corrected, unweighted (`nb06crops`) | 0.8310 | 0.8319 |
| **Corrected, class-weighted (`nb06crops_weighted`)** | **0.8283** | **0.8284** |

**Phase 1's automatic crops perform as well as oracle crops.** The 0.8577 in the sprint plan comes from `DS_28_10.ipynb`, which selected epochs on the test fold with the old `folds.csv`. It is optimistically biased and not a valid target.

### Operating point: high sensitivity (team decision)

Rule: per fold, the highest threshold reaching **sensitivity ≥ 0.90 on that fold's validation images**, applied unchanged to the held-out fold.

| | FP | FN | Sensitivity | Specificity |
|---|---|---|---|---|
| Oracle reference (nb05 @ 0.5) | 199 | 186 | 87.9% | 55.8% |
| **Chosen: `nb06crops_weighted`, validation sensitivity ≥ 0.90** | **213** | **162** | **89.4%** | **52.7%** |
| Alternative: `nb06crops` (unweighted), validation sensitivity ≥ 0.90 | 199 | 163 | 89.4% | 55.8% |

Fold thresholds: 0.2245 / 0.2479 / 0.1999 / 0.2739 / 0.5047. **Deployment threshold for the 5-model mean probability: 0.2479** (median of fold thresholds; derived from single-fold models and not validated on ensemble outputs).

This operating point favours sensitivity: 24 fewer missed pathological kidneys than the oracle baseline, at the cost of 14 more false positives. Other rules (Youden: 100 FP / 385 FN) are listed in `operating_points.csv`.

Class-weighted was the pre-specified choice (task p2-t2). The unweighted run is equivalent within fold-to-fold noise. Switching to it after seeing these outer-fold results would be selection on test data.

## What was wrong with notebook 07's first run

1. `timm.create_model(..., num_classes=2)` re-initialises the head with large weights: initial loss ≈ 3.7 vs 0.69.
2. No percentile intensity normalization.
3. `RandRotate90`: not a realistic ultrasound variation.
4. Weight decay 1e-4, 30 epochs, no LR schedule: overfitting.
5. ImageNet mean/std normalization (notebook 05 uses [0, 1]).

`baseline/`, `robust_augmentation/` and `sensitivity_target_analysis/` are superseded and kept for record.

## Caveats and remaining work

- **RadImageNet comparison (p2-t3) not done, by team decision**: RadImageNet publishes no EfficientNet-B0 weights (only ResNet50, DenseNet121, InceptionV3, IRV2), and pretraining one ourselves on 1.35 million images would take more than a week. The team chose to keep the ImageNet-pretrained EfficientNet-B0.
- Phase 1 provided no inner validation split (`inner_val_in_outer_folds` is empty). Inner validation uses the team's `nested_cv_splits.csv`; its test parts match `folds_master.csv` exactly, and no group crosses train/validation/test.
- Crops are OOF per image for segmentation, but the segmenter that cropped a classifier-training image may have trained on the classifier's outer-test fold. Fully nested end-to-end evaluation is not verified.
- KidneyUS has no pathology labels, so no external classifier validation is possible. These are internal results, not clinical performance claims.
