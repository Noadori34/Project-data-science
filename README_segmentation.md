# Phase 2: Kidney segmentation (branch `segmentation`)

Code for section 1 (shared protocol) and section 3 (Phase 2) of `plan.md`. Written before GPU access:
every step has been checked end to end on CPU with `--smoke` (tiny subset, 1 epoch, random weights),
but **no real results exist yet**. Run the full pipeline on the GPU to fill the tables.

## Quick start

```bash
pip install -r requirements-seg.txt        # install torch for your CUDA first
bash scripts/run_segmentation.sh --smoke   # ~5 min CPU check of every step
bash scripts/run_segmentation.sh           # full run (GPU)
```

Or open `notebooks/04_kidney_segmentation.ipynb` in Colab (it clones this branch).
Each step can also be run alone with `PYTHONPATH=src python -m kidney.<module>` (see below).

## What each module does

| Step | Module | Plan | Output |
|---|---|---|---|
| Split | `kidney.splits` | 1.2 | `results/splits/folds.csv` |
| Metrics | `kidney.metrics` | 1.3 | Dice, IoU [24], HD95 [S29], AUC, SP@95 [24], bootstrap CI [S27], DeLong [S26] |
| Mask prep | `kidney.prepare_masks` | 3.1 | `results/masks_png/`, `results/segmentation/mask_check.csv`, overlays |
| S-A, S-B | `kidney.train_seg --model unet_r34 / deeplabv3p_r50` | 3.2–3.3 | checkpoints, `results/pred_masks/{model}/{id}.png`, per-image CSV, `results/runs.csv` |
| S-C, S-D | `kidney.sam_prompt --model medsam / ultrasam` | 3.4 | masks for `gt` and `auto` box prompts |
| MedSAM fine-tune | `kidney.finetune_medsam` | 3.4 (optional) | `medsam_ft_gt`, `medsam_ft_auto` |
| External test | `kidney.eval_external` | 3.5 | `results/segmentation/external/` |
| Report | `kidney.report_seg` | 6 | `results/segmentation/summary.{csv,md}`, `results/kidney_boxes.csv`, failure grids |

## Decisions worth knowing

- **Split.** The team's leak-free split (`results/eda/folds.csv`) is adopted as the canonical
  `results/splits/folds.csv`: the 87 near-duplicate groups stay inside one fold, so a copy of a test
  image is never trained on. Fold balance is 77.3–77.4% pathological in every fold. Repetition *k*:
  test = fold *k*, validation = fold *k*+1 (mod 5), train = the rest, as in [24].
- **Box format** in the label files is `x y w h class`: 2,445 of 2,478 lesion-box centres fall inside
  the kidney mask with this reading.
- **Masks are not all one region:** 6 of 1,985 have more than one component and 2 have holes. The
  post-processing (largest region + fill holes) is still applied, as the plan says; raw scores are
  logged too (`dice_raw` etc.) so its effect is visible.
- **S-B encoder:** DeepLabV3+ with ResNet-50, the same depth as the URI-CADS backbone [24].
- **Selection:** early stopping on validation-fold Dice; the test fold is scored once, at the
  original image resolution.
- **Automatic SAM boxes** come from S-A's out-of-fold masks, so no test image leaks into the prompt.
- **UltraSam** is released as an MMDetection checkpoint. `sam_prompt.mmdet_to_sam` inverts the
  UltraSam repo's own SAM → MMDetection conversion (round-trip tested in `tests/`). The loader
  refuses to run if fewer than 95% of SAM weights are found in the checkpoint, so a mismatch fails
  loudly instead of giving numbers from a half-loaded model. This has not been tried on the real
  checkpoint yet (the download host is not reachable from where the code was written).
- **External test:** KidneyUS (487 images with a capsule outline from sonographer 1). The 5 fold
  models are scored separately (mean ± SD) and as an average-probability ensemble. Scores against an
  ellipse fitted to the capsule are reported next to the true-capsule scores.

## Not committed

`results/checkpoints/`, `results/cache/`, `results/masks_png/` and `weights/` are git-ignored
(large or regenerable). Get the MedSAM and UltraSam weights with `bash scripts/download_weights.sh`.
