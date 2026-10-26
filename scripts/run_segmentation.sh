#!/usr/bin/env bash
# Full Phase 2 pipeline (plan.md section 3), in the order of plan.md section 7.
# Usage:   bash scripts/run_segmentation.sh            # full run on the GPU
#          bash scripts/run_segmentation.sh --smoke    # quick CPU check of every step
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
SMOKE=""
if [ "${1:-}" = "--smoke" ]; then SMOKE="--smoke"; fi

echo "== 0. Tests";                     python -m pytest tests -q
echo "== 1. Shared split (folds.csv)";  python -m kidney.splits
echo "== 2. .mat -> PNG + mask checks"; python -m kidney.prepare_masks
echo "== 3. S-A U-Net ResNet-34";       python -m kidney.train_seg --model unet_r34 $SMOKE
echo "== 4. S-B DeepLabV3+";            python -m kidney.train_seg --model deeplabv3p_r50 $SMOKE
if [ -z "$SMOKE" ]; then bash scripts/download_weights.sh; fi
echo "== 5. S-C MedSAM (gt + auto box)";   python -m kidney.sam_prompt --model medsam $SMOKE
echo "== 6. S-D UltraSam (gt + auto box)"; python -m kidney.sam_prompt --model ultrasam $SMOKE || echo "UltraSam step failed; see message above"
echo "== 7. MedSAM decoder fine-tune";     python -m kidney.finetune_medsam $SMOKE
echo "== 8. External test on KidneyUS";    python -m kidney.eval_external --models unet_r34 deeplabv3p_r50 medsam ultrasam medsam_ft $SMOKE
echo "== 9. Tables, kidney_boxes.csv, failure overlays"; python -m kidney.report_seg
