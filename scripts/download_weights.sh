#!/usr/bin/env bash
# Download the foundation-model checkpoints for S-C (MedSAM [S3]) and S-D (UltraSam [S1]).
# They are large (~360 MB each) and are NOT committed (weights/ is in .gitignore).
# Also fetches our trained DeepLabV3+ (S-B) fold checkpoints from the GitHub release
# seg-weights-v1 into results/checkpoints/ (107 MB each, over GitHub's file limit).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights

if [ ! -s weights/medsam_vit_b.pth ]; then
  echo "Downloading MedSAM (Zenodo record 10689643)..."
  wget -q --show-progress -O weights/medsam_vit_b.pth \
    "https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1" \
  || { rm -f weights/medsam_vit_b.pth
       echo "MedSAM download failed. Get medsam_vit_b.pth from https://zenodo.org/records/10689643"
       echo "or the link in https://github.com/bowang-lab/MedSAM and put it in weights/."; }
fi

if [ ! -s weights/UltraSam.pth ]; then
  echo "Downloading UltraSam..."
  wget -q --show-progress -O weights/UltraSam.pth \
    "https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth" \
  || { rm -f weights/UltraSam.pth
       echo "UltraSam download failed. See https://github.com/CAMMA-public/UltraSam (README, checkpoint link)."; }
fi

RELEASE="https://github.com/Noadori34/Project-data-science/releases/download/seg-weights-v1"
mkdir -p results/checkpoints
for k in 0 1 2 3 4; do
  f="results/checkpoints/deeplabv3p_r50_fold${k}.pt"
  if [ ! -s "$f" ]; then
    echo "Downloading DeepLabV3+ fold ${k}..."
    wget -q --show-progress -O "$f" "${RELEASE}/deeplabv3p_r50_fold${k}.pt" \
    || { rm -f "$f"; echo "DeepLabV3+ fold ${k} download failed. See ${RELEASE%/download/*}/tag/seg-weights-v1"; }
  fi
done
(cd results/checkpoints && sha256sum -c --ignore-missing ../../scripts/deeplabv3p_r50.sha256)

ls -lh weights results/checkpoints
