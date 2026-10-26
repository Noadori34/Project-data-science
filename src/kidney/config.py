"""Paths and global settings. All paths are relative to the repo root (plan.md 1.4)."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("KIDNEY_ROOT", Path(__file__).resolve().parents[2]))

# Dataset A: URI-CADS [24]
DATA_A = ROOT / "bbdd_kidney"
A_IMAGES = DATA_A / "images"
A_LABELS = DATA_A / "labels"
A_MASKS_MAT = DATA_A / "masks"

# Dataset B: KidneyUS [18] (external test only)
DATA_B_IMAGES = ROOT / "kidneyUS_images_25_june_2025"
DATA_B_LABELS = ROOT / "kidneyUS-main" / "labels"
DATA_B_CAPSULE = DATA_B_LABELS / "reviewed_masks_1" / "capsule"
DATA_B_CSV = DATA_B_LABELS / "reviewed_labels_1.csv"

RESULTS = ROOT / "results"
SPLITS = RESULTS / "splits"
FOLDS_CSV = SPLITS / "folds.csv"
TEAM_FOLDS_CSV = RESULTS / "eda" / "folds.csv"  # leak-free, group-aware split already in the repo
MASKS_PNG = RESULTS / "masks_png"  # dataset A masks converted from .mat
PRED_MASKS = RESULTS / "pred_masks"  # out-of-fold predictions, one subfolder per model
CHECKPOINTS = RESULTS / "checkpoints"
RUNS_CSV = RESULTS / "runs.csv"
SEG_DIR = RESULTS / "segmentation"
WEIGHTS = ROOT / "weights"  # MedSAM / UltraSam checkpoints (not committed)

SEED = 42
N_FOLDS = 5

# Box classes 1-9 [24]
BOX_CLASSES = {1: "LIT", 2: "SCY", 3: "CCY", 4: "PYR", 5: "ANG", 6: "SRM", 7: "HYD", 8: "CT", 9: "CE"}


def seed_everything(seed: int = SEED) -> None:
    """Fixed seeds (plan.md 1.4)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def library_versions() -> dict:
    """Library versions recorded with every run (plan.md 1.4)."""
    import importlib

    out = {}
    for name in ["torch", "torchvision", "monai", "segmentation_models_pytorch", "timm", "numpy", "sklearn", "cv2"]:
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:
            out[name] = "n/a"
    return out


def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
