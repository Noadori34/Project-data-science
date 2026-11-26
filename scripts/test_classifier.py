
from pathlib import Path
import sys

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm

ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = ROOT / "results/phase5_demo/script_test/kidney_crop.jpg"
CHECKPOINT_PATH = (
    ROOT
    / "results/phase2_classifier/nb06crops_weighted/imagenet/fold_0.pt"
)


def preprocess(image_path):
    # Read the kidney crop as RGB.
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Per-image percentile intensity scaling, 1st–99th percentile.
    low, high = np.percentile(image, [1, 99])
    image = np.clip(
        (image - low) / max(high - low, 1e-6),
        0.0,
        1.0
    )

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # HWC -> CHW and add batch dimension.
    tensor = torch.from_numpy(
        image.transpose(2, 0, 1).copy()
    ).unsqueeze(0).float()

    return tensor


def make_model():
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=2
    )

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 2)
    )

    return model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=IMAGE_PATH,
        help="Path to an automatically generated kidney crop"
    )
    args = parser.parse_args()

    image_path = args.image
    assert image_path.is_file(), f"Missing crop: {image_path}"

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    x = preprocess(image_path).to(device)

    probabilities = []

    for fold in range(5):
        checkpoint_path = (
            ROOT
            / "results/phase2_classifier"
            / "nb06crops_weighted/imagenet"
            / f"fold_{fold}.pt"
        )

        assert checkpoint_path.is_file(), (
            f"Missing checkpoint: {checkpoint_path}"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        model = make_model()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()

        with torch.inference_mode():
            logits = model(x)
            probability = torch.softmax(
                logits, dim=1
            )[0, 1].item()

        probabilities.append(probability)

        print(
            f"Fold {fold}: pathology probability = "
            f"{probability:.4f}"
        )

        del model

    # Soft-voting ensemble
    ensemble_probability = float(np.mean(probabilities))

    # Provisional deployment threshold from Phase 2
    ensemble_threshold = 0.2478711456060409

    label = (
        "Pathological"
        if ensemble_probability >= ensemble_threshold
        else "Healthy"
    )

    print("\nFINAL ENSEMBLE RESULT")
    print("Device:", device)
    print("Mean pathology probability:", round(ensemble_probability, 4))
    print("Operating threshold:", round(ensemble_threshold, 4))
    print("Predicted label:", label)
    print(
        "Note: Ensemble threshold is provisional "
        "and has not been independently validated."
    )


if __name__ == "__main__":
    main()
