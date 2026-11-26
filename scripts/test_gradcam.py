
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from test_classifier import ROOT, preprocess, make_model


IMAGE_PATH = ROOT / "results/phase5_demo/1013/kidney_crop.jpg"
CHECKPOINT_PATH = (
    ROOT
    / "results/phase2_classifier/nb06crops_weighted/imagenet/fold_0.pt"
)
OUTPUT_PATH = ROOT / "results/phase5_demo/1013/gradcam_fold0.png"


def main():
    assert IMAGE_PATH.is_file(), f"Missing image: {IMAGE_PATH}"
    assert CHECKPOINT_PATH.is_file(), f"Missing checkpoint: {CHECKPOINT_PATH}"

    # MPS support in Grad-CAM can vary with PyTorch and package versions.
    # Start with CPU for a reproducible first test.
    device = torch.device("cpu")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model = make_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # Exact preprocessing used in the classifier.
    x = preprocess(IMAGE_PATH).to(device)

    # Grad-CAM on final convolutional features.
    target_layers = [model.conv_head]

    # Target the pathological class (class index 1).
    targets = [ClassifierOutputTarget(1)]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=x,
        targets=targets
    )[0]

    # RGB image corresponding to the classifier input.
    rgb = x[0].permute(1, 2, 0).detach().cpu().numpy()
    rgb = np.clip(rgb, 0, 1)

    # Create a color heatmap.
    heatmap = cv2.applyColorMap(
        np.uint8(255 * grayscale_cam),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend CAM with ultrasound crop.
    blended = (
        0.6 * rgb +
        0.4 * (heatmap.astype(np.float32) / 255.0)
    )
    blended = np.uint8(np.clip(blended, 0, 1) * 255)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(OUTPUT_PATH),
        cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    )

    print("Grad-CAM generated successfully.")
    print("Target class: Pathological (1)")
    print("CAM shape:", grayscale_cam.shape)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
