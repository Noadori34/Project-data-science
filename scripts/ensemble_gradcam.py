from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import timm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEFAULT_IMAGE = "results/phase5_demo/script_test/kidney_crop.jpg"
DEFAULT_CKPT_DIR = "results/phase2_classifier/nb06crops_weighted/imagenet"
DEFAULT_OUT_DIR = "results/phase5_demo/gradcam_ensemble"
IMAGE_SIZE = 224
TARGET_CLASS = 1  # pathological


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def get_device():
    # Grad-CAM is safest on CUDA or CPU.
    # On Mac, CPU is the most compatible fallback.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_model(device):
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=2
    )
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2)
    )
    return model.to(device)


def percentile_normalize(gray):
    """
    Phase 2 preprocessing:
    crop -> RGB -> percentile 1-99 -> resize 224 -> tensor [0,1]
    """
    a = gray.astype(np.float32)

    p1 = np.percentile(a, 1)
    p99 = np.percentile(a, 99)

    if p99 <= p1:
        a = np.clip(a, 0, 255)
        a = a / 255.0
    else:
        a = np.clip((a - p1) / (p99 - p1), 0.0, 1.0)

    return a


def load_and_preprocess_image(image_path, image_size=224):
    """
    Returns:
        rgb_for_overlay: HxWx3 float32 in [0,1]
        input_tensor: 1x3xHxW torch tensor
    """
    img = Image.open(image_path).convert("L")
    gray = np.array(img)

    norm = percentile_normalize(gray)
    rgb = np.stack([norm, norm, norm], axis=-1)

    rgb_resized = cv2.resize(
        rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

    input_tensor = torch.from_numpy(
        rgb_resized.transpose(2, 0, 1)
    ).float().unsqueeze(0)

    return rgb_resized, input_tensor


def load_checkpoint_into_model(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"'model_state_dict' not found in {checkpoint_path}")

    model.load_state_dict(state_dict)
    model.eval()
    return checkpoint


def predict_probability(model, input_tensor, device):
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return float(probs[1])  # pathology probability


def get_target_layer(model):
    # EfficientNet-B0 in timm: last convolutional layer before classifier
    if hasattr(model, "conv_head"):
        return [model.conv_head]

    # fallback
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("Could not find a convolutional layer for Grad-CAM.")
    return [conv_layers[-1]]


def save_heatmap_png(cam_map, save_path):
    """
    Save a colored heatmap image from a normalized CAM in [0,1].
    """
    heat_uint8 = np.uint8(255 * cam_map)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(str(save_path), heat_color)


# --------------------------------------------------
# Main Grad-CAM generation
# --------------------------------------------------

def generate_gradcam_for_fold(
    fold_idx,
    image_path,
    checkpoint_path,
    output_dir,
    device,
    target_class=1
):
    rgb_img, input_tensor = load_and_preprocess_image(image_path, IMAGE_SIZE)

    model = make_model(device)
    checkpoint = load_checkpoint_into_model(model, checkpoint_path, device)

    prob = predict_probability(model, input_tensor, device)

    target_layers = get_target_layer(model)
    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor.to(device),
            targets=targets
        )[0]

    grayscale_cam = np.clip(grayscale_cam, 0, 1)

    overlay = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    # Save outputs
    fold_prefix = f"fold_{fold_idx}"

    raw_resized_path = output_dir / f"{fold_prefix}_input_224.png"
    heatmap_path = output_dir / f"{fold_prefix}_heatmap.png"
    overlay_path = output_dir / f"{fold_prefix}_overlay.png"
    cam_npy_path = output_dir / f"{fold_prefix}_cam.npy"

    cv2.imwrite(
        str(raw_resized_path),
        cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )
    save_heatmap_png(grayscale_cam, heatmap_path)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    np.save(cam_npy_path, grayscale_cam)

    result = {
        "fold": fold_idx,
        "checkpoint": str(checkpoint_path),
        "probability_pathological": prob,
        "threshold": checkpoint.get("threshold"),
        "best_epoch": checkpoint.get("best_epoch"),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "cam_npy_path": str(cam_npy_path),
    }

    return grayscale_cam, rgb_img, result


def main():
    parser = argparse.ArgumentParser(description="Generate ensemble Grad-CAM for kidney classifier.")
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE,
        help="Path to kidney crop image"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=DEFAULT_CKPT_DIR,
        help="Directory containing fold_0.pt ... fold_4.pt"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=TARGET_CLASS,
        help="Target class for Grad-CAM (default: 1 pathological)"
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    assert image_path.exists(), f"Image not found: {image_path}"
    assert ckpt_dir.exists(), f"Checkpoint directory not found: {ckpt_dir}"

    device = get_device()
    print("Grad-CAM device:", device)
    print("Image:", image_path)
    print("Checkpoint directory:", ckpt_dir)
    print("Output directory:", out_dir)
    print()

    all_cams = []
    fold_results = []
    rgb_reference = None

    for fold_idx in range(5):
        checkpoint_path = ckpt_dir / f"fold_{fold_idx}.pt"
        assert checkpoint_path.exists(), f"Missing checkpoint: {checkpoint_path}"

        print(f"Generating Grad-CAM for fold {fold_idx} ...")
        cam_map, rgb_img, result = generate_gradcam_for_fold(
            fold_idx=fold_idx,
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            output_dir=out_dir,
            device=device,
            target_class=args.target_class
        )

        print(
            f"  pathology probability = {result['probability_pathological']:.4f} | "
            f"threshold = {result['threshold']}"
        )

        all_cams.append(cam_map)
        fold_results.append(result)
        rgb_reference = rgb_img

    # Ensemble average CAM
    ensemble_cam = np.mean(np.stack(all_cams, axis=0), axis=0)
    ensemble_cam = np.clip(ensemble_cam, 0, 1)

    ensemble_overlay = show_cam_on_image(
        rgb_reference,
        ensemble_cam,
        use_rgb=True
    )

    ensemble_heatmap_path = out_dir / "ensemble_heatmap.png"
    ensemble_overlay_path = out_dir / "ensemble_overlay.png"
    ensemble_cam_npy_path = out_dir / "ensemble_cam.npy"

    save_heatmap_png(ensemble_cam, ensemble_heatmap_path)
    cv2.imwrite(
        str(ensemble_overlay_path),
        cv2.cvtColor(ensemble_overlay, cv2.COLOR_RGB2BGR)
    )
    np.save(ensemble_cam_npy_path, ensemble_cam)

    # Save summary CSV
    csv_path = out_dir / "gradcam_fold_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "checkpoint",
                "probability_pathological",
                "threshold",
                "best_epoch",
                "heatmap_path",
                "overlay_path",
                "cam_npy_path",
            ]
        )
        writer.writeheader()
        writer.writerows(fold_results)

    print()
    print("Saved fold Grad-CAM outputs.")
    print("Saved ensemble heatmap:", ensemble_heatmap_path)
    print("Saved ensemble overlay:", ensemble_overlay_path)
    print("Saved ensemble raw CAM:", ensemble_cam_npy_path)
    print("Saved summary CSV:", csv_path)


if __name__ == "__main__":
    main()