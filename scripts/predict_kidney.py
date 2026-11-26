
from pathlib import Path
import sys
import json

import cv2
import numpy as np
import torch

# Allow imports from the existing src/kidney package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kidney.seg_common import build_model, predict_probs, postprocess


def predict_kidney(image_path, checkpoint_path, output_dir):
    """
    Predict a kidney mask from one ultrasound image.

    Uses the existing DeepLabV3+ model and preprocessing
    developed by the segmentation team.
    """

    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    assert image_path.is_file(), f"Missing image: {image_path}"
    assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ultrasound as grayscale
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # 2. Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 3. Load DeepLabV3+ checkpoint
    model = build_model(
        "deeplabv3p_r50",
        pretrained=False
    ).to(device)

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    # 4. Predict kidney segmentation mask
    probability_map = predict_probs(
        model,
        image,
        size=512,
        device=device
    )
    mask = postprocess(probability_map > 0.5)

    # 5. Save binary mask
    mask_path = output_dir / "kidney_mask.png"
    cv2.imwrite(
        str(mask_path),
        (mask * 255).astype(np.uint8)
    )

    # 6. Save overlay
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    overlay = image_color.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    overlay_path = output_dir / "kidney_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)



    # 7. Generate a kidney-focused crop with 15% padding
    h, w = image.shape

    if mask.any():
        ys, xs = np.where(mask > 0)

        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1

        box_width = x1 - x0
        box_height = y1 - y0

        pad_x = int(box_width * 0.15)
        pad_y = int(box_height * 0.15)

        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(w, x1 + pad_x)
        y1 = min(h, y1 + pad_y)

        crop = image[y0:y1, x0:x1]
    else:
        # If no kidney is detected, keep the full image.
        print("WARNING: Empty kidney mask; using full image.")
        crop = image.copy()

    crop_path = output_dir / "kidney_crop.png"
    if not cv2.imwrite(str(crop_path), crop):
        raise OSError(f"Could not save crop: {crop_path}")

    print("Crop shape:", crop.shape)
    print("Crop saved:", crop_path)

    # Save the training-compatible JPEG crop
    crop_jpg_path = output_dir / "kidney_crop.jpg"

    if not cv2.imwrite(str(crop_jpg_path), crop):
        raise OSError(f"Could not save JPEG crop: {crop_jpg_path}")

    print("JPEG crop saved:", crop_jpg_path)


    # Save crop coordinates for mapping Grad-CAM
    # back to the original ultrasound.
    if mask.any():
        crop_box = [int(x0), int(y0), int(x1), int(y1)]
    else:
        crop_box = [0, 0, int(w), int(h)]

    metadata = {
        "source_image": str(image_path),
        "original_width": int(w),
        "original_height": int(h),
        "crop_box_xyxy": crop_box,
        "padding_ratio": 0.15,
        "segmentation_checkpoint": str(checkpoint_path),
    }

    metadata_path = output_dir / "crop_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Crop metadata saved:", metadata_path)



    print("Device:", device)
    print("Predicted kidney pixels:", int(mask.sum()))
    print("Mask saved:", mask_path)
    print("Overlay saved:", overlay_path)

    return mask


if __name__ == "__main__":
    predict_kidney(
        image_path=ROOT / "bbdd_kidney/images/1013.jpg",
        checkpoint_path=ROOT / "results/checkpoints/deeplabv3p_r50_fold0.pt",
        output_dir=ROOT / "results/phase5_demo/script_test"
    )
