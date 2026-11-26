
from pathlib import Path
import argparse
import json

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Project kidney crop Grad-CAM onto original ultrasound"
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--cam",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )

    args = parser.parse_args()

    # 1. Read crop metadata
    with open(args.metadata) as f:
        metadata = json.load(f)

    image_path = Path(metadata["source_image"])
    x0, y0, x1, y1 = metadata["crop_box_xyxy"]

    # 2. Load original ultrasound
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    h, w = image.shape[:2]

    assert 0 <= x0 < x1 <= w
    assert 0 <= y0 < y1 <= h

    # 3. Load ensemble Grad-CAM
    cam = np.load(args.cam)

    assert cam.ndim == 2, "Grad-CAM must be a 2D array"

    # 4. Resize CAM to original crop dimensions
    crop_width = x1 - x0
    crop_height = y1 - y0

    cam_resized = cv2.resize(
        cam.astype(np.float32),
        (crop_width, crop_height),
        interpolation=cv2.INTER_LINEAR
    )

    cam_resized = np.clip(cam_resized, 0, 1)

    # 5. Place CAM in original-image coordinates
    full_cam = np.zeros((h, w), dtype=np.float32)
    full_cam[y0:y1, x0:x1] = cam_resized

    # 6. Create heatmap
    heatmap = cv2.applyColorMap(
        np.uint8(255 * full_cam),
        cv2.COLORMAP_JET
    )

    # 7. Blend heatmap only inside the crop.
    overlay = image.copy()

    blended_crop = cv2.addWeighted(
        image[y0:y1, x0:x1],
        0.6,
        heatmap[y0:y1, x0:x1],
        0.4,
        0
    )

    overlay[y0:y1, x0:x1] = blended_crop

    # Draw crop boundary for reference
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x1 - 1, y1 - 1),
        (0, 255, 255),
        2
    )

    # 8. Save result
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(args.output), overlay):
        raise OSError(f"Failed to save {args.output}")

    print("Original image:", image_path)
    print("Image dimensions:", (w, h))
    print("Crop box:", (x0, y0, x1, y1))
    print("CAM dimensions:", cam.shape)
    print("Projected Grad-CAM saved:", args.output)


if __name__ == "__main__":
    main()
