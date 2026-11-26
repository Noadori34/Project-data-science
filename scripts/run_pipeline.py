
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Kidney ultrasound inference pipeline"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to a kidney ultrasound image"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        choices=range(5),
        help="DeepLabV3+ segmentation fold (0–4)"
    )
    args = parser.parse_args()

    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    output_dir = (
        ROOT / "results/phase5_demo"
        / image_path.stem
    )

    checkpoint = (
        ROOT / "results/checkpoints"
        / f"deeplabv3p_r50_fold{args.fold}.pt"
    )

    # Step 1: Segmentation and kidney crop
    from predict_kidney import predict_kidney

    predict_kidney(
        image_path=image_path,
        checkpoint_path=checkpoint,
        output_dir=output_dir
    )

    crop_path = output_dir / "kidney_crop.jpg"
    if not crop_path.is_file():
        raise RuntimeError("Kidney crop was not generated")

    # Step 2: EfficientNet ensemble classification
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/test_classifier.py"),
            "--image",
            str(crop_path)
        ],
        check=True
    )



    # Step 3: Generate ensemble Grad-CAM
    gradcam_dir = output_dir / "gradcam"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/ensemble_gradcam.py"),
            "--image",
            str(crop_path),
            "--out-dir",
            str(gradcam_dir)
        ],
        check=True
    )

    # Step 4: Project ensemble Grad-CAM onto original ultrasound
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/project_gradcam.py"),
            "--metadata",
            str(output_dir / "crop_metadata.json"),
            "--cam",
            str(gradcam_dir / "ensemble_cam.npy"),
            "--output",
            str(output_dir / "gradcam_full_ultrasound.png")
        ],
        check=True
    )


    print("\nPipeline completed successfully.")
    print("Outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
