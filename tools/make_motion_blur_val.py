"""Create controlled synthetic motion-blur validation sets without overwriting source data."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import yaml

from befr.blur.motion_psf import apply_motion_blur


LEVEL_LENGTHS = {
    "light": (9, 11, 13),
    "medium": (15, 17, 19),
    "heavy": (21, 23, 25),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PSF motion-blur validation dataset.")
    parser.add_argument("--images", required=True, type=Path, help="Source validation image directory.")
    parser.add_argument("--labels", required=True, type=Path, help="Source YOLO validation label directory.")
    parser.add_argument("--out", required=True, type=Path, help="New dataset root.")
    parser.add_argument("--level", choices=["clear", "light", "medium", "heavy", "all"], default="heavy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yaml-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset-name", default=None)
    return parser.parse_args()


def blur_image(image: np.ndarray, length: int, theta: float) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    blurred = apply_motion_blur(tensor, length, theta)
    return (blurred.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def write_dataset_yaml(dataset_root: Path, yaml_path: Path) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_root = str(dataset_root.resolve()).replace("\\", "/")
    payload = {"path": resolved_root, "train": "images/val", "val": "images/val", "names": {0: "vehicle"}}
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def generate_level(
    images_dir: Path,
    labels_dir: Path,
    out_root: Path,
    level: str,
    rng: np.random.Generator,
) -> int:
    target_root = out_root / level
    target_images = target_root / "images" / "val"
    target_labels = target_root / "labels" / "val"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under {images_dir}")

    missing_labels = 0
    for source_image in image_paths:
        relative = source_image.relative_to(images_dir)
        destination_image = target_images / relative
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        if destination_image.exists():
            raise FileExistsError(f"Refusing to overwrite existing generated image: {destination_image}")
        if level == "clear":
            shutil.copy2(source_image, destination_image)
        else:
            image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"OpenCV could not read {source_image}")
            length = int(rng.choice(LEVEL_LENGTHS[level]))
            theta = float(rng.uniform(0.0, 180.0))
            output = blur_image(image, length, theta)
            if not cv2.imwrite(str(destination_image), output):
                raise OSError(f"OpenCV could not write {destination_image}")

        source_label = labels_dir / relative.with_suffix(".txt")
        destination_label = target_labels / relative.with_suffix(".txt")
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        if source_label.exists():
            shutil.copy2(source_label, destination_label)
        else:
            missing_labels += 1
            destination_label.touch()
    if missing_labels:
        print(f"Warning: {missing_labels} images had no label file; empty labels were created.")
    return len(image_paths)


def main() -> None:
    args = parse_args()
    if not args.images.is_dir() or not args.labels.is_dir():
        raise FileNotFoundError("Both --images and --labels must be existing directories.")
    levels = ["clear", "light", "medium", "heavy"] if args.level == "all" else [args.level]
    dataset_name = args.dataset_name or args.images.parents[1].name
    slug = dataset_name.lower().replace("-", "_").replace(" ", "_")
    for offset, level in enumerate(levels):
        count = generate_level(args.images, args.labels, args.out, level, np.random.default_rng(args.seed + offset))
        yaml_path = args.yaml_dir / f"{slug}_{level}.yaml"
        write_dataset_yaml(args.out / level, yaml_path)
        print(f"Created {count} {level} validation images and {yaml_path}")


if __name__ == "__main__":
    main()
