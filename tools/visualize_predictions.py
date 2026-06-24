"""Render side-by-side predictions for baseline and BE-FR ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib.pyplot as plt
import torch

from befr.evaluation import load_yolo


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare predictions from multiple checkpoints.")
    parser.add_argument(
        "--weights",
        action="append",
        required=True,
        help="Repeat as NAME=path/to/best.pt for YOLOv8, BE-only, FR-only, and BE-FR.",
    )
    parser.add_argument("--images", required=True, type=Path, help="One image or a directory.")
    parser.add_argument("--out", type=Path, default=Path("results/visualizations/predictions"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def named_weights(values: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(path).stem
        parsed.append((name, path))
    return parsed


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    paths = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No images found under {source}")
    return paths


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    models = [(name, load_yolo(path)) for name, path in named_weights(args.weights)]
    for image_path in image_paths(args.images):
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise ValueError(f"OpenCV could not read {image_path}")
        panels = [("Input", original)]
        for name, model in models:
            result = model.predict(source=original, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
            panels.append((name, result.plot()))
        figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), constrained_layout=True)
        if len(panels) == 1:
            axes = [axes]
        for axis, (title, panel) in zip(axes, panels):
            axis.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            axis.set_title(title)
            axis.axis("off")
        destination = args.out / f"{image_path.stem}_comparison.png"
        figure.savefig(destination, dpi=300, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
