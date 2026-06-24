"""Convert a selected BDD100K real-blur subset to one-class YOLO vehicle labels."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2


VEHICLE_CATEGORIES = {"car", "bus", "truck", "motor"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a YOLO-format BDD100K real-motion-blur vehicle subset.")
    parser.add_argument("--annotations", required=True, type=Path, help="BDD100K detection JSON.")
    parser.add_argument("--images", required=True, type=Path, help="BDD100K source image directory.")
    parser.add_argument("--out", required=True, type=Path, help="Output dataset root.")
    parser.add_argument("--image-list", type=Path, default=None, help="Optional text file of manually selected names.")
    parser.add_argument("--min-width", type=float, default=2.0)
    parser.add_argument("--min-height", type=float, default=2.0)
    return parser.parse_args()


def selected_names(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def clipped_yolo_box(box: dict, width: int, height: int, min_width: float, min_height: float) -> str | None:
    x1 = min(max(float(box["x1"]), 0.0), float(width))
    y1 = min(max(float(box["y1"]), 0.0), float(height))
    x2 = min(max(float(box["x2"]), 0.0), float(width))
    y2 = min(max(float(box["y2"]), 0.0), float(height))
    box_width, box_height = x2 - x1, y2 - y1
    if box_width < min_width or box_height < min_height:
        return None
    center_x = (x1 + x2) / (2.0 * width)
    center_y = (y1 + y2) / (2.0 * height)
    return f"0 {center_x:.8f} {center_y:.8f} {box_width / width:.8f} {box_height / height:.8f}"


def main() -> None:
    args = parse_args()
    names = selected_names(args.image_list)
    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Expected BDD100K annotation JSON to contain a list of image records.")
    output_images = args.out / "images" / "val"
    output_labels = args.out / "labels" / "val"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    retained_images = 0
    retained_boxes = 0
    missing_images = 0
    for record in records:
        name = record.get("name")
        if not name or (names is not None and name not in names):
            continue
        source = args.images / name
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            missing_images += 1
            continue
        height, width = image.shape[:2]
        labels = []
        for item in record.get("labels", []):
            if item.get("category") not in VEHICLE_CATEGORIES or "box2d" not in item:
                continue
            converted = clipped_yolo_box(item["box2d"], width, height, args.min_width, args.min_height)
            if converted is not None:
                labels.append(converted)
        shutil.copy2(source, output_images / name)
        (output_labels / Path(name).with_suffix(".txt")).write_text(
            "\n".join(labels) + ("\n" if labels else ""),
            encoding="utf-8",
        )
        retained_images += 1
        retained_boxes += len(labels)

    print(f"Retained images: {retained_images}")
    print(f"Retained vehicle boxes: {retained_boxes}")
    if missing_images:
        print(f"Warning: {missing_images} selected annotation records had unreadable/missing images.")
    if names is not None:
        print(f"Names requested by manual list: {len(names)}")


if __name__ == "__main__":
    main()
