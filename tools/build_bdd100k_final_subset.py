"""Rebuild the final 500-image BDD100K real motion-blur YOLO subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from befr.data.utils import find_images, image_label_relative_path, posix, transfer_file, validate_yolo_label


EXPECTED_FINAL = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a final BDD100K real-blur YOLO subset from a manifest.")
    parser.add_argument("--source-root", required=True, type=Path, help="Prepared 800-candidate YOLO dataset root.")
    parser.add_argument("--manifest", required=True, type=Path, help="Final image-name manifest.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output YOLO dataset root.")
    parser.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy")
    parser.add_argument("--expected-count", type=int, default=EXPECTED_FINAL)
    return parser.parse_args()


def image_root(source_root: Path) -> Path:
    root = source_root / "images"
    return root if root.is_dir() else source_root


def label_root(source_root: Path) -> Path:
    root = source_root / "labels"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {root}")
    return root


def build_image_index(source_root: Path) -> dict[str, Path]:
    images = find_images(image_root(source_root))
    index: dict[str, Path] = {}
    duplicates = []
    for image in images:
        if image.name in index:
            duplicates.append(image.name)
        index[image.name] = image
    if duplicates:
        raise ValueError(f"Duplicate image names in source root: {duplicates[:20]}")
    return index


def resolve_manifest_image(source_root: Path, image_index: dict[str, Path], entry: str) -> Path:
    entry_path = Path(entry)
    if entry_path.is_absolute():
        candidate = entry_path
    elif entry.startswith("images/") or entry.startswith("images\\"):
        candidate = source_root / entry_path
    else:
        candidate = image_index.get(entry_path.name)
        if candidate is None:
            candidate = image_root(source_root) / "val" / entry_path.name
    if candidate is None or not candidate.is_file():
        raise FileNotFoundError(f"Manifest image not found in source root: {entry}")
    return candidate


def corresponding_label(source_root: Path, image_path: Path) -> Path:
    images_base = image_root(source_root)
    labels_base = label_root(source_root)
    relative = image_path.relative_to(images_base)
    return labels_base / relative.with_suffix(".txt")


def main() -> int:
    args = parse_args()
    try:
        source_root = args.source_root.expanduser().resolve()
        output_root = args.output_root.expanduser().resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"Source root not found: {source_root}")
        if not args.manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {args.manifest}")
        entries = [line.strip() for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(entries) != args.expected_count:
            raise ValueError(f"Expected {args.expected_count} manifest entries, found {len(entries)}")
        if len(set(entries)) != len(entries):
            raise ValueError("Manifest contains duplicate entries.")

        image_index = build_image_index(source_root)
        copied_images = 0
        copied_labels = 0
        total_boxes = 0
        for entry in entries:
            image_path = resolve_manifest_image(source_root, image_index, entry)
            label_path = corresponding_label(source_root, image_path)
            validation = validate_yolo_label(label_path)
            if not validation.exists:
                raise FileNotFoundError(f"Missing label for {entry}: {label_path}")
            if validation.invalid_count:
                raise ValueError(f"Invalid YOLO label rows in {label_path}: {validation.invalid_count}")
            relative_under_images = image_path.relative_to(image_root(source_root))
            destination_image = output_root / "images" / "val" / relative_under_images.name
            destination_label = output_root / image_label_relative_path(Path("images") / "val" / relative_under_images.name)
            transfer_file(image_path, destination_image, args.mode)
            transfer_file(label_path, destination_label, args.mode)
            copied_images += 1
            copied_labels += 1
            total_boxes += validation.valid_count

        print(f"Images written: {copied_images}")
        print(f"Labels written: {copied_labels}")
        print(f"Valid vehicle boxes: {total_boxes}")
        print(f"Output images: {posix(output_root / 'images' / 'val')}")
        print(f"Output labels: {posix(output_root / 'labels' / 'val')}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
