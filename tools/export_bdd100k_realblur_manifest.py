"""Export a public manifest for a prepared BDD100K real-blur YOLO subset."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from befr.data.utils import natural_sorted


DEFAULT_DATASET_ROOT = Path(r"D:\BDD100K\bdd100k_realblur_vehicle_clean")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
KNOWN_SPLITS = {"train", "val", "valid", "validation", "test"}


@dataclass(frozen=True)
class LabelStats:
    label_exists: bool
    object_count: int
    vehicle_count: int
    empty_label: bool
    invalid_lines: tuple[str, ...]


@dataclass(frozen=True)
class ManifestRow:
    image_name: str
    relative_image_path: str
    relative_label_path: str
    split: str
    label_exists: bool
    object_count: int
    vehicle_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TXT/CSV manifests from a prepared BDD100K real-motion-blur YOLO dataset."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Prepared dataset root.")
    parser.add_argument("--txt-output", type=Path, required=True, help="Output TXT path.")
    parser.add_argument("--csv-output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--fail-on-missing-label",
        action="store_true",
        help="Return a non-zero exit code if any image lacks a matching YOLO label file.",
    )
    return parser.parse_args()


def posix(path: Path) -> str:
    return path.as_posix()


def find_image_root(dataset_root: Path) -> Path:
    image_root = dataset_root / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing images directory: {image_root}")
    return image_root


def find_label_root(dataset_root: Path) -> Path:
    label_root = dataset_root / "labels"
    if not label_root.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {label_root}")
    return label_root


def iter_images(image_root: Path) -> list[Path]:
    images = natural_sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No supported image files found under {image_root}")
    return images


def public_image_name(relative_to_images: Path) -> str:
    parts = relative_to_images.parts
    if parts and parts[0].lower() in KNOWN_SPLITS:
        return posix(Path(*parts[1:]))
    return posix(relative_to_images)


def split_name(relative_to_images: Path) -> str:
    parts = relative_to_images.parts
    if parts and parts[0].lower() in KNOWN_SPLITS:
        return "val" if parts[0].lower() == "validation" else parts[0]
    return "unknown"


def label_path_for(label_root: Path, relative_to_images: Path) -> Path:
    return label_root / relative_to_images.with_suffix(".txt")


def read_label_stats(path: Path) -> LabelStats:
    if not path.exists():
        return LabelStats(False, 0, 0, False, ())
    invalid_lines: list[str] = []
    object_count = 0
    vehicle_count = 0
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) < 5:
            invalid_lines.append(f"{path}:{line_number}: expected at least 5 YOLO columns")
            continue
        try:
            class_id = int(float(parts[0]))
            center_x, center_y, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            invalid_lines.append(f"{path}:{line_number}: non-numeric YOLO value")
            continue
        if width <= 0.0 or height <= 0.0:
            invalid_lines.append(f"{path}:{line_number}: non-positive box width/height")
            continue
        if not all(0.0 <= value <= 1.0 for value in (center_x, center_y, width, height)):
            invalid_lines.append(f"{path}:{line_number}: normalized box values outside [0, 1]")
            continue
        object_count += 1
        if class_id == 0:
            vehicle_count += 1
    return LabelStats(True, object_count, vehicle_count, not lines, tuple(invalid_lines))


def build_manifest(dataset_root: Path) -> tuple[list[ManifestRow], dict[str, int], list[str]]:
    dataset_root = dataset_root.expanduser().resolve()
    image_root = find_image_root(dataset_root)
    label_root = find_label_root(dataset_root)
    rows: list[ManifestRow] = []
    warnings: list[str] = []
    seen_names: set[str] = set()
    duplicates = 0
    missing_labels = 0
    empty_labels = 0
    invalid_label_files = 0

    for image_path in iter_images(image_root):
        relative_to_images = image_path.relative_to(image_root)
        image_name = public_image_name(relative_to_images)
        if image_name in seen_names:
            duplicates += 1
            warnings.append(f"Duplicate image name in public manifest: {image_name}")
        seen_names.add(image_name)

        label_path = label_path_for(label_root, relative_to_images)
        stats = read_label_stats(label_path)
        if not stats.label_exists:
            missing_labels += 1
            warnings.append(f"Missing label for {image_name}: {label_path}")
        if stats.empty_label:
            empty_labels += 1
            warnings.append(f"Empty label file for {image_name}: {label_path}")
        if stats.invalid_lines:
            invalid_label_files += 1
            warnings.extend(stats.invalid_lines)

        rows.append(
            ManifestRow(
                image_name=image_name,
                relative_image_path=posix(Path("images") / relative_to_images),
                relative_label_path=posix(Path("labels") / relative_to_images.with_suffix(".txt")),
                split=split_name(relative_to_images),
                label_exists=stats.label_exists,
                object_count=stats.object_count,
                vehicle_count=stats.vehicle_count,
            )
        )

    rows.sort(key=lambda row: row.image_name)
    summary = {
        "images": len(rows),
        "labels": sum(1 for row in rows if row.label_exists),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "duplicates": duplicates,
        "invalid_label_files": invalid_label_files,
        "total_objects": sum(row.object_count for row in rows),
        "total_vehicles": sum(row.vehicle_count for row in rows),
    }
    return rows, summary, warnings


def write_outputs(rows: list[ManifestRow], txt_output: Path, csv_output: Path) -> None:
    txt_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    txt_output.write_text("\n".join(row.image_name for row in rows) + "\n", encoding="utf-8")
    fieldnames = [
        "image_name",
        "relative_image_path",
        "relative_label_path",
        "split",
        "label_exists",
        "object_count",
        "vehicle_count",
    ]
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_name": row.image_name,
                    "relative_image_path": row.relative_image_path,
                    "relative_label_path": row.relative_label_path,
                    "split": row.split,
                    "label_exists": row.label_exists,
                    "object_count": row.object_count,
                    "vehicle_count": row.vehicle_count,
                }
            )


def main() -> int:
    args = parse_args()
    try:
        rows, summary, warnings = build_manifest(args.dataset_root)
        write_outputs(rows, args.txt_output, args.csv_output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"wrote_txt: {args.txt_output}")
    print(f"wrote_csv: {args.csv_output}")

    if summary["duplicates"] or summary["invalid_label_files"]:
        return 1
    if args.fail_on_missing_label and summary["missing_labels"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
