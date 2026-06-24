"""Convert UAVDT MOT-style annotations to one-class YOLO vehicle data."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from befr.data.utils import (
    clip_xywh_box,
    image_size,
    is_image,
    natural_sorted,
    posix,
    read_sequence_names,
    transfer_file,
    write_lines,
    yolo_line,
)


@dataclass
class AnnotationBox:
    frame_id: int
    left: float
    top: float
    width: float
    height: float
    category: int | None


@dataclass
class Stats:
    images_written: int = 0
    labels_written: int = 0
    boxes_written: int = 0
    boxes_deleted: int = 0
    boxes_clipped: int = 0
    malformed_rows: int = 0
    missing_annotation_frames: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UAVDT vehicle data in YOLO format.")
    parser.add_argument("--images-root", required=True, type=Path, help="Root containing UAVDT sequence image folders.")
    parser.add_argument("--annotations-root", required=True, type=Path, help="Root containing UAVDT GT text files.")
    parser.add_argument("--train-sequences", required=True, type=Path, help="Train sequence-name list.")
    parser.add_argument("--val-sequences", required=True, type=Path, help="Validation sequence-name list.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output YOLO dataset root.")
    parser.add_argument("--splits-output-dir", type=Path, default=Path("data_splits"))
    parser.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy")
    parser.add_argument("--min-width", type=float, default=1.0)
    parser.add_argument("--min-height", type=float, default=1.0)
    return parser.parse_args()


def find_sequence_dir(images_root: Path, sequence: str) -> Path:
    direct = images_root / sequence
    if direct.is_dir():
        return direct
    matches = [path for path in images_root.rglob(sequence) if path.is_dir()]
    if not matches:
        raise FileNotFoundError(f"Could not locate UAVDT sequence {sequence} under {images_root}")
    return natural_sorted(matches)[0]


def find_annotation_file(annotations_root: Path, sequence: str) -> Path:
    candidates = [
        annotations_root / f"{sequence}.txt",
        annotations_root / f"{sequence}_gt.txt",
        annotations_root / f"{sequence}_gt_whole.txt",
        annotations_root / sequence / "gt.txt",
        annotations_root / sequence / f"{sequence}.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = [path for path in annotations_root.rglob("*.txt") if path.stem.casefold().startswith(sequence.casefold())]
    if not matches:
        raise FileNotFoundError(f"Could not locate annotation file for {sequence} under {annotations_root}")
    return natural_sorted(matches)[0]


def frame_number(path: Path) -> int | None:
    digits = "".join(re.findall(r"\d+", path.stem))
    return int(digits) if digits else None


def parse_annotation_file(path: Path) -> tuple[dict[int, list[AnnotationBox]], int]:
    boxes: dict[int, list[AnnotationBox]] = defaultdict(list)
    malformed = 0
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part for part in re.split(r"[\s,]+", line) if part]
        if len(parts) < 6:
            malformed += 1
            continue
        try:
            frame_id = int(float(parts[0]))
            left, top, width, height = (float(value) for value in parts[2:6])
            category = int(float(parts[8])) if len(parts) > 8 else None
        except ValueError:
            malformed += 1
            continue
        if category is not None and category <= 0:
            continue
        boxes[frame_id].append(AnnotationBox(frame_id, left, top, width, height, category))
    return boxes, malformed


def convert_sequence(
    *,
    sequence: str,
    split: str,
    images_root: Path,
    annotations_root: Path,
    output_root: Path,
    mode: str,
    min_width: float,
    min_height: float,
) -> tuple[list[str], Stats]:
    sequence_dir = find_sequence_dir(images_root, sequence)
    annotation_file = find_annotation_file(annotations_root, sequence)
    annotations, malformed = parse_annotation_file(annotation_file)
    stats = Stats(malformed_rows=malformed)
    split_lines: list[str] = []
    image_paths = natural_sorted(path for path in sequence_dir.rglob("*") if is_image(path))
    frame_to_image = {frame_number(path): path for path in image_paths if frame_number(path) is not None}
    for image_path in image_paths:
        number = frame_number(image_path)
        if number is None:
            continue
        width, height = image_size(image_path)
        label_lines: list[str] = []
        for box in annotations.get(number, []):
            if box.width < min_width or box.height < min_height:
                stats.boxes_deleted += 1
                continue
            clipped = clip_xywh_box(box.left, box.top, box.width, box.height, width, height)
            if clipped is None:
                stats.boxes_deleted += 1
                continue
            if clipped.clipped:
                stats.boxes_clipped += 1
            label_lines.append(yolo_line(0, clipped, width, height))
        relative_image = Path("images") / split / sequence / image_path.name
        relative_label = Path("labels") / split / sequence / image_path.with_suffix(".txt").name
        transfer_file(image_path, output_root / relative_image, mode)
        (output_root / relative_label).parent.mkdir(parents=True, exist_ok=True)
        (output_root / relative_label).write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        split_lines.append(posix(relative_image))
        stats.images_written += 1
        stats.labels_written += 1
        stats.boxes_written += len(label_lines)
    stats.missing_annotation_frames = len(set(annotations) - set(frame_to_image))
    return split_lines, stats


def merge_stats(items: list[Stats]) -> Stats:
    total = Stats()
    for item in items:
        for field in total.__dataclass_fields__:
            setattr(total, field, getattr(total, field) + getattr(item, field))
    return total


def convert_split(args: argparse.Namespace, split: str, sequence_file: Path) -> tuple[list[str], Stats]:
    lines: list[str] = []
    stats_items: list[Stats] = []
    for sequence in read_sequence_names(sequence_file):
        sequence_lines, stats = convert_sequence(
            sequence=sequence,
            split=split,
            images_root=args.images_root,
            annotations_root=args.annotations_root,
            output_root=args.output_root,
            mode=args.mode,
            min_width=args.min_width,
            min_height=args.min_height,
        )
        lines.extend(sequence_lines)
        stats_items.append(stats)
    return [posix(path) for path in natural_sorted(Path(line) for line in lines)], merge_stats(stats_items)


def main() -> int:
    args = parse_args()
    try:
        train_lines, train_stats = convert_split(args, "train", args.train_sequences)
        val_lines, val_stats = convert_split(args, "val", args.val_sequences)
        write_lines(args.splits_output_dir / "uavdt_train.txt", train_lines)
        write_lines(args.splits_output_dir / "uavdt_val.txt", val_lines)
        total = merge_stats([train_stats, val_stats])
        print(f"Images written: {total.images_written}")
        print(f"Labels written: {total.labels_written}")
        print(f"Vehicle boxes written: {total.boxes_written}")
        print(f"Deleted boxes: {total.boxes_deleted}")
        print(f"Clipped boxes: {total.boxes_clipped}")
        print(f"Malformed annotation rows: {total.malformed_rows}")
        print(f"Annotation frames without image: {total.missing_annotation_frames}")
        return 0 if total.images_written else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
