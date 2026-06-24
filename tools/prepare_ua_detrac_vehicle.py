"""Convert UA-DETRAC XML annotations to one-class YOLO vehicle data."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from befr.data.utils import (
    BoxResult,
    box_inside,
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
class Stats:
    images_written: int = 0
    labels_written: int = 0
    boxes_written: int = 0
    boxes_deleted: int = 0
    boxes_clipped: int = 0
    boxes_ignored_region: int = 0
    missing_images: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UA-DETRAC vehicle data in YOLO format.")
    parser.add_argument("--train-images", required=True, type=Path, help="UA-DETRAC training image root.")
    parser.add_argument("--train-annotations", required=True, type=Path, help="UA-DETRAC training XML directory or file.")
    parser.add_argument("--val-images", required=True, type=Path, help="UA-DETRAC validation/test image root.")
    parser.add_argument("--val-annotations", required=True, type=Path, help="UA-DETRAC validation/test XML directory or file.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output YOLO dataset root.")
    parser.add_argument("--train-sequences", type=Path, default=None, help="Optional sequence-name list for train.")
    parser.add_argument("--val-sequences", type=Path, default=None, help="Optional sequence-name list for val.")
    parser.add_argument("--splits-output-dir", type=Path, default=Path("data_splits"))
    parser.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy")
    parser.add_argument("--min-width", type=float, default=1.0)
    parser.add_argument("--min-height", type=float, default=1.0)
    return parser.parse_args()


def annotation_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = natural_sorted(path.rglob("*.xml"))
    if not files:
        raise FileNotFoundError(f"No UA-DETRAC XML files found under {path}")
    return files


def sequence_name(xml_path: Path, root: ET.Element) -> str:
    return root.attrib.get("name") or root.attrib.get("sequence") or xml_path.stem


def find_sequence_dir(images_root: Path, sequence: str) -> Path | None:
    direct = images_root / sequence
    if direct.is_dir():
        return direct
    matches = [path for path in images_root.rglob(sequence) if path.is_dir()]
    return natural_sorted(matches)[0] if matches else None


def image_for_frame(sequence_dir: Path, frame_number: int) -> Path | None:
    candidates = [
        sequence_dir / f"img{frame_number:05d}{suffix}"
        for suffix in (".jpg", ".jpeg", ".png", ".bmp")
    ]
    candidates.extend(sequence_dir / f"img{frame_number:06d}{suffix}" for suffix in (".jpg", ".jpeg", ".png", ".bmp"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = [path for path in sequence_dir.iterdir() if is_image(path) and digits(path.stem) == frame_number]
    return natural_sorted(matches)[0] if matches else None


def digits(value: str) -> int | None:
    items = "".join(char for char in value if char.isdigit())
    return int(items) if items else None


def parse_box(element: ET.Element) -> tuple[float, float, float, float] | None:
    keys = ("left", "top", "width", "height")
    if not all(key in element.attrib for key in keys):
        return None
    return tuple(float(element.attrib[key]) for key in keys)  # type: ignore[return-value]


def ignored_regions(root: ET.Element, image_width: int, image_height: int) -> list[BoxResult]:
    regions = []
    for box in root.findall(".//ignored_region//box"):
        values = parse_box(box)
        if values is None:
            continue
        clipped = clip_xywh_box(*values, image_width, image_height)
        if clipped is not None:
            regions.append(clipped)
    return regions


def frame_targets(frame: ET.Element) -> list[ET.Element]:
    target_list = frame.find("target_list")
    if target_list is None:
        return []
    return list(target_list.findall("target"))


def convert_xml(
    *,
    xml_path: Path,
    images_root: Path,
    output_root: Path,
    split: str,
    sequence_filter: set[str] | None,
    mode: str,
    min_width: float,
    min_height: float,
) -> tuple[list[str], Stats]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sequence = sequence_name(xml_path, root)
    if sequence_filter is not None and sequence not in sequence_filter:
        return [], Stats()
    sequence_dir = find_sequence_dir(images_root, sequence)
    if sequence_dir is None:
        raise FileNotFoundError(f"Could not locate image sequence {sequence} under {images_root}")

    split_lines: list[str] = []
    stats = Stats()
    sample_image = next((path for path in natural_sorted(sequence_dir.iterdir()) if is_image(path)), None)
    if sample_image is None:
        raise FileNotFoundError(f"No images found in sequence directory {sequence_dir}")
    sample_width, sample_height = image_size(sample_image)
    ignored = ignored_regions(root, sample_width, sample_height)

    for frame in root.findall(".//frame"):
        frame_number = int(frame.attrib.get("num", frame.attrib.get("number", "0")))
        image_path = image_for_frame(sequence_dir, frame_number)
        if image_path is None:
            stats.missing_images += 1
            continue
        width, height = image_size(image_path)
        label_lines: list[str] = []
        for target in frame_targets(frame):
            box_element = target.find("box")
            if box_element is None:
                stats.boxes_deleted += 1
                continue
            values = parse_box(box_element)
            if values is None:
                stats.boxes_deleted += 1
                continue
            if values[2] < min_width or values[3] < min_height:
                stats.boxes_deleted += 1
                continue
            clipped = clip_xywh_box(*values, width, height)
            if clipped is None:
                stats.boxes_deleted += 1
                continue
            if any(box_inside(clipped, region) for region in ignored):
                stats.boxes_ignored_region += 1
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
    return split_lines, stats


def merge_stats(items: list[Stats]) -> Stats:
    total = Stats()
    for item in items:
        for field in total.__dataclass_fields__:
            setattr(total, field, getattr(total, field) + getattr(item, field))
    return total


def convert_split(
    *,
    split: str,
    images_root: Path,
    annotations: Path,
    output_root: Path,
    sequences: Path | None,
    mode: str,
    min_width: float,
    min_height: float,
) -> tuple[list[str], Stats]:
    sequence_filter = set(read_sequence_names(sequences)) if sequences else None
    split_lines: list[str] = []
    stats_items: list[Stats] = []
    for xml_path in annotation_files(annotations):
        lines, stats = convert_xml(
            xml_path=xml_path,
            images_root=images_root,
            output_root=output_root,
            split=split,
            sequence_filter=sequence_filter,
            mode=mode,
            min_width=min_width,
            min_height=min_height,
        )
        split_lines.extend(lines)
        stats_items.append(stats)
    return natural_sorted([Path(line) for line in split_lines]), merge_stats(stats_items)  # type: ignore[return-value]


def main() -> int:
    args = parse_args()
    try:
        train_lines, train_stats = convert_split(
            split="train",
            images_root=args.train_images,
            annotations=args.train_annotations,
            output_root=args.output_root,
            sequences=args.train_sequences,
            mode=args.mode,
            min_width=args.min_width,
            min_height=args.min_height,
        )
        val_lines, val_stats = convert_split(
            split="val",
            images_root=args.val_images,
            annotations=args.val_annotations,
            output_root=args.output_root,
            sequences=args.val_sequences,
            mode=args.mode,
            min_width=args.min_width,
            min_height=args.min_height,
        )
        write_lines(args.splits_output_dir / "ua_detrac_train.txt", [posix(path) for path in train_lines])
        write_lines(args.splits_output_dir / "ua_detrac_val.txt", [posix(path) for path in val_lines])
        total = merge_stats([train_stats, val_stats])
        print(f"Images written: {total.images_written}")
        print(f"Labels written: {total.labels_written}")
        print(f"Vehicle boxes written: {total.boxes_written}")
        print(f"Deleted boxes: {total.boxes_deleted}")
        print(f"Clipped boxes: {total.boxes_clipped}")
        print(f"Ignored-region boxes skipped: {total.boxes_ignored_region}")
        print(f"Missing frame images: {total.missing_images}")
        return 0 if total.images_written else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
