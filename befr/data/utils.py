"""Shared utilities for reproducible dataset preparation."""

from __future__ import annotations

import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class BoxResult:
    x1: float
    y1: float
    x2: float
    y2: float
    clipped: bool


@dataclass(frozen=True)
class LabelValidation:
    exists: bool
    valid_count: int
    invalid_count: int
    empty: bool


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def natural_key(value: str | Path) -> tuple[object, ...]:
    text = str(value.name if isinstance(value, Path) else value).casefold()
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


def natural_sorted(paths: Iterable[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: (natural_key(path.name), natural_key(path.as_posix())))


def find_images(root: Path) -> list[Path]:
    return natural_sorted(path for path in root.rglob("*") if is_image(path))


def posix(path: Path) -> str:
    return path.as_posix()


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to read image sizes for dataset conversion.") from exc
    with Image.open(path) as image:
        width, height = image.size
    return int(width), int(height)


def clip_xywh_box(left: float, top: float, width: float, height: float, image_width: int, image_height: int) -> BoxResult | None:
    x1 = max(0.0, min(float(left), float(image_width)))
    y1 = max(0.0, min(float(top), float(image_height)))
    x2 = max(0.0, min(float(left + width), float(image_width)))
    y2 = max(0.0, min(float(top + height), float(image_height)))
    if x2 <= x1 or y2 <= y1:
        return None
    clipped = x1 != float(left) or y1 != float(top) or x2 != float(left + width) or y2 != float(top + height)
    return BoxResult(x1=x1, y1=y1, x2=x2, y2=y2, clipped=clipped)


def box_inside(inner: BoxResult, outer: BoxResult) -> bool:
    return inner.x1 >= outer.x1 and inner.y1 >= outer.y1 and inner.x2 <= outer.x2 and inner.y2 <= outer.y2


def yolo_line(class_id: int, box: BoxResult, image_width: int, image_height: int) -> str:
    box_width = box.x2 - box.x1
    box_height = box.y2 - box.y1
    center_x = (box.x1 + box.x2) / (2.0 * image_width)
    center_y = (box.y1 + box.y2) / (2.0 * image_height)
    return f"{class_id} {center_x:.8f} {center_y:.8f} {box_width / image_width:.8f} {box_height / image_height:.8f}"


def validate_yolo_label(path: Path) -> LabelValidation:
    if not path.exists():
        return LabelValidation(False, 0, 0, False)
    valid = 0
    invalid = 0
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            invalid += 1
            continue
        try:
            int(float(parts[0]))
            x, y, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            invalid += 1
            continue
        if width <= 0.0 or height <= 0.0 or not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            invalid += 1
            continue
        valid += 1
    return LabelValidation(True, valid, invalid, not lines)


def transfer_file(source: Path, destination: Path, mode: str = "copy") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        os.symlink(source, destination)
    else:
        raise ValueError(f"Unsupported transfer mode: {mode}")


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(lines)
    path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_token_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    return [token for token in re.split(r"[\s,;，；、]+", text) if token]


def read_sequence_names(path: Path) -> list[str]:
    names = []
    for token in read_token_file(path):
        cleaned = token.strip().strip("\"'")
        if cleaned:
            names.append(cleaned)
    return names


def parse_integer_indices(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8-sig")
    return [int(token) for token in re.findall(r"\d+", text)]


def infer_index_base(indices: list[int], count: int) -> tuple[int, str]:
    if not indices:
        raise ValueError("Index file contains no integer indices.")
    if 0 in indices:
        base = 0
    elif all(1 <= index <= count for index in indices):
        base = 1
    else:
        base = 0
    valid_zero = all(0 <= index < count for index in indices)
    valid_one = all(1 <= index <= count for index in indices)
    if valid_zero and valid_one and 0 not in indices:
        note = "both 0-based and 1-based interpretations are in range; 1-based was used"
    else:
        note = f"{base}-based indices were inferred"
    return base, note


def image_label_relative_path(relative_image_path: Path) -> Path:
    parts = list(relative_image_path.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return Path("labels") / relative_image_path.with_suffix(".txt")
