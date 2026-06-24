"""Export deterministic train/val manifests from a YOLO dataset YAML."""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    split: str
    image_path: Path
    relative_image_path: str
    relative_label_path: str
    label_exists: bool
    object_count: int
    invalid_label: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export train/val image manifests from an Ultralytics YOLO YAML.")
    parser.add_argument("--data-yaml", required=True, type=Path, help="Dataset YAML such as data/ua_detrac_vehicle.yaml.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for split TXT files and report CSV.")
    parser.add_argument("--dataset-name", default=None, help="Manifest prefix. Defaults to YAML stem without _vehicle.")
    parser.add_argument("--fail-on-missing-label", action="store_true")
    return parser.parse_args()


def simple_yaml_load(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not raw_line.startswith((" ", "\t")) and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            current_key = key
            if value == "":
                payload[key] = {}
            else:
                payload[key] = parse_yaml_scalar(value)
        elif current_key and isinstance(payload.get(current_key), dict) and ":" in line:
            key, value = line.split(":", 1)
            payload[current_key][parse_yaml_scalar(key.strip())] = parse_yaml_scalar(value.strip())
    return payload


def parse_yaml_scalar(value: str) -> Any:
    value = value.strip().strip("\"'")
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    return value


def load_dataset_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return simple_yaml_load(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def default_dataset_name(data_yaml: Path) -> str:
    stem = data_yaml.stem
    for suffix in ("_vehicle", "-vehicle"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def resolve_dataset_root(data_yaml: Path, yaml_data: dict[str, Any]) -> Path:
    raw = yaml_data.get("path")
    if raw is None:
        return data_yaml.parent.resolve()
    root = Path(str(raw)).expanduser()
    if root.is_absolute():
        return root.resolve()
    cwd_candidate = (Path.cwd() / root).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    yaml_candidate = (data_yaml.parent / root).resolve()
    if yaml_candidate.exists():
        return yaml_candidate
    return cwd_candidate


def resolve_split_sources(data_yaml: Path, dataset_root: Path, value: Any) -> list[Path]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    resolved = []
    for item in values:
        path = Path(str(item)).expanduser()
        if path.is_absolute():
            resolved.append(path.resolve())
            continue
        root_candidate = (dataset_root / path).resolve()
        if root_candidate.exists():
            resolved.append(root_candidate)
        else:
            yaml_candidate = (data_yaml.parent / path).resolve()
            resolved.append(yaml_candidate if yaml_candidate.exists() else root_candidate)
    return resolved


def iter_images_from_source(source: Path) -> list[Path]:
    if source.is_file():
        return [resolve_listed_image(source, line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"Split source does not exist: {source}")


def resolve_listed_image(list_file: Path, line: str) -> Path:
    path = Path(line.strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (list_file.parent / path).resolve()


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def label_relative_path(relative_image_path: str) -> str:
    rel = Path(relative_image_path)
    parts = list(rel.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt").as_posix()
    return (Path("labels") / rel.with_suffix(".txt")).as_posix()


def count_valid_objects(label_path: Path) -> tuple[int, bool]:
    if not label_path.exists():
        return 0, False
    invalid = False
    count = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            invalid = True
            continue
        try:
            _class_id = int(float(parts[0]))
            _x, _y, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            invalid = True
            continue
        if width <= 0.0 or height <= 0.0:
            invalid = True
            continue
        count += 1
    return count, invalid


def label_path_from_relative(dataset_root: Path, relative_image_path: str) -> Path:
    return dataset_root / label_relative_path(relative_image_path)


def build_records(data_yaml: Path) -> tuple[str, dict[str, list[ImageRecord]], dict[str, int]]:
    data_yaml = data_yaml.resolve()
    yaml_data = load_dataset_yaml(data_yaml)
    dataset_root = resolve_dataset_root(data_yaml, yaml_data)
    records: dict[str, list[ImageRecord]] = {"train": [], "val": []}
    summary = {
        "train_images": 0,
        "val_images": 0,
        "missing_labels": 0,
        "invalid_label_files": 0,
        "duplicates": 0,
    }
    seen: set[str] = set()
    for split in ("train", "val"):
        split_sources = resolve_split_sources(data_yaml, dataset_root, yaml_data.get(split))
        if not split_sources:
            raise ValueError(f"Dataset YAML does not define {split}: {data_yaml}")
        images: list[Path] = []
        for source in split_sources:
            images.extend(iter_images_from_source(source))
        for image_path in sorted(set(path.resolve() for path in images)):
            relative_image = relative_to_root(image_path, dataset_root)
            key = f"{split}:{relative_image}"
            if key in seen:
                summary["duplicates"] += 1
                continue
            seen.add(key)
            label_path = label_path_from_relative(dataset_root, relative_image)
            object_count, invalid = count_valid_objects(label_path)
            label_exists = label_path.exists()
            if not label_exists:
                summary["missing_labels"] += 1
            if invalid:
                summary["invalid_label_files"] += 1
            records[split].append(
                ImageRecord(
                    split=split,
                    image_path=image_path,
                    relative_image_path=relative_image,
                    relative_label_path=label_relative_path(relative_image),
                    label_exists=label_exists,
                    object_count=object_count,
                    invalid_label=invalid,
                )
            )
        summary[f"{split}_images"] = len(records[split])
    return str(dataset_root), records, summary


def write_outputs(dataset_name: str, output_dir: Path, dataset_root: str, records: dict[str, list[ImageRecord]], summary: dict[str, int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        destination = output_dir / f"{dataset_name}_{split}.txt"
        destination.write_text("\n".join(row.relative_image_path for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    report = output_dir / f"{dataset_name}_manifest_report.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset_root",
            "split",
            "relative_image_path",
            "relative_label_path",
            "label_exists",
            "object_count",
            "invalid_label",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "val"):
            for row in records[split]:
                writer.writerow(
                    {
                        "dataset_root": dataset_root,
                        "split": split,
                        "relative_image_path": row.relative_image_path,
                        "relative_label_path": row.relative_label_path,
                        "label_exists": row.label_exists,
                        "object_count": row.object_count,
                        "invalid_label": row.invalid_label,
                    }
                )
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"report: {report}")


def main() -> int:
    args = parse_args()
    dataset_name = args.dataset_name or default_dataset_name(args.data_yaml)
    try:
        dataset_root, records, summary = build_records(args.data_yaml)
        write_outputs(dataset_name, args.output_dir, dataset_root, records, summary)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if summary["duplicates"] or summary["invalid_label_files"]:
        return 1
    if args.fail_on_missing_label and summary["missing_labels"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
