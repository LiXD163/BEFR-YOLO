"""Evaluate one checkpoint on clear/light/medium/heavy synthetic validation sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from befr.evaluation import validate, write_metrics_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all generated motion-blur levels.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--dataset-name", default="ua_detrac")
    parser.add_argument("--output", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def ensure_yaml(level_root: Path, dataset_name: str, level: str) -> Path:
    generated_dir = Path("results/metrics/generated_data_yamls")
    generated_dir.mkdir(parents=True, exist_ok=True)
    destination = generated_dir / f"{dataset_name}_{level}.yaml"
    resolved_root = str(level_root.resolve()).replace("\\", "/")
    payload = {"path": resolved_root, "train": "images/val", "val": "images/val", "names": {0: "vehicle"}}
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def main() -> None:
    args = parse_args()
    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    rows = []
    for level in ("clear", "light", "medium", "heavy"):
        level_root = args.data_root / level
        if not (level_root / "images" / "val").is_dir():
            raise FileNotFoundError(f"Missing generated level: {level_root / 'images' / 'val'}")
        data_yaml = ensure_yaml(level_root, args.dataset_name, level)
        row = {
            "dataset": args.dataset_name,
            "level": level,
            **validate(args.weights, data_yaml, imgsz=args.imgsz, batch=args.batch, device=device, workers=args.workers),
        }
        rows.append(row)
        print(row)
    output = args.output or f"results/metrics/{args.dataset_name}_befr.csv"
    write_metrics_csv(rows, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
