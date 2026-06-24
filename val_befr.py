"""Validate one baseline or BE-FR checkpoint and save reproducible metrics."""

from __future__ import annotations

import argparse

import torch

from befr.evaluation import validate, write_metrics_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLOv8 or BE-FR checkpoint.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="validation")
    parser.add_argument("--output", default="results/metrics/validation.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    row = {"name": args.name, **validate(args.weights, args.data, imgsz=args.imgsz, batch=args.batch, device=device)}
    write_metrics_csv([row], args.output)
    print(row)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
