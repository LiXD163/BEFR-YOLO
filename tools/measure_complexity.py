"""Measure model parameters, approximate FLOPs, raw inference FPS, and latency."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics.utils.torch_utils import get_flops, select_device

from befr.evaluation import load_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure YOLOv8/BE-FR inference complexity.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=300)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    device = select_device(args.device or "", verbose=False)
    model = load_yolo(args.weights).model.to(device).eval()
    image = torch.rand(args.batch, 3, args.imgsz, args.imgsz, device=device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.repeat):
            model(image)
        synchronize(device)
    elapsed = time.perf_counter() - start
    images_seen = args.repeat * args.batch
    latency_ms = elapsed * 1000.0 / images_seen
    result = {
        "params": int(sum(parameter.numel() for parameter in model.parameters())),
        "flops_g": float(get_flops(model, imgsz=args.imgsz)),
        "fps": images_seen / elapsed,
        "latency_ms": latency_ms,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": str(device),
    }
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result))
            writer.writeheader()
            writer.writerow(result)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
