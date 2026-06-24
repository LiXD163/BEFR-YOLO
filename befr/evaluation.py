"""Evaluation helpers shared by validation scripts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

# Import custom classes before loading a custom BE-FR checkpoint.
from befr.integration.model import BEFRDetectionModel  # noqa: F401


def load_yolo(weights: str | Path) -> YOLO:
    return YOLO(str(weights), task="detect")


def validate(
    weights: str | Path,
    data: str | Path,
    *,
    imgsz: int = 640,
    batch: int = 1,
    device: str = "cpu",
    workers: int = 4,
) -> dict[str, Any]:
    model = load_yolo(weights)
    raw_model = model.model
    params = int(sum(parameter.numel() for parameter in raw_model.parameters()))
    flops = float(get_flops(raw_model, imgsz=imgsz))
    metrics = model.val(data=str(data), imgsz=imgsz, batch=batch, device=device, workers=workers, verbose=False)
    latency = float(metrics.speed.get("inference", 0.0))
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "fps": 1000.0 / latency if latency > 0 else 0.0,
        "latency_ms": latency,
        "params": params,
        "flops_g": flops,
    }


def write_metrics_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No metric rows to write.")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
