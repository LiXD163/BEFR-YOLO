"""Generate a Grad-CAM-like qualitative heatmap for a selected YOLO feature layer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils.torch_utils import select_device

from befr.evaluation import load_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM-like YOLO heatmap.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=-2, help="Index in the Ultralytics model.model sequence.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("results/visualizations/gradcam.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device or "", verbose=False)
    yolo = load_yolo(args.weights)
    model = yolo.model.to(device).eval()
    target_layer = model.model[args.layer]
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Selected layer must output one tensor; choose a neck/backbone layer.")
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

    handle = target_layer.register_forward_hook(forward_hook)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read {args.image}")
    resized = cv2.resize(image, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).float().to(device) / 255.0
    tensor.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    output = model(tensor)
    predictions = output[0] if isinstance(output, tuple) else output
    if not isinstance(predictions, torch.Tensor) or predictions.ndim != 3:
        raise RuntimeError("Unexpected detection output; Grad-CAM expects a Bx(4+classes)xN tensor.")
    predictions[:, 4:].max().backward()
    handle.remove()

    activation = captured["activation"]
    gradient = captured["gradient"]
    weights = gradient.mean(dim=(-2, -1), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(args.imgsz, args.imgsz), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-6)
    heatmap = cv2.applyColorMap((cam.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(resized, 0.55, heatmap, 0.45, 0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
