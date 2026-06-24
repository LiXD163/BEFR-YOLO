"""Visualize P3/P4/P5 amplitude spectra before and after FR-Net."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics.utils.torch_utils import select_device

from befr.evaluation import load_yolo


LEVEL_INDEX = {"P3": 0, "P4": 1, "P5": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize feature spectra around FR-Net.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--level", choices=list(LEVEL_INDEX), default="P3")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("results/visualizations/fft_response.png"))
    return parser.parse_args()


def centered_spectrum(feature: torch.Tensor) -> torch.Tensor:
    amplitude = torch.fft.fft2(feature.float(), norm="ortho").abs().mean(dim=1)[0]
    return torch.log1p(torch.fft.fftshift(amplitude)).detach().cpu()


def main() -> None:
    args = parse_args()
    device = select_device(args.device or "", verbose=False)
    yolo = load_yolo(args.weights)
    model = yolo.model.to(device).eval()
    if not hasattr(model, "enable_feature_capture"):
        raise TypeError("This checkpoint does not contain BEFRDetectionModel feature capture.")
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read {args.image}")
    resized = cv2.resize(image, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).float().to(device) / 255.0
    model.enable_feature_capture(True)
    with torch.inference_mode():
        model(tensor)
    index = LEVEL_INDEX[args.level]
    before = centered_spectrum(model.last_pre_fr[index])
    after = centered_spectrum(model.last_post_fr[index])
    model.enable_feature_capture(False)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input")
    axes[1].imshow(before, cmap="magma")
    axes[1].set_title(f"{args.level} before FR-Net")
    axes[2].imshow(after, cmap="magma")
    axes[2].set_title(f"{args.level} after FR-Net")
    for axis in axes:
        axis.axis("off")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
