"""Run dependency imports and dummy BE-FR forward checks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
import torch
import ultralytics
import yaml

from befr.blur.blur_evolution import BlurEvolution
from befr.integration.model import BEFRDetectionModel
from befr.losses.spectral_consistency import SpectralConsistencyLoss
from befr.modules.frnet import FRNet


def main() -> None:
    print(f"torch={torch.__version__}, ultralytics={ultralytics.__version__}, cv2={cv2.__version__}")
    print(f"yaml={yaml.__version__}, matplotlib={matplotlib.__version__}")
    image = torch.rand(2, 3, 64, 64)
    blurred, metadata = BlurEvolution({"seed": 0})(image, epoch=99, total_epochs=100, moving_average_loss=1.0)
    print(f"BE-Net: {tuple(blurred.shape)}, first_sample={metadata[0]}")

    features = [
        torch.rand(1, 64, 80, 80),
        torch.rand(1, 128, 40, 40),
        torch.rand(1, 256, 20, 20),
    ]
    refined = FRNet()(features)
    spectral = SpectralConsistencyLoss()(features, refined)
    print(f"FR-Net: {[tuple(feature.shape) for feature in refined]}, spectral_loss={spectral.item():.6f}")

    model = BEFRDetectionModel("yolov8n.yaml", verbose=False).eval()
    model.enable_feature_capture(True)
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 128, 128))
    print(f"YOLOv8n BE-FR output: {tuple(output[0].shape)}")
    print(f"Captured scales: {[tuple(feature.shape) for feature in model.last_post_fr]}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
