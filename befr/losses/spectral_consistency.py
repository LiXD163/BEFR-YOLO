"""Multi-scale amplitude-spectrum consistency loss."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def normalized_amplitude_spectrum(feature: Tensor, eps: float = 1e-6) -> Tensor:
    """Return a scale-normalized log amplitude spectrum for a BCHW feature map."""
    amplitude = torch.fft.rfft2(feature.float(), norm="ortho").abs()
    amplitude = torch.log1p(amplitude)
    normalizer = amplitude.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return amplitude / normalizer


class SpectralConsistencyLoss(nn.Module):
    """Compare sharp-domain and refined blurred-domain spectra across P3/P4/P5."""

    def __init__(self, distance: str = "l1") -> None:
        super().__init__()
        if distance not in {"l1", "l2"}:
            raise ValueError("distance must be 'l1' or 'l2'.")
        self.distance = distance

    def forward(self, sharp_features: Sequence[Tensor], refined_blurred_features: Sequence[Tensor]) -> Tensor:
        if len(sharp_features) != len(refined_blurred_features):
            raise ValueError("Sharp and refined feature lists must have equal length.")
        losses = []
        for sharp, refined in zip(sharp_features, refined_blurred_features):
            sharp_spectrum = normalized_amplitude_spectrum(sharp.detach())
            refined_spectrum = normalized_amplitude_spectrum(refined)
            if self.distance == "l1":
                losses.append(F.l1_loss(refined_spectrum, sharp_spectrum))
            else:
                losses.append(F.mse_loss(refined_spectrum, sharp_spectrum))
        if not losses:
            raise ValueError("At least one feature level is required.")
        return torch.stack(losses).mean()
