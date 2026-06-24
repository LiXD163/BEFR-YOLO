"""Lightweight feature-level Frequency Refinement Network (FR-Net)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FRBlock(nn.Module):
    """Refine a detector feature map using a learnable real spectral template.

    This is feature-level spectral refinement for detection, not image restoration.
    A channel-shared template keeps each scale block lightweight.
    """

    def __init__(self, template_size: tuple[int, int], gamma_init: float = 0.1) -> None:
        super().__init__()
        height, width = template_size
        self.spectral_template = nn.Parameter(torch.empty(1, 1, height, width // 2 + 1))
        nn.init.normal_(self.spectral_template, mean=0.0, std=0.02)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _mask(self, target_size: tuple[int, int]) -> Tensor:
        target_h, target_w = target_size
        target_frequency_width = target_w // 2 + 1
        template = self.spectral_template
        if template.shape[-2:] != (target_h, target_frequency_width):
            template = F.interpolate(
                template,
                size=(target_h, target_frequency_width),
                mode="bilinear",
                align_corners=False,
            )
        # Positive bounded modulation avoids unstable complex amplification.
        return 2.0 * torch.sigmoid(template)

    def forward(self, x: Tensor) -> Tensor:
        original_dtype = x.dtype
        work = x.float()
        spectrum = torch.fft.rfft2(work, norm="ortho")
        refined_spectrum = spectrum * self._mask(x.shape[-2:])
        refined = torch.fft.irfft2(refined_spectrum, s=x.shape[-2:], norm="ortho")
        return x + self.gamma.to(original_dtype) * refined.to(original_dtype)


class FRNet(nn.Module):
    """Apply independent FR blocks to YOLO P3, P4, and P5 neck features."""

    def __init__(
        self,
        template_sizes: Sequence[tuple[int, int]] = ((80, 80), (40, 40), (20, 20)),
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        if len(template_sizes) != 3:
            raise ValueError("FRNet requires exactly three template sizes for P3, P4, and P5.")
        self.level_names = ("P3", "P4", "P5")
        self.blocks = nn.ModuleList([FRBlock(size, gamma_init=gamma_init) for size in template_sizes])

    def forward(self, features: Sequence[Tensor]) -> list[Tensor]:
        if len(features) != 3:
            raise ValueError(f"FRNet expected three feature maps, received {len(features)}.")
        return [block(feature) for block, feature in zip(self.blocks, features)]
