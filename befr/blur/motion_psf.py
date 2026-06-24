"""Linear point-spread-function (PSF) motion blur implemented with PyTorch."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def nearest_odd(value: float, minimum: int = 1, maximum: int | None = None) -> int:
    """Round a positive value to the nearest odd integer within optional bounds."""
    min_odd = minimum if minimum % 2 else minimum + 1
    max_odd = None if maximum is None else (maximum if maximum % 2 else maximum - 1)
    if max_odd is not None and max_odd < min_odd:
        raise ValueError(f"Range [{minimum}, {maximum}] contains no odd integer.")
    rounded = int(round(float(value)))
    if rounded % 2 == 0:
        lower, upper = rounded - 1, rounded + 1
        rounded = lower if abs(value - lower) <= abs(value - upper) else upper
    rounded = max(min_odd, rounded)
    if max_odd is not None:
        rounded = min(max_odd, rounded)
    return max(1, rounded)


def linear_motion_psf(
    length: float,
    theta_degrees: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Construct a normalized rasterized linear-motion PSF.

    The softly rasterized line avoids empty kernels at oblique angles while preserving
    the physical interpretation of integration along a straight motion trajectory.
    """
    size = nearest_odd(length)
    radius = size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    theta = math.radians(float(theta_degrees) % 180.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    along = xx * cos_t + yy * sin_t
    perpendicular = -xx * sin_t + yy * cos_t
    support = (along.abs() <= (size - 1) / 2 + 0.5).to(dtype)
    kernel = torch.exp(-0.5 * (perpendicular / 0.45).square()) * support
    return kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)


@torch.no_grad()
def apply_motion_blur(image: Tensor, length: float, theta_degrees: float) -> Tensor:
    """Apply one linear PSF to a CHW or BCHW normalized image tensor."""
    squeeze = image.ndim == 3
    if squeeze:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW image tensor, got shape {tuple(image.shape)}")

    batch, channels, height, width = image.shape
    kernel = linear_motion_psf(length, theta_degrees, device=image.device, dtype=torch.float32)
    pad = kernel.shape[-1] // 2
    padding_mode = "reflect" if height > pad and width > pad else "replicate"
    work = image.float()
    work = F.pad(work, (pad, pad, pad, pad), mode=padding_mode)
    weight = kernel.view(1, 1, *kernel.shape).repeat(channels, 1, 1, 1)
    blurred = F.conv2d(work, weight, groups=channels)
    blurred = blurred.to(image.dtype)
    return blurred.squeeze(0) if squeeze else blurred
