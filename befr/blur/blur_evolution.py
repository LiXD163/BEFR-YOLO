"""Training-only progressive Blur Evolution Network (BE-Net)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn

from .motion_psf import apply_motion_blur, nearest_odd


@dataclass
class BlurEvolutionConfig:
    """Configuration for progressive PSF-based degradation."""

    enabled: bool = True
    l_min: int = 9
    l_max: int = 25
    p_min: float = 0.2
    p_max: float = 0.8
    alpha: float = 2.0
    beta: float = 2.0
    pgn_hidden_dim: int = 32
    theta_min: float = 0.0
    theta_max: float = 180.0
    gaussian_noise_sigma: float = 0.01
    seed: int = 0

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "BlurEvolutionConfig":
        values = values or {}
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})

    def validate(self) -> None:
        if self.l_min <= 0 or self.l_max < self.l_min:
            raise ValueError("Require 0 < l_min <= l_max.")
        if (self.l_min if self.l_min % 2 else self.l_min + 1) > (
            self.l_max if self.l_max % 2 else self.l_max - 1
        ):
            raise ValueError("The blur-length range must contain at least one odd integer.")
        if not 0.0 <= self.p_min <= self.p_max <= 1.0:
            raise ValueError("Require 0 <= p_min <= p_max <= 1.")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative.")
        if self.theta_max <= self.theta_min:
            raise ValueError("theta_max must exceed theta_min.")
        if self.gaussian_noise_sigma < 0:
            raise ValueError("gaussian_noise_sigma must be non-negative.")


class ParameterGeneratorNetwork(nn.Module):
    """Two-layer PGN mapping normalized training state to blur factors in [0, 1]."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2), nn.Sigmoid())

    def forward(self, normalized_epoch: Tensor, normalized_loss: Tensor) -> Tensor:
        state = torch.stack((normalized_epoch, normalized_loss), dim=-1)
        return self.layers(state)


class BlurEvolution:
    """Generate curriculum-controlled blurred samples outside the inference model.

    PSF construction, odd rounding, Bernoulli injection, and convolution are
    intentionally non-differentiable training data-degradation operations.
    """

    def __init__(self, config: BlurEvolutionConfig | dict[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, BlurEvolutionConfig) else BlurEvolutionConfig.from_dict(config)
        self.config.validate()
        with torch.random.fork_rng():
            torch.manual_seed(self.config.seed)
            self.pgn = ParameterGeneratorNetwork(self.config.pgn_hidden_dim).eval()
        self._generators: dict[str, torch.Generator] = {}

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.config.seed + len(self._generators))
            self._generators[key] = generator
        return self._generators[key]

    def schedule(self, epoch: int, total_epochs: int) -> tuple[float, float, float]:
        """Return normalized progress, current maximum length, and blur probability."""
        denominator = max(total_epochs - 1, 1)
        progress = min(max(epoch / denominator, 0.0), 1.0)
        current_l_max = self.config.l_min + (self.config.l_max - self.config.l_min) * progress**self.config.alpha
        probability = self.config.p_min + (self.config.p_max - self.config.p_min) * progress**self.config.beta
        return progress, current_l_max, probability

    @torch.no_grad()
    def __call__(
        self,
        images: Tensor,
        *,
        epoch: int,
        total_epochs: int,
        moving_average_loss: float = 0.0,
    ) -> tuple[Tensor, list[dict[str, float | int | bool]]]:
        """Apply per-sample Bernoulli PSF blur and optional Gaussian sensor noise."""
        if not self.config.enabled:
            return images, []
        if images.ndim != 4:
            raise ValueError(f"Expected BCHW images, got shape {tuple(images.shape)}")

        progress, current_l_max, probability = self.schedule(epoch, total_epochs)
        batch = images.shape[0]
        device = images.device
        generator = self._generator(device)
        self.pgn.to(device)
        epoch_state = torch.full((batch,), progress, device=device, dtype=torch.float32)
        loss_value = abs(float(moving_average_loss))
        normalized_loss = loss_value / (1.0 + loss_value)
        loss_state = torch.full((batch,), normalized_loss, device=device, dtype=torch.float32)
        factors = self.pgn(epoch_state, loss_state)
        inject = torch.rand(batch, device=device, generator=generator) < probability

        output = images.clone()
        metadata: list[dict[str, float | int | bool]] = []
        minimum_odd = self.config.l_min if self.config.l_min % 2 else self.config.l_min + 1
        current_odd_max = max(minimum_odd, int(math.floor(current_l_max)))
        if current_odd_max % 2 == 0:
            current_odd_max -= 1
        current_odd_max = max(minimum_odd, current_odd_max)
        for index in range(batch):
            length_value = self.config.l_min + float(factors[index, 0]) * (current_l_max - self.config.l_min)
            length = nearest_odd(length_value, minimum_odd, current_odd_max)
            theta = self.config.theta_min + float(factors[index, 1]) * (
                self.config.theta_max - self.config.theta_min
            )
            applied = bool(inject[index])
            if applied:
                degraded = apply_motion_blur(output[index], length, theta)
                if self.config.gaussian_noise_sigma > 0:
                    noise = torch.randn(
                        degraded.shape,
                        device=device,
                        dtype=degraded.dtype,
                        generator=generator,
                    )
                    degraded = degraded + noise * self.config.gaussian_noise_sigma
                output[index] = degraded.clamp(0.0, 1.0)
            metadata.append(
                {
                    "applied": applied,
                    "length": length,
                    "theta": theta,
                    "probability": probability,
                    "current_l_max": current_l_max,
                }
            )
        return output, metadata
