from __future__ import annotations

import torch

from befr.blur.blur_evolution import BlurEvolution
from befr.blur.motion_psf import apply_motion_blur, linear_motion_psf, nearest_odd
from befr.losses.spectral_consistency import SpectralConsistencyLoss
from befr.modules.frnet import FRNet


def test_motion_psf_is_normalized_and_odd() -> None:
    assert nearest_odd(10.2) == 11
    kernel = linear_motion_psf(10.2, 37.0)
    assert kernel.shape == (11, 11)
    assert torch.isclose(kernel.sum(), torch.tensor(1.0), atol=1e-6)
    image = torch.rand(3, 32, 32)
    assert apply_motion_blur(image, 9, 45).shape == image.shape


def test_blur_evolution_is_seedable() -> None:
    image = torch.rand(2, 3, 32, 32)
    first = BlurEvolution({"seed": 123, "p_min": 1.0, "p_max": 1.0})
    second = BlurEvolution({"seed": 123, "p_min": 1.0, "p_max": 1.0})
    output_a, metadata_a = first(image, epoch=5, total_epochs=10, moving_average_loss=1.0)
    output_b, metadata_b = second(image, epoch=5, total_epochs=10, moving_average_loss=1.0)
    assert torch.allclose(output_a, output_b)
    assert metadata_a == metadata_b


def test_frnet_and_spectral_loss_have_gradients() -> None:
    features = [
        torch.rand(1, 8, 20, 20),
        torch.rand(1, 16, 10, 10),
        torch.rand(1, 32, 5, 5),
    ]
    module = FRNet()
    refined = module(features)
    loss = SpectralConsistencyLoss()(features, refined)
    loss.backward()
    assert loss.item() >= 0
    assert all(block.spectral_template.grad is not None for block in module.blocks)
