from __future__ import annotations

import pytest
import torch

pytest.importorskip("ultralytics")

from befr.integration.model import BEFRDetectionModel


def test_yolov8n_fr_hook_refines_three_scales() -> None:
    model = BEFRDetectionModel("yolov8n.yaml", verbose=False, use_fr=True, lambda_spec=0.1).eval()
    model.enable_feature_capture(True)
    with torch.inference_mode():
        model(torch.rand(1, 3, 128, 128))
    assert model.last_pre_fr is not None
    assert model.last_post_fr is not None
    assert len(model.last_pre_fr) == len(model.last_post_fr) == 3
    assert [x.shape for x in model.last_pre_fr] == [x.shape for x in model.last_post_fr]
