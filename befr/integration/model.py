"""Ultralytics DetectionModel with FR-Net immediately before Detect."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch import nn

from befr.losses.spectral_consistency import SpectralConsistencyLoss
from befr.modules.frnet import FRNet

try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError as exc:  # pragma: no cover - provides a useful import error without Ultralytics
    raise ImportError("Install ultralytics to use BEFRDetectionModel.") from exc


class BEFRDetectionModel(DetectionModel):
    """YOLOv8 detection model with optional P3/P4/P5 frequency refinement."""

    def __init__(
        self,
        cfg: str | dict[str, Any] = "yolov8n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        use_fr: bool = True,
        lambda_spec: float = 0.1,
        spectral_distance: str = "l1",
        template_sizes: tuple[tuple[int, int], ...] = ((80, 80), (40, 40), (20, 20)),
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.use_fr = bool(use_fr)
        self.lambda_spec = float(lambda_spec) if self.use_fr else 0.0
        self.fr_net = FRNet(template_sizes=template_sizes) if self.use_fr else nn.Identity()
        self.spectral_consistency = SpectralConsistencyLoss(distance=spectral_distance)
        self._capture_fr_features = False
        self.last_pre_fr: tuple[Tensor, ...] | None = None
        self.last_post_fr: tuple[Tensor, ...] | None = None
        self.last_spectral_loss = torch.tensor(0.0)
        self._fr_hook_handle = self.model[-1].register_forward_pre_hook(self._refine_detect_inputs)

    def _refine_detect_inputs(self, _module: torch.nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if not args or not isinstance(args[0], (list, tuple)):
            return None
        features = list(args[0])
        if len(features) != 3:
            return None
        refined = self.fr_net(features) if self.use_fr else features
        if self._capture_fr_features:
            self.last_pre_fr = tuple(features)
            self.last_post_fr = tuple(refined)
        return (refined, *args[1:])

    def enable_feature_capture(self, enabled: bool = True) -> None:
        self._capture_fr_features = enabled
        if not enabled:
            self.clear_feature_cache()

    def clear_feature_cache(self) -> None:
        self.last_pre_fr = None
        self.last_post_fr = None

    def loss(self, batch: dict[str, Any], preds: Any = None) -> tuple[Tensor, Tensor]:
        """Return standard YOLO loss plus optional multi-scale spectral consistency."""
        if self.lambda_spec <= 0.0 or not self.use_fr:
            return super().loss(batch, preds)
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        sharp_images = batch.get("sharp_img", batch["img"])
        originally_training = self.training
        self.enable_feature_capture(True)
        try:
            # The detached sharp target pass must not update BatchNorm running statistics.
            self.eval()
            with torch.no_grad():
                self.predict(sharp_images)
                if self.last_pre_fr is None:
                    raise RuntimeError("FR-Net feature hook did not capture sharp P3/P4/P5 features.")
                sharp_features = tuple(feature.detach() for feature in self.last_pre_fr)

            self.train(originally_training)
            predictions = self.predict(batch["img"]) if preds is None else preds
            if self.last_post_fr is None:
                raise RuntimeError("FR-Net feature hook did not capture refined blurred P3/P4/P5 features.")
            spectral_loss = self.spectral_consistency(sharp_features, self.last_post_fr)
            detection_loss, loss_items = self.criterion(predictions, batch)
            # Ultralytics detection loss is batch-scaled; match that convention here.
            spectral_term = self.lambda_spec * spectral_loss * batch["img"].shape[0]
            if detection_loss.ndim == 0:
                total_loss = detection_loss + spectral_term
            else:
                # Ultralytics 8.4 returns component losses and sums them in the trainer.
                # Add L_spec to one component so it contributes exactly once, not once per component.
                total_loss = detection_loss.clone()
                total_loss[0] = total_loss[0] + spectral_term
            self.last_spectral_loss = spectral_loss.detach()
            return total_loss, loss_items
        finally:
            self.train(originally_training)
            self.enable_feature_capture(False)
