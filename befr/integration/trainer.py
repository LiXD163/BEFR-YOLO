"""Custom Ultralytics trainer that applies training-only BE-Net degradation."""

from __future__ import annotations

from typing import Any

import torch

from befr.blur.blur_evolution import BlurEvolution, BlurEvolutionConfig
from befr.integration.model import BEFRDetectionModel

try:
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import RANK
    try:
        from ultralytics.utils.torch_utils import unwrap_model
    except ImportError:  # Ultralytics <= 8.3
        from ultralytics.utils.torch_utils import de_parallel as unwrap_model
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install ultralytics to use BEFRDetectionTrainer.") from exc


class BEFRDetectionTrainer(DetectionTrainer):
    """Train YOLOv8 with optional blur evolution, FR-Net, and spectral consistency."""

    def __init__(
        self,
        *args: Any,
        be_config: dict[str, Any] | None = None,
        fr_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.be_config = dict(be_config or {})
        self.fr_config = dict(fr_config or {})
        self.blur_evolution = BlurEvolution(BlurEvolutionConfig.from_dict(self.be_config))
        self._moving_average_loss = 0.0
        super().__init__(*args, **kwargs)

    def get_model(self, cfg: str | dict[str, Any] | None = None, weights: Any = None, verbose: bool = True):
        model = BEFRDetectionModel(
            cfg=cfg or "yolov8n.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            use_fr=bool(self.fr_config.get("enabled", True)),
            lambda_spec=float(self.fr_config.get("lambda_spec", 0.1)),
            spectral_distance=str(self.fr_config.get("spectral_distance", "l1")),
        )
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        use_be = bool(self.be_config.get("enabled", True))
        model = unwrap_model(self.model) if isinstance(self.model, torch.nn.Module) else None
        needs_sharp = use_be or bool(getattr(model, "lambda_spec", 0.0) > 0.0)
        if needs_sharp:
            batch["sharp_img"] = batch["img"].clone()
        if use_be:
            batch["img"], metadata = self.blur_evolution(
                batch["img"],
                epoch=int(self.epoch),
                total_epochs=int(self.epochs),
                moving_average_loss=self._moving_average_loss,
            )
            batch["be_metadata"] = metadata
        if getattr(self, "tloss", None) is not None:
            current = float(self.tloss.detach().mean().cpu())
            self._moving_average_loss = 0.9 * self._moving_average_loss + 0.1 * current
        return batch
