# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.modules import KeypointHeatmap
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import KeypointModel
from ultralytics.utils import DEFAULT_CFG, RANK


def _unwrap(model):
    """Return the underlying nn.Module, unwrapping DDP if needed."""
    return model.module if hasattr(model, "module") else model


class KeypointTrainer(DetectionTrainer):
    """Trainer for keypoint-only models that do not predict boxes or classes."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize keypoint-only trainer."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "keypoint"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> KeypointModel:
        """Get keypoint-only model with optional pretrained weights."""
        model = KeypointModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Attach dataset metadata to the model."""
        super().set_model_attributes()
        unwrapped = _unwrap(self.model)
        unwrapped.kpt_shape = self.data["kpt_shape"]
        unwrapped.kpt_names = self.data.get("kpt_names") or {
            i: [str(j) for j in range(unwrapped.kpt_shape[0])] for i in range(unwrapped.nc)
        }
        if isinstance(unwrapped.model[-1], KeypointHeatmap):
            head = unwrapped.model[-1]
            if self.args.hm_radius_add or not getattr(head, "hm_radius_add", 0):
                head.hm_radius_add = int(self.args.hm_radius_add)
            if self.args.hm_min_radius or not getattr(head, "hm_min_radius", 0):
                head.hm_min_radius = int(self.args.hm_min_radius)

    def get_validator(self):
        """Return validator for keypoint-only models."""
        unwrapped = _unwrap(self.model)
        self.loss_names = ("hm_loss", "off_loss") if isinstance(unwrapped.model[-1], KeypointHeatmap) else ("kpt_loss", "kobj_loss")
        return yolo.keypoint.KeypointValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """Load dataset metadata and require keypoint shape."""
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}.")
        return data
