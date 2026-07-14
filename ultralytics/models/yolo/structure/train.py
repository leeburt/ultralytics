# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.modules import StructureHeatmap
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import StructureModel
from ultralytics.utils import DEFAULT_CFG, RANK


def _unwrap(model):
    """Return the underlying nn.Module, unwrapping DDP if needed."""
    return model.module if hasattr(model, "module") else model


class StructureTrainer(DetectionTrainer):
    """Trainer for structure models."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize structure trainer."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "structure"
        super().__init__(cfg, overrides, _callbacks)
        self.epoch = 0

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> StructureModel:
        """Get structure model with optional pretrained weights."""
        model = StructureModel(
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

    def get_validator(self):
        """Return validator for structure models."""
        unwrapped = _unwrap(self.model)
        self.loss_names = (
            "comp_hm_loss",
            "comp_off_loss",
            "port_hm_loss",
            "port_off_loss",
            "dir_loss",
            "rho_loss",
            "endpoint_loss",
        ) if isinstance(unwrapped.model[-1], StructureHeatmap) else ("loss",)
        return yolo.structure.StructureValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """Load dataset metadata and require keypoint shape."""
        data = super().get_dataset()
        if "kpt_shape" not in data:
            data["kpt_shape"] = [25, 3]  # Default: 1 component + 24 ports
        return data

    def _do_train(self, world_size=1):
        """Override to pass epoch to loss criterion."""
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            # Update loss epoch for ramp-up
            unwrapped = _unwrap(self.model)
            if hasattr(unwrapped, "criterion") and hasattr(unwrapped.criterion, "set_epoch"):
                unwrapped.criterion.set_epoch(epoch)
            # Continue with normal training
            super()._do_train(world_size)
