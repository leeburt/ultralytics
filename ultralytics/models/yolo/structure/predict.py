# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
from .utils import radius_point_nms


class StructurePredictor(BasePredictor):
    """Predictor for structure models."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize structure predictor."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "structure"

    def postprocess(self, preds, img, orig_imgs):
        """Convert structure predictions to Results objects."""
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        component_nms_radius = 16.0
        port_nms_radius = 8.0
        max_det = self.args.max_det

        for b, pred_dict in enumerate(preds):
            orig_img = orig_imgs[b]
            img_path = self.batch[0][b] if isinstance(self.batch[0], list) else self.batch[0]

            # Process components
            components = pred_dict["components"]
            comp_conf = components[:, 2]
            comp_keep = comp_conf >= self.args.conf
            components = components[comp_keep]
            if components.shape[0]:
                components = radius_point_nms(components, component_nms_radius, max_det)
                comp_xy = components[:, :3].unsqueeze(1)
                comp_xy = ops.scale_coords(img.shape[2:], comp_xy, orig_img.shape)
            else:
                comp_xy = torch.zeros((0, 1, 3), device=components.device)

            # Process ports
            ports = pred_dict["ports"]
            port_conf = ports[:, 2]
            port_keep = port_conf >= self.args.conf
            ports = ports[port_keep]
            if ports.shape[0]:
                ports = radius_point_nms(ports, port_nms_radius, max_det)
                port_xy = ports[:, :3].unsqueeze(1)
                port_xy = ops.scale_coords(img.shape[2:], port_xy, orig_img.shape)
                pred_comp_xy = ports[:, 3:5].unsqueeze(1)
                pred_comp_xy = ops.scale_coords(img.shape[2:], pred_comp_xy, orig_img.shape)
            else:
                port_xy = torch.zeros((0, 1, 3), device=ports.device)
                pred_comp_xy = torch.zeros((0, 1, 2), device=ports.device)

            # Store in results
            results.append(Results(
                orig_img,
                path=img_path,
                names=self.model.names,
                keypoints={
                    "components": comp_xy,
                    "ports": port_xy,
                    "pred_components": pred_comp_xy,
                }
            ))
        return results
