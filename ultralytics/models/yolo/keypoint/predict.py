# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
from .utils import radius_point_nms


class KeypointPredictor(BasePredictor):
    """Predictor for keypoint-only models."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize keypoint-only predictor."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "keypoint"

    def postprocess(self, preds, img, orig_imgs):
        """Convert dense keypoint candidates to Results objects."""
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        nkpt = self.model.kpt_shape[0]
        max_det = self.args.max_det
        point_nms_radius = 8.0
        for i, (pred, orig_img) in enumerate(zip(preds, orig_imgs)):
            img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
            conf = pred[:, 2]
            keep = conf >= self.args.conf
            pred = pred[keep]
            if pred.shape[0]:
                pred = radius_point_nms(pred, point_nms_radius, max_det)
                kpts = pred[:, :3].view(-1, 1, 3)
                kpts = ops.scale_coords(img.shape[2:], kpts, orig_img.shape)
            else:
                kpts = torch.zeros((0, nkpt, 3), device=pred.device)
            results.append(Results(orig_img, path=img_path, names=self.model.names, keypoints=kpts))
        return results
