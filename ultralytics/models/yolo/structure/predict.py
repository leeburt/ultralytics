# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
from .utils import associate_ports_to_components, radius_point_nms, scale_link_geometry


class StructureResults(Results):
    """Inference result container for component, port, and link predictions."""

    def __init__(self, orig_img, path, names, structure, speed=None):
        super().__init__(orig_img=orig_img, path=path, names=names, speed=speed)
        self.structure = structure

    def __len__(self):
        return int(self.structure["components"].shape[0] + self.structure["ports"].shape[0])

    def _apply(self, fn: str, *args, **kwargs):
        if fn == "numpy":
            structure = {key: value.detach().cpu().numpy() for key, value in self.structure.items()}
        else:
            structure = {key: getattr(value, fn)(*args, **kwargs) for key, value in self.structure.items()}
        result = StructureResults(self.orig_img, self.path, self.names, structure, self.speed)
        result.save_dir = self.save_dir
        return result

    def verbose(self):
        return (
            f"{self.structure['components'].shape[0]} components, "
            f"{self.structure['ports'].shape[0]} ports, "
            f"{self.structure['links'].shape[0]} links, "
        )

    def plot(self, img=None, conf=True, show=False, save=False, filename=None, **kwargs):
        """Draw structure points and their component associations on a BGR image."""
        image = deepcopy(self.orig_img if img is None else img)
        as_numpy = lambda value: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        components = as_numpy(self.structure["components"])
        ports = as_numpy(self.structure["ports"])
        links = as_numpy(self.structure["links"])
        for link in links:
            port_idx, component_idx = int(link[0]), int(link[1])
            if port_idx < len(ports) and component_idx < len(components):
                start = tuple(ports[port_idx, 0, :2].astype(int))
                end = tuple(components[component_idx, 0, :2].astype(int))
                cv2.line(image, start, end, (0, 255, 80), 2)
        for component in components:
            x, y, score = component[0]
            cv2.circle(image, (int(x), int(y)), 3, (200, 50, 50), -1)
            if conf:
                cv2.putText(image, f"{score:.2f}", (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 50, 50), 1)
        for port in ports:
            x, y, score = port[0]
            cv2.rectangle(image, (int(x) - 4, int(y) - 4), (int(x) + 4, int(y) + 4), (50, 200, 50), -1)
            if conf:
                cv2.putText(image, f"{score:.2f}", (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 200, 50), 1)
        if save:
            cv2.imwrite(str(filename or f"results_{Path(self.path).name}"), image)
        if show:
            cv2.imshow(Path(self.path).name, image)
            cv2.waitKey(1)
        return image


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

            components = pred_dict["components"]
            components = components[components[:, 2] >= self.args.conf]
            if components.shape[0]:
                components = radius_point_nms(components, component_nms_radius, max_det)

            ports = pred_dict["ports"]
            ports = ports[ports[:, 2] >= self.args.conf]
            if ports.shape[0]:
                ports = radius_point_nms(ports, port_nms_radius, max_det)
            ports, links = associate_ports_to_components(components, ports)

            # Scale linked components and ports after canonical association/de-duplication.
            if components.shape[0]:
                comp_xy = components[:, :3].unsqueeze(1)
                comp_xy = ops.scale_coords(img.shape[2:], comp_xy, orig_img.shape)
            else:
                comp_xy = torch.zeros((0, 1, 3), device=components.device)

            if ports.shape[0]:
                port_xy = ports[:, :3].unsqueeze(1)
                port_xy = ops.scale_coords(img.shape[2:], port_xy, orig_img.shape)
                pred_comp_xy = ports[:, 3:5].unsqueeze(1)
                pred_comp_xy = ops.scale_coords(img.shape[2:], pred_comp_xy, orig_img.shape)
            else:
                port_xy = torch.zeros((0, 1, 3), device=ports.device)
                pred_comp_xy = torch.zeros((0, 1, 2), device=ports.device)

            scaled_port_data = torch.cat((port_xy.squeeze(1), pred_comp_xy.squeeze(1), ports[:, 5:]), dim=1)
            links = scale_link_geometry(links, scaled_port_data)
            result = StructureResults(
                orig_img,
                path=img_path,
                names=self.model.names,
                structure={
                    "components": comp_xy,
                    "ports": port_xy,
                    "pred_components": pred_comp_xy,
                    "links": links,
                },
            )
            results.append(result)
        return results
