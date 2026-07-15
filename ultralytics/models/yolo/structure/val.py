# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, RANK, ops
from ultralytics.utils.plotting import plot_images
from .utils import associate_ports_to_components, radius_point_nms, scale_link_geometry


class StructureMetrics:
    """Metrics for structure prediction with component, port, and link evaluation."""

    fixed_thresholds = (0.25, 0.30, 0.40, 0.45)

    keys = [
        "metrics/component_8px_F1",
        "metrics/port_8px_F1",
        "metrics/link_8px_F1",
        "metrics/link_association_accuracy",
        "metrics/angle_MAE",
        "metrics/endpoint_MAE",
        "metrics/strict_object_recall",
        "fitness",
    ]

    def __init__(self):
        """Initialize counters."""
        self.reset()
        self.speed = None
        self.save_dir = None

    def reset(self):
        """Reset accumulated metrics."""
        self.records = []
        self.num_gt_components = 0
        self.num_gt_ports = 0
        self.num_gt_links = 0

    def update(self, pred_components, pred_ports, pred_links, gt_components, gt_ports, gt_links):
        """Store one image of predictions and ground-truth."""
        self.records.append((
            pred_components.detach().float().cpu(),
            pred_ports.detach().float().cpu(),
            pred_links.detach().float().cpu() if pred_links is not None else None,
            gt_components.detach().float().cpu(),
            gt_ports.detach().float().cpu(),
            gt_links.detach().float().cpu() if gt_links is not None else None,
        ))
        self.num_gt_components += int(gt_components.shape[0])
        self.num_gt_ports += int(gt_ports.shape[0])
        self.num_gt_links += int(gt_links.shape[0]) if gt_links is not None else 0

    @property
    def stats(self) -> dict[str, Any]:
        """Return gather-able stats dict for DDP."""
        return {
            "records": self.records,
            "num_gt_components": self.num_gt_components,
            "num_gt_ports": self.num_gt_ports,
            "num_gt_links": self.num_gt_links,
        }

    @stats.setter
    def stats(self, value: dict[str, Any]) -> None:
        """Restore stats from gathered dict."""
        self.records = value["records"]
        self.num_gt_components = value["num_gt_components"]
        self.num_gt_ports = value["num_gt_ports"]
        self.num_gt_links = value["num_gt_links"]

    def clear_stats(self) -> None:
        """Clear accumulated stats after DDP gathering."""
        self.records = []
        self.num_gt_components = 0
        self.num_gt_ports = 0
        self.num_gt_links = 0

    def mean_results(self):
        """Return metrics in display order."""
        metrics = self._compute_metrics()
        # Fitness = 0.3 * component_F1 + 0.7 * link_F1
        fitness = 0.3 * metrics["component_8px_f1"] + 0.7 * metrics["link_8px_f1"]
        return [
            metrics["component_8px_f1"],
            metrics["port_8px_f1"],
            metrics["link_8px_f1"],
            metrics["link_association_accuracy"],
            metrics["angle_mae"],
            metrics["endpoint_mae"],
            metrics["strict_object_recall"],
            fitness,
        ]

    @property
    def results_dict(self):
        """Return metrics dict."""
        return dict(zip(self.keys, self.mean_results()))

    @staticmethod
    def _match_points(pred, gt, threshold):
        """Match predicted points to GT points with distance threshold."""
        if pred.numel() == 0:
            return [], [], list(range(gt.shape[0]))
        if gt.numel() == 0:
            return [], list(range(pred.shape[0])), []

        # Compute distance matrix
        pred_xy = pred[:, :2]
        gt_xy = gt[:, :2]
        dists = torch.cdist(pred_xy, gt_xy)

        # Greedily match highest confidence predictions first
        order = pred[:, 2].argsort(descending=True)
        matched_gt = set()
        matched_pred = set()
        matches = []

        for pred_idx in order:
            best_dist = float("inf")
            best_gt_idx = -1
            for gt_idx in range(gt.shape[0]):
                if gt_idx in matched_gt:
                    continue
                if dists[pred_idx, gt_idx] < threshold and dists[pred_idx, gt_idx] < best_dist:
                    best_dist = dists[pred_idx, gt_idx]
                    best_gt_idx = gt_idx
            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(int(pred_idx))
                matches.append((int(pred_idx), best_gt_idx))

        fp = [i for i in range(pred.shape[0]) if i not in matched_pred]
        fn = [i for i in range(gt.shape[0]) if i not in matched_gt]
        return matches, fp, fn

    def _compute_metrics(self):
        """Compute all structure metrics."""
        component_threshold = 8.0  # 2 feature cells at stride 4
        port_threshold = 8.0
        link_threshold = 8.0

        component_tp = 0
        component_fp = 0
        component_fn = 0
        port_tp = 0
        port_fp = 0
        port_fn = 0
        link_tp = 0
        link_fp = 0
        link_fn = 0
        correct_associations = 0
        association_total = 0
        angle_errors = []
        endpoint_errors = []
        strict_object_count = 0
        total_objects = 0

        for record in self.records:
            pred_comp, pred_port, pred_link, gt_comp, gt_port, gt_link = record

            # Component matching
            comp_matches, comp_fp_list, comp_fn_list = self._match_points(pred_comp, gt_comp, component_threshold)
            component_tp += len(comp_matches)
            component_fp += len(comp_fp_list)
            component_fn += len(comp_fn_list)

            # Port matching
            port_matches, port_fp_list, port_fn_list = self._match_points(pred_port, gt_port, port_threshold)
            port_tp += len(port_matches)
            port_fp += len(port_fp_list)
            port_fn += len(port_fn_list)

            # Build GT port->component links from data
            # Create map from gt_port_idx to gt_comp_idx
            gt_port_to_comp = {}
            if gt_link is not None:
                for link in gt_link:
                    port_idx = int(link[0])
                    comp_idx = int(link[1])
                    gt_port_to_comp[port_idx] = comp_idx

            # Build maps for matched preds to GT
            matched_pred_comp_to_gt_comp = {p_idx: g_idx for p_idx, g_idx in comp_matches}
            matched_pred_port_to_gt_port = {p_idx: g_idx for p_idx, g_idx in port_matches}

            # Track for each GT component how many ports are correctly linked
            gt_comp_port_counts = {}
            gt_comp_correct_port_counts = {}
            for port_idx, comp_idx in gt_port_to_comp.items():
                gt_comp_port_counts[comp_idx] = gt_comp_port_counts.get(comp_idx, 0) + 1

            total_objects += len(gt_comp_port_counts)

            # Evaluate links from predictions
            image_link_tp = 0
            if pred_link is not None:
                for link in pred_link:
                    pred_port_idx = int(link[0])
                    pred_comp_idx = int(link[1])
                    score = link[2]

                    # Check if port is matched to GT port
                    if pred_port_idx in matched_pred_port_to_gt_port:
                        gt_port_idx = matched_pred_port_to_gt_port[pred_port_idx]
                        # Check if this GT port should have a link
                        if gt_port_idx in gt_port_to_comp:
                            gt_comp_idx = gt_port_to_comp[gt_port_idx]
                            # Check if predicted component is matched to GT component
                            if pred_comp_idx in matched_pred_comp_to_gt_comp:
                                association_total += 1
                                matched_gt_comp_idx = matched_pred_comp_to_gt_comp[pred_comp_idx]
                                if matched_gt_comp_idx == gt_comp_idx:
                                    link_tp += 1
                                    image_link_tp += 1
                                    correct_associations += 1
                                    gt_comp_correct_port_counts[gt_comp_idx] = gt_comp_correct_port_counts.get(gt_comp_idx, 0) + 1
                                    # Compute angle error if available
                                    if link.shape[0] > 3:
                                        pred_dir = link[3:5]
                                        pred_angle = torch.atan2(pred_dir[1], pred_dir[0])
                                        # Get GT port and component positions to compute GT angle
                                        if pred_port_idx < len(pred_port) and gt_comp_idx < len(gt_comp):
                                            gt_p = gt_port[gt_port_idx, :2]
                                            gt_c = gt_comp[gt_comp_idx, :2]
                                            gt_delta = gt_c - gt_p
                                            gt_angle = torch.atan2(gt_delta[1], gt_delta[0])
                                            angle_diff = torch.abs(pred_angle - gt_angle)
                                            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                                            angle_errors.append(float(angle_diff))
                                    # Compute endpoint error if available
                                    if link.shape[0] > 5:
                                        pred_endpoint = link[5:7]
                                        if gt_comp_idx < len(gt_comp):
                                            gt_endpoint = gt_comp[gt_comp_idx, :2]
                                            endpoint_err = torch.norm(pred_endpoint - gt_endpoint)
                                            endpoint_errors.append(float(endpoint_err))
                                else:
                                    link_fp += 1
                            else:
                                link_fp += 1
                        else:
                            link_fp += 1
                    else:
                        link_fp += 1

            # Count FN links (per-image, not global)
            gt_link_count = len(gt_port_to_comp)
            link_fn += max(0, gt_link_count - image_link_tp)

            # Check strict object recall: all ports of component must be correctly linked
            for comp_idx in gt_comp_port_counts:
                if gt_comp_correct_port_counts.get(comp_idx, 0) == gt_comp_port_counts[comp_idx]:
                    strict_object_count += 1

        # Compute F1 scores
        def compute_f1(tp, fp, fn):
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            return precision, recall, f1

        comp_prec, comp_rec, comp_f1 = compute_f1(component_tp, component_fp, component_fn)
        port_prec, port_rec, port_f1 = compute_f1(port_tp, port_fp, port_fn)
        link_prec, link_rec, link_f1 = compute_f1(link_tp, link_fp, link_fn)

        assoc_acc = correct_associations / association_total if association_total else 0.0
        angle_mae = np.mean(angle_errors) if angle_errors else 0.0
        endpoint_mae = np.mean(endpoint_errors) if endpoint_errors else 0.0
        strict_recall = strict_object_count / max(total_objects, 1) if total_objects > 0 else 0.0

        return {
            "component_8px_f1": comp_f1,
            "port_8px_f1": port_f1,
            "link_8px_f1": link_f1,
            "link_association_accuracy": assoc_acc,
            "angle_mae": angle_mae,
            "endpoint_mae": endpoint_mae,
            "strict_object_recall": strict_recall,
        }


class StructureValidator(DetectionValidator):
    """Validator for structure models."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize structure validator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "structure"
        self.metrics = StructureMetrics()
        self.kpt_shape = None
        self.component_nms_radius = 16.0
        self.port_nms_radius = 8.0

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess images and keypoints."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    @staticmethod
    def _head_module(model: torch.nn.Module) -> torch.nn.Module:
        """Return the terminal structure head from wrapped validation models."""
        current = model
        for _ in range(5):
            if isinstance(current, torch.nn.Sequential):
                return current[-1]
            nested = getattr(current, "model", None)
            if nested is None or nested is current:
                break
            current = nested
        return current

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize validation metrics."""
        self.names = model.names
        self.nc = len(model.names)
        self.kpt_shape = self.data.get("kpt_shape", [25, 3])
        head = self._head_module(model)
        self.args.conf = max(float(self.args.conf or 0.0), 0.05) if head.__class__.__name__ == "StructureHeatmap" else self.args.conf
        self.metrics.reset()
        self.seen = 0

    def postprocess(self, preds: dict) -> list[dict]:
        """Filter structure predictions with confidence and radius NMS."""
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        outputs = []
        device = preds[0]["components"].device if isinstance(preds, list) else preds["components"].device

        for pred_dict in preds:
            components = pred_dict["components"]
            ports = pred_dict["ports"]

            # Filter by confidence
            comp_keep = components[:, 2] >= self.args.conf
            components = components[comp_keep]
            port_keep = ports[:, 2] >= self.args.conf
            ports = ports[port_keep]

            # NMS
            if components.shape[0]:
                components = radius_point_nms(components, self.component_nms_radius, self.args.max_det)
            if ports.shape[0]:
                ports = radius_point_nms(ports, self.port_nms_radius, self.args.max_det)

            ports, links = associate_ports_to_components(components, ports)

            outputs.append({
                "components": components,
                "ports": ports,
                "links": links,
            })
        return outputs

    def plot_predictions(self, batch, preds, batch_idx):
        """Override to handle structure-specific prediction format (no-op)."""
        pass

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare ground-truth for a batch image."""
        idx = batch["batch_idx"] == si
        kpts = batch["keypoints"][idx].clone()
        h, w = batch["img"].shape[2:]
        imgsz = torch.tensor((h, w), device=kpts.device, dtype=kpts.dtype)
        ori_shape = batch["ori_shape"][si]
        ratio_pad = batch["ratio_pad"][si]

        # Extract components (keypoint 0) and ports (keypoints 1-24)
        gt_components = []
        gt_ports = []
        gt_links = []

        for obj_i, obj_kpts in enumerate(kpts):
            # Component center is first keypoint
            comp_kpt = obj_kpts[0]
            if comp_kpt[2] > 0:
                comp_xy = comp_kpt[:2].clone()
                comp_xy[0] *= w
                comp_xy[1] *= h
                comp_xy = ops.scale_coords(batch["img"].shape[2:], comp_xy.view(1, 2), ori_shape, ratio_pad=ratio_pad).view(2)
                gt_components.append(torch.cat((comp_xy, torch.tensor([1.0], device=comp_xy.device))))
                comp_idx = len(gt_components) - 1

                # Ports are keypoints 1-24
                for port_offset, port_kpt in enumerate(obj_kpts[1:]):
                    if port_kpt[2] > 0:
                        port_xy = port_kpt[:2].clone()
                        port_xy[0] *= w
                        port_xy[1] *= h
                        port_xy = ops.scale_coords(batch["img"].shape[2:], port_xy.view(1, 2), ori_shape, ratio_pad=ratio_pad).view(2)
                        gt_ports.append(torch.cat((port_xy, torch.tensor([1.0], device=port_xy.device))))
                        port_idx = len(gt_ports) - 1
                        gt_links.append(torch.tensor([port_idx, comp_idx, 1.0], device=port_xy.device))

        if gt_components:
            gt_components = torch.stack(gt_components)
        else:
            gt_components = torch.zeros((0, 3), device=kpts.device)
        if gt_ports:
            gt_ports = torch.stack(gt_ports)
        else:
            gt_ports = torch.zeros((0, 3), device=kpts.device)
        if gt_links:
            gt_links = torch.stack(gt_links)
        else:
            gt_links = torch.zeros((0, 3), device=kpts.device)

        return {
            "components": gt_components,
            "ports": gt_ports,
            "links": gt_links,
            "ori_shape": ori_shape,
            "imgsz": batch["img"].shape[2:],
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def update_metrics(self, preds: list[dict], batch: dict[str, Any]) -> None:
        """Update structure metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            gt = self._prepare_batch(si, batch)

            # Scale predictions to original image coords
            pred_components = pred["components"]
            pred_ports = pred["ports"]
            pred_links = pred["links"]

            if pred_components.numel():
                pred_comp_xy = pred_components[:, :2].clone()
                pred_comp_xy = ops.scale_coords(batch["img"].shape[2:], pred_comp_xy, gt["ori_shape"], ratio_pad=gt["ratio_pad"])
                pred_components_scaled = torch.cat((pred_comp_xy, pred_components[:, 2:]), -1)
            else:
                pred_components_scaled = pred_components.new_zeros((0, 3))

            if pred_ports.numel():
                pred_port_xy = pred_ports[:, :2].clone()
                pred_port_xy = ops.scale_coords(batch["img"].shape[2:], pred_port_xy, gt["ori_shape"], ratio_pad=gt["ratio_pad"])
                # Also scale predicted component endpoints
                pred_port_predcomp = pred_ports[:, 3:5].clone()
                pred_port_predcomp = ops.scale_coords(batch["img"].shape[2:], pred_port_predcomp, gt["ori_shape"], ratio_pad=gt["ratio_pad"])
                pred_ports_scaled = torch.cat((pred_port_xy, pred_ports[:, 2:3], pred_port_predcomp, pred_ports[:, 5:]), -1)
            else:
                pred_ports_scaled = pred_ports.new_zeros((0, 8))

            pred_links_scaled = scale_link_geometry(pred_links, pred_ports_scaled)
            self.metrics.update(pred_components_scaled, pred_ports_scaled, pred_links_scaled, gt["components"], gt["ports"], gt["links"])

    def get_stats(self) -> dict[str, Any]:
        """Return validation statistics."""
        return self.metrics.results_dict

    def gather_stats(self) -> None:
        """Gather metrics from all DDP ranks."""
        if RANK == 0:
            gathered_stats = [None] * dist.get_world_size()
            dist.gather_object(self.metrics.stats, gathered_stats, dst=0)
            merged_records = []
            merged_nc = 0
            merged_np = 0
            merged_nl = 0
            for s in gathered_stats:
                merged_records.extend(s["records"])
                merged_nc += s["num_gt_components"]
                merged_np += s["num_gt_ports"]
                merged_nl += s["num_gt_links"]
            self.metrics.records = merged_records
            self.metrics.num_gt_components = merged_nc
            self.metrics.num_gt_ports = merged_np
            self.metrics.num_gt_links = merged_nl
            gathered_jdict = [None] * dist.get_world_size()
            dist.gather_object(self.jdict, gathered_jdict, dst=0)
            self.jdict = []
            for jdict in gathered_jdict:
                self.jdict.extend(jdict)
            self.seen = len(self.dataloader.dataset)
        elif RANK > 0:
            dist.gather_object(self.metrics.stats, None, dst=0)
            self.metrics.clear_stats()
            dist.gather_object(self.jdict, None, dst=0)
            self.jdict = []

    def get_desc(self) -> str:
        """Return validation progress header."""
        return ("%22s" + "%11s" * 8) % (
            "Class",
            "Images",
            "Comp_F1",
            "Port_F1",
            "Link_F1",
            "Assoc_Acc",
            "Angle_MAE",
            "Endpt_MAE",
            "Strict_Rec",
        )

    def print_results(self) -> None:
        """Print aggregate structure metrics."""
        comp_f1, port_f1, link_f1, assoc_acc, angle_mae, endpt_mae, strict_rec, _ = self.metrics.mean_results()
        LOGGER.info(
            ("%22s" + "%11i" + "%11.3g" * 7) % (
                "all",
                self.seen,
                comp_f1,
                port_f1,
                link_f1,
                assoc_acc,
                angle_mae,
                endpt_mae,
                strict_rec,
            )
        )

    def finalize_metrics(self) -> None:
        """Attach speed and save_dir to metrics."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
