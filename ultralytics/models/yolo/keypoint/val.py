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
from .utils import radius_point_nms


class KeypointOnlyMetrics:
    """Keypoint-only metrics using CenterNet-B style point matching."""

    fixed_thresholds = (0.25, 0.30, 0.40, 0.45)

    keys = [
        "metrics/center_3px_F1(K)",
        "metrics/center_5px_F1(K)",
        "metrics/center_10px_F1(K)",
        "metrics/center_5px_precision(K)",
        "metrics/center_5px_recall(K)",
        "metrics/center_5px_threshold(K)",
        "metrics/center_5px_F1_thr0.25(K)",
        "metrics/center_5px_F1_thr0.30(K)",
        "metrics/center_5px_F1_thr0.40(K)",
        "metrics/center_5px_F1_thr0.45(K)",
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
        self.num_gt = 0

    def update(self, pred_points: torch.Tensor, gt_points: torch.Tensor):
        """Store one image of predictions and ground-truth points in original-image pixels."""
        pred = pred_points.detach().float().cpu()
        gt = gt_points.detach().float().cpu()
        self.records.append((pred, gt))
        self.num_gt += int(gt.shape[0])

    @property
    def stats(self) -> dict[str, Any]:
        """Return gather-able stats dict for DDP."""
        return {"records": self.records, "num_gt": self.num_gt}

    @stats.setter
    def stats(self, value: dict[str, Any]) -> None:
        """Restore stats from gathered dict."""
        self.records = value["records"]
        self.num_gt = value["num_gt"]

    def clear_stats(self) -> None:
        """Clear accumulated stats after DDP gathering."""
        self.records = []
        self.num_gt = 0

    def mean_results(self):
        """Return metrics in display order."""
        metrics = self._best_metrics()
        center5 = metrics["center_5px"]
        fixed_center5 = self._fixed_center5_metrics()
        fitness = center5["f1"]
        return [
            metrics["center_3px"]["f1"],
            center5["f1"],
            metrics["center_10px"]["f1"],
            center5["precision"],
            center5["recall"],
            center5["threshold"],
            fixed_center5[0.25]["f1"],
            fixed_center5[0.30]["f1"],
            fixed_center5[0.40]["f1"],
            fixed_center5[0.45]["f1"],
            fitness,
        ]

    @property
    def results_dict(self):
        """Return metrics dict."""
        return dict(zip(self.keys, self.mean_results()))

    @staticmethod
    def _thresholds() -> np.ndarray:
        """Return the same confidence sweep used by the CenterNet-B point evaluator."""
        return np.concatenate([np.arange(0.01, 0.10, 0.01), np.arange(0.10, 0.95, 0.05)])

    def _best_metrics(self) -> dict[str, dict[str, float]]:
        """Compute best F1 for 3/5/10 px center-distance thresholds."""
        return {
            "center_3px": self._best_metric(3.0),
            "center_5px": self._best_metric(5.0),
            "center_10px": self._best_metric(10.0),
        }

    def _best_metric(self, distance_threshold: float) -> dict[str, float]:
        """Return the best threshold-swept point matching metric for one distance threshold."""
        scores = [self._score_f1(float(threshold), distance_threshold) for threshold in self._thresholds()]
        return max(scores, key=lambda item: item["f1"])

    def _fixed_center5_metrics(self) -> dict[float, dict[str, float]]:
        """Return center-5px metrics at fixed deployment-style confidence thresholds."""
        return {threshold: self._score_f1(threshold, 5.0) for threshold in self.fixed_thresholds}

    def _score_f1(self, threshold: float, distance_threshold: float) -> dict[str, float]:
        """Match predictions to GT like B experiment: score-sorted nearest unmatched GT within pixel radius."""
        tp = 0
        fp = 0
        for pred_points, gt_points in self.records:
            pred = pred_points[pred_points[:, 2] >= threshold]
            if pred.numel():
                pred = pred[pred[:, 2].argsort(descending=True)]
            matched = set()
            for point in pred:
                best_index = -1
                best_dist = float("inf")
                for gt_index, gt_point in enumerate(gt_points):
                    if gt_index in matched:
                        continue
                    dist = float(torch.linalg.vector_norm(point[:2] - gt_point[:2]).item())
                    if dist <= distance_threshold and dist < best_dist:
                        best_dist = dist
                        best_index = gt_index
                if best_index >= 0:
                    tp += 1
                    matched.add(best_index)
                else:
                    fp += 1
        fn = self.num_gt - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "threshold": float(threshold),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }


class KeypointValidator(DetectionValidator):
    """Validator for keypoint-only models."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize keypoint-only validator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "keypoint"
        self.metrics = KeypointOnlyMetrics()
        self.kpt_shape = None
        self.match_threshold = 0.02
        self.point_nms_radius = 8.0

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess images and keypoints."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    @staticmethod
    def _head_module(model: torch.nn.Module) -> torch.nn.Module:
        """Return the terminal keypoint head from wrapped validation models."""
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
        self.kpt_shape = self.data["kpt_shape"]
        head = self._head_module(model)
        self.args.conf = max(float(self.args.conf or 0.0), 0.05) if head.__class__.__name__ == "KeypointHeatmap" else self.args.conf
        self.metrics.reset()
        self.seen = 0

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Filter dense point candidates with confidence and radius NMS."""
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        outputs = []
        for pred in preds:
            keep = pred[:, 2] >= self.args.conf
            pred = pred[keep]
            if pred.shape[0]:
                pred = radius_point_nms(pred, self.point_nms_radius, self.args.max_det)
            outputs.append(
                {"keypoints": pred[:, :3].view(-1, 1, 3), "conf": pred[:, 2] if pred.numel() else pred.new_zeros(0)}
            )
        return outputs

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare ground-truth keypoints for a batch image."""
        idx = batch["batch_idx"] == si
        kpts = batch["keypoints"][idx].clone()
        h, w = batch["img"].shape[2:]
        if kpts.numel():
            kpts[..., 0] *= w
            kpts[..., 1] *= h
        return {
            "keypoints": kpts,
            "ori_shape": batch["ori_shape"][si],
            "imgsz": batch["img"].shape[2:],
            "ratio_pad": batch["ratio_pad"][si],
            "im_file": batch["im_file"][si],
        }

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update keypoint metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            gt = pbatch["keypoints"]
            pred_kpts = pred["keypoints"]
            if gt.numel():
                visible = gt[..., 2] > 0 if gt.shape[-1] == 3 else torch.ones_like(gt[..., 0], dtype=torch.bool)
                gt_kpts = gt[visible].view(-1, 1, gt.shape[-1])
                gt_xy = ops.scale_coords(
                    pbatch["imgsz"],
                    gt_kpts.clone(),
                    pbatch["ori_shape"],
                    ratio_pad=pbatch["ratio_pad"],
                ).view(-1, gt.shape[-1])[..., :2]
            else:
                gt_xy = gt.new_zeros((0, 2))
            if pred_kpts.numel():
                predn = {
                    "keypoints": ops.scale_coords(
                        pbatch["imgsz"], pred_kpts.clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
                    )
                }
                pred_points = torch.cat(
                    (predn["keypoints"][..., :2].reshape(-1, 2), pred["conf"].reshape(-1, 1)),
                    dim=1,
                )
            else:
                predn = {"keypoints": pred_kpts}
                pred_points = pred_kpts.new_zeros((0, 3))
            self.metrics.update(pred_points, gt_xy)
            if self.args.save_txt and pred_points.numel():
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def get_stats(self) -> dict[str, Any]:
        """Return validation statistics."""
        return self.metrics.results_dict

    def gather_stats(self) -> None:
        """Gather keypoint metrics from all DDP ranks."""
        if RANK == 0:
            gathered_stats = [None] * dist.get_world_size()
            dist.gather_object(self.metrics.stats, gathered_stats, dst=0)
            merged_records = []
            merged_num_gt = 0
            for s in gathered_stats:
                merged_records.extend(s["records"])
                merged_num_gt += s["num_gt"]
            self.metrics.records = merged_records
            self.metrics.num_gt = merged_num_gt
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
        return ("%22s" + "%11s" * 11) % (
            "Class",
            "Images",
            "Points",
            "F1@3",
            "F1@5",
            "F1@10",
            "P@5",
            "R@5",
            "F1@.25",
            "F1@.30",
            "F1@.40",
            "F1@.45",
        )

    def print_results(self) -> None:
        """Print aggregate keypoint metrics."""
        f1_3, f1_5, f1_10, precision_5, recall_5, _, f1_025, f1_030, f1_040, f1_045, _ = (
            self.metrics.mean_results()
        )
        LOGGER.info(
            ("%22s" + "%11i" * 2 + "%11.3g" * 9)
            % (
                "all",
                self.seen,
                self.metrics.num_gt,
                f1_3,
                f1_5,
                f1_10,
                precision_5,
                recall_5,
                f1_025,
                f1_030,
                f1_040,
                f1_045,
            )
        )

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        """Plot predicted keypoints."""
        if not preds:
            return
        labels = {"keypoints": torch.cat([p["keypoints"] for p in preds], 0)}
        labels["batch_idx"] = torch.cat(
            [torch.full((p["keypoints"].shape[0],), i, device=p["keypoints"].device) for i, p in enumerate(preds)], 0
        )
        plot_images(
            images=batch["img"],
            labels=labels,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def finalize_metrics(self) -> None:
        """Attach speed and save_dir to metrics."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
