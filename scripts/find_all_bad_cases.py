#!/usr/bin/env python3
"""
Find bad cases for the full training/validation set used by the best merge_v4 model.
Match exactly the preprocessing/postprocessing logic used during training/validation.
"""

import os
import shutil
import csv
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# Configuration
MODEL_PATH = "/data-ssd/libo/ultralytics/runs/keypoint/merge_v4_yolo26_1536_scratch_e200_gpu34/weights/best.pt"
DATA_ROOT = "/data-ssd/libo/p100/yolo_utils/dataset"
OUTPUT_ROOT = "/data-ssd/libo/p100/yolo_utils/dataset/external_inline/merge_config_v2_bad_cases"

# Full dataset paths from merge_config_v2.yaml
DATA_PATHS = {
    "train": [
        "external_inline/datasets_in_line_7k_train_nonempty/images/train",
        "external_inline/pose_compare_inline_diff_b_500/images/train",
        "external_inline/paper_graph/images/train"
    ],
    "val": [
        "external_inline/datasets_in_line_7k_train_nonempty/images/val",
        "external_inline/datasets_in_line_7k_background_only/images/val",
        "external_inline/paper_graph/images/train"
    ]
}

CONF_THRESHOLD = 0.3
MATCH_DISTANCE_PX = 5
NMS_RADIUS_PX = 8
MAX_DET = 300
DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"


def letterbox(
    img: np.ndarray,
    new_shape: int = 1280,
    color: tuple = (114, 114, 114),
) -> Tuple[np.ndarray, float, int, int]:
    """Match Ultralytics validation resize + letterbox for fixed square input."""
    h0, w0 = img.shape[:2]
    r0 = new_shape / max(h0, w0)
    if r0 != 1:
        new_w = min(int(np.ceil(w0 * r0)), new_shape)
        new_h = min(int(np.ceil(h0 * r0)), new_shape)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        new_h, new_w = h0, w0

    dw = (new_shape - new_w) / 2
    dh = (new_shape - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    gain = new_h / h0
    return padded, gain, left, top


def radius_point_nms(points: np.ndarray, radius: float, max_det: int = 300) -> np.ndarray:
    """Keep highest confidence point within each local radius (numpy version)."""
    if points.shape[0] == 0:
        return points

    # Sort by confidence descending
    idx = np.argsort(points[:, 2])[::-1]
    points = points[idx]
    radius2 = radius ** 2
    keep = []

    for i, p in enumerate(points):
        if len(keep) >= max_det:
            break
        too_close = False
        for j in keep:
            if (p[0] - points[j, 0]) ** 2 + (p[1] - points[j, 1]) ** 2 < radius2:
                too_close = True
                break
        if not too_close:
            keep.append(i)

    return points[keep]


def load_gt_keypoints(label_path: Path, img_width: int, img_height: int) -> List[Tuple[float, float]]:
    """Load ground truth keypoints from YOLO txt label, convert to absolute coordinates."""
    keypoints = []
    if not label_path.exists():
        return keypoints

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            # YOLO keypoint format: <class> <x> <y> <w> <h> <kpt_x> <kpt_y> <kpt_vis>
            if len(parts) >= 8:
                x = parts[5] * img_width
                y = parts[6] * img_height
                keypoints.append((x, y))
            # Fallback: old format <class> <x> <y> <vis>
            elif len(parts) == 4:
                x = parts[1] * img_width
                y = parts[2] * img_height
                keypoints.append((x, y))

    return keypoints


def match_points(preds: List[Tuple[float, float, float]], gts: List[Tuple[float, float]], distance_thr: float) -> Tuple[int, int, int]:
    """
    Match predicted points to ground truth points.
    Returns (tp_count, fp_count, fn_count)
    """
    if not preds and not gts:
        return 0, 0, 0
    if not preds:
        return 0, 0, len(gts)
    if not gts:
        return 0, len(preds), 0

    # Sort predictions by confidence descending
    preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)
    matched_gt = set()
    tp = 0

    for p in preds_sorted:
        px, py, _ = p
        best_dist = float("inf")
        best_gt_idx = -1

        for gt_idx, (gx, gy) in enumerate(gts):
            if gt_idx in matched_gt:
                continue
            dist = ((px - gx)**2 + (py - gy)**2) ** 0.5
            if dist < best_dist and dist <= distance_thr:
                best_dist = dist
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            tp += 1

    fp = len(preds_sorted) - tp
    fn = len(gts) - tp

    return tp, fp, fn


def process_dataset(split: str, img_rel_paths: List[str], model, output_root: Path, data_root: Path) -> Dict:
    """Process all images in a dataset split (train/val) and collect bad cases."""
    split_stats = {
        "total_images": 0,
        "bad_cases": 0,
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0
    }

    for img_rel_path in img_rel_paths:
        img_full_path = data_root / img_rel_path
        if not img_full_path.exists():
            continue

        # Get corresponding label path: replace images/ -> labels/ and .jpg -> .txt
        label_rel_path = img_rel_path.replace("images/", "labels/").replace(".jpg", ".txt")
        label_full_path = data_root / label_rel_path

        # Load image
        img = cv2.imread(str(img_full_path))
        if img is None:
            print(f"Warning: Could not read image {img_full_path}, skipping.")
            continue
        h, w = img.shape[:2]
        split_stats["total_images"] += 1

        # Preprocess (match validation preprocessing)
        img_letterbox, ratio, dw, dh = letterbox(img, 1280)
        # BGR -> RGB, HWC -> CHW, uint8 -> normalized float32
        img_input = img_letterbox[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = torch.from_numpy(img_input).unsqueeze(0).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            raw_pred = model(img_input)[0]  # raw decode output [1, 300, 4]

        # Postprocess
        pred_np = raw_pred.cpu().numpy()[0]  # [300, 4]
        # Filter by confidence
        mask = pred_np[:, 2] >= CONF_THRESHOLD
        pred_points = pred_np[mask]
        if len(pred_points) == 0:
            pred_points_processed = []
        else:
            # Apply radius NMS
            pred_points = radius_point_nms(pred_points, NMS_RADIUS_PX, MAX_DET)
            # Convert back to original image coordinates
            pred_points_processed = []
            for p in pred_points:
                x, y, conf, cls = p
                x_orig = (x - dw) / ratio
                y_orig = (y - dh) / ratio
                if 0 <= x_orig < w and 0 <= y_orig < h:
                    pred_points_processed.append((float(x_orig), float(y_orig), float(conf)))

        # Load ground truth
        gt_points = load_gt_keypoints(label_full_path, w, h)

        # Match points
        tp, fp, fn = match_points(pred_points_processed, gt_points, MATCH_DISTANCE_PX)
        split_stats["total_tp"] += tp
        split_stats["total_fp"] += fp
        split_stats["total_fn"] += fn

        # If there are mismatches, copy to output directory
        if fp > 0 or fn > 0:
            split_stats["bad_cases"] += 1
            # Create output directories preserving the original path structure
            out_img_path = output_root / img_rel_path
            out_label_path = output_root / label_rel_path
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            out_label_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy image and label
            shutil.copy2(img_full_path, out_img_path)
            if label_full_path.exists():
                shutil.copy2(label_full_path, out_label_path)

            # Write to stats csv
            stats_writer.writerow([
                img_rel_path,
                len(gt_points),
                len(pred_points_processed),
                tp,
                fp,
                fn
            ])

    return split_stats


if __name__ == "__main__":
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    yolo_model = YOLO(MODEL_PATH)
    model = yolo_model.model.to(DEVICE)
    model.eval()

    # Setup output directories
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    # Stats file
    stats_file = open(output_root / "bad_cases_stats.csv", "w", newline="")
    stats_writer = csv.writer(stats_file)
    stats_writer.writerow(["relative_path", "num_gt", "num_pred", "tp", "fp", "fn"])

    all_stats = {}
    for split in ["train", "val"]:
        print(f"\n=== Processing {split} set ===")
        # Collect all image paths for this split
        all_img_paths = []
        for rel_dir in DATA_PATHS[split]:
            full_dir = Path(DATA_ROOT) / rel_dir
            if full_dir.exists():
                all_img_paths.extend([str(Path(rel_dir) / p.name) for p in full_dir.glob("*.jpg")])

        print(f"Found {len(all_img_paths)} images in {split} set")

        # Process this split
        split_stats = process_dataset(split, all_img_paths, model, output_root, Path(DATA_ROOT))
        all_stats[split] = split_stats

        # Print summary
        print(f"{split} set summary:")
        print(f"  Total images: {split_stats['total_images']}")
        print(f"  Bad cases: {split_stats['bad_cases']} ({split_stats['bad_cases']/split_stats['total_images']*100:.2f}%)")
        print(f"  Total TP: {split_stats['total_tp']}, FP: {split_stats['total_fp']}, FN: {split_stats['total_fn']}")
        precision = split_stats['total_tp'] / (split_stats['total_tp'] + split_stats['total_fp']) if split_stats['total_tp'] + split_stats['total_fp'] else 0
        recall = split_stats['total_tp'] / (split_stats['total_tp'] + split_stats['total_fn']) if split_stats['total_tp'] + split_stats['total_fn'] else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    stats_file.close()
    print(f"\nAll bad cases saved to: {OUTPUT_DIR}")
