#!/usr/bin/env python3
"""
Find training samples where model prediction differs from ground truth (FP/FN).
Fixed version: uses model raw forward + custom postprocess instead of YOLO.predict.
"""

import os
import shutil
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# Configuration
MODEL_PATH = "/data-ssd/libo/ultralytics/runs/keypoint/merge_v4_yolo26_1536_scratch_e200_gpu34/weights/best.pt"
TRAIN_DATASET_PATH = "/data-ssd/libo/p100/yolo_utils/dataset/external_inline/datasets_in_line_7k_train_nonempty"
OUTPUT_DIR = "/data-ssd/libo/p100/yolo_utils/dataset/external_inline/datasets_in_line_7k_train_nonempty_bad_cases_fixed"
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


def main():
    # Setup output directories
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    yolo_model = YOLO(MODEL_PATH)
    model = yolo_model.model.to(DEVICE)
    model.eval()

    # Get all training images
    images_dir = Path(TRAIN_DATASET_PATH) / "images" / "train"
    labels_dir = Path(TRAIN_DATASET_PATH) / "labels" / "train"
    image_paths = list(sorted(images_dir.glob("*.jpg")))
    print(f"Found {len(image_paths)} training images.")

    # Stats file
    stats_file = open(output_dir / "bad_cases_stats.csv", "w", newline="")
    stats_writer = csv.writer(stats_file)
    stats_writer.writerow(["filename", "num_gt", "num_pred", "tp", "fp", "fn"])

    bad_cases_count = 0

    # Process all images
    for img_path in tqdm(image_paths, desc="Processing"):
        # Get label path
        label_path = labels_dir / (img_path.stem + ".txt")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping.")
            continue
        h, w = img.shape[:2]

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
        gt_points = load_gt_keypoints(label_path, w, h)

        # Match points
        tp, fp, fn = match_points(pred_points_processed, gt_points, MATCH_DISTANCE_PX)

        # If there are mismatches, copy to output directory
        if fp > 0 or fn > 0:
            bad_cases_count += 1
            # Copy image
            shutil.copy2(img_path, output_dir / "images" / "train" / img_path.name)
            # Copy label
            if label_path.exists():
                shutil.copy2(label_path, output_dir / "labels" / "train" / label_path.name)
            # Write stats
            stats_writer.writerow([
                img_path.name,
                len(gt_points),
                len(pred_points_processed),
                tp,
                fp,
                fn
            ])

    stats_file.close()
    print(f"\nDone! Found {bad_cases_count} bad cases out of {len(image_paths)} images.")
    print(f"Bad cases saved to {OUTPUT_DIR}")
    print(f"Stats file: {output_dir / 'bad_cases_stats.csv'}")


if __name__ == "__main__":
    main()
