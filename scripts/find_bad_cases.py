#!/usr/bin/env python3
"""
Find training samples where model prediction differs from ground truth (FP/FN).
Output bad cases in the same dataset format for manual annotation.
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
OUTPUT_DIR = "/data-ssd/libo/p100/yolo_utils/dataset/external_inline/datasets_in_line_7k_train_nonempty_bad_cases"
CONF_THRESHOLD = 0.3
MATCH_DISTANCE_PX = 5
BATCH_SIZE = 32
DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"


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
            # We only care about kpt_x and kpt_y which are normalized 0-1
            if len(parts) >= 8:
                x = parts[5] * img_width
                y = parts[6] * img_height
                keypoints.append((x, y))
            # Fallback: if label is only x, y (old format)
            elif len(parts) == 4:
                x = parts[1] * img_width
                y = parts[2] * img_height
                keypoints.append((x, y))

    return keypoints


def match_points(preds: List[Tuple[float, float, float]], gts: List[Tuple[float, float]], distance_thr: float) -> Tuple[int, int]:
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
    model = YOLO(MODEL_PATH)

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

        # Load image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping.")
            continue
        h, w = img.shape[:2]

        # Run prediction
        results = model(
            img_path,
            imgsz=1280,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False
        )[0]

        # Extract predicted keypoints
        pred_points = []
        if results.keypoints is not None and len(results.keypoints) > 0:
            kpts = results.keypoints.xy[0].cpu().numpy()
            confs = results.keypoints.conf[0].cpu().numpy()
            for (x, y), conf in zip(kpts, confs):
                pred_points.append((float(x), float(y), float(conf)))

        # Load ground truth
        gt_points = load_gt_keypoints(label_path, w, h)

        # Match points
        tp, fp, fn = match_points(pred_points, gt_points, MATCH_DISTANCE_PX)

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
                len(pred_points),
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
