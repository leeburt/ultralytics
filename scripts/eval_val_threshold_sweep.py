#!/usr/bin/env python3
"""
在验证集上搜索最优阈值组合（conf, nms_radius, match_distance）。
"""
import cv2
import numpy as np
import torch
import csv
import random
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

MODEL_PATH = "/data-ssd/libo/ultralytics/runs/keypoint/merge_v4_human_checked_lr0005_e200_gpu5/weights/best.pt"
DATA_YAML = "/data-ssd/libo/p100/yolo_utils/dataset/external_inline/merge_config_v2.yaml"
OUTPUT_ROOT = Path("/data-ssd/libo/p100/yolo_utils/dataset/external_inline/val_threshold_sweep")
NUM_SAMPLES = 100
SEED = 42
DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"

random.seed(SEED)


def letterbox(img: np.ndarray, new_shape: int = 1280):
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
    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, new_h / h0, left, top


def radius_point_nms(points: np.ndarray, radius: float, max_det: int = 300):
    if points.shape[0] == 0:
        return points
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


def load_gt_keypoints(label_path: Path, img_width: int, img_height: int):
    keypoints = []
    if not label_path.exists():
        return np.array(keypoints, dtype=np.float32)
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            if len(parts) >= 8:
                x = parts[5] * img_width
                y = parts[6] * img_height
                keypoints.append((x, y))
            elif len(parts) == 4:
                x = parts[1] * img_width
                y = parts[2] * img_height
                keypoints.append((x, y))
    return np.array(keypoints, dtype=np.float32)


def match_points(preds: np.ndarray, gts: np.ndarray, distance_thr: float):
    tp_pred, fp_pred = [], []
    fn_gt = gts.copy()
    preds = preds[np.argsort(preds[:, 2])[::-1]]
    for p in preds:
        px, py = p[:2]
        best_dist, best_idx = float("inf"), -1
        for idx, (gx, gy) in enumerate(fn_gt):
            dist = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
            if dist < best_dist and dist <= distance_thr:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0:
            tp_pred.append((px, py))
            fn_gt = np.delete(fn_gt, best_idx, axis=0)
        else:
            fp_pred.append((px, py))
    return np.array(tp_pred, dtype=np.float32), np.array(fp_pred, dtype=np.float32), fn_gt


def find_val_images(data_yaml: Path):
    import yaml
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    root = Path(data.get("path", "/data-ssd/libo/p100/yolo_utils/dataset"))
    images = []
    for v in data.get("val", []):
        img_dir = root / v
        label_dir = root / v.replace("images/", "labels/")
        if img_dir.exists():
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG", ".PNG"):
                for img_path in img_dir.glob(f"*{ext}"):
                    label_path = label_dir / f"{img_path.stem}.txt"
                    images.append((img_path, label_path))
    return images


def evaluate_on_samples(model, samples, conf, nms_r, match_d):
    total_tp = total_fp = total_fn = 0
    for img_path, label_path in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        img_lb, ratio, dw, dh = letterbox(img, 1280)
        img_input = img_lb[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = torch.from_numpy(img_input).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            raw_pred = model(img_input)[0].cpu().numpy()[0]

        mask = raw_pred[:, 2] >= conf
        pred_points = raw_pred[mask]
        pred_points_processed = []
        if len(pred_points) > 0:
            pred_points = radius_point_nms(pred_points, nms_r)
            for p in pred_points:
                x, y, confv, cls = p
                x_orig = (x - dw) / ratio
                y_orig = (y - dh) / ratio
                if 0 <= x_orig < w and 0 <= y_orig < h:
                    pred_points_processed.append((x_orig, y_orig, confv))
        pred_points_processed = np.array(pred_points_processed, dtype=np.float32)
        gt_points = load_gt_keypoints(label_path, w, h)

        if len(pred_points_processed) == 0:
            tp_c = 0
            fp_c = 0
            fn_c = len(gt_points)
        else:
            tp_pts, fp_pts, fn_pts = match_points(pred_points_processed, gt_points, match_d)
            tp_c = len(tp_pts)
            fp_c = len(fp_pts)
            fn_c = len(fn_pts)
        total_tp += tp_c
        total_fp += fp_c
        total_fn += fn_c

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, total_tp, total_fp, total_fn


def main():
    print(f"加载模型: {MODEL_PATH}")
    yolo_model = YOLO(MODEL_PATH)
    model = yolo_model.model.to(DEVICE)
    model.eval()

    val_images = find_val_images(Path(DATA_YAML))
    print(f"验证集总数: {len(val_images)}")

    if len(val_images) < NUM_SAMPLES:
        selected = val_images
    else:
        selected = random.sample(val_images, NUM_SAMPLES)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    conf_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    nms_radii = [4, 6, 8, 10, 12]
    match_distances = [3, 5, 7, 10]

    results = []
    total_combos = len(conf_thresholds) * len(nms_radii) * len(match_distances)
    pbar = tqdm(total=total_combos, desc="验证集阈值搜索")

    for conf in conf_thresholds:
        for nms_r in nms_radii:
            for match_d in match_distances:
                precision, recall, f1, tp, fp, fn = evaluate_on_samples(model, selected, conf, nms_r, match_d)
                results.append({
                    "conf": conf,
                    "nms_radius": nms_r,
                    "match_distance": match_d,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                })
                pbar.update(1)
    pbar.close()

    results.sort(key=lambda x: x["f1"], reverse=True)

    csv_path = OUTPUT_ROOT / "threshold_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n===== 验证集最优 Top 10 =====")
    for i, r in enumerate(results[:10], 1):
        print(f"{i}. conf={r['conf']}, nms={r['nms_radius']}, match={r['match_distance']} => "
              f"P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f} (TP={r['tp']}, FP={r['fp']}, FN={r['fn']})")

    print(f"\nCSV 保存到: {csv_path}")


if __name__ == "__main__":
    main()
