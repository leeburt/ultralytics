#!/usr/bin/env python3
"""
随机抽取训练集200个样本，评估模型在训练集上的拟合情况。
"""
import cv2
import numpy as np
import torch
import json
import csv
import random
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# 配置
MODEL_PATH = "/data-ssd/libo/ultralytics/runs/keypoint/merge_v4_human_checked_lr0005_e200_gpu5/weights/best.pt"
DATA_YAML = "/data-ssd/libo/p100/yolo_utils/dataset/external_inline/merge_config_v2.yaml"
OUTPUT_ROOT = Path("/data-ssd/libo/p100/yolo_utils/dataset/external_inline/train_random_200_analysis")
NUM_SAMPLES = 200
SEED = 42

CONF_THRESHOLD = 0.3
NMS_RADIUS_PX = 8
MATCH_DISTANCE_PX = 5
DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"

COLOR_TP = (0, 255, 0)
COLOR_FP = (0, 255, 255)
COLOR_FN = (0, 0, 255)

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


def draw_points(img: np.ndarray, points: np.ndarray, color: tuple, radius: int = 6):
    for (x, y) in points:
        cv2.circle(img, (int(round(x)), int(round(y))), radius, color, 2)
        cv2.circle(img, (int(round(x)), int(round(y))), radius // 2, color, -1)


def find_train_images(data_yaml: Path):
    import yaml
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    root = Path(data.get("path", "/data-ssd/libo/p100/yolo_utils/dataset"))
    images = []
    for t in data.get("train", []):
        img_dir = root / t
        label_dir = root / t.replace("images/", "labels/")
        if img_dir.exists():
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG", ".PNG"):
                for img_path in img_dir.glob(f"*{ext}"):
                    label_path = label_dir / f"{img_path.stem}.txt"
                    images.append((img_path, label_path))
    return images


def main():
    print(f"加载模型: {MODEL_PATH}")
    yolo_model = YOLO(MODEL_PATH)
    model = yolo_model.model.to(DEVICE)
    model.eval()

    train_images = find_train_images(Path(DATA_YAML))
    print(f"训练集总数: {len(train_images)}")

    if len(train_images) < NUM_SAMPLES:
        selected = train_images
    else:
        selected = random.sample(train_images, NUM_SAMPLES)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    total_tp = total_fp = total_fn = 0

    for idx, (img_path, label_path) in enumerate(tqdm(selected, desc="分析训练集拟合")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        vis_img = img.copy()

        # 推理
        img_lb, ratio, dw, dh = letterbox(img, 1280)
        img_input = img_lb[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = torch.from_numpy(img_input).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            raw_pred = model(img_input)[0].cpu().numpy()[0]

        mask = raw_pred[:, 2] >= CONF_THRESHOLD
        pred_points = raw_pred[mask]
        pred_points_processed = []
        if len(pred_points) > 0:
            pred_points = radius_point_nms(pred_points, NMS_RADIUS_PX)
            for p in pred_points:
                x, y, conf, cls = p
                x_orig = (x - dw) / ratio
                y_orig = (y - dh) / ratio
                if 0 <= x_orig < w and 0 <= y_orig < h:
                    pred_points_processed.append((x_orig, y_orig, conf))
        pred_points_processed = np.array(pred_points_processed, dtype=np.float32)

        # 加载GT
        gt_points = load_gt_keypoints(label_path, w, h)

        # 匹配
        if len(pred_points_processed) == 0:
            tp_points = np.array([])
            fp_points = np.array([])
            fn_points = gt_points
        else:
            tp_points, fp_points, fn_points = match_points(pred_points_processed, gt_points, MATCH_DISTANCE_PX)

        tp_c = len(tp_points)
        fp_c = len(fp_points)
        fn_c = len(fn_points)
        total_tp += tp_c
        total_fp += fp_c
        total_fn += fn_c

        is_bad = fp_c > 0 or fn_c > 0

        # 画图
        draw_points(vis_img, fn_points, COLOR_FN, radius=8)
        draw_points(vis_img, fp_points, COLOR_FP, radius=7)
        draw_points(vis_img, tp_points, COLOR_TP, radius=6)

        text = f"TP:{tp_c} FP:{fp_c} FN:{fn_c}"
        cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out_name = f"{idx:03d}_{img_path.stem}_TP{tp_c}_FP{fp_c}_FN{fn_c}.jpg"
        out_path = OUTPUT_ROOT / out_name
        cv2.imwrite(str(out_path), vis_img)

        results.append({
            "idx": idx,
            "image": str(img_path),
            "label": str(label_path),
            "width": w,
            "height": h,
            "gt_count": len(gt_points),
            "pred_count": len(pred_points_processed),
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "precision": round(tp_c / (tp_c + fp_c), 4) if (tp_c + fp_c) > 0 else 0,
            "recall": round(tp_c / (tp_c + fn_c), 4) if (tp_c + fn_c) > 0 else 0,
            "f1": round(2 * tp_c / (2 * tp_c + fp_c + fn_c), 4) if (2 * tp_c + fp_c + fn_c) > 0 else 0,
            "is_bad": is_bad,
            "vis_path": str(out_path),
        })

    # 保存CSV
    csv_path = OUTPUT_ROOT / "train_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    bad_count = sum(1 for r in results if r["is_bad"])
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n训练集分析完成！")
    print(f"样本数: {len(results)}")
    print(f"Bad case: {bad_count}/{len(results)} ({bad_count/len(results)*100:.1f}%)")
    print(f"总TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"可视化图保存到: {OUTPUT_ROOT}")
    print(f"统计CSV: {csv_path}")


if __name__ == "__main__":
    main()
