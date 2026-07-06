#!/usr/bin/env python3
"""
评估 paper_benchmark_100 数据集，尝试多组阈值。
"""
import cv2
import numpy as np
import torch
import csv
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

MODEL_PATH = "/data-ssd/libo/ultralytics/runs/keypoint/merge_v4_human_checked_lr0005_e200_gpu5/weights/best.pt"
IMG_DIR = Path("/data-ssd/libo/p100/yolo_utils/dataset/external_inline/paper_benchmark_100/images/val")
LABEL_DIR = Path("/data-ssd/libo/p100/yolo_utils/dataset/external_inline/paper_benchmark_100/labels/val")
OUTPUT_ROOT = Path("/data-ssd/libo/p100/yolo_utils/dataset/external_inline/paper_benchmark_100_analysis")
DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"


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


def evaluate_dataset(model, img_dir, label_dir, conf, nms_r, match_d, draw_vis=False, vis_dir=None):
    total_tp = total_fp = total_fn = 0
    results = []
    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpeg"))
    for img_path in tqdm(img_paths, desc=f"conf={conf},nms={nms_r},match={match_d}"):
        label_path = label_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        vis_img = img.copy() if draw_vis else None

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
            tp_pts = np.array([])
            fp_pts = np.array([])
            fn_pts = gt_points
        else:
            tp_pts, fp_pts, fn_pts = match_points(pred_points_processed, gt_points, match_d)
            tp_c = len(tp_pts)
            fp_c = len(fp_pts)
            fn_c = len(fn_pts)

        total_tp += tp_c
        total_fp += fp_c
        total_fn += fn_c

        is_bad = fp_c > 0 or fn_c > 0

        if draw_vis and vis_dir:
            COLOR_TP = (0, 255, 0)
            COLOR_FP = (0, 255, 255)
            COLOR_FN = (0, 0, 255)
            for (x, y) in fn_pts:
                cv2.circle(vis_img, (int(round(x)), int(round(y))), 8, COLOR_FN, 2)
                cv2.circle(vis_img, (int(round(x)), int(round(y))), 4, COLOR_FN, -1)
            for (x, y) in fp_pts:
                cv2.circle(vis_img, (int(round(x)), int(round(y))), 7, COLOR_FP, 2)
                cv2.circle(vis_img, (int(round(x)), int(round(y))), 3, COLOR_FP, -1)
            for (x, y) in tp_pts:
                cv2.circle(vis_img, (int(round(x)), int(round(y))), 6, COLOR_TP, 2)
                cv2.circle(vis_img, (int(round(x)), int(round(y))), 3, COLOR_TP, -1)
            text = f"TP:{tp_c} FP:{fp_c} FN:{fn_c}"
            cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out_path = vis_dir / f"{img_path.stem}_TP{tp_c}_FP{fp_c}_FN{fn_c}.jpg"
            cv2.imwrite(str(out_path), vis_img)

        results.append({
            "image": str(img_path),
            "gt_count": len(gt_points),
            "pred_count": len(pred_points_processed),
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "is_bad": is_bad,
        })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, total_tp, total_fp, total_fn, results


def main():
    print(f"加载模型: {MODEL_PATH}")
    yolo_model = YOLO(MODEL_PATH)
    model = yolo_model.model.to(DEVICE)
    model.eval()

    img_paths = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png")) + sorted(IMG_DIR.glob("*.jpeg"))
    print(f"图像总数: {len(img_paths)}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 测试多组阈值
    configs = [
        ("default", 0.3, 8, 5),
        ("train_best", 0.4, 8, 7),
        ("val_best", 0.25, 8, 10),
        ("paper_onnx_aligned", 0.25, 8, 10),
    ]

    summary = []
    for name, conf, nms_r, match_d in configs:
        vis_dir = OUTPUT_ROOT / name
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n>>> 评估配置: {name} (conf={conf}, nms={nms_r}, match={match_d})")
        precision, recall, f1, tp, fp, fn, results = evaluate_dataset(
            model, IMG_DIR, LABEL_DIR, conf, nms_r, match_d,
            draw_vis=True, vis_dir=vis_dir
        )
        bad_count = sum(1 for r in results if r["is_bad"])
        print(f"  Bad case: {bad_count}/{len(results)} ({bad_count/len(results)*100:.1f}%)")
        print(f"  TP={tp}, FP={fp}, FN={fn}")
        print(f"  P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

        # 保存CSV
        csv_path = vis_dir / "stats.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        summary.append({
            "name": name,
            "conf": conf,
            "nms_radius": nms_r,
            "match_distance": match_d,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "bad_count": bad_count,
            "total": len(results),
        })

    # 保存汇总
    summary_path = OUTPUT_ROOT / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print("\n===== 汇总 =====")
    for s in summary:
        print(f"{s['name']:20s} conf={s['conf']} match={s['match_distance']} => "
              f"P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f} "
              f"(bad={s['bad_count']}/{s['total']})")
    print(f"\n结果保存到: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
