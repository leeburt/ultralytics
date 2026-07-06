#!/usr/bin/env python3
"""
重新生成FP/FN对比图，这次会自动创建目录
"""
import cv2
import numpy as np
import torch
import csv
import json
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# 配置
MODEL_PATH = "/data-ssd/libo/ultralytics/runs/keypoint/merge_v4_yolo26_1536_scratch_e200_gpu34/weights/best.pt"
STATS_CSV = "/data-ssd/libo/p100/yolo_utils/dataset/external_inline/merge_config_v2_bad_cases/bad_cases_stats.csv"
DATA_ROOT = Path("/data-ssd/libo/p100/yolo_utils/dataset")
OUTPUT_ROOT = Path("/data-ssd/libo/p100/yolo_utils/dataset/external_inline/merge_config_v2_bad_cases/visualizations/")
CONF_THRESHOLD = 0.3
NMS_RADIUS_PX = 8
MATCH_DISTANCE_PX = 5
DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"

# 颜色定义
COLOR_TP = (0, 255, 0)   # 绿色，正确预测
COLOR_FP = (0, 255, 255) # 黄色，误检
COLOR_FN = (0, 0, 255)   # 红色，漏检

def letterbox(
    img: np.ndarray,
    new_shape: int = 1280,
    color: tuple = (114, 114, 114),
) -> tuple[np.ndarray, float, int, int]:
    """匹配验证时的预处理逻辑"""
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
    """和推理时保持一致的NMS逻辑"""
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

def load_gt_keypoints(label_path: Path, img_width: int, img_height: int) -> np.ndarray:
    """加载GT关键点"""
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

def match_points(preds: np.ndarray, gts: np.ndarray, distance_thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """匹配预测和GT，返回TP、FP、FN的坐标"""
    tp_pred = [] # 预测中正确的点
    fp_pred = [] # 预测中误检的点
    fn_gt = gts.copy() # GT中漏检的点

    preds = preds[np.argsort(preds[:, 2])[::-1]]  # 按置信度从高到低匹配

    for p in preds:
        px, py, conf = p[:3]
        best_dist = float("inf")
        best_idx = -1
        for idx, (gx, gy) in enumerate(fn_gt):
            dist = ((px - gx)**2 + (py - gy)**2) ** 0.5
            if dist < best_dist and dist <= distance_thr:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0:
            tp_pred.append((px, py))
            fn_gt = np.delete(fn_gt, best_idx, axis=0)
        else:
            fp_pred.append((px, py))
    return np.array(tp_pred, dtype=np.float32), np.array(fp_pred, dtype=np.float32), fn_gt

def draw_points(img: np.ndarray, points: np.ndarray, color: tuple, radius: int = 6, thickness: int = 2):
    """在图上画点"""
    for (x, y) in points:
        cv2.circle(img, (int(round(x)), int(round(y))), radius, color, thickness)
        cv2.circle(img, (int(round(x)), int(round(y))), radius//2, color, -1)

# 移除文本绘制功能，只保留点标注

def main():
    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    yolo_model = YOLO(MODEL_PATH)
    model = yolo_model.model.to(DEVICE)
    model.eval()

    # 读取bad case统计
    with open(STATS_CSV, "r") as f:
        reader = csv.DictReader(f)
        bad_cases = list(reader)
    print(f"待处理bad case: {len(bad_cases)} 个")

    success = 0
    for case in tqdm(bad_cases, desc="生成对比图"):
        rel_path = case["relative_path"]
        tp = int(case["tp"])
        fp = int(case["fp"])
        fn = int(case["fn"])

        # 图片和标注路径
        img_path = DATA_ROOT / rel_path
        label_path = DATA_ROOT / rel_path.replace("images/", "labels/").replace(".jpg", ".txt")
        # 输出路径：统一放到visualizations目录下，保持原有的目录结构
        output_path = OUTPUT_ROOT / rel_path.replace(".jpg", "_pred_compare.jpg")

        if not img_path.exists():
            print(f"[跳过] 图片不存在: {img_path}")
            continue
        if not label_path.exists():
            print(f"[跳过] 标注不存在: {label_path}")
            continue

        # 提前创建输出目录（关键修复！）
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 加载图片
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        vis_img = img.copy()

        # 推理预测
        img_letterbox, ratio, dw, dh = letterbox(img, 1280)
        # BGR->RGB, HWC->CHW, 归一化
        img_input = img_letterbox[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = torch.from_numpy(img_input).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            raw_pred = model(img_input)[0].cpu().numpy()[0]  # [300, 4]

        # 后处理
        mask = raw_pred[:, 2] >= CONF_THRESHOLD
        pred_points = raw_pred[mask]
        pred_points_processed = []
        if len(pred_points) > 0:
            # 应用半径NMS
            pred_points = radius_point_nms(pred_points, NMS_RADIUS_PX)
            # 还原到原图坐标
            for p in pred_points:
                x, y, conf, cls = p
                x_orig = (x - dw) / ratio
                y_orig = (y - dh) / ratio
                if 0 <= x_orig < w and 0 <= y_orig < h:
                    pred_points_processed.append((x_orig, y_orig, conf))
        pred_points_processed = np.array(pred_points_processed, dtype=np.float32)

        # 加载GT
        gt_points = load_gt_keypoints(label_path, w, h)

        # 匹配得到TP/FP/FN
        if len(pred_points_processed) == 0:
            tp_points = np.array([])
            fp_points = np.array([])
            fn_points = gt_points
        else:
            tp_points, fp_points, fn_points = match_points(pred_points_processed, gt_points, MATCH_DISTANCE_PX)

        # 画图 - 只保留颜色点标注，不添加任何文字
        draw_points(vis_img, fn_points, COLOR_FN, radius=8) # 漏检的GT点（红色）画大一点
        draw_points(vis_img, fp_points, COLOR_FP, radius=7) # 误检测点（黄色）
        draw_points(vis_img, tp_points, COLOR_TP, radius=6) # 正确预测点（绿色）

        # 保存对比图到visualizations目录
        cv2.imwrite(str(output_path), vis_img)

        # 同时保存对比图到ckt_netlist_json目录，和JSON文件在一起
        ckt_output_path = Path(str(output_path).replace("visualizations/", "").replace("/images/", "/ckt_netlist_json/"))
        ckt_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ckt_output_path), vis_img)

        # 生成详细预测结果JSON
        json_path = ckt_output_path.with_suffix('.json').name.replace("_pred_compare.json", "_pred_result.json")
        json_path = ckt_output_path.parent / json_path

        # 计算统计指标
        tp_count = len(tp_points)
        fp_count = len(fp_points)
        fn_count = len(fn_points)
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # 构造JSON数据
        result = {
            "image_info": {
                "path": str(img_path),
                "width": w,
                "height": h
            },
            "gt_keypoints": [{"x": float(p[0]), "y": float(p[1])} for p in gt_points],
            "pred_keypoints": [{"x": float(p[0]), "y": float(p[1]), "confidence": float(p[2])} for p in pred_points_processed],
            "tp_points": [{"x": float(p[0]), "y": float(p[1])} for p in tp_points],
            "fp_points": [{"x": float(p[0]), "y": float(p[1])} for p in fp_points],
            "fn_points": [{"x": float(p[0]), "y": float(p[1])} for p in fn_points],
            "statistics": {
                "tp_count": tp_count,
                "fp_count": fp_count,
                "fn_count": fn_count,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            }
        }

        # 保存JSON文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        success += 1

    print(f"完成: 成功生成 {success}/{len(bad_cases)} 张对比图")
    print(f"对比图统一保存在 {OUTPUT_ROOT} 目录下，保持原数据集的目录结构")

if __name__ == "__main__":
    main()
