#!/usr/bin/env python3
"""
为每个bad case生成FP/FN对比可视化图，保存到对应ckt_netlist_json目录下。
绿色=TP（正确预测），黄色=FP（误检），红色=FN（漏检）
"""
import cv2
import numpy as np
import torch
import csv
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# 配置
MODEL_PATH = "/data-ssd/libo/ultralytics/runs/keypoint/merge_v4_yolo26_1536_scratch_e200_gpu34/weights/best.pt"
STATS_CSV = "/data-ssd/libo/p100/yolo_utils/dataset/external_inline/merge_config_v2_bad_cases/bad_cases_stats.csv"
DATA_ROOT = Path("/data-ssd/libo/p100/yolo_utils/dataset")
OUTPUT_DIR = Path("/data-ssd/libo/p100/yolo_utils/dataset/external_inline/merge_config_v2_bad_cases/")
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

def draw_text_box(img: np.ndarray, text: str, pos: tuple, font_scale: float = 0.6, thickness: int = 2):
    """画带背景的文本框"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # 画背景框
    cv2.rectangle(img, (x, y - text_h - baseline), (x + text_w, y + baseline), (0,0,0), -1)
    # 画文本
    cv2.putText(img, text, (x, y), font, font_scale, (255,255,255), thickness)

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
        # 输出路径（存到对应ckt_netlist_json目录下）
        output_dir = (DATA_ROOT / rel_path.replace("images/", "ckt_netlist_json/")).parent
        output_path = output_dir / f"{img_path.stem}_pred_compare.jpg"

        if not img_path.exists():
            print(f"[跳过] 图片不存在: {img_path}")
            continue
        if not label_path.exists():
            print(f"[跳过] 标注不存在: {label_path}")
            continue

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
        if len(pred_points) == 0:
            pred_points_processed = np.array([])
        else:
            pred_points = radius_point_nms(pred_points, NMS_RADIUS_PX)
            # 还原到原图坐标
            pred_points_processed = []
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

        # 画图
        draw_points(vis_img, fn_points, COLOR_FN, radius=8) # 漏检的GT点画大一点
        draw_points(vis_img, fp_points, COLOR_FP, radius=7) # 误检点
        draw_points(vis_img, tp_points, COLOR_TP, radius=6) # 正确预测点

        # 画统计文本
        text = f"TP: {tp:2d} | FP: {fp:2d} | FN: {fn:2d}"
        draw_text_box(vis_img, text, (10, 30), font_scale=0.8)

        # 画图例
        draw_text_box(vis_img, "🟢 TP (Correct)", (10, 60), font_scale=0.5)
        draw_text_box(vis_img, "🟡 FP (False Positive)", (10, 85), font_scale=0.5)
        draw_text_box(vis_img, "🔴 FN (False Negative)", (10, 110), font_scale=0.5)

        # 保存
        cv2.imwrite(str(output_path), vis_img)
        success += 1

    print(f"完成: 成功生成 {success}/{len(bad_cases)} 张对比图")
    print(f"对比图保存在各ckt_netlist_json目录下，命名为 xxx_pred_compare.jpg")

if __name__ == "__main__":
    main()
