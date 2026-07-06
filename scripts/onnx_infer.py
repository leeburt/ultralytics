#!/usr/bin/env python3
"""
ONNX inference for keypoint-only CenterNet-style heatmap model.

Input:  (1, 3, 1280, 1280) float32, RGB, normalized to [0, 1]
Output: (1, 300, 4) [x, y, score, class]

Usage:
    python onnx_infer.py --model best.onnx --image input.jpg --thr 0.30
    python onnx_infer.py --model best.onnx --dir ./images/ --thr 0.30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(
    img: np.ndarray,
    new_shape: int = 1280,
    color: tuple = (114, 114, 114),
) -> tuple[np.ndarray, float, int, int]:
    """Match Ultralytics validation resize + letterbox for a fixed square ONNX input."""
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

    # Ultralytics ops.scale_coords uses ratio_pad[0][0] for both axes.
    gain = new_h / h0
    return padded, gain, left, top


def preprocess(image_path: str | Path, imgsz: int = 1280) -> tuple[np.ndarray, float, int, int, np.ndarray]:
    """Load and preprocess image: BGR letterbox -> RGB CHW normalized float32 tensor."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig_img = img.copy()
    img, ratio, dw, dh = letterbox(img, imgsz)
    # HWC BGR -> CHW RGB, uint8 -> normalized float32
    img = img[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img, ratio, dw, dh, orig_img


def radius_point_nms(points: np.ndarray, radius: float, max_det: int = 300) -> np.ndarray:
    """Keep highest confidence point within each local radius (numpy version)."""
    if points.shape[0] == 0:
        return points
    # Sort by score descending
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


def postprocess(
    output: np.ndarray,
    ratio: float,
    dw: int,
    dh: int,
    orig_h: int,
    orig_w: int,
    conf_thr: float = 0.30,
    nms_radius: float = 8.0,
) -> list[dict]:
    """Decode ONNX output points back to original image coordinates.

    Points that map outside the valid image region (letterbox padding area) are discarded.
    """
    # Filter by confidence
    mask = output[0, :, 2] >= conf_thr
    pts = output[0, mask]
    if pts.shape[0] == 0:
        return []
    pts = radius_point_nms(pts, nms_radius)
    points = []
    for point in pts:
        x, y, score, cls = point
        # Undo letterbox padding and scale
        x_orig = (x - dw) / ratio
        y_orig = (y - dh) / ratio
        # Discard points that fall in padding area (outside valid image bounds)
        if x_orig < 0 or x_orig >= orig_w or y_orig < 0 or y_orig >= orig_h:
            continue
        points.append({"x": float(x_orig), "y": float(y_orig), "score": float(score), "class": int(cls)})
    return points


def draw_points(img: np.ndarray, points: list[dict]) -> np.ndarray:
    """Draw predicted points on image."""
    vis = img.copy()
    for p in points:
        x, y = int(p["x"]), int(p["y"])
        cv2.circle(vis, (x, y), 6, (0, 255, 0), 2)
        cv2.putText(vis, f"{p['score']:.2f}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis


def load_model(model_path: str) -> ort.InferenceSession:
    """Load ONNX model with CUDA if available, otherwise CPU."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"[Model]  Input : {session.get_inputs()[0].name}  shape={session.get_inputs()[0].shape}")
    print(f"[Model]  Output: {session.get_outputs()[0].name}  shape={session.get_outputs()[0].shape}")
    return session


def run_inference(
    model_path: str,
    image_path: str,
    imgsz: int = 1280,
    conf_thr: float = 0.30,
    save_vis: bool = True,
) -> list[dict]:
    """Run end-to-end inference on a single image."""
    model = load_model(model_path)
    tensor, ratio, dw, dh, orig_img = preprocess(image_path, imgsz)
    input_name = model.get_inputs()[0].name
    output = model.run(None, {input_name: tensor})[0]
    points = postprocess(output, ratio, dw, dh, orig_img.shape[0], orig_img.shape[1], conf_thr)
    if save_vis:
        out_path = Path(image_path).stem + "_pred.jpg"
        cv2.imwrite(out_path, draw_points(orig_img, points))
        print(f"[Visualization] saved to {out_path}")
    print(f"[Results] {len(points)} points detected (conf_thr={conf_thr})")
    for p in sorted(points, key=lambda x: x["score"], reverse=True):
        print(f"  x={p['x']:.1f}  y={p['y']:.1f}  score={p['score']:.4f}  class={p['class']}")
    return points


def main():
    parser = argparse.ArgumentParser(description="Keypoint heatmap ONNX inference")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input size (square)")
    parser.add_argument("--thr", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--no-vis", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    if args.dir:
        image_dir = Path(args.dir)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])
        if not images:
            raise FileNotFoundError(f"No images found in {args.dir}")
        model = load_model(args.model)
        total = 0
        for img_path in images:
            tensor, ratio, dw, dh, orig_img = preprocess(img_path, args.imgsz)
            output = model.run(None, {model.get_inputs()[0].name: tensor})[0]
            points = postprocess(output, ratio, dw, dh, orig_img.shape[0], orig_img.shape[1], args.thr)
            total += len(points)
            if not args.no_vis:
                out_path = img_path.stem + "_pred.jpg"
                cv2.imwrite(out_path, draw_points(orig_img, points))
            print(f"{img_path.name}: {len(points)} points")
        print(f"Total: {total} points across {len(images)} images")
    elif args.image:
        run_inference(args.model, args.image, args.imgsz, args.thr, save_vis=not args.no_vis)
    else:
        parser.error("Either --image or --dir is required")


if __name__ == "__main__":
    main()
