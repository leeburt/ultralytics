from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO


IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(data_yaml: Path) -> list[Path]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = Path(data.get("path", data_yaml.parent))
    val = data["val"]
    val_dirs = val if isinstance(val, list) else [val]
    images: list[Path] = []
    for item in val_dirs:
        directory = Path(item)
        if not directory.is_absolute():
            directory = root / directory
        if directory.is_file():
            with directory.open("r", encoding="utf-8") as f:
                images.extend(Path(line.strip()) for line in f if line.strip())
        else:
            images.extend(p for p in directory.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)
    return sorted(dict.fromkeys(p.resolve() for p in images))


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_gt_points(image_path: Path, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    path = label_path(image_path)
    points = []
    if not path.exists():
        return np.zeros((0, 2), dtype=np.float32)
    for line in path.read_text(encoding="utf-8").splitlines():
        vals = line.strip().split()
        if len(vals) < 7:
            continue
        nums = [float(x) for x in vals]
        kpts = nums[5:]
        for i in range(0, len(kpts), 3):
            if i + 2 >= len(kpts):
                break
            x, y, visible = kpts[i : i + 3]
            if visible > 0:
                points.append((x * w, y * h))
    return np.asarray(points, dtype=np.float32)


def result_points(result) -> np.ndarray:
    if result.keypoints is None or result.keypoints.data is None:
        return np.zeros((0, 3), dtype=np.float32)
    data = result.keypoints.data.detach().cpu().numpy()
    if data.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return data.reshape(-1, data.shape[-1])[:, :3].astype(np.float32)


def match_points(pred: np.ndarray, gt: np.ndarray, threshold: float):
    if len(pred) == 0 or len(gt) == 0:
        return [], list(range(len(pred))), list(range(len(gt)))
    dist = np.linalg.norm(pred[:, None, :2] - gt[None, :, :2], axis=2)
    candidates = []
    for pi in range(dist.shape[0]):
        gi = int(dist[pi].argmin())
        d = float(dist[pi, gi])
        if d <= threshold:
            candidates.append((d, pi, gi))
    candidates.sort()
    used_p, used_g, matches = set(), set(), []
    for d, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, d))
    fp = [i for i in range(len(pred)) if i not in used_p]
    fn = [i for i in range(len(gt)) if i not in used_g]
    return matches, fp, fn


def draw_case(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, matches: list, fp: list, fn: list) -> np.ndarray:
    vis = image.copy()
    matched_p = {m[0] for m in matches}
    matched_g = {m[1] for m in matches}

    for i, (x, y) in enumerate(gt):
        color = (0, 220, 0) if i in matched_g else (0, 0, 255)
        cv2.circle(vis, (round(x), round(y)), 8, color, 2)
        if i not in matched_g:
            cv2.putText(vis, "FN", (round(x) + 8, round(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    for i, row in enumerate(pred):
        x, y, conf = row[:3]
        color = (255, 180, 0) if i in matched_p else (255, 0, 255)
        cv2.drawMarker(vis, (round(x), round(y)), color, cv2.MARKER_CROSS, 16, 2)
        text = f"{conf:.2f}" if i in matched_p else f"FP {conf:.2f}"
        cv2.putText(vis, text, (round(x) + 8, round(y) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(vis, "GT green/red(FN), Pred cyan/magenta(FP)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 3)
    cv2.putText(vis, "GT green/red(FN), Pred cyan/magenta(FP)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    return vis


def side_by_side(original: np.ndarray, annotated: np.ndarray) -> np.ndarray:
    left = original.copy()
    right = annotated.copy()
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right
    cv2.putText(canvas, "Original", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3)
    cv2.putText(canvas, "Original", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
    cv2.putText(
        canvas,
        "Prediction / GT",
        (left.shape[1] + 12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        3,
    )
    cv2.putText(
        canvas,
        "Prediction / GT",
        (left.shape[1] + 12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        1,
    )
    return canvas


def save_case(out_dir: Path, image_path: Path, image: np.ndarray, gt: np.ndarray, pred: np.ndarray, matches, fp, fn) -> Path:
    annotated = draw_case(image, gt, pred, matches, fp, fn)
    panel = side_by_side(image, annotated)
    name = f"{image_path.parent.name}_{image_path.stem}_gt{len(gt)}_pred{len(pred)}_fp{len(fp)}_fn{len(fn)}.jpg"
    out_path = out_dir / name
    cv2.imwrite(str(out_path), panel)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--conf", default=0.2, type=float)
    parser.add_argument("--imgsz", default=1280, type=int)
    parser.add_argument("--max-cases", default=50, type=int)
    parser.add_argument("--match-threshold", default=0.02, type=float, help="Normalized by max(original height, width).")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fp_dir = args.out / "fp_cases"
    fn_dir = args.out / "fn_cases"
    both_dir = args.out / "fp_fn_cases"
    for directory in (fp_dir, fn_dir, both_dir):
        directory.mkdir(parents=True, exist_ok=True)

    images = iter_images(args.data)
    model = YOLO(str(args.weights), task="keypoint")
    rows = []
    fp_saved = fn_saved = both_saved = 0

    for result in model.predict(source=[str(p) for p in images], imgsz=args.imgsz, conf=args.conf, batch=16, stream=True, verbose=False):
        image_path = Path(result.path)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        gt = read_gt_points(image_path, image.shape[:2])
        pred = result_points(result)
        threshold = args.match_threshold * max(image.shape[:2])
        matches, fp, fn = match_points(pred, gt, threshold)
        rows.append(
            {
                "image": str(image_path),
                "gt": len(gt),
                "pred": len(pred),
                "tp": len(matches),
                "fp": len(fp),
                "fn": len(fn),
                "threshold_px": round(threshold, 3),
            }
        )
        if fp and fn and both_saved < args.max_cases:
            save_case(both_dir, image_path, image, gt, pred, matches, fp, fn)
            both_saved += 1
        if fp and fp_saved < args.max_cases:
            save_case(fp_dir, image_path, image, gt, pred, matches, fp, fn)
            fp_saved += 1
        if fn and fn_saved < args.max_cases:
            save_case(fn_dir, image_path, image, gt, pred, matches, fp, fn)
            fn_saved += 1

    with (args.out / "case_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gt", "pred", "tp", "fp", "fn", "threshold_px"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"images={len(rows)} fp_cases={fp_saved} fn_cases={fn_saved} fp_fn_cases={both_saved}")
    print(args.out)


if __name__ == "__main__":
    main()
