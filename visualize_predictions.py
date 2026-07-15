#!/usr/bin/env python3
"""Visualize YOLO11l predictions on ALL validation images with GT comparison."""

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics.models.yolo.structure.utils import associate_ports_to_components, radius_point_nms
from ultralytics.nn.tasks import load_checkpoint


def draw_predictions(img, components, ports, links):
    """Draw predictions: components=3px dots, ports=filled rectangles."""
    vis = img.copy()
    for comp in components:
        x, y, conf = int(comp[0]), int(comp[1]), comp[2]
        cv2.circle(vis, (x, y), 3, (200, 50, 50), -1)
        cv2.putText(vis, f"{conf:.2f}", (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 50, 50), 1)
    for port in ports:
        x, y, conf, pred_cx, pred_cy = int(port[0]), int(port[1]), port[2], int(port[3]), int(port[4])
        cv2.rectangle(vis, (x - 4, y - 4), (x + 4, y + 4), (50, 200, 50), -1)
        cv2.putText(vis, f"{conf:.2f}", (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 200, 50), 1)
        cv2.line(vis, (x, y), (pred_cx, pred_cy), (200, 180, 80), 1)
    if links is not None and len(links) > 0:
        for link in links:
            pi, ci = int(link[0]), int(link[1])
            if pi < len(ports) and ci < len(components):
                cv2.line(vis, (int(ports[pi][0]), int(ports[pi][1])),
                         (int(components[ci][0]), int(components[ci][1])), (0, 255, 80), 2)
    cv2.putText(vis, f"PRED: C:{len(components)} P:{len(ports)} L:{len(links) if links is not None else 0}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def draw_ground_truth(img, gt_components, gt_ports, gt_links):
    """Draw ground truth: components=3px dots, ports=filled rectangles."""
    vis = img.copy()
    for comp in gt_components:
        x, y = int(comp[0]), int(comp[1])
        cv2.circle(vis, (x, y), 3, (50, 50, 200), -1)
    for port in gt_ports:
        x, y = int(port[0]), int(port[1])
        cv2.rectangle(vis, (x - 4, y - 4), (x + 4, y + 4), (50, 200, 50), -1)
    for link in gt_links:
        pi, ci = int(link[0]), int(link[1])
        if pi < len(gt_ports) and ci < len(gt_components):
            cv2.line(vis, (int(gt_ports[pi][0]), int(gt_ports[pi][1])),
                     (int(gt_components[ci][0]), int(gt_components[ci][1])), (0, 200, 80), 2)
    cv2.putText(vis, f"GT: C:{len(gt_components)} P:{len(gt_ports)} L:{len(gt_links)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def main():
    model_path = "runs/structure/runs/structure/train/weights/best.pt"
    print(f"Loading: {model_path}")
    model, ckpt = load_checkpoint(model_path, device="cuda:3")
    model = model.cuda().eval()

    val_img_dir = Path("datasets/device_ports/images/val")
    val_label_dir = Path("datasets/device_ports/labels/val")
    image_paths = sorted(val_img_dir.glob("*.png"))
    total = len(image_paths)
    print(f"Processing all {total} validation images...")

    out_dir = Path("predictions/yolo11l_val_all")
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_match_count = 0
    port_match_count = 0
    link_match_count = 0

    for idx, img_path in enumerate(image_paths):
        img_name = img_path.stem
        label_path = val_label_dir / f"{img_name}.txt"
        img = cv2.imread(str(img_path))
        if img is None or not label_path.exists():
            continue
        h, w = img.shape[:2]

        with open(label_path) as f:
            lines = f.readlines()
        gt_components, gt_ports, gt_links = [], [], []
        for line in lines:
            parts = list(map(float, line.strip().split()))
            kpts_raw = parts[5:]
            nk = len(kpts_raw) // 3
            if nk >= 1 and kpts_raw[2] > 0:
                gt_components.append([kpts_raw[0] * w, kpts_raw[1] * h])
                ci = len(gt_components) - 1
                for i in range(1, nk):
                    if i * 3 + 2 < len(kpts_raw) and kpts_raw[i * 3 + 2] > 0:
                        gt_ports.append([kpts_raw[i * 3] * w, kpts_raw[i * 3 + 1] * h])
                        gt_links.append([len(gt_ports) - 1, ci])

        img_640 = cv2.resize(img, (640, 640))
        t = torch.from_numpy(img_640).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            pd = model(t.cuda())[0][0]

        components = pd["components"]
        ports = pd["ports"]
        components = components[components[:, 2] >= 0.1]
        ports = ports[ports[:, 2] >= 0.1]
        if components.shape[0]:
            components = radius_point_nms(components, 16.0, 300)
        if ports.shape[0]:
            ports = radius_point_nms(ports, 8.0, 300)
        ports, link_tensor = associate_ports_to_components(components, ports)

        sx, sy = w / 640.0, h / 640.0
        components[:, 0] *= sx; components[:, 1] *= sy
        ports[:, 0] *= sx; ports[:, 1] *= sy
        ports[:, 3] *= sx; ports[:, 4] *= sy

        pred_comps = components.cpu().numpy()
        pred_ports = ports.cpu().numpy()
        pred_links = link_tensor[:, :2].cpu().numpy() if link_tensor.numel() else np.zeros((0, 2))

        n_gt_c, n_gt_p, n_gt_l = len(gt_components), len(gt_ports), len(gt_links)
        n_pd_c, n_pd_p, n_pd_l = len(pred_comps), len(pred_ports), len(pred_links)

        if n_pd_c == n_gt_c: comp_match_count += 1
        if n_pd_p >= n_gt_p: port_match_count += 1
        if n_pd_l >= n_gt_l: link_match_count += 1

        vis_pred = draw_predictions(img, pred_comps, pred_ports, pred_links)
        vis_gt = draw_ground_truth(img, gt_components, gt_ports, gt_links)
        combined = np.hstack([vis_gt, vis_pred])

        cv2.putText(combined, "GROUND TRUTH", (w // 2 - 80, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "PREDICTION", (w + w // 2 - 60, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imwrite(str(out_dir / f"{img_name}.png"), combined)

        if (idx + 1) % 50 == 0:
            pc = comp_match_count / (idx + 1) * 100
            pp = port_match_count / (idx + 1) * 100
            pl = link_match_count / (idx + 1) * 100
            print(f"  [{idx+1}/{total}] comp_exact={pc:.0f}% port_ge={pp:.0f}% link_ge={pl:.0f}%")

    pc = comp_match_count / total * 100
    pp = port_match_count / total * 100
    pl = link_match_count / total * 100
    print(f"\nDone! {total} images saved to {out_dir}/")
    print(f"Exact component count match: {comp_match_count}/{total} ({pc:.1f}%)")
    print(f"Port count >= GT:           {port_match_count}/{total} ({pp:.1f}%)")
    print(f"Link count >= GT:           {link_match_count}/{total} ({pl:.1f}%)")


if __name__ == "__main__":
    main()
