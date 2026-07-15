"""Convert device_ports dataset to YOLO structure format."""

from pathlib import Path
import json
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def convert_dataset(
    src_json="/data-ssd/libo/p100/StructureDetector/zujian_det/datasets/device_ports_train_4.23.json",
    src_img_dir="/data-ssd/libo/p100/StructureDetector/zujian_det/datasets/train_data_260423",
    dst_root="/data-ssd/libo/ultralytics/datasets/device_ports",
    train_ratio=0.9,
):
    """Convert device_ports JSON dataset to YOLO structure format."""
    dst_root = Path(dst_root)
    (dst_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dst_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dst_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dst_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    with open(src_json, "r") as f:
        data = json.load(f)

    # Split dataset
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    split = int(len(data) * train_ratio)
    train_indices = indices[:split]
    val_indices = indices[split:]

    def process_sample(idx, sample, split_name):
        """Process one sample and save to split."""
        if not sample.get("images"):
            return
        img_path = Path(sample["images"][0])
        if not img_path.exists():
            # Try to find in alternative locations
            alt_path = Path(src_img_dir) / img_path.name
            if alt_path.exists():
                img_path = alt_path
            else:
                return

        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            return
        h, w = img.shape[:2]

        # Parse annotations
        conversations = sample.get("conversations", [])
        if len(conversations) < 2:
            return
        gt_text = conversations[1].get("value", "")

        # Extract JSON from GT text
        try:
            if gt_text.startswith("{"):
                gt_json = json.loads(gt_text)
            else:
                # Find JSON substring
                start = gt_text.find("{")
                end = gt_text.rfind("}")
                if start >= 0 and end > start:
                    gt_json = json.loads(gt_text[start:end+1])
                else:
                    return
        except json.JSONDecodeError:
            return

        device_ports = gt_json.get("device_ports", {})
        if not device_ports:
            return

        # Create labels: one object per component, with keypoints [center, port1, port2, ...]
        labels = []
        for comp_id, comp_data in device_ports.items():
            # Component bbox (optional, used for Gaussian radius)
            bbox = comp_data.get("bbox", [])
            if len(bbox) == 4 and bbox[0] == "top_left":
                # Format: ["top_left", [x1, y1], "bottom_right", [x2, y2]]
                x1, y1 = bbox[1]
                x2, y2 = bbox[3]
                # Convert to normalized xywh
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                bx = (x1 + x2) / 2 / w
                by = (y1 + y2) / 2 / h
            else:
                bx, by, bw, bh = 0.5, 0.5, 0.0, 0.0  # dummy if no bbox

            # Initialize keypoints: [25, 3] = 1 center + 24 ports
            keypoints = np.zeros((25, 3), dtype=np.float32)

            # Component center (compute from bbox if available, or use port centroid)
            ports = comp_data.get("Ports", {})
            port_coords = []
            for port_name, port_xy in ports.items():
                if len(port_xy) == 2:
                    port_coords.append(port_xy)

            if len(port_coords) > 0:
                comp_center = np.mean(port_coords, axis=0)
            elif len(bbox) == 4 and bbox[0] == "top_left":
                comp_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            else:
                continue  # skip if no info

            # Component center as keypoint 0
            keypoints[0, 0] = comp_center[0] / w  # normalized x
            keypoints[0, 1] = comp_center[1] / h  # normalized y
            keypoints[0, 2] = 2.0  # visible

            # Ports as keypoints 1-24
            for i, (port_name, port_xy) in enumerate(list(ports.items())[:24]):
                if len(port_xy) == 2:
                    keypoints[1 + i, 0] = port_xy[0] / w
                    keypoints[1 + i, 1] = port_xy[1] / h
                    keypoints[1 + i, 2] = 2.0  # visible

            # Format: class_id, x, y, w, h, kpt0_x, kpt0_y, kpt0_v, kpt1_x, kpt1_y, kpt1_v, ...
            label_line = [0, bx, by, bw, bh]
            for kpt in keypoints:
                label_line.extend([kpt[0], kpt[1], kpt[2]])

            labels.append(label_line)

        # Save label file
        if labels:
            # Copy image only when we have valid labels
            dst_img = dst_root / "images" / split_name / f"{idx:06d}.png"
            shutil.copy2(img_path, dst_img)

            dst_label = dst_root / "labels" / split_name / f"{idx:06d}.txt"
            with open(dst_label, "w") as f:
                for label_line in labels:
                    f.write(" ".join([f"{x:.6f}" for x in label_line]) + "\n")
            return True
        return False

    # Process all samples
    train_count = 0
    val_count = 0
    for i, idx in enumerate(tqdm(train_indices, desc="Converting train")):
        if process_sample(idx, data[idx], "train"):
            train_count += 1
    for i, idx in enumerate(tqdm(val_indices, desc="Converting val")):
        if process_sample(idx, data[idx], "val"):
            val_count += 1

    print(f"Converted {train_count} train, {val_count} val samples")


if __name__ == "__main__":
    convert_dataset()
