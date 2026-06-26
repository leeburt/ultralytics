# Keypoint Heatmap Model for Inline Point Detection

## Overview

CenterNet-style keypoint heatmap model for detecting in-line points in document layout images. Single class, single keypoint per instance.

The model predicts a dense heatmap (confidence) and local offsets at 1/4 input resolution, then extracts local maxima with sub-pixel refinement via `topk` + offset regression.

## Architecture

- **Backbone:** YOLO11-n with P2 output (high-resolution branch)
- **Neck:** P2 → P5 FPN with lateral fusion via `KeypointHeatmap` head
- **Head:** 1-channel heatmap + 2-channel offset, fused from P2/P3/P4/P5 features
- **Parameters:** 2.35M
- **GFLOPs:** 9.7 @ 1280

```
Input (3, 1280, 1280)
  → YOLO11 backbone (P2/4, P3/8, P4/16, P5/32)
  → Lateral conv + bilinear upsampling fusion
  → Refine block (3x3 conv × 2)
  → Heatmap (1-ch) + Offset (2-ch)
  → TopK peaks → (x, y, score) points
```

## Training

| Item | Detail |
|------|--------|
| Pretrain | DA external port (line detection pretrained) |
| Resolution | 1536×1536 |
| Epochs | 200 (best: 157, F1@5=0.942) |
| Batch | 20 (2× A800 80GB) |
| Optimizer | AdamW, lr=0.01, lrf=0.01 |
| Warmup | 5 epochs |
| Augmentation | mosaic (off after epoch 15), random affine, HSV |
| Loss | Focal loss (heatmap) + L1 (offset) |
| Config | `ultralytics/cfg/models/11/yolo11-keypoint-heatmap-p2.yaml` |

## Performance (paper_benchmark_100)

| Resolution | F1@3 | F1@5 | F1@10 | P@5 | R@5 |
|-----------:|:----:|:----:|:-----:|:---:|:---:|
| PyTorch @1536 | 0.997 | 0.997 | 0.997 | 0.996 | 0.998 |
| PyTorch @1280 | 0.999 | 0.999 | 0.999 | 1.000 | 0.998 |
| ONNX @1280 | 0.999 | 0.999 | 0.999 | 1.000 | 0.998 |

Best model: `runs/keypoint/merge_v4_yolo26_1536_scratch_e200_gpu34/weights/best.pt`

## ONNX Export

```bash
yolo export \
  model=runs/keypoint/merge_v4_yolo26_1536_scratch_e200_gpu34/weights/best.pt \
  format=onnx imgsz=1280
```

Output: `best.onnx` (9.1 MB)
- Input: `(1, 3, 1280, 1280)` float32, BGR
- Output: `(1, 300, 4)` — per-point `[x, y, score, class]`

## ONNX Inference

```bash
# Single image
python scripts/onnx_infer.py \
  --model best.onnx \
  --image input.jpg \
  --thr 0.30

# Batch directory
python scripts/onnx_infer.py \
  --model best.onnx \
  --dir ./images/ \
  --thr 0.30
```

## Training Quick Start

```bash
yolo keypoint train \
  model=ultralytics/cfg/models/11/yolo11-keypoint-heatmap-p2.yaml \
  pretrained=<pretrained.pt> \
  data=<dataset.yaml> \
  epochs=200 imgsz=1536 batch=20 device=0,1 workers=16 \
  optimizer=AdamW lr0=0.01 lrf=0.01 warmup_epochs=5 \
  close_mosaic=15 patience=50 amp=false deterministic=true \
  hm_radius_add=2 hm_min_radius=0
```

## Validation

```bash
# PyTorch
yolo keypoint val model=best.pt data=<dataset.yaml> imgsz=1280

# ONNX
yolo val task=keypoint model=best.onnx data=<dataset.yaml> imgsz=1280
```
