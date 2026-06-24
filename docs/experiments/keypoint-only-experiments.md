# Keypoint-only Experiments

## Objective

Build and compare YOLO-based keypoint-only models for `in_line` point localization, without using the original YOLOPose bbox + pose head as the final model structure.

The target dataset is:

```text
/data-ssd/libo/p100/yolo_utils/dataset/external_inline/merge_config_yolo26_0609.yaml
```

The main pretrained checkpoint is:

```text
/data-ssd/libo/p100/ci2n/models/da_external_port_20260205.pt
```

This checkpoint is a YOLO pose model. The experiments reuse compatible backbone/neck weights and reinitialize incompatible keypoint-only heads.

## Common Training Setup

Unless noted otherwise:

- Environment: `yolo_latest`
- Input size: `1280`
- Batch size: `16`
- Epochs: `100`
- AMP: disabled
- Dataset `kpt_shape`: `[1, 3]`
- Task name: `keypoint`
- Main metrics:
  - `precision(K)`: matched predictions / all predicted points
  - `recall(K)`: matched GT points / all GT points
  - `mean_dist(K)`: normalized distance over matched points

Current branch for the strict CenterNet-loss and multi-scale-fusion work:

```text
exp/keypoint-centernet-strict-loss
```

## Experiment A: Direct Regression Baseline

### Purpose

Create the simplest pure keypoint-only baseline:

- no bbox branch
- no class branch
- each dense anchor predicts `(x, y, confidence)`

This validates whether the existing YOLO backbone/neck can learn point localization without YOLOPose instance boxes.

### Model Design

Config:

```text
ultralytics/cfg/models/11/yolo11-keypoint.yaml
```

Head:

```text
ultralytics/nn/modules/head.py::KeypointDetect
```

Output:

```text
(x, y, conf, keypoint_index)
```

Training loss:

```text
ultralytics/utils/loss.py::KeypointOnlyLoss
```

The loss assigns each visible GT keypoint to the nearest anchor. Assigned anchors are positive; all other anchors are negative. Location is trained with SmoothL1 on decoded image-space coordinates. Confidence is trained with BCE.

### Known Behavior

The model learns point locations quickly, but produces many dense duplicate candidates. This causes high recall and low precision when `max_det=300`.

Snapshot:

```text
run: /data-ssd/libo/ultralytics/runs/keypoint/merge_yolo26_0609_1280_da_pretrain
tmux: kp1280
epoch 8:
  train/kpt_loss: 1.16418
  train/kobj_loss: 0.15496
  precision(K): 0.05075
  recall(K): 0.98606
  mean_dist(K): 0.00119
```

## Experiment B: Direct Regression + Point NMS

### Purpose

Reduce duplicate dense predictions from Experiment A by adding point-level local suppression.

This tests whether the low precision is mainly a postprocessing issue rather than a model-learning issue.

### Model Design

Base model is the same as Experiment A.

Additional postprocess:

```text
ultralytics/models/yolo/keypoint/utils.py::radius_point_nms
```

Validator and predictor call point-NMS:

```text
ultralytics/models/yolo/keypoint/val.py
ultralytics/models/yolo/keypoint/predict.py
```

Current radius:

```text
8 px
```

Training-side assignment was also adjusted so that if multiple GT points of the same keypoint type map to the same nearest anchor, later points are assigned to the nearest unused anchor. All unassigned anchors remain negative.

### Known Behavior

The change is structurally correct, but radius `8 px` is not enough to eliminate dense duplicates in early epochs.

Snapshot:

```text
run: /data-ssd/libo/ultralytics/runs/keypoint/merge_yolo26_0609_1280_da_pretrain_pnms_gpu7
tmux: kp1280_pnms_gpu7
epoch 3:
  train/kpt_loss: 1.41006
  train/kobj_loss: 0.23655
  precision(K): 0.04958
  recall(K): 0.95387
  mean_dist(K): 0.00191
```

### Follow-up Checks

- Try larger point-NMS radii: `16`, `24`, `32`.
- Track precision/recall tradeoff against `conf` threshold.
- Reduce `max_det` for keypoint task if each image has a small expected number of points.

## Experiment C: CenterNet-style Heatmap Baseline

### Purpose

Replace dense coordinate regression with heatmap-based point modeling.

The goal is to make the model learn local peaks directly, rather than relying on a dense anchor confidence head plus post-hoc suppression.

### Model Design

Config:

```text
ultralytics/cfg/models/11/yolo11-keypoint-heatmap.yaml
```

Head:

```text
ultralytics/nn/modules/head.py::KeypointHeatmap
```

Current head uses P3/8 only:

```text
hm:     (B, K, H, W)
offset: (B, K * 2, H, W)
```

Decode:

1. apply sigmoid to heatmap logits
2. keep 3x3 local maxima
3. top-k candidate selection
4. add local offset
5. multiply by stride to image coordinates

Loss:

```text
ultralytics/utils/loss.py::KeypointHeatmapLoss
```

Current loss components:

- Gaussian heatmap target
- CenterNet-style modified focal loss for heatmap
- SmoothL1 offset loss at center cells only

Current differences from original CenterNet:

- no wh/size branch, because this is point-only
- Gaussian radius is fixed at `2` feature cells
- offset uses `sigmoid()` and SmoothL1
- original CenterNet usually uses direct L1 offset regression
- single-scale P3/8 heatmap only

### Current Run

```text
run: /data-ssd/libo/ultralytics/runs/keypoint/merge_yolo26_0609_1280_da_pretrain_heatmap_gpu6
tmux: kp1280_heatmap_gpu6
GPU: physical 6, UUID GPU-3251...
pretrained transfer: 894/922
```

Snapshot:

```text
epoch 1:
  train/hm_loss: 0.60516
  train/off_loss: 0.01582
  precision(K): 0.15668
  recall(K): 0.99616
  mean_dist(K): 0.00173
  val/hm_loss: 7.30661
  val/off_loss: 0.0064
```

### Follow-up Checks

- Continue training and compare epoch 3-5 metrics with Experiment A/B.
- If precision is still low, inspect predicted heatmap peaks and candidate counts per image.
- Consider making `topk`, `conf`, and `max_det` stricter for heatmap decoding.
- Align offset more closely with CenterNet:
  - remove sigmoid from offset
  - use L1 loss
  - optionally initialize offset bias near `0.5`
- Test different Gaussian radius values: `1`, `2`, `3`, `4`.

## Experiment D: Strict CenterNet Loss + Multi-scale Fusion

### Purpose

Align the heatmap version more closely with CenterNet, and add feature fusion so point heatmaps use both fine spatial detail and deeper semantic context.

This is the active branch:

```text
exp/keypoint-centernet-strict-loss
```

### Model Design

Config:

```text
ultralytics/cfg/models/11/yolo11-keypoint-heatmap.yaml
```

The heatmap head now consumes three neck outputs:

```text
P3: layer 16, stride 8
P4: layer 19, stride 16
P5: layer 22, stride 32
```

Fusion:

```text
1. project P3/P4/P5 with 1x1 Conv to the same channel width
2. upsample P4 and P5 to P3 resolution
3. sum the projected features
4. refine with two 3x3 Conv blocks
5. predict heatmap logits and raw local offsets on the fused P3/8 map
```

YAML endpoint:

```text
[[16, 19, 22], 1, KeypointHeatmap, [kpt_shape, 300]]
```

Loss changes versus Experiment C:

- heatmap remains CenterNet modified focal loss
- Gaussian radius is computed from bbox size on the output heatmap, using the CenterNet radius formula
- offset branch predicts raw offsets directly, without sigmoid
- offset loss is masked L1 at positive center cells only
- no width/height branch is added, because this remains point-only

Decode path:

```text
sigmoid heatmap -> 3x3 local-max peak selection -> top-k -> add raw offset -> image coordinates
```

### Current Run

```text
run: /data-ssd/libo/ultralytics/runs/keypoint/merge_yolo26_0609_1280_da_pretrain_heatmap_strict_ms_gpu5
tmux: kp1280_heatmap_strict_ms_gpu5
GPU: physical 5, UUID GPU-72c3...
input size: 1280
batch size: 16
pretrained: /data-ssd/libo/p100/ci2n/models/da_external_port_20260205.pt
pretrained transfer: 894/928
```

Startup snapshot:

```text
model last layer:
  [16, 19, 22] -> KeypointHeatmap [[1, 3], 300, [256, 512, 512]]
GPU memory:
  about 78.4 GiB used on physical GPU 5
epoch 1 early training:
  hm_loss decreased from about 4.45 to about 1.60 by step 109/564
  off_loss decreased from about 0.54 to about 0.36 by step 109/564
```

## Code Entry Points

Task registration:

```text
ultralytics/models/yolo/model.py
ultralytics/models/yolo/__init__.py
ultralytics/cfg/__init__.py
```

Model/task implementation:

```text
ultralytics/nn/tasks.py::KeypointModel
ultralytics/nn/modules/head.py::KeypointDetect
ultralytics/nn/modules/head.py::KeypointHeatmap
```

Losses:

```text
ultralytics/utils/loss.py::KeypointOnlyLoss
ultralytics/utils/loss.py::KeypointHeatmapLoss
```

Training/validation/prediction:

```text
ultralytics/models/yolo/keypoint/train.py
ultralytics/models/yolo/keypoint/val.py
ultralytics/models/yolo/keypoint/predict.py
```

Dataset compatibility:

```text
ultralytics/data/dataset.py
ultralytics/engine/trainer.py
```

## Operational Notes

Active tmux sessions at the time this document was created:

```text
kp1280                direct regression baseline
kp1280_pnms_gpu7      direct regression + point-NMS
kp1280_heatmap_gpu6   CenterNet-style heatmap
```

Useful commands:

```bash
tmux attach -t kp1280
tmux attach -t kp1280_pnms_gpu7
tmux attach -t kp1280_heatmap_gpu6
```

Check latest metrics:

```bash
tail -n 10 /data-ssd/libo/ultralytics/runs/keypoint/merge_yolo26_0609_1280_da_pretrain/results.csv
tail -n 10 /data-ssd/libo/ultralytics/runs/keypoint/merge_yolo26_0609_1280_da_pretrain_pnms_gpu7/results.csv
tail -n 10 /data-ssd/libo/ultralytics/runs/keypoint/merge_yolo26_0609_1280_da_pretrain_heatmap_gpu6/results.csv
```

## Decision Criteria

Prefer the model variant that achieves:

1. high recall without relying on very large `max_det`
2. substantially better precision than direct dense regression
3. stable low `mean_dist(K)`
4. robust behavior on close points
5. simple inference postprocessing

The direct-regression baseline proves localization is learnable, but the dense duplicate prediction problem is significant. The heatmap model is expected to be a better long-term direction if it forms sparse peaks reliably.
