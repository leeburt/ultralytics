#!/usr/bin/env python3
"""
在最好模型基础上，使用人工check后的数据继续训练（低学习率微调）
"""
from ultralytics import YOLO

BEST_MODEL = "/data-ssd/libo/ultralytics/runs/keypoint/merge_v4_yolo26_1536_scratch_e200_gpu34/weights/best.pt"
DATA_YAML = "/data-ssd/libo/p100/yolo_utils/dataset/external_inline/merge_config_v2.yaml"

model = YOLO(BEST_MODEL)

results = model.train(
    data=DATA_YAML,
    epochs=200,
    imgsz=1536,
    batch=20,
    device=5,
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    pose=12.0,
    kobj=1.0,
    patience=50,
    workers=16,
    deterministic=True,
    seed=0,
    close_mosaic=15,
    project="/data-ssd/libo/ultralytics/runs/keypoint",
    name="merge_v4_human_checked_lr0005_e200_gpu5",
    exist_ok=True,
    val=True,
    plots=True,
    amp=False,
    save=True,
    verbose=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    auto_augment="randaugment",
    erasing=0.4,
)
