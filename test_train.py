#!/usr/bin/env python3
# Ultralytics structure test training - quick GPU test

from ultralytics import YOLO

if __name__ == "__main__":
    # Load model
    model = YOLO("ultralytics/cfg/models/11/yolo11n-structure.yaml")

    # Quick test with GPU
    results = model.train(
        data="datasets/device_ports.yaml",
        epochs=1,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
    )

    print("Test training complete!")
