#!/usr/bin/env python3
"""Quick small dataset test to verify training + validation pipeline."""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/11/yolo11n-structure.yaml")

    results = model.train(
        data="datasets/device_ports_small.yaml",
        epochs=2,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        plots=False,
    )

    print("\n===== Pipeline verification PASSED! =====")
    print(f"Final metrics: {results}")
