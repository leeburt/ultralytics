#!/usr/bin/env python3
"""Ultralytics structure model training script."""

from ultralytics import YOLO


def main():
    # Load model from YAML config
    model = YOLO("ultralytics/cfg/models/11/yolo11l-structure.yaml")

    # Train with GPU
    results = model.train(
        data="datasets/device_ports.yaml",
        epochs=200,          # Train longer
        imgsz=640,           # Input size
        batch=8,             # Batch size per GPU (reduced for large model)
        device=0,            # GPU device (set to "cpu" for CPU-only)
        workers=8,           # Data loading workers
        lr0=0.01,            # Initial learning rate
        lrf=0.01,            # Final learning rate factor
        warmup_epochs=3,     # Warmup epochs
        cos_lr=True,         # Cosine LR schedule
        patience=30,         # Early stopping patience
        save=True,           # Save checkpoints
        save_period=5,       # Save every 5 epochs
        project="runs/structure",
        name="train",        # Experiment name
        exist_ok=True,       # Overwrite existing directory
    )

    print("Training complete!")
    print(f"Best model saved at: {model.trainer.best}")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
