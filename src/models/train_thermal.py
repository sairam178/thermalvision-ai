"""
train_thermal.py
================
Phase 4 — Standalone Training Pipeline for 1-Channel Thermal YOLOv8.

Trains the adapted 1-channel YOLOv8 thermal pedestrian detector on the processed
FLIR dataset using Ultralytics with thermal-specific hyperparameters:
- Disables HSV hue/saturation jitter (hsv_h=0.0, hsv_s=0.0) invalid for infrared.
- Retains value jitter (hsv_v=0.2) to simulate sensor gain variations.
- Uses multi-scale mosaic and horizontal flips.
- Saves best/last model checkpoints, loss curves, confusion matrices, and metrics.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # e:\CV_Project
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "yolov8n_thermal_1ch.pt"
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "data.yaml"
DEFAULT_RUNS = PROJECT_ROOT / "runs" / "detect"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("train_thermal")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def check_gpu_environment(logger: logging.Logger) -> str:
    """Audits acceleration hardware (CUDA GPU or CPU fallback)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        device_str = "0"
        logger.info(f"Hardware Acceleration: CUDA GPU detected -> {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        device_str = "cpu"
        logger.warning("Hardware Acceleration: No GPU detected. Running on CPU (training will be slow).")
    return device_str


def train_1ch_model(
    weights_path: str = str(DEFAULT_WEIGHTS),
    data_path: str = str(DEFAULT_DATA),
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    project: str = str(DEFAULT_RUNS),
    name: str = "train_thermal_1ch",
    device: str = None,
    save_period: int = 5,
    patience: int = 15,
) -> Any:
    """
    Executes training using Ultralytics YOLO with 1-channel thermal settings.
    """
    logger = setup_logger()
    logger.info("=" * 70)
    logger.info("PHASE 4: 1-CHANNEL THERMAL YOLOv8 TRAINING PIPELINE")
    logger.info("=" * 70)

    # 1. Device detection
    if device is None:
        device = check_gpu_environment(logger)

    # 2. Verify model and data paths
    weights_p = Path(weights_path)
    data_p = Path(data_path)
    assert weights_p.exists(), f"Weights file not found: {weights_p}"
    assert data_p.exists(), f"Dataset config not found: {data_p}"

    logger.info(f"Loading checkpoint: {weights_p}")
    model = YOLO(str(weights_p))

    # Audit Layer 0 conv
    first_conv = model.model.model[0].conv
    logger.info(f"Verified Layer 0 Conv: {first_conv}")
    assert first_conv.in_channels == 1, f"Expected 1 channel input, got {first_conv.in_channels}"

    # 3. Configure Thermal-Specific Hyperparameters
    # Hue and saturation jitter MUST be 0.0 for 1-channel infrared data
    train_args = {
        "data": str(data_p),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": device,
        "project": project,
        "name": name,
        "save": True,
        "save_period": save_period,
        "patience": patience,
        "plots": True,
        "val": True,
        # Thermal specific augmentations:
        "hsv_h": 0.0,      # Disable hue jitter (not applicable to LWIR)
        "hsv_s": 0.0,      # Disable saturation jitter (not applicable to LWIR)
        "hsv_v": 0.2,      # Value/gain jitter for sensor dynamic range variations
        "fliplr": 0.5,     # Horizontal flip
        "flipud": 0.0,     # Disable vertical flip
        "mosaic": 1.0,     # Mosaic augmentation
        "close_mosaic": 10,# Close mosaic in final 10 epochs
        "degrees": 0.0,    # Rotation degrees
        "translate": 0.1,  # Translation
        "scale": 0.5,      # Scale jitter
        "optimizer": "auto",
        "verbose": True,
    }

    logger.info("Training Hyperparameters Configured:")
    for k, v in train_args.items():
        logger.info(f"  {k:15s}: {v}")

    logger.info("\n[!] Starting training session...")
    results = model.train(**train_args)
    logger.info("[+] Training completed successfully.")

    # 4. Run test split validation
    logger.info("\n[!] Evaluating on test split...")
    test_results = model.val(data=str(data_p), split="test")
    logger.info(f"[+] Test mAP@0.5: {test_results.box.map50:.4f} | mAP@0.5:0.95: {test_results.box.map:.4f}")

    return results, test_results


def main():
    parser = argparse.ArgumentParser(description="Train 1-Channel Thermal YOLOv8")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to 1-ch checkpoint")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default=None, help="Device ('0', 'cpu', etc.)")
    parser.add_argument("--project", type=str, default=str(DEFAULT_RUNS), help="Runs directory")
    parser.add_argument("--name", type=str, default="train_thermal_1ch", help="Run name")

    args = parser.parse_args()
    train_1ch_model(
        weights_path=args.weights,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()

