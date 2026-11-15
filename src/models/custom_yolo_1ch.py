"""
custom_yolo_1ch.py
==================
Phase 3 — 1-Channel YOLOv8 Thermal Architecture & Weight Transplantation.

Objective
---------
Adapts the pretrained Ultralytics YOLOv8 network to accept genuine single-channel
thermal infrared (LWIR) images (input shape: [B, 1, H, W]) without duplicate RGB
channels, preserving all compatible pretrained feature representations via
channel-averaged weight transplantation.

Mathematical Principle
----------------------
The first convolutional layer in standard YOLOv8n has kernel weights:
    W_3ch in R^{C_out x 3 x K_h x K_w} (where C_out=16, K=3)

For genuine single-channel grayscale thermal input, we require:
    W_1ch in R^{C_out x 1 x K_h x K_w}

We transplant weights via channel-averaging:
    W_1ch = (1 / 3) * sum_{c=0}^2 W_3ch[:, c:c+1, :, :] = W_3ch.mean(dim=1, keepdim=True)

This mathematically preserves:
1. The average activation response across edge and gradient filters.
2. Filter variance and scale across all 16 output channels.
3. The pretrained feature representations learned across the COCO dataset.

Compatible backbone (layers 1-9), neck (layers 10-21), and bbox regression
heads (cv2) are transferred with 100% exact fidelity. The classification head
(cv3) is configured for single-class pedestrian detection (nc=1).
"""

import os
import sys
import copy
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("custom_yolo_1ch")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def build_1ch_thermal_yolo(
    weights_path: str = "yolov8n.pt",
    cfg_path: str = "configs/yolov8n_1ch.yaml",
    save_path: str = "weights/yolov8n_thermal_1ch.pt",
) -> Tuple[DetectionModel, Dict[str, Any]]:
    """
    Builds a genuine 1-channel YOLOv8 model and transplants pretrained weights.

    Parameters:
        weights_path: Path to pretrained 3-channel weights (e.g. yolov8n.pt)
        cfg_path: Path to 1-channel YOLOv8 YAML architecture
        save_path: Destination path for the adapted checkpoint (.pt)

    Returns:
        model: Adapted DetectionModel instance
        stats: Dictionary containing parameter and layer transplantation statistics
    """
    logger = setup_logger()
    logger.info("=" * 70)
    logger.info("YOLOv8 ONE-CHANNEL THERMAL ARCHITECTURE ADAPTATION")
    logger.info("=" * 70)

    # 1. Load pretrained 3-channel source model
    logger.info(f"Loading pretrained source model: {weights_path}")
    source_yolo = YOLO(weights_path)
    source_model = source_yolo.model
    source_dict = source_model.state_dict()

    # 2. Instantiate target 1-channel, 1-class DetectionModel
    logger.info(f"Instantiating target 1-channel architecture from: {cfg_path}")
    target_model = DetectionModel(cfg=cfg_path, ch=1, nc=1)
    target_dict = target_model.state_dict()

    # Set metadata
    target_model.nc = 1
    target_model.names = {0: "person"}
    target_model.args = {"ch": 1, "nc": 1}

    # Verify first convolution specification
    first_conv_orig = source_model.model[0].conv
    first_conv_target = target_model.model[0].conv
    logger.info(f"Source first conv: {first_conv_orig} -> Weight: {first_conv_orig.weight.shape}")
    logger.info(f"Target first conv: {first_conv_target} -> Weight: {first_conv_target.weight.shape}")

    # 3. Intelligent Weight Transplantation
    transferred_weights = {}
    matched_count = 0
    adapted_count = 0
    stats = {
        "total_target_parameters": sum(p.numel() for p in target_model.parameters()),
        "directly_matched_parameters": 0,
        "adapted_parameters": 0,
        "adapted_layers": [],
        "first_conv_source_shape": list(first_conv_orig.weight.shape),
        "first_conv_target_shape": list(first_conv_target.weight.shape),
    }

    logger.info("Executing weight transplantation across layers...")
    for key, tgt_param in target_dict.items():
        if key not in source_dict:
            transferred_weights[key] = tgt_param
            continue

        src_param = source_dict[key]

        # Case A: Exact shape match (Backbone layers 1-9, Neck 10-21, Head cv2)
        if src_param.shape == tgt_param.shape:
            transferred_weights[key] = src_param.clone()
            matched_count += tgt_param.numel()

        # Case B: First convolutional layer (in_channels: 3 -> 1)
        elif key == "model.0.conv.weight":
            # W_1ch = mean(W_3ch, dim=1, keepdim=True)
            w_1ch = src_param.mean(dim=1, keepdim=True)
            transferred_weights[key] = w_1ch
            adapted_count += w_1ch.numel()
            stats["adapted_layers"].append({
                "layer": key,
                "type": "channel_average",
                "source_shape": list(src_param.shape),
                "target_shape": list(w_1ch.shape),
            })
            logger.info(f"  [ADAPTED] {key}: {list(src_param.shape)} -> {list(w_1ch.shape)} via channel-averaging")

        # Case C: Classification head (cv3) sub-tensor adaptation
        elif "cv3" in key:
            # Slices compatible feature dimensions from pretrained COCO head
            slices = tuple(slice(0, min(s_d, t_d)) for s_d, t_d in zip(src_param.shape, tgt_param.shape))
            new_param = tgt_param.clone()
            new_param[slices] = src_param[slices]
            transferred_weights[key] = new_param
            adapted_count += new_param.numel()
            stats["adapted_layers"].append({
                "layer": key,
                "type": "subtensor_slice",
                "source_shape": list(src_param.shape),
                "target_shape": list(new_param.shape),
            })
        else:
            transferred_weights[key] = tgt_param

    # Load adapted state dict into target model
    target_model.load_state_dict(transferred_weights)

    stats["directly_matched_parameters"] = matched_count
    stats["adapted_parameters"] = adapted_count
    stats["total_transferred"] = matched_count + adapted_count
    stats["transferred_percentage"] = (stats["total_transferred"] / stats["total_target_parameters"]) * 100

    logger.info("Transplantation Summary:")
    logger.info(f"  Total target parameters     : {stats['total_target_parameters']:,}")
    logger.info(f"  Directly matched parameters : {stats['directly_matched_parameters']:,} ({stats['directly_matched_parameters']/stats['total_target_parameters']*100:.2f}%)")
    logger.info(f"  Adapted parameters (Conv1+H): {stats['adapted_parameters']:,} ({stats['adapted_parameters']/stats['total_target_parameters']*100:.2f}%)")
    logger.info(f"  Total transferred coverage  : {stats['transferred_percentage']:.2f}%")

    # 4. Save Ultralytics-compatible checkpoint
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": target_model,
        "yaml": cfg_path,
        "epoch": -1,
        "train_args": {"ch": 1, "nc": 1},
        "names": {0: "person"},
    }
    torch.save(ckpt, save_path)
    logger.info(f"Successfully saved adapted 1-channel checkpoint to: {save_path}")

    return target_model, stats


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 1-Channel Thermal Architecture Adapter")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to pretrained YOLOv8 weights")
    parser.add_argument("--cfg", type=str, default="configs/yolov8n_1ch.yaml", help="Path to 1-channel YAML config")
    parser.add_argument("--save", type=str, default="weights/yolov8n_thermal_1ch.pt", help="Path to save adapted checkpoint")

    args = parser.parse_args()
    build_1ch_thermal_yolo(weights_path=args.weights, cfg_path=args.cfg, save_path=args.save)


if __name__ == "__main__":
    main()

