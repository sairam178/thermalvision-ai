import os
import sys
import time
import json
import platform
import psutil
import torch
import cv2
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, r'E:\CV_Project')
from backend.app.model_loader import global_model_manager

# Destination directory
OUTPUT_DIR = Path(r"E:\CV_Project\evaluation_results\image_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hardware Info
HW_INFO = {
    "os": platform.platform(),
    "processor": platform.processor(),
    "physical_cores": psutil.cpu_count(logical=False),
    "logical_cores": psutil.cpu_count(logical=True),
    "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_used": "cpu"
}

# Image Sources
RAW_IMAGES = [
    {
        "id": "IMG_01",
        "name": "media_1794594968087_907cde47.webp",
        "path": r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794594968087_907cde47.webp",
        "source_modality": "LWIR Thermal",
        "modality_proof": "FLIR FC-Series SPS commercial uncooled microbolometer security camera (header string: 'FC-Series SPS S/N: ...')",
        "is_thermal": True,
        "is_composite": False
    },
    {
        "id": "IMG_02",
        "name": "media_1794594968088_63cb97ba.jpg",
        "path": r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794594968088_63cb97ba.jpg",
        "source_modality": "Synthetic RGB Illustration",
        "modality_proof": "Digital concept graphic with artificial CRT raster scanlines and rainbow gradient outlines. Not a radiometric sensor capture.",
        "is_thermal": False,
        "is_composite": False
    },
    {
        "id": "IMG_03",
        "name": "media_1794594968129_034d6b9b.jpg",
        "path": r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794594968129_034d6b9b.jpg",
        "source_modality": "LWIR Thermal (Radiometric)",
        "modality_proof": "Official FLIR logo in upper right, radiometric calibrated temperature scale bar (-20.3 C to 17.2 C) in Ironbow palette.",
        "is_thermal": True,
        "is_composite": False
    },
    {
        "id": "IMG_04_COMPOSITE",
        "name": "media_1794598536358_16bc4685.jpg",
        "path": r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794598536358_16bc4685.jpg",
        "source_modality": "Composite Figure (FIR Thermal Top + NIR Night Vision Bottom)",
        "modality_proof": "Scientific comparison figure. Top panel: passive Far-Infrared (FIR 8-14 um). Bottom panel: active Near-Infrared (NIR ~850 nm with illuminators).",
        "is_thermal": False, # Must be separated!
        "is_composite": True
    },
    {
        "id": "IMG_05",
        "name": "media_1794598536328_e6f019d6.jpg",
        "path": r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794598536328_e6f019d6.jpg",
        "source_modality": "LWIR Thermal",
        "modality_proof": "Smart city thermal traffic sensor (e.g. FLIR TrafiSense uncooled LWIR sensor) mounted near crosswalk signal.",
        "is_thermal": True,
        "is_composite": False
    },
    {
        "id": "IMG_06",
        "name": "media_1794598536330_637e5cf3.jpg",
        "path": r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794598536330_637e5cf3.jpg",
        "source_modality": "NIR Night Vision",
        "modality_proof": "Figure caption: 'Fig. 4: NIR-NVS Illustration from Mercedes S-Class'. Active Near-Infrared Night View Assist using 850nm IR headlamps.",
        "is_thermal": False,
        "is_composite": False
    }
]

def calculate_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

# Ground Truth Bounding Boxes [x1, y1, x2, y2] for genuine thermal images
GROUND_TRUTHS = {
    "thermal_01_parking_lot": [
        {"id": "GT1", "bbox": [464, 224, 555, 395], "description": "Adult pedestrian walking across parking lot lane"}
    ],
    "thermal_02_winter_crosswalk": [
        {"id": "GT1", "bbox": [28, 70, 75, 205], "description": "Far left pedestrian walking left"},
        {"id": "GT2", "bbox": [122, 115, 195, 310], "description": "Left foreground pedestrian walking left with bag"},
        {"id": "GT3", "bbox": [190, 80, 226, 165], "description": "Background pedestrian crossing street"},
        {"id": "GT4", "bbox": [218, 95, 280, 270], "description": "Mid-left pedestrian in dark coat"},
        {"id": "GT5", "bbox": [245, 145, 330, 360], "description": "Center pedestrian in heavy winter hooded parka"},
        {"id": "GT6", "bbox": [360, 95, 435, 270], "description": "Background walker behind center group"},
        {"id": "GT7", "bbox": [375, 100, 520, 430], "description": "Foreground right pedestrian walking with thermal leg signature"},
        {"id": "GT8", "bbox": [560, 110, 640, 440], "description": "Far right pedestrian on crosswalk"}
    ],
    "thermal_03_automotive_fir": [
        {"id": "GT1", "bbox": [610, 165, 662, 312], "description": "Pedestrian standing on right shoulder (FIR thermal emission)"}
    ],
    "thermal_04_traffic_sensor": [
        {"id": "GT1", "bbox": [395, 215, 465, 390], "description": "Pedestrian walking across zebra crosswalk"}
    ]
}

def run_verification():
    print(f"[+] Loading model from: {global_model_manager.model_path}")
    print(f"[+] Model loaded status: {global_model_manager.is_loaded}")
    print(f"[+] Hardware: {HW_INFO['processor']} ({HW_INFO['physical_cores']}C/{HW_INFO['logical_cores']}T, {HW_INFO['ram_gb']}GB RAM)")

    # 1. Separate Composite Image
    comp_path = r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794598536358_16bc4685.jpg"
    comp_img = cv2.imread(comp_path)
    
    # Split top (FIR) and bottom (NIR)
    top_fir = comp_img[0:370, :]
    bottom_nir = comp_img[405:, :]
    
    cv2.imwrite(str(OUTPUT_DIR / "separated_automotive_fir_top.jpg"), top_fir)
    cv2.imwrite(str(OUTPUT_DIR / "separated_automotive_nir_bottom.jpg"), bottom_nir)
    print("[+] Successfully separated composite figure into FIR top and NIR bottom panels.")

    # 2. Genuine Thermal Dataset for Verification
    thermal_dataset = [
        {
            "key": "thermal_01_parking_lot",
            "name": "Thermal Image 1: FLIR FC-Series SPS Parking Lot",
            "source_file": "media_1794594968087_907cde47.webp",
            "image": cv2.imread(r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794594968087_907cde47.webp"),
            "modality": "LWIR Thermal (Uncooled Microbolometer)",
            "ground_truth": GROUND_TRUTHS["thermal_01_parking_lot"]
        },
        {
            "key": "thermal_02_winter_crosswalk",
            "name": "Thermal Image 2: FLIR Radiometric Winter Crosswalk Crowd",
            "source_file": "media_1794594968129_034d6b9b.jpg",
            "image": cv2.imread(r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794594968129_034d6b9b.jpg"),
            "modality": "LWIR Thermal (Radiometric Ironbow, -20.3C to 17.2C)",
            "ground_truth": GROUND_TRUTHS["thermal_02_winter_crosswalk"]
        },
        {
            "key": "thermal_03_automotive_fir",
            "name": "Thermal Image 3: Automotive Far-Infrared (FIR) Night Vision Panel",
            "source_file": "media_1794598536358_16bc4685.jpg [Top Panel rows 0:370]",
            "image": top_fir,
            "modality": "FIR Thermal (Automotive uncooled 8-14 um)",
            "ground_truth": GROUND_TRUTHS["thermal_03_automotive_fir"]
        },
        {
            "key": "thermal_04_traffic_sensor",
            "name": "Thermal Image 4: Smart City Thermal Traffic Sensor Crosswalk",
            "source_file": "media_1794598536328_e6f019d6.jpg",
            "image": cv2.imread(r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794598536328_e6f019d6.jpg"),
            "modality": "LWIR Thermal (Overhead Traffic Sensor)",
            "ground_truth": GROUND_TRUTHS["thermal_04_traffic_sensor"]
        }
    ]

    # Benchmark Latency & Accuracy
    results_summary = []

    for item in thermal_dataset:
        img = item["image"]
        h, w = img.shape[:2]
        gt_boxes = item["ground_truth"]
        
        # Warmup inference
        _ = global_model_manager.predict_frame(img, conf_threshold=0.25)

        # Timed benchmark: 10 repeated runs
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            pred = global_model_manager.predict_frame(img, conf_threshold=0.25)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_latency = float(np.mean(latencies))
        std_latency = float(np.std(latencies))

        # Predictions at conf=0.25
        pred_25 = global_model_manager.predict_frame(img, conf_threshold=0.25)
        boxes_25 = pred_25["boxes"]

        # Predictions at conf=0.15
        pred_15 = global_model_manager.predict_frame(img, conf_threshold=0.15)
        boxes_15 = pred_15["boxes"]

        # Evaluate against Ground Truth (using conf=0.25, with IoU threshold 0.5)
        # Match predicted boxes to ground truth
        matched_gt = set()
        matched_pred = set()
        iou_pairs = []

        for p_idx, p in enumerate(boxes_25):
            p_box = [int(p['x1']), int(p['y1']), int(p['x2']), int(p['y2'])]
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g in enumerate(gt_boxes):
                iou = calculate_iou(p_box, g['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx
            
            if best_iou >= 0.50:
                if best_g_idx not in matched_gt:
                    matched_gt.add(best_g_idx)
                    matched_pred.add(p_idx)
                    iou_pairs.append((best_iou, p['confidence']))
                else:
                    # Duplicate detection for same GT
                    pass

        tp = len(matched_gt)
        fp = len(boxes_25) - len(matched_pred)
        fn = len(gt_boxes) - tp

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        avg_iou = float(np.mean([p[0] for p in iou_pairs])) if len(iou_pairs) > 0 else 0.0

        # Annotate comparison: Ground Truth in Green (0, 255, 0), Predictions in Cyan (255, 229, 0)
        annotated = img.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        # Draw Ground Truth in Emerald Green
        for g in gt_boxes:
            gx1, gy1, gx2, gy2 = g['bbox']
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 230, 0), 2)
            cv2.putText(annotated, f"GT:{g['id']}", (gx1, max(15, gy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 0), 1, cv2.LINE_AA)

        # Draw Predictions in Neon Cyan
        for p in boxes_25:
            px1, py1, px2, py2 = int(p['x1']), int(p['y1']), int(p['x2']), int(p['y2'])
            conf = p['confidence']
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 229, 0), 2)
            cv2.putText(annotated, f"YOLO:{int(conf*100)}%", (px1, min(h - 5, py2 + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 229, 0), 1, cv2.LINE_AA)

        annotated_path = OUTPUT_DIR / f"{item['key']}_verified.jpg"
        cv2.imwrite(str(annotated_path), annotated)

        # Also save raw input image in output dir
        raw_path = OUTPUT_DIR / f"{item['key']}_raw.jpg"
        cv2.imwrite(str(raw_path), img)

        res_record = {
            "key": item["key"],
            "name": item["name"],
            "source_file": item["source_file"],
            "modality": item["modality"],
            "dimensions": f"{w}x{h}",
            "gt_count": len(gt_boxes),
            "pred_count_conf25": len(boxes_25),
            "pred_count_conf15": len(boxes_15),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mean_iou": round(avg_iou, 4),
            "mean_latency_ms": round(mean_latency, 2),
            "std_latency_ms": round(std_latency, 2),
            "predictions_25": boxes_25,
            "annotated_path": str(annotated_path)
        }
        results_summary.append(res_record)
        print(f"\n[Verified] {item['name']}: GT={len(gt_boxes)}, TP={tp}, FP={fp}, FN={fn}, Prec={precision:.1%}, Rec={recall:.1%}, Latency={mean_latency:.1f}ms")

    # Save verification JSON
    json_path = OUTPUT_DIR / "verification_metrics.json"
    with open(json_path, "w") as f:
        json.dump({"hardware": HW_INFO, "results": results_summary}, f, indent=2)
    print(f"[+] Saved metrics JSON to: {json_path}")

    return results_summary

if __name__ == "__main__":
    run_verification()

