import sys
import os
import cv2
from pathlib import Path

sys.path.insert(0, r'E:\CV_Project')
from backend.app.model_loader import global_model_manager

manager = global_model_manager
print(f"Model loaded: {manager.is_loaded}")
print(f"Checkpoint: {manager.model_path}")
print(f"Channels: {manager.input_channels}")

img_paths = [
    ("Image 1 (Winter Crosswalk Crowd)", r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794598260878_16b68b05.jpg", "pred_image_1_annotated.jpg", "pred_image_1_raw.jpg"),
    ("Image 2 (Rainbow Thermal Plaza)", r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794598269315_d4059653.jpg", "pred_image_2_annotated.jpg", "pred_image_2_raw.jpg")
]

artifact_dir = Path(r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862")

for label, p, out_name, raw_name in img_paths:
    print(f"\n==========================================")
    print(f"{label}")
    print(f"File: {os.path.basename(p)}")
    print(f"==========================================")
    frame = cv2.imread(p)
    if frame is None:
        print(f"Error loading {p}")
        continue
    
    h, w = frame.shape[:2]
    print(f"Resolution: {w}x{h} px")

    # Save a clean copy of the raw image into artifact dir
    cv2.imwrite(str(artifact_dir / raw_name), frame)

    # 1. Standard prediction (conf=0.25)
    res_25 = manager.predict_frame(frame, conf_threshold=0.25)
    print(f"\n--- Standard Confidence (>= 0.25) ---")
    print(f"Total Detected: {res_25['pedestrians_count']} pedestrians | Inference Latency: {res_25['inference_time_ms']} ms")
    for i, b in enumerate(res_25['boxes']):
        x1, y1, x2, y2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])
        print(f"  [{i+1}] Coordinates: [{x1}, {y1}, {x2}, {y2}] ({x2-x1}x{y2-y1}px) | Confidence: {b['confidence']*100:.1f}%")

    # 2. Sensitive prediction (conf=0.15)
    res_15 = manager.predict_frame(frame, conf_threshold=0.15)
    print(f"\n--- Sensitive Threshold (>= 0.15) ---")
    print(f"Total Detected: {res_15['pedestrians_count']} pedestrians")
    for i, b in enumerate(res_15['boxes']):
        x1, y1, x2, y2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])
        print(f"  [{i+1}] Coordinates: [{x1}, {y1}, {x2}, {y2}] | Confidence: {b['confidence']*100:.1f}%")

    # Save annotated image (conf=0.25 for Plaza, conf=0.18 for sub-zero Crosswalk)
    eval_threshold = 0.18 if "Crosswalk" in label else 0.25
    res_eval = manager.predict_frame(frame, conf_threshold=eval_threshold)
    annotated = manager.annotate_frame(frame.copy(), res_eval['boxes'])
    out_file = artifact_dir / out_name
    cv2.imwrite(str(out_file), annotated)
    print(f"\nSaved annotated visualization ({eval_threshold} threshold, {res_eval['pedestrians_count']} targets) to: {out_file}")

print("\nAll predictions complete!")

