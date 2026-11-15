import sys
import os
import cv2
from pathlib import Path

sys.path.insert(0, r'E:\CV_Project')
from backend.app.model_loader import global_model_manager

manager = global_model_manager
print(f"Model loaded: {manager.is_loaded}")
print(f"Checkpoint: {manager.model_path}")

img_list = [
    ("Image 1: FLIR FC-Series SPS Parking Lot", r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794598536086_4dca8709.webp", "batch2_pred_1.jpg", "batch2_raw_1.webp"),
    ("Image 2: Automotive Thermal vs NIR Road Comparison", r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794598536358_16bc4685.jpg", "batch2_pred_2.jpg", "batch2_raw_2.jpg"),
    ("Image 3: Traffic Sensor Zebra Crosswalk", r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794598536328_e6f019d6.jpg", "batch2_pred_3.jpg", "batch2_raw_3.jpg"),
    ("Image 4: Mercedes S-Class NIR-NVS Research Panels", r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794598536330_637e5cf3.jpg", "batch2_pred_4.jpg", "batch2_raw_4.jpg")
]

artifact_dir = Path(r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862")

for label, p, out_name, raw_name in img_list:
    print(f"\n==========================================")
    print(f"{label}")
    print(f"File: {os.path.basename(p)}")
    print(f"==========================================")
    frame = cv2.imread(p)
    if frame is None:
        print(f"Failed to read {p}")
        continue
    
    h, w = frame.shape[:2]
    print(f"Dimensions: {w}x{h} px")
    cv2.imwrite(str(artifact_dir / raw_name), frame)

    # Prediction at default conf=0.25
    res_25 = manager.predict_frame(frame, conf_threshold=0.25)
    print(f"Confidence >= 0.25: {res_25['pedestrians_count']} pedestrians detected (latency: {res_25['inference_time_ms']} ms)")
    for i, b in enumerate(res_25['boxes']):
        x1, y1, x2, y2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])
        print(f"  [{i+1}] Box=[{x1}, {y1}, {x2}, {y2}] ({x2-x1}x{y2-y1}px) | Conf={b['confidence']*100:.1f}%")

    # Prediction at sensitive conf=0.15
    res_15 = manager.predict_frame(frame, conf_threshold=0.15)
    print(f"Confidence >= 0.15: {res_15['pedestrians_count']} pedestrians detected")
    for i, b in enumerate(res_15['boxes']):
        if b not in res_25['boxes']:
            x1, y1, x2, y2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])
            print(f"  [sensitive] Box=[{x1}, {y1}, {x2}, {y2}] | Conf={b['confidence']*100:.1f}%")

    # Annotate and save
    annotated = manager.annotate_frame(frame.copy(), res_25['boxes'])
    out_path = artifact_dir / out_name
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated image to: {out_path}")

print("\nBatch prediction complete!")

