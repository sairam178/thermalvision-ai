import sys
import os
from pathlib import Path
import cv2

sys.path.insert(0, r'E:\CV_Project')
from backend.app.model_loader import global_model_manager

manager = global_model_manager
print(f"Model successfully loaded: {manager.is_loaded}")
print(f"Model path: {manager.model_path}")
print(f"Input channels: {manager.input_channels}")

img_paths = [
    (1, r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794594968087_907cde47.webp"),
    (2, r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794594968088_63cb97ba.jpg"),
    (3, r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794594968129_034d6b9b.jpg")
]

artifact_dir = Path(r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862")

for idx, p in img_paths:
    print(f"\n==========================================")
    print(f"IMAGE {idx}: {os.path.basename(p)}")
    print(f"==========================================")
    frame = cv2.imread(p)
    if frame is None:
        print(f"Error loading {p}")
        continue
    
    h, w, c = frame.shape
    print(f"Resolution: {w}x{h}, Channels: {c}")
    
    # 1. Standard confidence threshold = 0.25
    res_25 = manager.predict_frame(frame, conf_threshold=0.25)
    print(f"\n--- Results at conf_threshold = 0.25 ---")
    print(f"Pedestrians detected: {res_25['pedestrians_count']}")
    print(f"Inference Latency: {res_25['inference_time_ms']} ms")
    for i, b in enumerate(res_25["boxes"]):
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        print(f"  [{i+1}] BBox=[{x1}, {y1}, {x2}, {y2}] ({x2-x1}x{y2-y1}) | Conf={b['confidence']*100:.1f}% | Class={b['class_name']}")
    
    # Save standard annotated output
    annotated_25 = manager.annotate_frame(frame.copy(), res_25["boxes"])
    out_25 = artifact_dir / f"eval_detected_image_{idx}.jpg"
    cv2.imwrite(str(out_25), annotated_25)
    print(f"Saved annotated image (conf 0.25) to: {out_25}")

    # 2. Sensitivity test at conf_threshold = 0.15
    res_15 = manager.predict_frame(frame, conf_threshold=0.15)
    print(f"\n--- Sensitivity check at conf_threshold = 0.15 ---")
    print(f"Pedestrians detected at 0.15: {res_15['pedestrians_count']}")
    for i, b in enumerate(res_15["boxes"]):
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        print(f"  [{i+1}] BBox=[{x1}, {y1}, {x2}, {y2}] | Conf={b['confidence']*100:.1f}%")

    # 3. Precision test at conf_threshold = 0.40
    res_40 = manager.predict_frame(frame, conf_threshold=0.40)
    print(f"Pedestrians detected at 0.40 (High Confidence): {res_40['pedestrians_count']}")

print("\nEvaluation inference complete!")
