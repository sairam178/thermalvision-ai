import sys
import cv2
import numpy as np

sys.path.insert(0, r'E:\CV_Project')
from backend.app.model_loader import global_model_manager

# 1. Parking Lot Image
p1 = r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794594968087_907cde47.webp"
img1 = cv2.imread(p1)
res1 = global_model_manager.predict_frame(img1, conf_threshold=0.15)
print("=== Parking Lot Image ===")
print("Dimensions:", img1.shape)
for b in res1['boxes']:
    print(f"Pred: [{int(b['x1'])}, {int(b['y1'])}, {int(b['x2'])}, {int(b['y2'])}] conf={b['confidence']*100:.1f}%")

# 2. Winter Crosswalk Image
p2 = r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794594968129_034d6b9b.jpg"
img2 = cv2.imread(p2)
res2 = global_model_manager.predict_frame(img2, conf_threshold=0.10)
print("\n=== Winter Crosswalk Image ===")
print("Dimensions:", img2.shape)
for b in res2['boxes']:
    print(f"Pred: [{int(b['x1'])}, {int(b['y1'])}, {int(b['x2'])}, {int(b['y2'])}] conf={b['confidence']*100:.1f}%")

# 3. Automotive Top Panel (FIR)
p3 = r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794598536358_16bc4685.jpg"
img3 = cv2.imread(p3)
top_fir = img3[0:370, :]
res3 = global_model_manager.predict_frame(top_fir, conf_threshold=0.15)
print("\n=== Automotive Top Panel (FIR) ===")
print("Dimensions:", top_fir.shape)
for b in res3['boxes']:
    print(f"Pred: [{int(b['x1'])}, {int(b['y1'])}, {int(b['x2'])}, {int(b['y2'])}] conf={b['confidence']*100:.1f}%")

# 4. Traffic Sensor Zebra Crosswalk
p4 = r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794598536328_e6f019d6.jpg"
img4 = cv2.imread(p4)
res4 = global_model_manager.predict_frame(img4, conf_threshold=0.15)
print("\n=== Traffic Sensor Zebra Crosswalk ===")
print("Dimensions:", img4.shape)
for b in res4['boxes']:
    print(f"Pred: [{int(b['x1'])}, {int(b['y1'])}, {int(b['x2'])}, {int(b['y2'])}] conf={b['confidence']*100:.1f}%")

