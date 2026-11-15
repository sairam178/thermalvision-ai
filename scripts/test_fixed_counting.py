import cv2
import sys
from pathlib import Path
sys.path.insert(0, r'E:\CV_Project')
from backend.app.model_loader import global_model_manager

def merge_sub_body_boxes(boxes):
    if not boxes or len(boxes) <= 1:
        return boxes
    sorted_b = sorted(boxes, key=lambda b: (b['x2'] - b['x1']) * (b['y2'] - b['y1']), reverse=True)
    merged = []
    for b in sorted_b:
        bw = b['x2'] - b['x1']
        bh = b['y2'] - b['y1']
        is_part = False
        for m in merged:
            mw = m['x2'] - m['x1']
            mh = m['y2'] - m['y1']
            x_left = max(b['x1'], m['x1'])
            x_right = min(b['x2'], m['x2'])
            x_overlap = max(0, x_right - x_left)
            x_ratio = x_overlap / min(bw, mw) if min(bw, mw) > 0 else 0
            if x_ratio > 0.45:
                y_top = max(b['y1'], m['y1'])
                y_bot = min(b['y2'], m['y2'])
                y_overlap = max(0, y_bot - y_top)
                if y_overlap > 0 or abs(b['y1'] - m['y2']) < 60 or abs(m['y1'] - b['y2']) < 60:
                    m['x1'] = min(m['x1'], b['x1'])
                    m['y1'] = min(m['y1'], b['y1'])
                    m['x2'] = max(m['x2'], b['x2'])
                    m['y2'] = max(m['y2'], b['y2'])
                    m['confidence'] = max(m['confidence'], b['confidence'])
                    is_part = True
                    break
        if not is_part:
            merged.append(b.copy())
    return merged

video_path = r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794635711855_7c775cdc.mp4"

cap = cv2.VideoCapture(video_path)
global_model_manager.reset_tracker()

tracks = {}
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    # 1. Run inference
    raw_det = global_model_manager.predict_frame(frame, conf_threshold=0.25)
    # 2. Merge sub-body parts into complete human entities
    clean_boxes = merge_sub_body_boxes(raw_det['boxes'])
    
    # 3. Associate tracking
    for b in clean_boxes:
        cx = (b['x1'] + b['x2']) / 2.0
        cy = (b['y1'] + b['y2']) / 2.0
        matched_tid = None
        best_dist = 999999
        for tid, t_data in tracks.items():
            last_f = t_data['frames'][-1]
            if frame_idx - last_f <= 45: # Within 1.5 seconds
                last_cx, last_cy = t_data['centroids'][-1]
                dist = ((cx - last_cx)**2 + (cy - last_cy)**2)**0.5
                if dist < 250 and dist < best_dist:
                    best_dist = dist
                    matched_tid = tid
        if matched_tid is not None:
            tracks[matched_tid]['frames'].append(frame_idx)
            tracks[matched_tid]['centroids'].append((cx, cy))
            tracks[matched_tid]['boxes'].append([b['x1'], b['y1'], b['x2'], b['y2']])
        else:
            new_tid = len(tracks) + 1
            tracks[new_tid] = {
                'frames': [frame_idx],
                'centroids': [(cx, cy)],
                'boxes': [[b['x1'], b['y1'], b['x2'], b['y2']]]
            }

cap.release()

print(f"Total video frames processed: {frame_idx}")
print("--- ALL GENERATED TRACKLETS ---")
for tid, t in sorted(tracks.items()):
    cnt = len(t['frames'])
    print(f"Track #{tid}: {cnt} frames (from frame {t['frames'][0]} to {t['frames'][-1]})")

# Filter tracks that persist for at least 15 frames (~0.5s) to eliminate noise
real_pedestrians = [tid for tid, t in tracks.items() if len(t['frames']) >= 15]
print(f"\n=======================================================")
print(f">>> REAL UNIQUE PEDESTRIANS DETECTED: {len(real_pedestrians)}")
print(f"=======================================================")
for pid in real_pedestrians:
    t = tracks[pid]
    print(f"Pedestrian #{pid}: Active across {len(t['frames'])} frames ({t['frames'][0]} -> {t['frames'][-1]})")

