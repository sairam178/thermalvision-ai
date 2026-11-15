import cv2
import sys
from pathlib import Path
sys.path.insert(0, r'E:\CV_Project')
from backend.app.model_loader import global_model_manager

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
    
    det = global_model_manager.track_frame(frame, conf_threshold=0.25, persist=True)
    boxes = det['boxes']
    
    # 1. Filter out simultaneous sub-body boxes (e.g. torso inside full body)
    boxes_sorted = sorted(boxes, key=lambda b: (b['x2']-b['x1'])*(b['y2']-b['y1']), reverse=True)
    kept_boxes = []
    for b in boxes_sorted:
        area_b = (b['x2']-b['x1'])*(b['y2']-b['y1'])
        is_sub_box = False
        for k in kept_boxes:
            ix1 = max(b['x1'], k['x1'])
            iy1 = max(b['y1'], k['y1'])
            ix2 = min(b['x2'], k['x2'])
            iy2 = min(b['y2'], k['y2'])
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                if area_b > 0 and (inter_area / area_b) > 0.60:
                    is_sub_box = True
                    break
        if not is_sub_box:
            kept_boxes.append(b)

    for b in kept_boxes:
        tid = b.get('track_id')
        if tid is not None:
            if tid not in tracks:
                tracks[tid] = {'frames': [], 'centroids': []}
            tracks[tid]['frames'].append(frame_idx)
            cx = (b['x1'] + b['x2']) / 2.0
            cy = (b['y1'] + b['y2']) / 2.0
            tracks[tid]['centroids'].append((cx, cy))

cap.release()

print(f"Total video frames: {frame_idx}")
print("--- RAW TRACKS AFTER SUB-BODY NMS ---")
for tid, d in sorted(tracks.items()):
    print(f"Track #{tid}: count={len(d['frames'])} frames, span={d['frames'][0]} -> {d['frames'][-1]}")

# 2. Filter out transient noise (must appear for >= 12 frames, ~0.4s)
confirmed_tracks = {tid: d for tid, d in tracks.items() if len(d['frames']) >= 12}
print(f"\nConfirmed tracks (>=12 frames): {list(confirmed_tracks.keys())}")

# 3. Tracklet Merging across Time & Space (Continuous Trajectory)
track_clusters = []
for tid, d in sorted(confirmed_tracks.items(), key=lambda item: item[1]['frames'][0]):
    start_f = d['frames'][0]
    start_pt = d['centroids'][0]
    merged = False
    for cluster in track_clusters:
        last_f = cluster['last_frame']
        last_pt = cluster['last_pt']
        frame_gap = start_f - last_f
        spatial_dist = ((start_pt[0] - last_pt[0])**2 + (start_pt[1] - last_pt[1])**2)**0.5
        
        # If the gap between tracks is <= 60 frames (2.0 sec) and within reasonable walking distance
        if 0 <= frame_gap <= 60 and spatial_dist < 350:
            cluster['tids'].append(tid)
            cluster['last_frame'] = d['frames'][-1]
            cluster['last_pt'] = d['centroids'][-1]
            merged = True
            break
            
    if not merged:
        track_clusters.append({
            'tids': [tid],
            'last_frame': d['frames'][-1],
            'last_pt': d['centroids'][-1]
        })

print(f"\n==============================================")
print(f">>> FINAL CALCULATED UNIQUE PEDESTRIANS: {len(track_clusters)}")
print(f"==============================================")
for i, c in enumerate(track_clusters):
    print(f"Pedestrian Cluster {i+1}: Member Track IDs={c['tids']}")

