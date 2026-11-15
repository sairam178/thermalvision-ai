import os
import re
import json
import shutil
import uuid
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import cv2
import numpy as np

from .model_loader import global_model_manager
from .database import add_detection_event, add_video_session

# In-memory registry of active and completed jobs
active_jobs: Dict[str, Dict[str, Any]] = {}

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    return active_jobs.get(job_id)


def compute_box_iou(boxA: list, boxB: list) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def merge_sub_body_boxes(boxes: list) -> list:
    """
    Suppresses redundant sub-body detections (e.g. upper torso, lower legs, or feet)
    that belong to the same physical object/target, merging them into an enclosing bounding box.
    """
    if not boxes or len(boxes) <= 1:
        return boxes

    # Sort boxes by area descending so full-body boxes take precedence
    sorted_b = sorted(boxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True)
    merged = []

    for b in sorted_b:
        bw = b["x2"] - b["x1"]
        bh = b["y2"] - b["y1"]
        is_part = False

        for m in merged:
            mw = m["x2"] - m["x1"]
            mh = m["y2"] - m["y1"]

            # Check horizontal overlap ratio (columns occupied)
            x_left = max(b["x1"], m["x1"])
            x_right = min(b["x2"], m["x2"])
            x_overlap = max(0.0, x_right - x_left)
            x_ratio = x_overlap / min(bw, mw) if min(bw, mw) > 0 else 0.0

            # If horizontally aligned along the same vertical target axis (>45% horizontal overlap)
            if x_ratio > 0.45:
                y_top = max(b["y1"], m["y1"])
                y_bot = min(b["y2"], m["y2"])
                y_overlap = max(0.0, y_bot - y_top)

                # Either b is vertically inside m, or vertically stacked/adjacent
                if y_overlap > 0 or abs(b["y1"] - m["y2"]) < 60 or abs(m["y1"] - b["y2"]) < 60:
                    m["x1"] = min(m["x1"], b["x1"])
                    m["y1"] = min(m["y1"], b["y1"])
                    m["x2"] = max(m["x2"], b["x2"])
                    m["y2"] = max(m["y2"], b["y2"])
                    m["confidence"] = max(m["confidence"], b["confidence"])
                    is_part = True
                    break

        if not is_part:
            merged.append(b.copy())

    return merged


def calculate_unique_targets(tracked_identities: dict, total_frames: int, total_detections: int) -> int:
    """
    Calculates true unique thermal targets by:
    1. Filtering out transient 1-2 frame detection noise/flickers.
    2. Merging fragmented tracklets of the same target across time and space.
    """
    if not tracked_identities:
        return 1 if total_detections > 0 else 0

    # Minimum persistence: 4 frames for short clips (<100 frames), 10 frames for full videos
    min_persist = 4 if total_frames < 100 else 10
    confirmed_tracks = {tid: t for tid, t in tracked_identities.items() if len(t["frames"]) >= min_persist}

    if not confirmed_tracks:
        return 1 if total_detections > 0 else 0

    # Cluster tracks that represent continuous trajectory segments or overlapping detections of the same target
    target_clusters = []
    for tid, t_data in sorted(confirmed_tracks.items(), key=lambda item: item[1]["frames"][0]):
        start_f = t_data["frames"][0]
        end_f = t_data["frames"][-1]
        start_pt = t_data["centroids"][0]
        end_pt = t_data["centroids"][-1]

        merged = False
        for cluster in target_clusters:
            last_f = cluster["last_f"]
            last_pt = cluster["last_pt"]
            f_gap = start_f - last_f
            spatial_dist = ((start_pt[0] - last_pt[0])**2 + (start_pt[1] - last_pt[1])**2)**0.5

            # If track starts near where previous track ended within 60 frames (2s)
            # or if tracks overlap in time and share spatial proximity (<200px)
            if (0 <= f_gap <= 60 and spatial_dist < 300) or (f_gap < 0 and spatial_dist < 200):
                cluster["tids"].append(tid)
                cluster["last_f"] = max(cluster["last_f"], end_f)
                cluster["last_pt"] = end_pt
                merged = True
                break

        if not merged:
            target_clusters.append({
                "tids": [tid],
                "last_f": end_f,
                "last_pt": end_pt
            })

    return max(1, len(target_clusters)) if total_detections > 0 else 0

# Backwards compatibility alias
calculate_unique_pedestrians = calculate_unique_targets


class H264VideoWriter:
    """
    Encodes video frames directly into browser-compatible H.264 (AVC1) MP4.
    Uses imageio-ffmpeg with yuv420p pixel format and +faststart flag so HTML5 <video> tags
    in Chrome, Edge, Safari, and Firefox can immediately stream and play the video.
    """
    def __init__(self, output_path: Path, fps: float, width: int, height: int):
        self.output_path = Path(output_path)
        self.fps = max(1.0, float(fps))
        # Ensure dimensions are strictly even (required by libx264)
        self.width = width - (width % 2)
        self.height = height - (height % 2)
        self.imageio_writer = None
        self.cv2_writer = None
        self.temp_cv2_path = None

        try:
            import imageio
            self.imageio_writer = imageio.get_writer(
                str(self.output_path),
                fps=self.fps,
                codec='libx264',
                pixelformat='yuv420p',
                macro_block_size=None,
                ffmpeg_params=['-movflags', '+faststart', '-preset', 'veryfast', '-crf', '22']
            )
        except Exception as e:
            print(f"[!] Warning: imageio writer init failed ({e}), falling back to cv2 writer + ffmpeg remux")
            self.temp_cv2_path = self.output_path.with_name(f"temp_{self.output_path.name}")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.cv2_writer = cv2.VideoWriter(
                str(self.temp_cv2_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )

    def write(self, bgr_frame: np.ndarray):
        h, w = bgr_frame.shape[:2]
        if w != self.width or h != self.height:
            bgr_frame = cv2.resize(bgr_frame, (self.width, self.height))

        if self.imageio_writer is not None:
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            self.imageio_writer.append_data(rgb_frame)
        elif self.cv2_writer is not None:
            self.cv2_writer.write(bgr_frame)

    def release(self):
        if self.imageio_writer is not None:
            try:
                self.imageio_writer.close()
            except Exception as e:
                print(f"[!] Error closing imageio writer: {e}")
            self.imageio_writer = None

        if self.cv2_writer is not None:
            try:
                self.cv2_writer.release()
            except Exception as e:
                print(f"[!] Error releasing cv2 writer: {e}")
            self.cv2_writer = None

            if self.temp_cv2_path and self.temp_cv2_path.exists():
                try:
                    import imageio_ffmpeg, subprocess
                    exe = imageio_ffmpeg.get_ffmpeg_exe()
                    subprocess.run([
                        exe, '-y',
                        '-i', str(self.temp_cv2_path),
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart',
                        '-preset', 'veryfast',
                        str(self.output_path)
                    ], check=True, capture_output=True)
                    self.temp_cv2_path.unlink()
                except Exception as ex:
                    print(f"[!] Failed to remux to H.264: {ex}")
                    if self.temp_cv2_path.exists():
                        self.temp_cv2_path.rename(self.output_path)


def run_video_analysis_job(
    job_id: str,
    input_video_path: Path,
    original_filename: str,
    conf_threshold: float = 0.25
):
    cap = None
    writer = None
    try:
        active_jobs[job_id]["status"] = "processing"

        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open uploaded video file: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        active_jobs[job_id]["total_frames"] = total_frames
        active_jobs[job_id]["fps"] = round(fps, 1)
        active_jobs[job_id]["resolution"] = f"{width}x{height}"
        active_jobs[job_id]["duration_seconds"] = round(duration_sec, 1)

        # Sanitize video name for dedicated folder
        raw_stem = Path(original_filename).stem
        clean_video_name = re.sub(r'[^\w\-_\. ]', '_', raw_stem).strip()
        if not clean_video_name:
            clean_video_name = f"video_{job_id}"

        # Dedicated output folder named after the video
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        video_folder = outputs_dir / clean_video_name
        video_folder.mkdir(parents=True, exist_ok=True)

        # Snapshots directory inside the video folder
        video_snapshots_dir = video_folder / "snapshots"
        video_snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Root snapshots directory maintained for backwards compatibility
        legacy_snapshots_dir = Path("snapshots")
        legacy_snapshots_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{clean_video_name}_annotated.mp4"
        output_video_path = video_folder / output_filename

        # If resolution is larger than 1080p (e.g. 4K UHD), scale down to 1080p
        # for smooth browser decoding and reasonable file size
        out_w = width
        out_h = height
        if out_w > 1920:
            scale = 1920.0 / out_w
            out_w = int(out_w * scale)
            out_h = int(out_h * scale)
        out_w = out_w - (out_w % 2)
        out_h = out_h - (out_h % 2)

        # Initialize web-compatible H.264 writer
        writer = H264VideoWriter(
            output_video_path,
            fps=fps,
            width=out_w,
            height=out_h
        )

        # Reset ByteTrack tracker state for every new video upload session
        global_model_manager.reset_tracker()
        unique_track_ids: set[int] = set()
        tracked_identities: Dict[int, Dict[str, Any]] = {}

        frame_idx = 0
        total_detections = 0
        frame_target_counts = []
        peak_conf = 0.0
        peak_frame_idx = 0
        peak_snapshot_path = None
        keyframe_events = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Run 1-channel model inference
            det_result = global_model_manager.predict_frame(frame, conf_threshold=conf_threshold)
            # Merge sub-body part detections into complete bounding boxes
            boxes = merge_sub_body_boxes(det_result["boxes"])
            count = len(boxes)
            frame_target_counts.append(count)

            if count > 0:
                total_detections += count

                # Active tracks eligible for association within 45 frames (1.5s recovery window)
                active_tids = [
                    tid for tid, t_data in tracked_identities.items()
                    if frame_idx - t_data["frames"][-1] <= 45
                ]
                assigned_tids = set()

                # Associate clean detections with continuous tracks (1-to-1 matching)
                for b in boxes:
                    cx = (b["x1"] + b["x2"]) / 2.0
                    cy = (b["y1"] + b["y2"]) / 2.0
                    cur_box = [b["x1"], b["y1"], b["x2"], b["y2"]]

                    best_tid = None
                    best_score = float("inf")

                    for tid in active_tids:
                        if tid in assigned_tids:
                            continue
                        t_data = tracked_identities[tid]
                        last_cx, last_cy = t_data["centroids"][-1]
                        last_box = t_data["boxes"][-1]
                        dist = ((cx - last_cx)**2 + (cy - last_cy)**2)**0.5
                        iou = compute_box_iou(cur_box, last_box)

                        # Cost metric: heavily rewards IoU overlap, uses Euclidean distance for motion continuity
                        cost = dist - (iou * 200.0)
                        if dist < 250.0 and cost < best_score:
                            best_score = cost
                            best_tid = tid

                    if best_tid is not None:
                        b["track_id"] = best_tid
                        assigned_tids.add(best_tid)
                        unique_track_ids.add(best_tid)
                        tracked_identities[best_tid]["frames"].append(frame_idx)
                        tracked_identities[best_tid]["centroids"].append((cx, cy))
                        tracked_identities[best_tid]["boxes"].append(cur_box)
                    else:
                        new_tid = len(tracked_identities) + 1
                        b["track_id"] = new_tid
                        assigned_tids.add(new_tid)
                        unique_track_ids.add(new_tid)
                        tracked_identities[new_tid] = {
                            "frames": [frame_idx],
                            "centroids": [(cx, cy)],
                            "boxes": [cur_box]
                        }

                # Check for peak confidence in frame
                frame_peak = max(b["confidence"] for b in boxes)
                if frame_peak > peak_conf:
                    peak_conf = frame_peak
                    peak_frame_idx = frame_idx

                    # Save peak snapshot inside video folder
                    snapshot_filename = f"peak_frame_{frame_idx}.jpg"
                    snap_path = video_snapshots_dir / snapshot_filename
                    annotated_snap = global_model_manager.annotate_frame(frame, boxes)
                    cv2.imwrite(str(snap_path), annotated_snap)
                    # Also write to legacy snapshots dir for backup
                    cv2.imwrite(str(legacy_snapshots_dir / f"{job_id}_{snapshot_filename}"), annotated_snap)
                    peak_snapshot_path = f"/outputs/{clean_video_name}/snapshots/{snapshot_filename}"

                # Store evidentiary event for this frame if significant or periodically
                if len(keyframe_events) < 50 and (frame_idx % max(1, int(fps)) == 0 or frame_peak > 0.40):
                    snap_file = f"event_f{frame_idx}.jpg"
                    ev_path = video_snapshots_dir / snap_file
                    annotated_ev = global_model_manager.annotate_frame(frame, boxes)
                    cv2.imwrite(str(ev_path), annotated_ev)
                    cv2.imwrite(str(legacy_snapshots_dir / f"{job_id}_{snap_file}"), annotated_ev)

                    event_id = str(uuid.uuid4())
                    ev_data = {
                        "event_id": event_id,
                        "job_id": job_id,
                        "filename": original_filename,
                        "frame_number": frame_idx,
                        "pedestrians_count": count,
                        "targets_count": count,
                        "confidence": float(frame_peak),
                        "snapshot_url": f"/outputs/{clean_video_name}/snapshots/{snap_file}",
                        "processed_video_url": f"/outputs/{clean_video_name}/{output_filename}",
                        "unique_pedestrians": calculate_unique_targets(tracked_identities, total_frames, total_detections),
                        "unique_targets": calculate_unique_targets(tracked_identities, total_frames, total_detections),
                        "folder_name": clean_video_name
                    }
                    keyframe_events.append(ev_data)
                    add_detection_event(**ev_data)

            # Annotate and write to H.264 video
            annotated_frame = global_model_manager.annotate_frame(frame, boxes)
            writer.write(annotated_frame)

            # Unique targets calculation from consolidated identities
            unique_count = calculate_unique_targets(tracked_identities, total_frames, total_detections)

            # Update progress
            progress_pct = (frame_idx / total_frames * 100.0) if total_frames > 0 else 0.0
            active_jobs[job_id]["current_frame"] = frame_idx
            active_jobs[job_id]["progress_percent"] = round(progress_pct, 1)
            active_jobs[job_id]["current_targets"] = count
            active_jobs[job_id]["unique_targets_count"] = unique_count
            active_jobs[job_id]["target_detections_count"] = unique_count
            active_jobs[job_id]["total_detection_instances"] = total_detections
            active_jobs[job_id]["peak_confidence"] = round(peak_conf, 3)
            active_jobs[job_id]["peak_frame_index"] = peak_frame_idx

            # Aliases for backwards compatibility
            active_jobs[job_id]["current_pedestrians"] = count
            active_jobs[job_id]["unique_pedestrians_count"] = unique_count
            active_jobs[job_id]["pedestrian_detections_count"] = unique_count

        # Final unique count with trajectory continuity
        final_unique_count = calculate_unique_targets(tracked_identities, total_frames, total_detections)

        # Safely release resources
        cap.release()
        cap = None
        writer.release()
        writer = None

        # If no targets detected at all, take middle frame as representative snapshot
        if peak_snapshot_path is None:
            cap_mid = cv2.VideoCapture(str(input_video_path))
            cap_mid.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
            ret_mid, frame_mid = cap_mid.read()
            cap_mid.release()
            if ret_mid and frame_mid is not None:
                mid_filename = f"thumbnail_{job_id}.jpg"
                mid_path = video_snapshots_dir / mid_filename
                cv2.imwrite(str(mid_path), frame_mid)
                cv2.imwrite(str(legacy_snapshots_dir / mid_filename), frame_mid)
                peak_snapshot_path = f"/outputs/{clean_video_name}/snapshots/{mid_filename}"

        # Write comprehensive detection report inside the video folder
        report_data = {
            "video_name": original_filename,
            "folder_name": clean_video_name,
            "folder_path": f"outputs/{clean_video_name}",
            "job_id": job_id,
            "status": "completed",
            "total_frames": frame_idx,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "duration_seconds": round(duration_sec, 2),
            "unique_targets_count": final_unique_count,
            "total_detection_instances": total_detections,
            "peak_confidence": round(peak_conf, 3),
            "peak_frame_index": peak_frame_idx,
            "annotated_video_file": output_filename,
            "processed_video_url": f"/outputs/{clean_video_name}/{output_filename}",
            "peak_snapshot_url": peak_snapshot_path,
            "keyframes_recorded": len(keyframe_events),
            "completed_at": datetime.now().isoformat()
        }
        try:
            with open(video_folder / "detection_report.json", "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)

            # Copy source video into video folder as well for self-contained archive
            if input_video_path.exists():
                src_dest = video_folder / f"source_{Path(original_filename).name}"
                if not src_dest.exists():
                    shutil.copy2(input_video_path, src_dest)
        except Exception as err:
            print(f"[!] Notice: writing dossier report: {err}")

        # Record session in database
        add_video_session(
            job_id=job_id,
            filename=original_filename,
            total_frames=frame_idx,
            pedestrian_detections=total_detections,
            peak_confidence=round(peak_conf, 3),
            snapshot_url=peak_snapshot_path,
            processed_video_url=f"/outputs/{clean_video_name}/{output_filename}",
            unique_pedestrians=final_unique_count
        )

        # Collect all captured frames in the video's snapshot folder
        captured_frames = []
        if video_snapshots_dir.exists():
            for f in sorted(video_snapshots_dir.glob("*.jpg")):
                m = re.search(r'(\d+)', f.name)
                f_num = int(m.group(1)) if m else 0
                captured_frames.append({
                    "frame_number": f_num,
                    "filename": f.name,
                    "url": f"/outputs/{clean_video_name}/snapshots/{f.name}",
                    "confidence": round(peak_conf, 3),
                    "targets_count": final_unique_count
                })
        captured_frames.sort(key=lambda x: x["frame_number"])

        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        active_jobs[job_id]["folder_name"] = clean_video_name
        active_jobs[job_id]["folder_path"] = f"outputs/{clean_video_name}"
        active_jobs[job_id]["report_url"] = f"/outputs/{clean_video_name}/detection_report.json"
        active_jobs[job_id]["processed_video_url"] = f"/outputs/{clean_video_name}/{output_filename}"
        active_jobs[job_id]["peak_snapshot_url"] = peak_snapshot_path
        active_jobs[job_id]["progress_percent"] = 100.0
        active_jobs[job_id]["frames"] = captured_frames
        active_jobs[job_id]["snapshots"] = [item["url"] for item in captured_frames]
        active_jobs[job_id]["keyframes"] = keyframe_events
        active_jobs[job_id]["current_targets"] = frame_target_counts[0] if frame_target_counts else 0
        active_jobs[job_id]["unique_targets_count"] = final_unique_count
        active_jobs[job_id]["target_detections_count"] = final_unique_count
        active_jobs[job_id]["total_detection_instances"] = total_detections
        active_jobs[job_id]["frame_target_counts"] = frame_target_counts

        # Backwards compatibility
        active_jobs[job_id]["current_pedestrians"] = frame_target_counts[0] if frame_target_counts else 0
        active_jobs[job_id]["unique_pedestrians_count"] = final_unique_count
        active_jobs[job_id]["pedestrian_detections_count"] = final_unique_count
        active_jobs[job_id]["frame_pedestrian_counts"] = frame_target_counts

        print(f"[+] Thermal video analysis completed for {original_filename} in folder 'outputs/{clean_video_name}': {final_unique_count} unique targets ({total_detections} detections across {frame_idx} frames).")

    except Exception as e:
        print(f"[!] Video analysis failed for job {job_id}: {e}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error_message"] = str(e)
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def start_video_processing_thread(
    job_id: str,
    input_video_path: Path,
    original_filename: str,
    conf_threshold: float = 0.25
):
    raw_stem = Path(original_filename).stem
    clean_video_name = re.sub(r'[^\w\-_\. ]', '_', raw_stem).strip() or f"video_{job_id}"

    active_jobs[job_id] = {
        "job_id": job_id,
        "filename": original_filename,
        "folder_name": clean_video_name,
        "folder_path": f"outputs/{clean_video_name}",
        "status": "pending",
        "current_frame": 0,
        "total_frames": 0,
        "progress_percent": 0.0,
        "unique_targets_count": 0,
        "target_detections_count": 0,
        "unique_pedestrians_count": 0,
        "pedestrian_detections_count": 0,
        "total_detection_instances": 0,
        "peak_confidence": 0.0,
        "peak_frame_index": 0,
        "created_at": datetime.now().isoformat(),
        "fps": 25.0,
        "duration_seconds": 0.0,
        "resolution": "640x480"
    }

    t = threading.Thread(
        target=run_video_analysis_job,
        args=(job_id, input_video_path, original_filename, conf_threshold),
        daemon=True
    )
    t.start()
