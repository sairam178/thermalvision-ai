import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.model_loader import global_model_manager
from backend.app.video_processor import run_video_analysis_job, active_jobs
from backend.app.database import get_overview_stats, get_all_events

def test_video_inference():
    print("=" * 75)
    print("TESTING BACKEND VIDEO INFERENCE WITH 1-CHANNEL YOLOV8 (best.pt)")
    print("=" * 75)

    # 1. Verify model loader & checkpoint
    print(f"[+] 1. Model Loader Status: {'Loaded' if global_model_manager.is_loaded else 'Offline'}")
    print(f"       - Model Checkpoint Path: {global_model_manager.model_path}")
    print(f"       - Input Channels       : {global_model_manager.input_channels} (VERIFIED 1-CHANNEL)")
    print(f"       - Classes Target       : {global_model_manager.model.names if hasattr(global_model_manager.model, 'names') else 'person'}")
    assert global_model_manager.is_loaded, "Model manager must be loaded"
    assert global_model_manager.input_channels == 1, "Must have exactly 1 input channel"

    # 2. Verify test video
    input_video = Path("uploads/test_thermal_patrol.mp4")
    if not input_video.exists():
        print(f"[FAIL] Test video not found at: {input_video}")
        sys.exit(1)

    cap_check = cv2.VideoCapture(str(input_video))
    total_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_check.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_check.release()

    print(f"\n[+] 2. Target Test Video: {input_video.resolve()}")
    print(f"       - Dimensions : {w}x{h}")
    print(f"       - Total Frames: {total_frames}")
    print(f"       - Framerate   : {fps} FPS")

    # 3. Execute video analysis job
    job_id = "test_vid_001"
    active_jobs[job_id] = {
        "job_id": job_id,
        "filename": input_video.name,
        "status": "pending",
        "current_frame": 0,
        "total_frames": total_frames,
        "progress_percent": 0.0,
        "pedestrian_detections_count": 0,
        "peak_confidence": 0.0,
        "peak_frame_index": 0,
        "peak_snapshot_url": None,
        "processed_video_url": None,
        "fps": fps,
        "resolution": f"{w}x{h}",
        "duration_seconds": total_frames / fps
    }

    print(f"\n[+] 3. Running Video Analysis Pipeline (Job ID: {job_id})...")
    start_time = time.perf_counter()

    # Run the worker function directly
    run_video_analysis_job(
        job_id=job_id,
        input_video_path=input_video,
        original_filename=input_video.name,
        conf_threshold=0.25
    )

    elapsed_sec = time.perf_counter() - start_time
    job_result = active_jobs[job_id]

    print("\n" + "=" * 75)
    print("VIDEO PROCESSING JOB RESULTS")
    print("=" * 75)
    print(f"  Status                  : {job_result.get('status')}")
    print(f"  Total Frames Processed  : {job_result.get('current_frame')} / {job_result.get('total_frames')}")
    print(f"  Elapsed Processing Time : {elapsed_sec:.2f} seconds ({total_frames / elapsed_sec:.1f} FPS)")
    print(f"  Total Pedestrian Targets: {job_result.get('pedestrian_detections_count')} detections")
    print(f"  Peak Confidence Score   : {job_result.get('peak_confidence') * 100:.1f}% (Frame #{job_result.get('peak_frame_index')})")
    print(f"  Peak Snapshot Path      : {job_result.get('peak_snapshot_url')}")
    print(f"  Processed Output Video  : {job_result.get('processed_video_url')}")

    assert job_result.get("status") == "completed", "Job did not complete successfully"
    assert job_result.get("current_frame") == total_frames, "All frames must be processed"
    assert job_result.get("pedestrian_detections_count") > 0, "Expected pedestrian detections in test video"

    # 4. Verify output video file on disk
    out_video_rel = job_result.get("processed_video_url").lstrip("/")
    out_video_path = Path(out_video_rel)
    assert out_video_path.exists(), f"Output video file not found at: {out_video_path}"
    out_size_mb = out_video_path.stat().st_size / (1024 * 1024)
    print(f"\n[+] 4. Processed Output Video File Verified:")
    print(f"       - Path: {out_video_path.resolve()}")
    print(f"       - Size: {out_size_mb:.2f} MB")

    # Open output video and verify frame count
    out_cap = cv2.VideoCapture(str(out_video_path))
    out_frame_count = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_cap.release()
    print(f"       - Encoded Output Frames: {out_frame_count} (100% frame retention)")
    assert out_frame_count == total_frames, f"Expected {total_frames} frames in output, got {out_frame_count}"

    # 5. Verify database logging
    db_stats = get_overview_stats()
    events = get_all_events(limit=10)
    print(f"\n[+] 5. Database Logging Verification:")
    print(f"       - Total Recorded Sessions : {db_stats['total_videos_analyzed']}")
    print(f"       - Total Logged Detections : {db_stats['total_pedestrian_detections']}")
    print(f"       - Archived Keyframe Events: {len(events)} events in Evidence Vault")

    print("\n" + "=" * 75)
    print("BACKEND VIDEO INFERENCE TEST: 100% PASSED")
    print("=" * 75)

if __name__ == "__main__":
    test_video_inference()

