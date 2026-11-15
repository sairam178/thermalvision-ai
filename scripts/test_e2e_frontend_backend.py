import json
import time
import base64
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def run_e2e_verification():
    print("[*] Starting End-to-End Frontend & Backend Verification...")

    # 1. Check Frontend Static Serving
    print("\n[1/7] Testing Frontend SPA Root Serving (GET /)...")
    res = requests.get(f"{BASE_URL}/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "THERMAL VISION" in res.text or "<div id=\"root\">" in res.text
    print("   [PASS] Frontend SPA HTML successfully served.")

    # 2. Check Health Endpoint
    print("\n[2/7] Testing Health Endpoint (GET /api/health)...")
    res = requests.get(f"{BASE_URL}/api/health")
    assert res.status_code == 200
    health = res.json()
    print(f"   [PASS] Service: {health.get('service')}, Model Loaded: {health.get('model_loaded')}")
    assert health.get("status") == "healthy"

    # 3. Check Model Status Endpoint
    print("\n[3/7] Testing Model Status (GET /api/model/status)...")
    res = requests.get(f"{BASE_URL}/api/model/status")
    assert res.status_code == 200
    m_status = res.json()
    print(f"   [PASS] Architecture: {m_status.get('architecture')}, Channels: {m_status.get('input_channels')}, Classes: {m_status.get('classes')}")
    assert m_status.get("input_channels") == 1
    assert m_status.get("loaded") is True

    # 4. Check Overview Statistics Endpoint
    print("\n[4/7] Testing Overview Statistics (GET /api/overview/stats)...")
    res = requests.get(f"{BASE_URL}/api/overview/stats")
    assert res.status_code == 200
    stats = res.json()
    print(f"   [PASS] Videos Analyzed: {stats.get('total_videos_analyzed')}, Frames: {stats.get('total_frames_processed')}, Detections: {stats.get('total_pedestrian_detections')}")
    assert "total_frames_processed" in stats
    assert "recent_sessions" in stats

    # 5. Check Image Inference (Simulates Live Camera Frame)
    print("\n[5/7] Testing Image Frame Detection (POST /api/detect/image)...")
    # Find test thermal image
    img_candidates = list(Path("test_extract/test/images").glob("*.png")) + list(Path("snapshots").glob("*.jpg"))
    assert len(img_candidates) > 0, "No test image found!"
    test_img_path = img_candidates[0]
    
    with open(test_img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    res = requests.post(f"{BASE_URL}/api/detect/image", json={"image": img_b64, "conf": 0.25})
    assert res.status_code == 200
    img_result = res.json()
    print(f"   [PASS] Pedestrians Detected: {img_result.get('pedestrians_count')}, Inference Time: {img_result.get('inference_time_ms'):.1f} ms")
    assert "boxes" in img_result
    assert img_result.get("pedestrians_count") >= 0

    # 6. Check Video Inference Workflow
    print("\n[6/7] Testing Video Ingestion & Processing Workflow (POST /api/detect/video)...")
    video_path = Path("uploads/test_thermal_patrol.mp4")
    assert video_path.exists(), "test_thermal_patrol.mp4 missing!"
    
    with open(video_path, "rb") as vf:
        res = requests.post(
            f"{BASE_URL}/api/detect/video",
            files={"video": ("test_thermal_patrol.mp4", vf, "video/mp4")},
            data={"conf": 0.25}
        )
    assert res.status_code == 200
    job_info = res.json()
    job_id = job_info["job_id"]
    print(f"   [PASS] Job submitted successfully. Job ID: {job_id}")

    # Poll status
    print("   Polling job status...")
    max_wait = 90
    start_t = time.time()
    last_pct = -1
    final_status = None
    while time.time() - start_t < max_wait:
        res = requests.get(f"{BASE_URL}/api/detect/video/{job_id}/status")
        assert res.status_code == 200
        status_data = res.json()
        pct = int(status_data.get("progress_percent", 0))
        if pct != last_pct and pct % 20 == 0:
            print(f"     -> Progress: {pct}% (Frame {status_data.get('current_frame')}/{status_data.get('total_frames')})")
            last_pct = pct
        if status_data.get("status") in ["completed", "failed"]:
            final_status = status_data
            break
        time.sleep(1)

    assert final_status is not None, "Job timed out"
    assert final_status["status"] == "completed", f"Job failed: {final_status.get('error_message')}"
    print(f"   [PASS] Video processing completed! Frames: {final_status['total_frames']}, Pedestrian Detections: {final_status['pedestrian_detections_count']}, Peak Confidence: {final_status['peak_confidence']*100:.1f}%")
    print(f"          Processed video URL: {final_status['processed_video_url']}")
    print(f"          Peak snapshot URL:   {final_status['peak_snapshot_url']}")

    # 7. Check History Vault
    print("\n[7/7] Testing Detection History Vault (GET /api/history)...")
    res = requests.get(f"{BASE_URL}/api/history")
    assert res.status_code == 200
    history = res.json()
    print(f"   [PASS] Retrieved {len(history)} archived detection events.")
    assert len(history) > 0
    recent = history[0]
    print(f"          Latest Event: {recent.get('filename')}, Frame #{recent.get('frame_number')}, Confidence: {recent.get('confidence')*100:.1f}%")

    print("\n" + "="*60)
    print("🎉 ALL 7 END-TO-END INTEGRATION CHECKS PASSED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    run_e2e_verification()
