import time
from pathlib import Path
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_api_video_upload_flow():
    print("=" * 75)
    print("TESTING FASTAPI VIDEO UPLOAD & ASYNC INFERENCE LIFECYCLE")
    print("=" * 75)

    test_video_path = Path("uploads/test_thermal_patrol.mp4")
    assert test_video_path.exists(), "Test video must exist"

    # 1. POST /api/detect/video
    print(f"[+] 1. Submitting Video to POST /api/detect/video: {test_video_path.name}")
    with open(test_video_path, "rb") as f:
        response = client.post(
            "/api/detect/video",
            files={"video": ("test_thermal_patrol.mp4", f, "video/mp4")},
            data={"conf": "0.25"}
        )

    assert response.status_code == 200, f"Upload failed: {response.text}"
    upload_res = response.json()
    job_id = upload_res["job_id"]
    print(f"       -> Job ID Assigned: {job_id}")
    print(f"       -> Server Message : {upload_res['message']}")

    # 2. Poll GET /api/detect/video/{job_id}/status
    print("\n[+] 2. Polling GET /api/detect/video/{job_id}/status until completion...")
    max_wait = 60
    start = time.time()
    last_reported_frame = -1

    while time.time() - start < max_wait:
        st_res = client.get(f"/api/detect/video/{job_id}/status")
        assert st_res.status_code == 200, f"Status check failed: {st_res.text}"
        st_data = st_res.json()

        cur_frame = st_data.get("current_frame", 0)
        tot_frames = st_data.get("total_frames", 0)
        pct = st_data.get("progress_percent", 0.0)
        status = st_data.get("status")

        if cur_frame != last_reported_frame and cur_frame % 15 == 0:
            print(f"       -> Status: {status:<10} | Progress: {pct:>5.1f}% | Frame: {cur_frame}/{tot_frames} | Detections: {st_data.get('pedestrian_detections_count', 0)}")
            last_reported_frame = cur_frame

        if status == "completed":
            print(f"       -> [DONE] Status: {status} | 100% Processed ({cur_frame}/{tot_frames} frames)")
            break
        elif status == "failed":
            print(f"       -> [FAIL] Processing failed: {st_data.get('error_message')}")
            sys.exit(1)

        time.sleep(0.5)

    assert st_data.get("status") == "completed", "Job did not complete within timeout"

    # 3. GET /api/detect/video/{job_id}/result
    print("\n[+] 3. Fetching Final Result via GET /api/detect/video/{job_id}/result...")
    res_response = client.get(f"/api/detect/video/{job_id}/result")
    assert res_response.status_code == 200, f"Result endpoint failed: {res_response.text}"
    final_data = res_response.json()

    print(f"       -> Status               : {final_data['status']}")
    print(f"       -> Processed Video URL  : {final_data['processed_video_url']}")
    print(f"       -> Total Detections     : {final_data['details']['pedestrian_detections_count']}")
    print(f"       -> Peak Confidence Score: {final_data['details']['peak_confidence'] * 100:.1f}%")
    print(f"       -> Peak Snapshot URL    : {final_data['details']['peak_snapshot_url']}")

    # 4. Verify file exists on disk
    out_file = Path(final_data['processed_video_url'].lstrip("/"))
    assert out_file.exists(), f"Output video file missing: {out_file}"
    print(f"\n[+] 4. Processed MP4 File Confirmed on Disk: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")

    print("\n" + "=" * 75)
    print("FASTAPI VIDEO INFERENCE LIFECYCLE: 100% PASSED")
    print("=" * 75)

if __name__ == "__main__":
    test_api_video_upload_flow()

