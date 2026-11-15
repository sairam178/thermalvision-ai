import json
import base64
import numpy as np
import cv2
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_endpoints():
    print("=" * 60)
    print("TESTING BACKEND API ENDPOINTS")
    print("=" * 60)

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("[PASS] GET /api/health ->", res.json())

    # 2. Model status
    res = client.get("/api/model/status")
    assert res.status_code == 200, f"Model status failed: {res.text}"
    status = res.json()
    assert status["input_channels"] == 1, f"Expected 1 input channel, got {status['input_channels']}"
    print("[PASS] GET /api/model/status ->", status["architecture"], f"(Channels: {status['input_channels']})")

    # 3. Overview stats
    res = client.get("/api/overview/stats")
    assert res.status_code == 200, f"Overview stats failed: {res.text}"
    stats = res.json()
    print("[PASS] GET /api/overview/stats -> Total videos:", stats["total_videos_analyzed"], "Total detections:", stats["total_pedestrian_detections"])

    # 4. Image detection (base64)
    dummy_img = np.zeros((480, 640), dtype=np.uint8)
    # Add a bright simulated thermal pedestrian blob
    cv2.ellipse(dummy_img, (320, 240), (25, 60), 0, 0, 360, 255, -1)
    cv2.circle(dummy_img, (320, 160), 18, 255, -1)
    _, buffer = cv2.imencode(".jpg", dummy_img)
    b64_str = base64.b64encode(buffer).decode("utf-8")

    res = client.post("/api/detect/image", json={"image": b64_str, "conf": 0.15})
    assert res.status_code == 200, f"Detect image failed: {res.text}"
    det = res.json()
    print(f"[PASS] POST /api/detect/image -> Pedestrians: {det['pedestrians_count']}, Latency: {det['inference_time_ms']} ms")

    # 5. History snapshot save
    files = {"file": ("test_snapshot.jpg", buffer.tobytes(), "image/jpeg")}
    data = {"pedestrians_count": "1", "confidence": "0.92"}
    res = client.post("/api/history/snapshot", files=files, data=data)
    assert res.status_code == 200, f"Save snapshot failed: {res.text}"
    print("[PASS] POST /api/history/snapshot ->", res.json())

    # 6. History fetch
    res = client.get("/api/history")
    assert res.status_code == 200, f"Get history failed: {res.text}"
    events = res.json()
    assert len(events) > 0, "Expected at least 1 history event"
    print(f"[PASS] GET /api/history -> Retrieved {len(events)} events.")

    # 7. Delete event
    event_id = events[0]["id"]
    res = client.delete(f"/api/history/{event_id}")
    assert res.status_code == 200, f"Delete event failed: {res.text}"
    print(f"[PASS] DELETE /api/history/{event_id} -> Deleted successfully.")

    print("=" * 60)
    print("ALL API ENDPOINT INTEGRATION TESTS PASSED (100%)")
    print("=" * 60)

if __name__ == "__main__":
    test_endpoints()

