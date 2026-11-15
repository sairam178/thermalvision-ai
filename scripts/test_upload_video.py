import urllib.request
import json
import time

# 1. Check model status
resp = urllib.request.urlopen("http://127.0.0.1:8000/api/model/status")
print("Model status:", json.loads(resp.read().decode("utf-8")))

# 2. Test HTTP multipart upload of Video_Project_2.mp4
video_path = r"C:/Users/tharu/.gemini/antigravity/brain/d8f625c7-709c-41a4-b60a-929dcbda5862/.user_uploaded/media_1794635711855_7c775cdc.mp4"

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
with open(video_path, "rb") as f:
    file_bytes = f.read()

part1 = f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; filename=\"Video_Project_2.mp4\"\r\nContent-Type: video/mp4\r\n\r\n".encode("utf-8")
part2 = f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"conf\"\r\n\r\n0.25\r\n--{boundary}--\r\n".encode("utf-8")
body = part1 + file_bytes + part2

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/detect/video",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)

upload_resp = urllib.request.urlopen(req)
res_data = json.loads(upload_resp.read().decode("utf-8"))
print("Upload response:", res_data)
job_id = res_data["job_id"]

# Poll status until done
while True:
    time.sleep(2)
    status_resp = urllib.request.urlopen(f"http://127.0.0.1:8000/api/detect/video/{job_id}/status")
    status_data = json.loads(status_resp.read().decode("utf-8"))
    pct = status_data.get("progress_percent", 0)
    cur = status_data.get("current_frame", 0)
    st = status_data.get("status")
    uniq = status_data.get("unique_pedestrians_count", 0)
    det = status_data.get("total_detection_instances", 0)
    print(f"Status: {st} | Progress: {pct:.1f}% | Frame: {cur} | Unique: {uniq} | Detections: {det}", flush=True)
    if st in ["completed", "failed"]:
        print("Final Job Result:", json.dumps(status_data, indent=2))
        break

