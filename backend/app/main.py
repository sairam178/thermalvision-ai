import os
import uuid
import base64
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from .model_loader import global_model_manager
from .video_processor import start_video_processing_thread, get_job_status
from .nlp_engine import global_nlp_engine
from .database import (
    init_db,
    add_detection_event,
    get_all_events,
    delete_event,
    clear_all_events,
    get_overview_stats
)

# Initialize database tables
init_db()

# Ensure directories exist
for d in ["uploads", "outputs", "snapshots", "models"]:
    Path(d).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="ThermalVision AI - Thermal Object & Target Detection System",
    description="Backend API for 1-Channel LWIR Thermal Object and Target Detection using YOLOv8",
    version="2.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts for video playback and snapshot images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")

# Pydantic models
class ImageDetectRequest(BaseModel):
    image: str # Base64 encoded JPEG/PNG
    conf: Optional[float] = 0.25

class NLPChatRequest(BaseModel):
    query: str
    context: Optional[dict] = None

class NLPReportRequest(BaseModel):
    context: dict

@app.post("/api/nlp/chat")
def nlp_chat_query(req: NLPChatRequest):
    try:
        res = global_nlp_engine.answer_query(req.query, req.context)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/nlp/generate-report")
def nlp_generate_report(req: NLPReportRequest):
    try:
        report = global_nlp_engine.generate_narrative_report(req.context)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints
@app.get("/api/health")
def get_health():
    return {
        "status": "healthy",
        "service": "ThermalVision AI Target Detection",
        "model_loaded": global_model_manager.is_loaded
    }

@app.get("/api/model/status")
def get_model_status():
    return global_model_manager.get_status()

@app.get("/api/overview/stats")
def get_overview_statistics():
    db_stats = get_overview_stats()
    return {
        "model_status": global_model_manager.get_status(),
        "total_videos_analyzed": db_stats["total_videos_analyzed"],
        "total_targets_detected": db_stats.get("total_targets_detected", db_stats["total_pedestrian_detections"]),
        "total_unique_targets": db_stats.get("total_unique_targets", db_stats.get("total_unique_pedestrians", 0)),
        "total_detections": db_stats.get("total_detections", db_stats["total_pedestrian_detections"]),
        "total_pedestrian_detections": db_stats["total_pedestrian_detections"], # legacy key
        "total_frames_processed": db_stats["total_frames_processed"],
        "recent_sessions": db_stats["recent_sessions"]
    }

@app.post("/api/detect/image")
async def detect_image_base64(req: ImageDetectRequest):
    try:
        # Decode base64 string
        encoded_data = req.image
        if "," in encoded_data:
            encoded_data = encoded_data.split(",")[1]

        image_bytes = base64.b64decode(encoded_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode base64 image")

        res = global_model_manager.predict_frame(frame, conf_threshold=req.conf or 0.25)
        
        # Render annotated frame with high-tech brackets
        annotated = global_model_manager.annotate_frame(frame, res["boxes"])
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        res["annotated_image"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        res["image_width"] = frame.shape[1]
        res["image_height"] = frame.shape[0]
        return res

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect/image-file")
async def detect_image_file(
    file: UploadFile = File(...),
    conf: float = Form(0.25)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        res = global_model_manager.predict_frame(frame, conf_threshold=conf)

        # Render annotated frame with high-tech brackets
        annotated = global_model_manager.annotate_frame(frame, res["boxes"])
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        res["annotated_image"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        res["image_width"] = frame.shape[1]
        res["image_height"] = frame.shape[0]
        res["filename"] = file.filename
        return res

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/samples")
def list_sample_images():
    samples_dir = Path("frontend/public/samples")
    if not samples_dir.exists():
        samples_dir = Path("public/samples")
    
    if samples_dir.exists():
        samples = [f"/samples/{f.name}" for f in samples_dir.glob("*.png") if f.is_file()]
        return {"samples": sorted(samples)}
    return {"samples": []}

@app.post("/api/detect/video")
async def detect_video(
    video: UploadFile = File(...),
    conf: float = Form(0.25)
):
    # Validate extension
    ext = Path(video.filename).suffix.lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: '{ext}'. Please upload an MP4, AVI, MOV, or WEBM file."
        )

    job_id = str(uuid.uuid4())[:8]
    save_filename = f"{job_id}_{video.filename}"
    upload_path = Path("uploads") / save_filename

    try:
        with open(upload_path, "wb") as f:
            while chunk := await video.read(1024 * 1024): # 1MB chunk
                f.write(chunk)

        # Launch background processing worker
        start_video_processing_thread(
            job_id=job_id,
            input_video_path=upload_path,
            original_filename=video.filename,
            conf_threshold=conf
        )

        return {
            "job_id": job_id,
            "filename": video.filename,
            "message": "Video successfully uploaded. 1-Channel analysis started."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

@app.get("/api/detect/video/{job_id}/status")
def get_video_status(job_id: str):
    status = get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return status

@app.get("/api/detect/video/{job_id}/result")
def get_video_result(job_id: str):
    status = get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")

    if status.get("status") != "completed":
        return JSONResponse(status_code=202, content=status)

    out_url = status.get("processed_video_url")
    return {
        "status": "completed",
        "job_id": job_id,
        "processed_video_url": out_url,
        "folder_name": status.get("folder_name"),
        "folder_path": status.get("folder_path"),
        "report_url": status.get("report_url"),
        "details": status
    }

@app.get("/api/detect/video/{job_id}/folder")
def get_video_folder_contents(job_id: str):
    status = get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")

    folder_name = status.get("folder_name")
    if not folder_name:
        raise HTTPException(status_code=400, detail="Folder name not assigned yet")

    folder_dir = Path("outputs") / folder_name
    if not folder_dir.exists():
        raise HTTPException(status_code=404, detail=f"Folder 'outputs/{folder_name}' not found")

    files = []
    for item in folder_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(folder_dir).as_posix()
            files.append({
                "name": item.name,
                "relative_path": rel_path,
                "url": f"/outputs/{folder_name}/{rel_path}",
                "size_bytes": item.stat().st_size
            })

    return {
        "job_id": job_id,
        "folder_name": folder_name,
        "folder_path": f"outputs/{folder_name}",
        "files_count": len(files),
        "files": files
    }

@app.get("/api/folder/{folder_name}")
def get_folder_by_name(folder_name: str):
    clean_name = re.sub(r'[^\w\-_\. ]', '_', folder_name).strip()
    folder_dir = Path("outputs") / clean_name
    if not folder_dir.exists():
        raise HTTPException(status_code=404, detail=f"Folder 'outputs/{clean_name}' not found")

    files = []
    frames = []
    for item in sorted(folder_dir.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(folder_dir).as_posix()
            file_info = {
                "name": item.name,
                "relative_path": rel_path,
                "url": f"/outputs/{clean_name}/{rel_path}",
                "size_bytes": item.stat().st_size
            }
            files.append(file_info)
            if item.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                m = re.search(r'(\d+)', item.name)
                f_num = int(m.group(1)) if m else 0
                frames.append({
                    "frame_number": f_num,
                    "filename": item.name,
                    "url": file_info["url"],
                    "relative_path": rel_path,
                    "size_bytes": file_info["size_bytes"]
                })

    frames.sort(key=lambda x: x["frame_number"])
    return {
        "folder_name": clean_name,
        "folder_path": f"outputs/{clean_name}",
        "files_count": len(files),
        "frames_count": len(frames),
        "files": files,
        "frames": frames
    }

@app.get("/api/history")
def get_history_events():
    return get_all_events()

@app.post("/api/history/snapshot")
async def save_live_snapshot(
    file: UploadFile = File(...),
    filename: Optional[str] = Form("Photo Thermal Detection"),
    pedestrians_count: int = Form(0),
    targets_count: Optional[int] = Form(None),
    confidence: float = Form(0.0)
):
    try:
        snap_id = str(uuid.uuid4())
        save_filename = f"capture_{snap_id[:8]}.jpg"
        save_path = Path("snapshots") / save_filename

        count = targets_count if targets_count is not None else pedestrians_count

        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)

        add_detection_event(
            event_id=snap_id,
            job_id=None,
            filename=filename or "Photo Thermal Detection",
            frame_number=1,
            pedestrians_count=count,
            confidence=confidence,
            snapshot_url=f"/snapshots/{save_filename}",
            processed_video_url=None,
            unique_pedestrians=count
        )

        return {"success": True, "snapshot_url": f"/snapshots/{save_filename}", "event_id": snap_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/history/{event_id}")
def delete_single_event(event_id: str):
    success = delete_event(event_id)
    if not success:
        raise HTTPException(status_code=404, detail="History event not found")
    return {"success": True, "deleted_id": event_id}

@app.delete("/api/history")
def clear_history():
    count = clear_all_events()
    return {"success": True, "cleared_count": count}

# Optional: Mount frontend build dist for seamless single-origin deployment
dist_path = Path("frontend/dist")
if dist_path.exists():
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")

