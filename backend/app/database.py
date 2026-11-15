import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_DIR = Path("backend/data")
DB_PATH = DB_DIR / "surveillance.db"

def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Table for individual evidentiary frame captures
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detection_events (
        id TEXT PRIMARY KEY,
        job_id TEXT,
        filename TEXT NOT NULL,
        frame_number INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        pedestrians_count INTEGER NOT NULL,
        confidence REAL NOT NULL,
        snapshot_url TEXT NOT NULL,
        processed_video_url TEXT,
        unique_pedestrians INTEGER DEFAULT 0
    )
    """)

    # Table for video analysis sessions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS video_sessions (
        job_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        total_frames INTEGER NOT NULL,
        pedestrian_detections INTEGER NOT NULL,
        peak_confidence REAL NOT NULL,
        snapshot_url TEXT,
        processed_video_url TEXT,
        unique_pedestrians INTEGER DEFAULT 0
    )
    """)

    # Automatic column migration if database already exists
    try:
        cursor.execute("ALTER TABLE video_sessions ADD COLUMN unique_pedestrians INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE detection_events ADD COLUMN unique_pedestrians INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

# Auto-initialize database on import
init_db()

def add_detection_event(
    event_id: str,
    job_id: Optional[str] = None,
    filename: str = "",
    frame_number: int = 1,
    pedestrians_count: int = 0,
    confidence: float = 0.0,
    snapshot_url: str = "",
    processed_video_url: Optional[str] = None,
    timestamp: Optional[str] = None,
    unique_pedestrians: int = 0,
    **kwargs
):
    if not timestamp:
        timestamp = datetime.now().isoformat()

    # Support aliases
    count = kwargs.get("targets_count", pedestrians_count)
    unique_cnt = kwargs.get("unique_targets", unique_pedestrians)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO detection_events (id, job_id, filename, frame_number, timestamp, pedestrians_count, confidence, snapshot_url, processed_video_url, unique_pedestrians)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (event_id, job_id, filename, frame_number, timestamp, count, confidence, snapshot_url, processed_video_url, unique_cnt))
    conn.commit()
    conn.close()

def add_video_session(
    job_id: str,
    filename: str,
    total_frames: int,
    pedestrian_detections: int,
    peak_confidence: float,
    snapshot_url: Optional[str] = None,
    processed_video_url: Optional[str] = None,
    timestamp: Optional[str] = None,
    unique_pedestrians: int = 0,
    **kwargs
):
    if not timestamp:
        timestamp = datetime.now().isoformat()

    total_dets = kwargs.get("total_detections", kwargs.get("target_detections", pedestrian_detections))
    unique_cnt = kwargs.get("unique_targets", unique_pedestrians)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO video_sessions (job_id, filename, timestamp, total_frames, pedestrian_detections, peak_confidence, snapshot_url, processed_video_url, unique_pedestrians)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (job_id, filename, timestamp, total_frames, total_dets, peak_confidence, snapshot_url, processed_video_url, unique_cnt))
    conn.commit()
    conn.close()

def get_all_events(limit: int = 100) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, job_id, filename, frame_number, timestamp, pedestrians_count, confidence, snapshot_url, processed_video_url, unique_pedestrians
    FROM detection_events
    ORDER BY timestamp DESC
    LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    events = []
    for r in rows:
        d = dict(r)
        d["targets_count"] = d.get("pedestrians_count", 0)
        d["unique_targets"] = d.get("unique_pedestrians", 0)
        events.append(d)
    return events

def delete_event(event_id: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detection_events WHERE id = ?", (event_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def clear_all_events() -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detection_events")
    cursor.execute("DELETE FROM video_sessions")
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count

def get_overview_stats() -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM video_sessions")
    total_videos = cursor.fetchone()[0]

    cursor.execute("SELECT COALESCE(SUM(pedestrian_detections), 0) FROM video_sessions")
    total_pedestrians = cursor.fetchone()[0]

    cursor.execute("SELECT COALESCE(SUM(unique_pedestrians), 0) FROM video_sessions")
    total_unique_pedestrians = cursor.fetchone()[0]

    # Also count live snapshot events if any
    cursor.execute("SELECT COALESCE(SUM(pedestrians_count), 0) FROM detection_events WHERE job_id IS NULL OR job_id = ''")
    live_pedestrians = cursor.fetchone()[0]
    total_pedestrians += live_pedestrians

    cursor.execute("SELECT COALESCE(SUM(total_frames), 0) FROM video_sessions")
    total_frames = cursor.fetchone()[0]

    cursor.execute("""
    SELECT job_id, filename, timestamp, total_frames, pedestrian_detections, peak_confidence, snapshot_url, processed_video_url, unique_pedestrians
    FROM video_sessions
    ORDER BY timestamp DESC
    LIMIT 5
    """)
    recent_sessions = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return {
        "total_videos_analyzed": total_videos,
        "total_targets_detected": total_pedestrians,
        "total_unique_targets": total_unique_pedestrians,
        "total_detections": total_pedestrians,
        "total_pedestrian_detections": total_pedestrians,
        "total_unique_pedestrians": total_unique_pedestrians,
        "total_frames_processed": total_frames,
        "recent_sessions": recent_sessions
    }
