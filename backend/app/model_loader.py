import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class ThermalModelManager:
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = None
        self.is_loaded = False
        self.input_channels = 1
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.load_model(model_path)

    def find_model_checkpoint(self, preferred_path: str = None) -> Path:
        candidates = []
        if preferred_path:
            candidates.append(Path(preferred_path))

        env_path = os.environ.get("MODEL_PATH")
        if env_path:
            candidates.append(Path(env_path))

        candidates.extend([
            Path("backend/models/best.pt"),
            Path("models/best.pt"),
            Path("weights/best.pt"),
            Path("weights/yolov8n_thermal_1ch.pt"),
            Path("models/yolov8n_thermal_1ch.pt"),
            Path("yolov8n.pt")
        ])

        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()

        return Path("models/best.pt").resolve()

    def load_model(self, model_path: str = None):
        target_path = self.find_model_checkpoint(model_path)
        self.model_path = str(target_path)

        if not target_path.exists():
            print(f"[!] Warning: Model checkpoint not found at {target_path}. Running in standby.")
            self.is_loaded = False
            return

        try:
            print(f"[+] Loading thermal model from: {target_path}")
            self.model = YOLO(str(target_path))
            
            # Inspect first layer in_channels
            try:
                first_conv = self.model.model.model[0].conv
                self.input_channels = getattr(first_conv, "in_channels", 1)
            except Exception:
                self.input_channels = 1

            self.is_loaded = True
            print(f"[+] Model loaded successfully. Input channels: {self.input_channels}, Device: {self.device}")
        except Exception as e:
            print(f"[!] Error loading model from {target_path}: {e}")
            self.is_loaded = False

    def preprocess_to_1channel(self, frame: np.ndarray) -> np.ndarray:
        """
        Guarantees single-channel grayscale (H, W) input for thermal model.
        """
        if frame is None:
            raise ValueError("Empty frame provided for preprocessing")

        if len(frame.shape) == 3:
            # Convert BGR/RGB to genuine single-channel grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 2:
            gray = frame
        else:
            gray = frame.squeeze()

        return gray

    def reset_tracker(self):
        """
        Resets ByteTrack tracking state and STrack ID counters for a clean video session.
        """
        if self.model is not None:
            if hasattr(self.model, 'predictor') and self.model.predictor is not None:
                self.model.predictor.trackers = []
        try:
            from ultralytics.trackers.byte_tracker import STrack
            STrack.reset_id()
        except Exception as e:
            print(f"[!] Warning: STrack.reset_id() notice: {e}")

    def track_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        persist: bool = True
    ) -> Dict[str, Any]:
        """
        Runs Ultralytics ByteTrack on 1-channel thermal input.
        Maintains consistent tracking IDs across consecutive frames.
        """
        if not self.is_loaded or self.model is None:
            return {
                "targets_count": 0,
                "detections_count": 0,
                "pedestrians_count": 0,
                "boxes": [],
                "inference_time_ms": 0.0,
                "error": "Model not loaded"
            }

        start_time = time.perf_counter()

        # Convert to 1-channel grayscale
        gray_frame = self.preprocess_to_1channel(frame)

        # Run ByteTrack tracking
        results = self.model.track(
            source=gray_frame,
            persist=persist,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            imgsz=640,
            device=self.device,
            verbose=False
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        boxes_list: List[Dict[str, Any]] = []
        if results and len(results) > 0:
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                track_ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else [None] * len(xyxy)

                for box, conf, cls_id, tid in zip(xyxy, confs, clss, track_ids):
                    boxes_list.append({
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "class_name": "target",
                        "track_id": int(tid) if tid is not None else None
                    })

        return {
            "targets_count": len(boxes_list),
            "detections_count": len(boxes_list),
            "pedestrians_count": len(boxes_list),
            "boxes": boxes_list,
            "inference_time_ms": round(elapsed_ms, 2)
        }

    def predict_frame(self, frame: np.ndarray, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Runs 1-channel inference on a frame and returns structured detections.
        """
        if not self.is_loaded or self.model is None:
            return {
                "targets_count": 0,
                "detections_count": 0,
                "pedestrians_count": 0,
                "boxes": [],
                "inference_time_ms": 0.0,
                "error": "Model not loaded"
            }

        start_time = time.perf_counter()

        # Convert to 1-channel grayscale
        gray_frame = self.preprocess_to_1channel(frame)

        # Run inference
        results = self.model.predict(
            source=gray_frame,
            conf=conf_threshold,
            imgsz=640,
            device=self.device,
            verbose=False
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        boxes_list: List[Dict[str, Any]] = []
        if results and len(results) > 0:
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(xyxy, confs, clss):
                    boxes_list.append({
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "class_name": "target"
                    })

        return {
            "targets_count": len(boxes_list),
            "detections_count": len(boxes_list),
            "pedestrians_count": len(boxes_list),
            "boxes": boxes_list,
            "inference_time_ms": round(elapsed_ms, 2)
        }

    def annotate_frame(self, frame: np.ndarray, boxes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Renders cyber/surveillance neon cyan bounding brackets and confidence labels.
        """
        annotated = frame.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        h, w = annotated.shape[:2]

        for b in boxes:
            x1, y1 = int(round(b["x1"])), int(round(b["y1"]))
            x2, y2 = int(round(b["x2"])), int(round(b["y2"]))
            conf = b.get("confidence", 0.0)

            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            box_w = x2 - x1
            box_h = y2 - y1

            # Neon Cyan Color in BGR: (255, 229, 0) -> #00e5ff
            cyan_color = (255, 229, 0)
            corner_color = (250, 189, 56) # Sky Blue

            # Main bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), cyan_color, 2)

            # High-tech corner brackets
            corner_len = min(14, max(5, box_w // 4))
            thickness = 3
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), corner_color, thickness)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), corner_color, thickness)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), corner_color, thickness)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), corner_color, thickness)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), corner_color, thickness)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), corner_color, thickness)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), corner_color, thickness)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), corner_color, thickness)

            # Text label: Never show internal tracking IDs on visual output
            label = f"Target {int(conf * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

            label_y = max(th + 6, y1 - 6)
            cv2.rectangle(
                annotated,
                (x1, label_y - th - 4),
                (x1 + tw + 8, label_y + 4),
                (14, 10, 7), # Dark navy/black background
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y - 2),
                font,
                font_scale,
                cyan_color,
                1,
                cv2.LINE_AA
            )

        return annotated

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model_name": Path(self.model_path).name if self.model_path else "Custom YOLOv8 1-Channel",
            "model_path": self.model_path,
            "architecture": "YOLOv8n 1-Channel (Adapted Layer 0 Conv2d(1, 16, 3, 2, 1))",
            "input_channels": self.input_channels,
            "classes": ["target"],
            "device": self.device,
            "dataset_name": "FLIR Thermal Dataset (26,061 frames)"
        }

# Global singleton model manager
global_model_manager = ThermalModelManager()

