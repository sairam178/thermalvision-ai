export interface ModelStatus {
  loaded: boolean;
  model_name: string;
  model_path: string;
  architecture: string;
  input_channels: number;
  classes: string[];
  device: string;
  dataset_name: string;
}

export interface OverviewStats {
  model_status: ModelStatus;
  total_videos_analyzed: number;
  total_targets_detected?: number;
  total_unique_targets?: number;
  total_detections?: number;
  total_pedestrian_detections: number; // legacy alias
  total_unique_pedestrians?: number;
  total_frames_processed: number;
  recent_sessions: DetectionSessionSummary[];
}

export interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  class_id: number;
  class_name: string;
  track_id?: number | null;
}

export interface FrameDetectionResult {
  frame_index?: number;
  targets_count?: number;
  detections_count?: number;
  pedestrians_count: number;
  boxes: DetectionBox[];
  inference_time_ms: number;
  annotated_image?: string;
  annotated_image_base64?: string;
  image_width?: number;
  image_height?: number;
  filename?: string;
}

export interface VideoFrameItem {
  frame_number: number;
  filename: string;
  url: string;
  confidence?: number;
  targets_count?: number;
  size_bytes?: number;
}

export interface FolderFileInfo {
  name: string;
  relative_path: string;
  url: string;
  size_bytes: number;
}

export interface VideoJobProgress {
  job_id: string;
  filename: string;
  folder_name?: string;
  folder_path?: string;
  report_url?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  current_frame: number;
  total_frames: number;
  progress_percent: number;
  unique_targets_count?: number;
  current_targets?: number;
  target_detections_count?: number;
  frame_target_counts?: number[];
  unique_pedestrians_count: number;
  current_pedestrians?: number;
  pedestrian_detections_count: number;
  total_detection_instances: number;
  frame_pedestrian_counts?: number[];
  peak_confidence: number;
  peak_frame_index: number;
  peak_snapshot_url?: string;
  processed_video_url?: string;
  frames?: VideoFrameItem[];
  snapshots?: string[];
  folder_files?: FolderFileInfo[];
  error_message?: string;
  created_at: string;
  completed_at?: string;
  fps: number;
  duration_seconds: number;
  resolution: string;
}

export interface DetectionEvent {
  id: string;
  job_id: string;
  filename: string;
  frame_number: number;
  timestamp: string;
  targets_count?: number;
  pedestrians_count: number;
  confidence: number;
  snapshot_url: string;
  processed_video_url?: string;
  folder_name?: string;
  unique_targets?: number;
  unique_pedestrians?: number;
}

export interface DetectionSessionSummary {
  job_id: string;
  filename: string;
  timestamp: string;
  total_frames: number;
  target_detections?: number;
  pedestrian_detections: number;
  unique_targets?: number;
  unique_pedestrians?: number;
  peak_confidence: number;
  snapshot_url?: string;
  processed_video_url?: string;
  folder_name?: string;
}
