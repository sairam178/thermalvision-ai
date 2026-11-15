import type { ModelStatus, OverviewStats, FrameDetectionResult, VideoJobProgress, DetectionEvent } from '../types';

const BACKEND_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const API_BASE = BACKEND_BASE ? `${BACKEND_BASE}/api` : '/api';

export function getMediaUrl(path: string | undefined | null): string {
  if (!path) return '';
  if (path.startsWith('http://') || path.startsWith('https://') || path.startsWith('data:')) {
    return path;
  }
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${BACKEND_BASE}${cleanPath}`;
}

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Backend health check failed');
  return res.json();
}

export async function fetchModelStatus(): Promise<ModelStatus> {
  const res = await fetch(`${API_BASE}/model/status`);
  if (!res.ok) throw new Error('Failed to fetch model status');
  return res.json();
}

export async function fetchOverviewStats(): Promise<OverviewStats> {
  const res = await fetch(`${API_BASE}/overview/stats`);
  if (!res.ok) throw new Error('Failed to fetch overview stats');
  return res.json();
}

export async function detectImageFrame(
  imageBlobOrBase64: Blob | string,
  confThreshold: number = 0.25
): Promise<FrameDetectionResult> {
  let body: FormData | string;
  let headers: HeadersInit = {};

  if (typeof imageBlobOrBase64 === 'string') {
    body = JSON.stringify({ image: imageBlobOrBase64, conf: confThreshold });
    headers = { 'Content-Type': 'application/json' };
    const res = await fetch(`${API_BASE}/detect/image`, {
      method: 'POST',
      headers,
      body,
    });
    if (!res.ok) throw new Error('Inference request failed');
    return res.json();
  } else {
    const formData = new FormData();
    formData.append('file', imageBlobOrBase64, 'frame.jpg');
    formData.append('conf', confThreshold.toString());
    const res = await fetch(`${API_BASE}/detect/image-file`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Inference request failed');
    return res.json();
  }
}

export async function startVideoDetection(
  file: File,
  confThreshold: number = 0.25
): Promise<{ job_id: string; message: string }> {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('conf', confThreshold.toString());

  const res = await fetch(`${API_BASE}/detect/video`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to submit video for detection');
  }
  return res.json();
}

export async function fetchVideoJobStatus(jobId: string): Promise<VideoJobProgress> {
  const res = await fetch(`${API_BASE}/detect/video/${jobId}/status`);
  if (!res.ok) throw new Error('Failed to fetch video job status');
  return res.json();
}

export async function fetchHistory(): Promise<DetectionEvent[]> {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) throw new Error('Failed to fetch detection history');
  return res.json();
}

export async function deleteHistoryItem(id: string): Promise<{ success: boolean }> {
  const res = await fetch(`${API_BASE}/history/${id}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to delete history item');
  return res.json();
}

export async function clearAllHistory(): Promise<{ success: boolean; cleared_count: number }> {
  const res = await fetch(`${API_BASE}/history`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to clear history');
  return res.json();
}

export async function sendNLPChat(
  query: string,
  context?: Record<string, unknown>
): Promise<{ reply: string; suggestions?: string[]; context_type?: string }> {
  const res = await fetch(`${API_BASE}/nlp/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, context }),
  });
  if (!res.ok) throw new Error('NLP query failed');
  return res.json();
}

export async function generateNLPNarrativeReport(
  context: Record<string, unknown>
): Promise<{
  threat_level: string;
  verdict_headline: string;
  summary_text: string;
  detailed_narrative: string;
  recommendations: string[];
  markdown_report: string;
  generated_at: string;
}> {
  const res = await fetch(`${API_BASE}/nlp/generate-report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ context }),
  });
  if (!res.ok) throw new Error('NLP report generation failed');
  return res.json();
}

export async function fetchSampleImages(): Promise<string[]> {
  try {
    const res = await fetch(`${API_BASE}/samples`);
    if (res.ok) {
      const data = await res.json();
      if (data.samples && data.samples.length > 0) return data.samples;
    }
  } catch {
    // fallback
  }
  return [
    '/samples/thermal_sample_1.png',
    '/samples/thermal_sample_2.png',
    '/samples/thermal_sample_3.png',
    '/samples/thermal_sample_4.png',
  ];
}

export async function fetchVideoFolderFiles(jobId: string): Promise<{
  folder_name: string;
  folder_path: string;
  files: Array<{ name: string; relative_path: string; url: string; size_bytes: number }>;
}> {
  const res = await fetch(`${API_BASE}/detect/video/${jobId}/folder`);
  if (!res.ok) throw new Error('Failed to fetch video folder files');
  return res.json();
}

export async function fetchFolderByName(folderName: string): Promise<{
  folder_name: string;
  folder_path: string;
  files_count: number;
  frames_count: number;
  files: Array<{ name: string; relative_path: string; url: string; size_bytes: number }>;
  frames: Array<{ frame_number: number; filename: string; url: string; relative_path: string; size_bytes: number }>;
}> {
  const res = await fetch(`${API_BASE}/folder/${encodeURIComponent(folderName)}`);
  if (!res.ok) throw new Error('Failed to fetch folder');
  return res.json();
}

export async function saveSnapshotEvent(
  imageBlob: Blob,
  metadata: {
    filename?: string;
    frame_number?: number;
    pedestrians_count?: number;
    targets_count?: number;
    confidence: number;
  }
): Promise<{ success: boolean; id?: string; snapshot_url: string }> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'snapshot.jpg');
  formData.append('filename', metadata.filename || 'Photo Thermal Detection');
  formData.append('frame_number', (metadata.frame_number || 1).toString());
  formData.append('pedestrians_count', (metadata.targets_count ?? metadata.pedestrians_count ?? 0).toString());
  formData.append('targets_count', (metadata.targets_count ?? metadata.pedestrians_count ?? 0).toString());
  formData.append('confidence', metadata.confidence.toString());

  const res = await fetch(`${API_BASE}/history/snapshot`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Failed to save snapshot to history');
  return res.json();
}
