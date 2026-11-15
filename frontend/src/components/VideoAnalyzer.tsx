import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
  Upload,
  Film,
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Download,
  Trash2,
  FileText,
  ShieldAlert,
  ShieldCheck,
  AlertCircle,
  Activity,
  Sliders,
  Sparkles,
  FolderOpen,
  Image as ImageIcon,
  Layers,
  Eye,
  X,
  ChevronRight,
  Maximize2,
} from 'lucide-react';
import {
  startVideoDetection,
  fetchVideoJobStatus,
  fetchVideoFolderFiles,
  getMediaUrl,
  generateNLPNarrativeReport
} from '../services/api';
import type { VideoJobProgress, VideoFrameItem, FolderFileInfo } from '../types';

export const VideoAnalyzer: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState<VideoJobProgress | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [folderFiles, setFolderFiles] = useState<FolderFileInfo[]>([]);
  const [folderActiveTab, setFolderActiveTab] = useState<'frames' | 'files'>('frames');
  const [selectedFramePreview, setSelectedFramePreview] = useState<VideoFrameItem | null>(null);
  const [aiReport, setAiReport] = useState<{
    threat_level: string;
    verdict_headline: string;
    summary_text: string;
    detailed_narrative: string;
    recommendations: string[];
  } | null>(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState<boolean>(false);

  // Playback state
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isMuted, setIsMuted] = useState<boolean>(true);
  const [currentFrameNum, setCurrentFrameNum] = useState<number>(0);
  const [currentTimeSec, setCurrentTimeSec] = useState<number>(0);
  const [videoDuration, setVideoDuration] = useState<number>(0);

  // Current targets in the active playback frame
  const currentTargetsInFrame = jobProgress?.frame_target_counts && jobProgress.frame_target_counts.length > 0
    ? (jobProgress.frame_target_counts[Math.min(Math.max(0, currentFrameNum - 1), jobProgress.frame_target_counts.length - 1)] ?? 0)
    : (jobProgress?.current_targets ?? jobProgress?.current_pedestrians ?? 0);

  // Poll video processing status
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;

    if (jobId && jobProgress?.status !== 'completed' && jobProgress?.status !== 'failed') {
      interval = setInterval(async () => {
        try {
          const status = await fetchVideoJobStatus(jobId);
          setJobProgress(status);

          if (status.status === 'completed' || status.status === 'failed') {
            if (status.status === 'completed') {
              try {
                const folderRes = await fetchVideoFolderFiles(jobId);
                if (folderRes && folderRes.files) {
                  setFolderFiles(folderRes.files);
                }
              } catch (fErr) {
                console.log('Notice: folder files query:', fErr);
              }
            } else if (status.status === 'failed') {
              setErrorMsg(status.error_message || 'Video processing encountered an issue. Please try again.');
            }
            clearInterval(interval);
          }
        } catch (err) {
          console.error('Polling job status failed:', err);
        }
      }, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [jobId, jobProgress?.status]);

  // Load folder files if completed
  useEffect(() => {
    if (jobProgress?.status === 'completed' && jobId && folderFiles.length === 0) {
      fetchVideoFolderFiles(jobId)
        .then((res) => {
          if (res?.files) setFolderFiles(res.files);
        })
        .catch(() => {});
    }
  }, [jobProgress?.status, jobId, folderFiles.length]);

  // Derived list of all frames captured for this video folder
  const videoFrames: VideoFrameItem[] = useMemo(() => {
    if (jobProgress?.frames && jobProgress.frames.length > 0) {
      return jobProgress.frames;
    }
    // Extract image files from folderFiles
    const imageFiles = folderFiles.filter(
      (f) =>
        /\.(jpe?g|png)$/i.test(f.name) &&
        (f.relative_path.includes('snapshots') || f.name.includes('frame') || f.name.includes('event'))
    );
    if (imageFiles.length > 0) {
      return imageFiles
        .map((f) => {
          const m = f.name.match(/(\d+)/);
          const fNum = m ? parseInt(m[1], 10) : 0;
          return {
            frame_number: fNum,
            filename: f.name,
            url: f.url,
            confidence: jobProgress?.peak_confidence,
            targets_count: jobProgress?.unique_targets_count ?? jobProgress?.unique_pedestrians_count ?? 1,
            size_bytes: f.size_bytes,
          };
        })
        .sort((a, b) => a.frame_number - b.frame_number);
    }
    // Fallback to peak snapshot
    if (jobProgress?.peak_snapshot_url) {
      return [
        {
          frame_number: jobProgress.peak_frame_index || 1,
          filename: 'peak_frame.jpg',
          url: jobProgress.peak_snapshot_url,
          confidence: jobProgress.peak_confidence,
          targets_count: jobProgress.unique_targets_count ?? jobProgress.unique_pedestrians_count ?? 1,
        },
      ];
    }
    return [];
  }, [
    jobProgress?.frames,
    jobProgress?.peak_snapshot_url,
    jobProgress?.peak_frame_index,
    jobProgress?.peak_confidence,
    jobProgress?.unique_targets_count,
    jobProgress?.unique_pedestrians_count,
    folderFiles,
  ]);

  // Handle frame jump
  const handleJumpToFrame = (frameNum: number) => {
    if (!videoRef.current) return;
    const fps = jobProgress?.fps || 25;
    const targetSec = frameNum / fps;
    videoRef.current.currentTime = Math.max(0, targetSec);
    setCurrentTimeSec(targetSec);
    setCurrentFrameNum(frameNum);
    videoRef.current.play().catch(() => {});
    setIsPlaying(true);
  };

  // Handle file select
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setJobId(null);
      setJobProgress(null);
      setErrorMsg(null);
      setFolderFiles([]);
      setSelectedFramePreview(null);
    }
  };

  // Drag and drop
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setJobId(null);
      setJobProgress(null);
      setErrorMsg(null);
      setFolderFiles([]);
      setSelectedFramePreview(null);
    }
  };

  // Start Analysis
  const handleStartAnalysis = async () => {
    if (!selectedFile) return;

    try {
      setIsUploading(true);
      setErrorMsg(null);
      const res = await startVideoDetection(selectedFile, confThreshold);
      setJobId(res.job_id);
      setIsUploading(false);
    } catch (err: unknown) {
      setIsUploading(false);
      setErrorMsg((err as Error).message || 'Failed to start video processing');
    }
  };

  // Reset & Upload another video
  const handleResetVideo = () => {
    setSelectedFile(null);
    setJobId(null);
    setJobProgress(null);
    setErrorMsg(null);
    setIsPlaying(false);
    setFolderFiles([]);
    setSelectedFramePreview(null);
    setAiReport(null);
  };

  // Video playback controls
  const togglePlayPause = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleReplay = () => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = 0;
    videoRef.current.play();
    setIsPlaying(true);
  };

  const handleTimeUpdate = () => {
    if (!videoRef.current) return;
    const time = videoRef.current.currentTime;
    setCurrentTimeSec(time);
    const fps = jobProgress?.fps || 25;
    setCurrentFrameNum(Math.round(time * fps));
  };

  const handleLoadedMetadata = () => {
    if (!videoRef.current) return;
    setVideoDuration(videoRef.current.duration);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!videoRef.current) return;
    const time = parseFloat(e.target.value);
    videoRef.current.currentTime = time;
    setCurrentTimeSec(time);
  };

  // Export JSON dossier
  const handleExportReport = () => {
    if (!jobProgress) return;
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(jobProgress, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `${jobProgress.filename}_detection_dossier.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handleGenerateAIReport = async () => {
    if (!jobProgress) return;
    try {
      setIsGeneratingReport(true);
      setErrorMsg(null);
      const rep = await generateNLPNarrativeReport({
        video_name: jobProgress.filename,
        unique_targets_count: jobProgress.unique_targets_count ?? jobProgress.unique_pedestrians_count,
        total_detection_instances: jobProgress.total_detection_instances,
        peak_confidence: jobProgress.peak_confidence,
        peak_frame_index: jobProgress.peak_frame_index,
        total_frames: jobProgress.total_frames,
        fps: jobProgress.fps,
        duration_seconds: jobProgress.duration_seconds,
        folder_name: jobProgress.folder_name,
      });
      setAiReport(rep);
    } catch (err) {
      console.error('NLP Report error:', err);
      setErrorMsg('Failed to generate tactical narrative report');
    } finally {
      setIsGeneratingReport(false);
    }
  };

  return (
    <div className="space-y-6 pb-16">
      {/* Top Banner when a video is loaded / analyzing / completed */}
      {selectedFile && (
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-4 rounded-2xl bg-[#0a0d16] border border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-cyan-400">
              <Film className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center flex-wrap gap-2">
                <span className="font-bold text-sm text-white">{selectedFile.name}</span>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${jobProgress?.status === 'completed'
                      ? 'bg-emerald-950 border border-emerald-500/40 text-emerald-300'
                      : jobProgress?.status === 'processing'
                        ? 'bg-cyan-950 border border-cyan-500/40 text-cyan-300 animate-pulse'
                        : 'bg-slate-800 text-slate-300'
                    }`}
                >
                  {jobProgress?.status?.toUpperCase() || 'LOADED'}
                </span>

                {jobProgress?.status === 'completed' && (
                  <span className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-cyan-950/90 border border-cyan-500/50 text-cyan-300 flex items-center gap-1.5 shadow-sm">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                    TOTAL UNIQUE TARGETS: {jobProgress?.unique_targets_count ?? jobProgress?.unique_pedestrians_count ?? 0}
                  </span>
                )}
                {jobProgress?.folder_name && (
                  <span className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-[#0d1c33] border border-blue-500/40 text-blue-300 flex items-center gap-1.5">
                    📂 outputs/{jobProgress.folder_name}/
                  </span>
                )}
              </div>
              <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                {jobProgress ? (
                  <>
                    {jobProgress.fps || 25.0} FPS • {jobProgress.resolution || '640x480'} • Thermal Video Stream • Folder: outputs/{jobProgress.folder_name || '...'}
                  </>
                ) : (
                  <>Ready for 1-channel thermal analysis • {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center flex-wrap gap-2.5">
            <button
              onClick={handleResetVideo}
              className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-slate-900 hover:bg-red-950/40 border border-slate-800 hover:border-red-500/40 text-slate-300 hover:text-red-300 text-xs font-medium transition-colors cursor-pointer"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Remove Video & Upload Another
            </button>

            {jobProgress?.status === 'completed' && (
              <>
                <button
                  onClick={handleExportReport}
                  className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-cyan-300 text-xs font-semibold transition-colors cursor-pointer"
                >
                  <FileText className="w-3.5 h-3.5 text-cyan-400" />
                  Export Report (JSON)
                </button>

                {jobProgress.processed_video_url && (
                  <a
                    href={getMediaUrl(jobProgress.processed_video_url)}
                    download={`processed_${selectedFile.name}`}
                    className="flex items-center gap-2 px-4 py-1.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-slate-950 text-xs font-bold transition-colors shadow-lg shadow-cyan-950/40 cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download Video
                  </a>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* Main Container */}
      {!selectedFile ? (
        /* Video Upload Dropzone */
        <div className="max-w-3xl mx-auto py-10">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="p-12 rounded-3xl bg-[#090d16] border-2 border-dashed border-slate-800 hover:border-cyan-500/60 transition-all text-center flex flex-col items-center justify-center relative overflow-hidden group shadow-2xl"
          >
            <div className="absolute inset-0 surveillance-grid opacity-20 pointer-events-none" />

            <div className="w-20 h-20 rounded-2xl bg-slate-900/90 border border-slate-800 flex items-center justify-center text-cyan-400 mb-6 group-hover:scale-105 transition-transform shadow-lg shadow-cyan-950/30">
              <Upload className="w-10 h-10" />
            </div>

            <h2 className="text-xl font-bold text-white mb-2">
              Ingest Thermal Surveillance Footage
            </h2>
            <p className="text-xs text-slate-400 max-w-md mb-6 leading-relaxed">
              Upload MP4, AVI or MOV thermal video files. The backend feeds every frame through your genuine 1-channel YOLOv8 model for thermal target detection and stores all output files into a folder named after your video.
            </p>

            <label className="cursor-pointer px-6 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs tracking-wider uppercase transition-all shadow-lg shadow-cyan-950/40 mb-4 inline-flex items-center gap-2">
              <Film className="w-4 h-4" />
              Select Video from Computer
              <input
                type="file"
                accept="video/*,.mp4,.avi,.mov,.mkv,.webm,video/mp4,video/avi,video/quicktime,video/x-msvideo"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>

            <div className="text-[11px] text-slate-500 font-mono">
              Supported Formats: MP4, AVI, MOV • Model: YOLOv8 1-Channel LWIR Target Detection
            </div>
          </div>
        </div>
      ) : jobProgress?.status === 'processing' || isUploading ? (
        /* Progress Screen */
        <div className="max-w-2xl mx-auto p-8 rounded-3xl bg-[#090d16] border border-slate-800 text-center space-y-6">
          <div className="w-16 h-16 rounded-2xl bg-cyan-950/40 border border-cyan-500/40 flex items-center justify-center mx-auto text-cyan-400 animate-pulse">
            <Activity className="w-8 h-8" />
          </div>

          <div>
            <h3 className="text-lg font-bold text-white mb-1">
              Processing Thermal Video Frame-by-Frame
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Converting frames to 1-channel grayscale matrices, running YOLOv8 inference, and organizing files into folder 'outputs/{jobProgress?.folder_name || selectedFile.name.replace(/\.[^/.]+$/, '')}/'...
            </p>
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">
                Frame: {jobProgress?.current_frame || 0} / {jobProgress?.total_frames || '...'}
              </span>
              <span className="text-cyan-400 font-bold">
                {jobProgress?.progress_percent ? jobProgress.progress_percent.toFixed(1) : 0}%
              </span>
            </div>

            <div className="w-full h-3 rounded-full bg-slate-900 border border-slate-800 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-300 shadow-lg shadow-cyan-500/50"
                style={{ width: `${jobProgress?.progress_percent || 5}%` }}
              />
            </div>
          </div>

          {/* Live detection counter during processing */}
          <div className="p-4 rounded-xl bg-[#06080e] border border-slate-800 flex items-center justify-around text-xs font-mono">
            <div>
              <div className="text-slate-500">Current in Frame</div>
              <div className="text-xl font-black text-amber-400">
                {jobProgress?.current_targets ?? jobProgress?.current_pedestrians ?? 0}
              </div>
            </div>
            <div className="w-px h-8 bg-slate-800" />
            <div>
              <div className="text-slate-500">Total Unique Targets</div>
              <div className="text-xl font-black text-cyan-400">
                {jobProgress?.unique_targets_count ?? jobProgress?.unique_pedestrians_count ?? 0}
              </div>
            </div>
            <div className="w-px h-8 bg-slate-800" />
            <div>
              <div className="text-slate-500">Peak Confidence</div>
              <div className="text-xl font-black text-white">
                {jobProgress?.peak_confidence ? (jobProgress.peak_confidence * 100).toFixed(1) : 0}%
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Video Player & Surveillance Telemetry Split View (Matches Screenshot 6) */
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Video Player Feed (2 Cols) */}
          <div className="lg:col-span-2 space-y-4">
            <div className="relative aspect-video rounded-2xl bg-[#06080e] border border-slate-800 overflow-hidden shadow-2xl flex items-center justify-center">
              <div className="absolute inset-0 surveillance-grid opacity-30 pointer-events-none" />

              {/* Native Video Element */}
              {jobProgress?.processed_video_url ? (
                <video
                  ref={videoRef}
                  key={jobProgress.processed_video_url}
                  className="w-full h-full object-contain cursor-pointer"
                  onTimeUpdate={handleTimeUpdate}
                  onLoadedMetadata={handleLoadedMetadata}
                  onEnded={() => setIsPlaying(false)}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onClick={togglePlayPause}
                  muted={isMuted}
                  playsInline
                  preload="auto"
                  controls
                >
                  <source src={getMediaUrl(jobProgress.processed_video_url)} type="video/mp4" />
                  Your browser does not support HTML5 video playback.
                </video>
              ) : (
                <div className="text-center p-6 space-y-4">
                  <div className="w-12 h-12 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center mx-auto text-slate-500">
                    <Film className="w-6 h-6" />
                  </div>
                  <div>
                    <div className="text-sm font-bold text-white mb-1">
                      Ready to Analyze: {selectedFile.name}
                    </div>
                    <div className="text-xs text-slate-400">
                      Configure minimum confidence and click "Start Analysis" below.
                    </div>
                  </div>

                  {/* Confidence Slider before analysis */}
                  <div className="max-w-xs mx-auto text-left pt-2">
                    <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                      <span className="flex items-center gap-1 font-mono">
                        <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                        Min Confidence Threshold:
                      </span>
                      <span className="font-mono text-cyan-400 font-bold">
                        {(confThreshold * 100).toFixed(0)}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.10"
                      max="0.80"
                      step="0.05"
                      value={confThreshold}
                      onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                      className="w-full accent-cyan-400 cursor-pointer"
                    />
                  </div>

                  <button
                    onClick={handleStartAnalysis}
                    className="px-6 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs uppercase tracking-wider transition-all shadow-lg shadow-cyan-950/40 cursor-pointer"
                  >
                    Start 1-Channel Analysis
                  </button>
                </div>
              )}

              {/* Overlay HUD Tags */}
              {jobProgress?.status === 'completed' && (
                <>
                  <div className="absolute top-4 left-4 flex flex-col sm:flex-row items-start sm:items-center gap-2 pointer-events-none">
                    <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-black/85 backdrop-blur-md border border-red-500/50 text-xs font-mono font-bold text-red-400 shadow-xl">
                      <span className="w-2 h-2 rounded-full bg-red-400 animate-pulse" />
                      TOTAL UNIQUE TARGETS: {jobProgress?.unique_targets_count ?? jobProgress?.unique_pedestrians_count ?? 0}
                    </div>
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/85 backdrop-blur-md border border-amber-500/40 text-xs font-mono font-bold text-amber-300 shadow-xl">
                      <span className="w-2 h-2 rounded-full bg-amber-400" />
                      CURRENT IN FRAME: {currentTargetsInFrame}
                    </div>
                  </div>

                  <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/80 backdrop-blur-md border border-slate-800 text-[10px] font-mono text-slate-300 pointer-events-none">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                    BYTETRACK • 1-CH FEED
                  </div>
                </>
              )}
            </div>

            {/* Custom Playback Controls Bar */}
            {jobProgress?.status === 'completed' && (
              <div className="p-3.5 rounded-xl bg-[#0a0d16] border border-slate-800 flex items-center justify-between gap-4">
                <div className="flex items-center gap-2">
                  <button
                    onClick={togglePlayPause}
                    className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-white transition-colors cursor-pointer"
                    title={isPlaying ? 'Pause' : 'Play'}
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 text-cyan-400" />}
                  </button>

                  <button
                    onClick={handleReplay}
                    className="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors cursor-pointer"
                    title="Replay from start"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>

                  <button
                    onClick={() => setIsMuted(!isMuted)}
                    className="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors cursor-pointer"
                    title={isMuted ? 'Unmute' : 'Mute'}
                  >
                    {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4 text-cyan-400" />}
                  </button>
                </div>

                {/* Scrubber slider */}
                <div className="flex-1 flex items-center gap-3">
                  <input
                    type="range"
                    min="0"
                    max={videoDuration || 100}
                    step="0.04"
                    value={currentTimeSec}
                    onChange={handleSeek}
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>

                {/* Frame Counter */}
                <div className="text-xs font-mono text-slate-400 whitespace-nowrap">
                  Frame: <span className="text-white font-bold">{currentFrameNum}</span> / {jobProgress.total_frames}
                </div>
              </div>
            )}
          </div>

          {/* Right: Surveillance Telemetry & Snapshot Dossier (1 Col) */}
          <div className="space-y-5">
            {/* Telemetry Card */}
            <div className="p-5 rounded-2xl bg-[#0a0d18] border border-slate-800 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <span className="text-xs font-mono tracking-wider text-slate-400 uppercase font-semibold">
                  THERMAL TARGET TELEMETRY
                </span>
                <span className="px-2 py-0.5 rounded text-[9px] font-mono font-bold bg-emerald-950 border border-emerald-500/50 text-emerald-400">
                  LIVE SENSOR
                </span>
              </div>

              {/* Folder Location Box */}
              {jobProgress?.folder_name && (
                <div className="p-3 rounded-xl bg-[#070e1b] border border-blue-500/30 flex items-center justify-between text-xs font-mono">
                  <div className="flex items-center gap-2 text-blue-300">
                    <span className="text-sm">📁</span>
                    <div>
                      <div className="text-[10px] text-slate-400 uppercase font-bold">OUTPUT DOSSIER FOLDER</div>
                      <div className="text-white font-bold truncate max-w-[220px]">
                        outputs/{jobProgress.folder_name}/
                      </div>
                    </div>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-blue-950/80 text-blue-300 text-[10px] font-bold">
                    ORGANIZED
                  </span>
                </div>
              )}

              {/* Prominent Unique Targets Display */}
              <div className="p-4 rounded-xl bg-gradient-to-br from-[#0c1322] to-[#070b14] border border-cyan-500/40 shadow-lg relative overflow-hidden">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-[10px] font-mono tracking-wider text-cyan-400 uppercase font-semibold">
                      TOTAL UNIQUE TARGETS (BYTETRACK)
                    </div>
                    <div className="text-2xl sm:text-3xl font-black text-white font-mono mt-1 flex items-baseline gap-2">
                      <span className={jobProgress && (jobProgress.unique_targets_count ?? jobProgress.unique_pedestrians_count ?? 0) > 0 ? "text-red-400" : "text-emerald-400"}>
                        Total Unique Targets: {jobProgress?.unique_targets_count ?? jobProgress?.unique_pedestrians_count ?? 0}
                      </span>
                    </div>
                    <div className="text-[11px] font-mono text-slate-300 mt-1 flex items-center gap-3">
                      <span>Current in Frame: <span className="text-amber-400 font-bold">{currentTargetsInFrame}</span></span>
                      <span className="text-slate-600">•</span>
                      <span>Total Detections: <span className="text-slate-200 font-semibold">{jobProgress?.total_detection_instances || jobProgress?.target_detections_count || jobProgress?.pedestrian_detections_count || 0}</span></span>
                    </div>
                  </div>
                  <div className={`px-2.5 py-1.5 rounded-lg border text-[11px] font-mono font-bold flex items-center gap-1.5 ${(jobProgress?.unique_targets_count ?? jobProgress?.unique_pedestrians_count ?? 0) > 0
                      ? 'bg-red-950/80 border-red-500/50 text-red-300'
                      : 'bg-emerald-950/80 border-emerald-500/50 text-emerald-300'
                    }`}>
                    <span className={`w-2 h-2 rounded-full ${(jobProgress?.unique_targets_count ?? jobProgress?.unique_pedestrians_count ?? 0) > 0 ? 'bg-red-400 animate-pulse' : 'bg-emerald-400'
                      }`} />
                    {(jobProgress?.unique_targets_count ?? jobProgress?.unique_pedestrians_count ?? 0) > 0 ? 'ACTIVE TARGETS' : 'PATH CLEAR'}
                  </div>
                </div>
              </div>

              {/* Scene Verdict Box */}
              <div className="p-4 rounded-xl bg-[#06080e] border border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div
                    className={`w-9 h-9 rounded-lg flex items-center justify-center ${(jobProgress?.target_detections_count || jobProgress?.pedestrian_detections_count || 0) > 0
                        ? 'bg-red-950/60 text-red-400 border border-red-500/40'
                        : 'bg-emerald-950/60 text-emerald-400 border border-emerald-500/40'
                      }`}
                  >
                    {(jobProgress?.target_detections_count || jobProgress?.pedestrian_detections_count || 0) > 0 ? (
                      <ShieldAlert className="w-5 h-5" />
                    ) : (
                      <ShieldCheck className="w-5 h-5" />
                    )}
                  </div>
                  <div>
                    <div className="text-[10px] font-mono text-slate-400 uppercase">SCENE VERDICT</div>
                    <div className="text-sm font-bold text-white">
                      {(jobProgress?.target_detections_count || jobProgress?.pedestrian_detections_count || 0) > 0
                        ? 'Thermal Targets Detected'
                        : 'Clear Surveillance Corridor'}
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-[10px] font-mono text-slate-400 uppercase">PEAK CONFIDENCE</div>
                  <div className="text-base font-black text-cyan-400 font-mono">
                    {jobProgress?.peak_confidence
                      ? `${(jobProgress.peak_confidence * 100).toFixed(1)}%`
                      : '0.0%'}
                  </div>
                </div>
              </div>

              {/* Attention Index Progress Bar */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">Thermal Activity Index:</span>
                  <span
                    className={`font-bold ${(jobProgress?.target_detections_count || jobProgress?.pedestrian_detections_count || 0) > 0
                        ? 'text-red-400'
                        : 'text-emerald-400'
                      }`}
                  >
                    {(jobProgress?.target_detections_count || jobProgress?.pedestrian_detections_count || 0) > 0
                      ? 'Elevated / Active Heat Signatures'
                      : 'Baseline / Quiet Corridor'}
                  </span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-900 border border-slate-800 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${(jobProgress?.target_detections_count || jobProgress?.pedestrian_detections_count || 0) > 0
                        ? 'bg-gradient-to-r from-amber-500 to-red-500'
                        : 'bg-emerald-500'
                      }`}
                    style={{
                      width: `${Math.min(100, Math.max(10, (jobProgress?.peak_confidence || 0.1) * 100))}%`,
                    }}
                  />
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div className="p-3 rounded-lg bg-[#06080e] border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">CURRENT TARGETS</div>
                  <div className="text-sm font-bold text-amber-400 mt-0.5">
                    {currentTargetsInFrame}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-[#06080e] border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">TOTAL UNIQUE TARGETS</div>
                  <div className="text-sm font-bold text-cyan-400 mt-0.5">
                    {jobProgress?.unique_targets_count ?? jobProgress?.unique_pedestrians_count ?? 0}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-[#06080e] border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">TOTAL INSTANCES</div>
                  <div className="text-sm font-bold text-white mt-0.5">
                    {jobProgress?.total_detection_instances || jobProgress?.target_detections_count || jobProgress?.pedestrian_detections_count || 0}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-[#06080e] border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">PEAK CONFIDENCE</div>
                  <div className="text-sm font-bold text-cyan-400 mt-0.5">
                    {jobProgress?.peak_confidence ? `${(jobProgress.peak_confidence * 100).toFixed(1)}%` : '0.0%'}
                  </div>
                </div>
              </div>

              {/* ByteTrack Tracking Architecture Note */}
              <div className="p-3 rounded-xl bg-[#080c16] border border-slate-800/80 text-[10px] font-mono text-slate-400 leading-relaxed">
                <span className="text-cyan-300 font-semibold">ByteTrack Object Tracking:</span> Maintains persistent target IDs across frames using motion affinity. Files are stored inside folder <span className="text-white font-semibold">outputs/{jobProgress?.folder_name || '...'}/</span>.
              </div>

              {/* Log Dossier CTA */}
              <div className="space-y-2">
                <button
                  onClick={handleGenerateAIReport}
                  disabled={!jobProgress || jobProgress.status !== 'completed' || isGeneratingReport}
                  className="w-full py-3 rounded-xl bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-600 hover:from-indigo-500 hover:to-cyan-500 disabled:opacity-50 text-white font-extrabold text-xs uppercase tracking-wider transition-all shadow-lg shadow-indigo-950/50 flex items-center justify-center gap-2 cursor-pointer"
                >
                  <Sparkles className="w-4 h-4 text-cyan-300 animate-pulse" />
                  <span>{isGeneratingReport ? 'SYNTHESIZING TACTICAL REPORT...' : 'GENERATE AI TACTICAL REPORT (NLP)'}</span>
                </button>

                <button
                  onClick={handleExportReport}
                  disabled={!jobProgress || jobProgress.status !== 'completed'}
                  className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white font-extrabold text-xs uppercase tracking-wider transition-all shadow-lg shadow-cyan-950/50 flex items-center justify-center gap-2 cursor-pointer"
                >
                  <FileText className="w-4 h-4" />
                  EXPORT DOSSIER REPORT (JSON)
                </button>
              </div>

              {/* AI Narrative Tactical Report Box */}
              {aiReport && (
                <div className="p-4 rounded-xl bg-[#081122] border border-cyan-500/40 text-xs space-y-2.5 animate-in fade-in">
                  <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                    <span className="text-[10px] font-mono uppercase font-bold text-cyan-300 flex items-center gap-1">
                      <Sparkles className="w-3.5 h-3.5" />
                      TACTICAL SURVEILLANCE REPORT
                    </span>
                    <span className="px-2 py-0.5 rounded text-[9px] font-mono font-bold bg-cyan-950 border border-cyan-500/50 text-cyan-300">
                      {aiReport.threat_level}
                    </span>
                  </div>

                  <div>
                    <h4 className="font-bold text-white text-xs">{aiReport.verdict_headline}</h4>
                    <p className="text-[11px] text-slate-300 mt-1 leading-relaxed">
                      {aiReport.summary_text}
                    </p>
                  </div>

                  <div className="p-2.5 rounded-lg bg-black/40 border border-slate-800 text-[11px] text-slate-400 font-mono leading-relaxed">
                    {aiReport.detailed_narrative}
                  </div>

                  <div className="space-y-1">
                    <span className="text-[10px] font-mono uppercase text-slate-400 font-bold">TACTICAL ACTIONS:</span>
                    <ul className="space-y-1 text-[11px] text-slate-300">
                      {aiReport.recommendations.map((rec, i) => (
                        <li key={i} className="flex items-start gap-1">
                          <span className="text-cyan-400 font-bold">•</span>
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>

            {/* Evidentiary Snapshot Card */}
            <div className="p-5 rounded-2xl bg-[#0a0d18] border border-slate-800 space-y-3">
              <div className="text-xs font-mono tracking-wider text-slate-400 uppercase font-semibold">
                CAPTURED EVIDENTIARY SNAPSHOT
              </div>

              <div className="w-full aspect-video rounded-xl bg-[#06080e] border border-slate-800 overflow-hidden flex items-center justify-center relative">
                {jobProgress?.peak_snapshot_url ? (
                  <img
                    src={getMediaUrl(jobProgress.peak_snapshot_url)}
                    alt="Peak target capture"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="text-center p-4">
                    <Film className="w-6 h-6 text-slate-600 mx-auto mb-1" />
                    <span className="text-xs text-slate-500 font-mono">No snapshot captured yet</span>
                  </div>
                )}
                {jobProgress?.peak_frame_index && (
                  <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/80 text-[10px] font-mono text-cyan-400">
                    Frame #{jobProgress.peak_frame_index}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ========================================================================= */}
          {/* DEDICATED VIDEO OUTPUT FOLDER: outputs/<video_name>/ WITH FRAME GALLERY  */}
          {/* ========================================================================= */}
          <div className="lg:col-span-3 mt-4 rounded-2xl bg-[#070b16] border border-cyan-500/40 overflow-hidden shadow-2xl space-y-0">
            {/* Folder Titlebar */}
            <div className="px-6 py-4 bg-gradient-to-r from-[#0b1424] via-[#09101d] to-[#0c1626] border-b border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex items-center gap-3.5">
                <div className="w-11 h-11 rounded-xl bg-cyan-950/90 border border-cyan-400/60 flex items-center justify-center text-cyan-400 shadow-lg shadow-cyan-950/60">
                  <FolderOpen className="w-6 h-6 text-cyan-300" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono uppercase font-bold text-cyan-400 tracking-wider">
                      DEDICATED VIDEO DOSSIER FOLDER
                    </span>
                    <span className="px-2 py-0.5 rounded text-[9px] font-mono font-bold bg-cyan-950/80 border border-cyan-500/40 text-cyan-300">
                      {videoFrames.length} FRAMES ARCHIVED
                    </span>
                  </div>
                  <h3 className="text-base sm:text-lg font-black text-white font-mono flex items-center gap-2 mt-0.5">
                    <span className="text-blue-400">outputs/{jobProgress?.folder_name || 'video_folder'}/</span>
                    <span className="text-slate-500 text-xs font-normal hidden sm:inline">
                      ({folderFiles.length} files total)
                    </span>
                  </h3>
                </div>
              </div>

              {/* View Mode Toggle: Frames Gallery vs File Manifest */}
              <div className="flex items-center gap-2 bg-[#05070d] p-1.5 rounded-xl border border-slate-800">
                <button
                  onClick={() => setFolderActiveTab('frames')}
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-bold transition-all flex items-center gap-1.5 cursor-pointer ${
                    folderActiveTab === 'frames'
                      ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-950/60 font-black'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <ImageIcon className="w-3.5 h-3.5" />
                  Frames Gallery ({videoFrames.length})
                </button>
                <button
                  onClick={() => setFolderActiveTab('files')}
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-mono font-bold transition-all flex items-center gap-1.5 cursor-pointer ${
                    folderActiveTab === 'files'
                      ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-950/60 font-black'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Layers className="w-3.5 h-3.5" />
                  Dossier Files ({folderFiles.length})
                </button>
              </div>
            </div>

            {/* Path Sub-Header */}
            <div className="px-6 py-2.5 bg-[#050811] border-b border-slate-800/80 flex items-center justify-between text-xs font-mono text-slate-400">
              <div className="flex items-center gap-1.5 text-slate-400 text-[11px] truncate">
                <span className="text-slate-600">Root</span>
                <ChevronRight className="w-3 h-3 text-slate-600" />
                <span className="text-slate-400">outputs</span>
                <ChevronRight className="w-3 h-3 text-slate-600" />
                <span className="text-cyan-300 font-bold">{jobProgress?.folder_name || 'video_output'}</span>
                <ChevronRight className="w-3 h-3 text-slate-600" />
                <span className="text-white font-semibold">
                  {folderActiveTab === 'frames' ? 'snapshots / (Detection Frames)' : 'files / (All Manifests)'}
                </span>
              </div>
              <div className="text-[10px] text-cyan-400 font-bold hidden sm:block">
                CLICK ANY FRAME TO INSPECT OR JUMP PLAYBACK
              </div>
            </div>

            {/* Folder Body Content */}
            <div className="p-6">
              {folderActiveTab === 'frames' ? (
                /* Tab 1: Frames Gallery Grid */
                videoFrames.length > 0 ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {videoFrames.map((frame, idx) => (
                      <div
                        key={idx}
                        className="group relative rounded-xl bg-[#04060d] border border-slate-800 hover:border-cyan-400/80 transition-all duration-200 overflow-hidden shadow-lg flex flex-col justify-between"
                      >
                        {/* Frame Image Container */}
                        <div
                          className="relative aspect-video bg-black overflow-hidden cursor-pointer"
                          onClick={() => setSelectedFramePreview(frame)}
                        >
                          <img
                            src={getMediaUrl(frame.url)}
                            alt={`Frame #${frame.frame_number}`}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                            loading="lazy"
                          />
                          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/40 pointer-events-none" />

                          {/* Top Badges */}
                          <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/85 backdrop-blur-md border border-cyan-500/40 text-[10px] font-mono font-bold text-cyan-300">
                            Frame #{frame.frame_number}
                          </div>

                          {frame.confidence && (
                            <div className="absolute top-2 right-2 px-2 py-0.5 rounded bg-black/85 backdrop-blur-md border border-emerald-500/40 text-[9px] font-mono font-bold text-emerald-400">
                              {(frame.confidence * 100).toFixed(1)}%
                            </div>
                          )}

                          {/* Hover Overlay Icon */}
                          <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/40 backdrop-blur-[2px]">
                            <div className="p-2 rounded-full bg-cyan-500/90 text-slate-950 font-bold shadow-lg">
                              <Maximize2 className="w-4 h-4" />
                            </div>
                          </div>
                        </div>

                        {/* Frame Info Footer */}
                        <div className="p-3 bg-[#070b14] border-t border-slate-800/80 flex items-center justify-between text-xs font-mono">
                          <div>
                            <div className="text-[11px] font-bold text-white flex items-center gap-1">
                              <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
                              <span>{frame.targets_count || 1} Target(s)</span>
                            </div>
                            <div className="text-[10px] text-slate-400 mt-0.5">
                              {((frame.frame_number || 1) / (jobProgress?.fps || 25)).toFixed(2)}s in video
                            </div>
                          </div>

                          <div className="flex items-center gap-1.5">
                            <button
                              onClick={() => handleJumpToFrame(frame.frame_number)}
                              className="p-1.5 rounded-lg bg-slate-800 hover:bg-cyan-500 hover:text-slate-950 text-cyan-300 transition-colors cursor-pointer"
                              title="Jump Video to Frame"
                            >
                              <Play className="w-3.5 h-3.5" />
                            </button>
                            <button
                              onClick={() => setSelectedFramePreview(frame)}
                              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors cursor-pointer"
                              title="Inspect Full Resolution"
                            >
                              <Eye className="w-3.5 h-3.5" />
                            </button>
                            <a
                              href={getMediaUrl(frame.url)}
                              download={frame.filename || `frame_${frame.frame_number}.jpg`}
                              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors cursor-pointer"
                              title="Download Frame"
                            >
                              <Download className="w-3.5 h-3.5" />
                            </a>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12 text-slate-500 font-mono text-xs">
                    No frames detected yet.
                  </div>
                )
              ) : (
                /* Tab 2: All Dossier Files Table */
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 text-[10px] uppercase">
                        <th className="pb-3 font-semibold">File Name</th>
                        <th className="pb-3 font-semibold">Relative Path in Folder</th>
                        <th className="pb-3 font-semibold">Size</th>
                        <th className="pb-3 text-right font-semibold">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {folderFiles.length > 0 ? (
                        folderFiles.map((f, i) => (
                          <tr key={i} className="hover:bg-slate-900/40 transition-colors">
                            <td className="py-2.5 font-bold text-white flex items-center gap-2">
                              {f.name.endsWith('.mp4') ? (
                                <Film className="w-4 h-4 text-cyan-400" />
                              ) : f.name.endsWith('.json') ? (
                                <FileText className="w-4 h-4 text-amber-400" />
                              ) : (
                                <ImageIcon className="w-4 h-4 text-blue-400" />
                              )}
                              <span>{f.name}</span>
                            </td>
                            <td className="py-2.5 text-slate-400">{f.relative_path}</td>
                            <td className="py-2.5 text-slate-400">
                              {(f.size_bytes / 1024).toFixed(1)} KB
                            </td>
                            <td className="py-2.5 text-right">
                              <a
                                href={getMediaUrl(f.url)}
                                download={f.name}
                                className="inline-flex items-center gap-1 px-3 py-1 rounded-lg bg-slate-800 hover:bg-cyan-500 hover:text-slate-950 text-cyan-300 font-bold transition-colors cursor-pointer"
                              >
                                <Download className="w-3 h-3" />
                                <span>Download</span>
                              </a>
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={4} className="py-8 text-center text-slate-500">
                            Folder files loading...
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Lightbox Modal for Frame High-Resolution Inspection */}
      {selectedFramePreview && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-in fade-in">
          <div className="max-w-4xl w-full p-6 rounded-2xl bg-[#0a0f1d] border border-cyan-500/50 shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-cyan-950 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
                  <ImageIcon className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white font-mono">
                    outputs/{jobProgress?.folder_name || 'video_folder'}/snapshots/{selectedFramePreview.filename}
                  </h3>
                  <div className="text-[11px] text-slate-400 font-mono flex items-center gap-3 mt-0.5">
                    <span className="text-cyan-400 font-bold">Frame #{selectedFramePreview.frame_number}</span>
                    <span>•</span>
                    <span>Time: {((selectedFramePreview.frame_number || 1) / (jobProgress?.fps || 25)).toFixed(2)}s</span>
                    <span>•</span>
                    <span className="text-emerald-400 font-bold">
                      Confidence: {selectedFramePreview.confidence ? (selectedFramePreview.confidence * 100).toFixed(1) : (jobProgress?.peak_confidence ? (jobProgress.peak_confidence * 100).toFixed(1) : 0)}%
                    </span>
                  </div>
                </div>
              </div>

              <button
                onClick={() => setSelectedFramePreview(null)}
                className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="relative aspect-video rounded-xl bg-black overflow-hidden border border-slate-800 flex items-center justify-center">
              <img
                src={getMediaUrl(selectedFramePreview.url)}
                alt={`Frame #${selectedFramePreview.frame_number}`}
                className="w-full h-full object-contain"
              />
            </div>

            <div className="flex items-center justify-between pt-2">
              <div className="text-xs font-mono text-slate-400">
                Evidentiary frame extracted from video archive
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => {
                    handleJumpToFrame(selectedFramePreview.frame_number);
                    setSelectedFramePreview(null);
                  }}
                  className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs uppercase tracking-wider transition-colors flex items-center gap-2 cursor-pointer shadow-lg shadow-cyan-950/40"
                >
                  <Play className="w-3.5 h-3.5" />
                  Jump Player to Frame
                </button>
                <a
                  href={getMediaUrl(selectedFramePreview.url)}
                  download={selectedFramePreview.filename || `frame_${selectedFramePreview.frame_number}.jpg`}
                  className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-bold text-xs uppercase tracking-wider transition-colors flex items-center gap-2 cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download Frame
                </a>
              </div>
            </div>
          </div>
        </div>
      )}

      {errorMsg && (
        <div className="p-4 rounded-xl bg-red-950/60 border border-red-500/50 text-red-300 text-xs font-mono flex items-center gap-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
};
