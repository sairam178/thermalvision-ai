import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  Camera,
  VideoOff,
  Play,
  Square,
  AlertTriangle,
  ShieldAlert,
  Volume2,
  VolumeX,
  Camera as SnapshotIcon,
  Sliders,
  CheckCircle2,
  RefreshCw,
} from 'lucide-react';
import { detectImageFrame } from '../services/api';
import type { DetectionBox } from '../types';

export const LiveMonitor: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [isWebcamConnected, setIsWebcamConnected] = useState(false);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [detectedBoxes, setDetectedBoxes] = useState<DetectionBox[]>([]);
  const [pedestrianCount, setPedestrianCount] = useState<number>(0);
  const [inferenceTimeMs, setInferenceTimeMs] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [audioAlert, setAudioAlert] = useState<boolean>(true);
  const [isSimulatingFeed, setIsSimulatingFeed] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('Camera standby');
  const [lastSnapshotSaved, setLastSnapshotSaved] = useState<string | null>(null);

  const monitoringRef = useRef(false);
  monitoringRef.current = isMonitoring;

  // Sound alert beep using Web Audio API
  const playAlertSound = useCallback(() => {
    if (!audioAlert) return;
    try {
      const audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5
      gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.2);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.2);
    } catch {
      // AudioContext unavailable or blocked by autoplay
    }
  }, [audioAlert]);

  // Connect Webcam
  const handleConnectWebcam = async () => {
    try {
      setStatusMessage('Requesting camera access...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsWebcamConnected(true);
        setIsSimulatingFeed(false);
        setStatusMessage('Webcam connected. Single-channel grayscale ready.');
      }
    } catch (err) {
      console.error('Camera connection failed:', err);
      setStatusMessage('Failed to access webcam. You can try the Sample Thermal Feed below.');
    }
  };

  // Disconnect Webcam
  const handleDisconnectWebcam = () => {
    setIsMonitoring(false);
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsWebcamConnected(false);
    setIsSimulatingFeed(false);
    setDetectedBoxes([]);
    setPedestrianCount(0);
    setStatusMessage('Camera disconnected');
  };

  // Load Synthetic / Sample Thermal Simulation Feed
  const handleLoadSimulationFeed = () => {
    handleDisconnectWebcam();
    setIsSimulatingFeed(true);
    setStatusMessage('Simulation feed active: synthetic pedestrian test pattern');
  };

  // Main Detection Loop
  useEffect(() => {
    let animationFrameId: number;
    let lastTime = performance.now();
    let frameCounter = 0;
    let isProcessing = false;

    const processFrame = async () => {
      if (!monitoringRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const hiddenCanvas = hiddenCanvasRef.current;

      if (video && canvas && hiddenCanvas && isWebcamConnected && video.readyState === 4) {
        const width = video.videoWidth || 640;
        const height = video.videoHeight || 480;

        canvas.width = width;
        canvas.height = height;
        hiddenCanvas.width = width;
        hiddenCanvas.height = height;

        const ctx = canvas.getContext('2d');
        const hiddenCtx = hiddenCanvas.getContext('2d');

        if (ctx && hiddenCtx) {
          // Draw video to hidden canvas
          hiddenCtx.drawImage(video, 0, 0, width, height);

          // Convert to 1-channel Grayscale representation
          const imgData = hiddenCtx.getImageData(0, 0, width, height);
          const data = imgData.data;
          for (let i = 0; i < data.length; i += 4) {
            const gray = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114) | 0;
            data[i] = gray;
            data[i + 1] = gray;
            data[i + 2] = gray;
          }
          hiddenCtx.putImageData(imgData, 0, 0);

          // Render grayscale image onto display canvas
          ctx.drawImage(hiddenCanvas, 0, 0, width, height);

          // Send to backend model if not already processing previous frame
          if (!isProcessing) {
            isProcessing = true;

            hiddenCanvas.toBlob(
              async (blob) => {
                if (blob && monitoringRef.current) {
                  try {
                    const result = await detectImageFrame(blob, confThreshold);
                    setDetectedBoxes(result.boxes);
                    setPedestrianCount(result.pedestrians_count);
                    setInferenceTimeMs(result.inference_time_ms);

                    if (result.pedestrians_count > 0) {
                      playAlertSound();
                    }
                  } catch (err) {
                    console.error('Frame inference error:', err);
                  }
                }
                frameCounter++;
                const now = performance.now();
                if (now - lastTime >= 1000) {
                  setFps(Math.round((frameCounter * 1000) / (now - lastTime)));
                  frameCounter = 0;
                  lastTime = now;
                }
                isProcessing = false;
              },
              'image/jpeg',
              0.85
            );
          }

          // Render Bounding Boxes on main display canvas
          for (const box of detectedBoxes) {
            ctx.strokeStyle = '#00e5ff';
            ctx.lineWidth = 2.5;
            ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);

            // Corner brackets
            const cornerLen = Math.min(14, (box.x2 - box.x1) / 4);
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 3.5;
            // Top-left
            ctx.beginPath();
            ctx.moveTo(box.x1, box.y1 + cornerLen);
            ctx.lineTo(box.x1, box.y1);
            ctx.lineTo(box.x1 + cornerLen, box.y1);
            ctx.stroke();
            // Top-right
            ctx.beginPath();
            ctx.moveTo(box.x2 - cornerLen, box.y1);
            ctx.lineTo(box.x2, box.y1);
            ctx.lineTo(box.x2, box.y1 + cornerLen);
            ctx.stroke();

            // Label background
            const label = `Pedestrian ${(box.confidence * 100).toFixed(0)}%`;
            ctx.font = 'bold 12px monospace';
            const textWidth = ctx.measureText(label).width;
            ctx.fillStyle = 'rgba(7, 10, 18, 0.85)';
            ctx.fillRect(box.x1, Math.max(0, box.y1 - 22), textWidth + 12, 20);

            // Label text
            ctx.fillStyle = '#00e5ff';
            ctx.fillText(label, box.x1 + 6, Math.max(14, box.y1 - 7));
          }
        }
      }

      if (monitoringRef.current) {
        animationFrameId = requestAnimationFrame(processFrame);
      }
    };

    if (isMonitoring && isWebcamConnected) {
      animationFrameId = requestAnimationFrame(processFrame);
    }

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isMonitoring, isWebcamConnected, confThreshold, detectedBoxes, playAlertSound]);

  // Capture Evidentiary Snapshot
  const handleCaptureSnapshot = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      try {
        const formData = new FormData();
        formData.append('file', blob, `live_snapshot_${Date.now()}.jpg`);
        formData.append('pedestrians_count', pedestrianCount.toString());
        formData.append('confidence', detectedBoxes.length > 0 ? detectedBoxes[0].confidence.toString() : '0');

        const res = await fetch('/api/history/snapshot', {
          method: 'POST',
          body: formData,
        });
        if (res.ok) {
          setLastSnapshotSaved(`Snapshot saved to vault at ${new Date().toLocaleTimeString()}`);
          setTimeout(() => setLastSnapshotSaved(null), 4000);
        }
      } catch (err) {
        console.error('Failed to save snapshot:', err);
      }
    });
  };

  return (
    <div className="space-y-6 pb-16">
      {/* Hidden processing canvas */}
      <canvas ref={hiddenCanvasRef} className="hidden" />

      {/* Top Controls Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-2xl bg-[#0a0d16] border border-slate-800">
        <div>
          <h1 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight flex items-center gap-2.5">
            <span className="w-3 h-3 rounded-full bg-cyan-400 animate-hud-blink" />
            Live Camera Surveillance Stream
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            {statusMessage}
          </p>
        </div>

        <div className="flex items-center flex-wrap gap-2.5">
          {!isWebcamConnected ? (
            <>
              <button
                onClick={handleConnectWebcam}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-950/40 transition-all cursor-pointer"
              >
                <Camera className="w-4 h-4" />
                Connect Webcam
              </button>
              <button
                onClick={handleLoadSimulationFeed}
                className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 font-medium text-xs transition-colors cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5 text-cyan-400" />
                Load Camera Sim
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => setIsMonitoring(!isMonitoring)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl font-bold text-xs transition-all shadow-md cursor-pointer ${
                  isMonitoring
                    ? 'bg-amber-600 hover:bg-amber-500 text-white'
                    : 'bg-emerald-600 hover:bg-emerald-500 text-white'
                }`}
              >
                {isMonitoring ? (
                  <>
                    <Square className="w-3.5 h-3.5" />
                    Pause Monitoring
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5" />
                    Start Monitoring
                  </>
                )}
              </button>

              <button
                onClick={handleDisconnectWebcam}
                className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-red-950/40 border border-slate-700 hover:border-red-500/40 text-slate-300 hover:text-red-300 text-xs font-medium transition-colors cursor-pointer"
              >
                <VideoOff className="w-4 h-4" />
                Disconnect
              </button>
            </>
          )}
        </div>
      </div>

      {/* Prominent Disclaimer Banner */}
      <div className="p-3.5 rounded-xl bg-[#0e1628] border border-blue-500/30 flex items-start gap-3">
        <AlertTriangle className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
        <div className="text-xs text-slate-300 leading-relaxed">
          <span className="font-bold text-cyan-300">Technical Modality Notice:</span> Real-time webcam frames are converted to genuine 1-channel grayscale matrices to match the input layer of the custom YOLOv8 model.
          This mode is designed for <span className="text-white font-semibold">interface verification and algorithmic testing</span>, not authentic long-wave infrared (LWIR) thermal hardware.
        </div>
      </div>

      {/* Main Viewport & Telemetry Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Surveillance Viewport (3 Cols) */}
        <div className="lg:col-span-3 space-y-3">
          <div className="relative w-full aspect-video rounded-2xl bg-[#06080e] border border-slate-800 overflow-hidden shadow-2xl flex items-center justify-center">
            {/* Ambient surveillance grid */}
            <div className="absolute inset-0 surveillance-grid opacity-30 pointer-events-none" />

            {/* Hidden native video element */}
            <video ref={videoRef} className="hidden" playsInline muted />

            {/* Active Drawing Canvas */}
            {isWebcamConnected ? (
              <canvas ref={canvasRef} className="w-full h-full object-contain" />
            ) : isSimulatingFeed ? (
              /* Simulated Thermal Camera View */
              <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center">
                <div className="w-16 h-16 rounded-full bg-cyan-950/40 border border-cyan-500/30 flex items-center justify-center mb-4">
                  <Camera className="w-8 h-8 text-cyan-400 animate-pulse" />
                </div>
                <div className="text-sm font-bold text-white mb-1">
                  Synthetic Camera Simulation Active
                </div>
                <div className="text-xs text-slate-400 max-w-sm mb-4">
                  Connect a physical webcam or switch to the Video Analyzer tab to ingest real FLIR thermal MP4 files with 100% genuine thermal gradients.
                </div>
                <button
                  onClick={handleConnectWebcam}
                  className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs transition-colors cursor-pointer"
                >
                  Connect Local Webcam Device
                </button>
              </div>
            ) : (
              /* Camera Standby View */
              <div className="text-center p-6">
                <div className="w-14 h-14 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center mx-auto mb-3 text-slate-600">
                  <Camera className="w-7 h-7" />
                </div>
                <div className="text-sm font-bold text-slate-300 mb-1">Camera Stream Standby</div>
                <div className="text-xs text-slate-500 mb-4">Click "Connect Webcam" above to initialize video stream</div>
                <button
                  onClick={handleConnectWebcam}
                  className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-cyan-300 text-xs font-semibold transition-colors cursor-pointer"
                >
                  Request Camera Permission
                </button>
              </div>
            )}

            {/* Top Right HUD Pill */}
            <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/80 backdrop-blur-md border border-slate-800 text-[10px] font-mono">
              <span
                className={`w-2 h-2 rounded-full ${
                  isMonitoring ? 'bg-red-500 animate-pulse' : 'bg-slate-500'
                }`}
              />
              <span className="text-slate-300 font-bold">
                {isMonitoring ? 'SURVEILLANCE HUD • ACTIVE' : 'STREAM PAUSED'}
              </span>
            </div>

            {/* Top Left Live Modality Tag */}
            <div className="absolute top-4 left-4 px-2.5 py-1 rounded-md bg-black/80 backdrop-blur-md border border-slate-800 text-[10px] font-mono text-cyan-400">
              1-CH LUMINANCE PIPELINE
            </div>

            {/* Bottom Overlay Target Stats */}
            {isWebcamConnected && (
              <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between pointer-events-none">
                <div className="flex items-center gap-2">
                  <div className="px-3 py-1.5 rounded-lg bg-black/85 backdrop-blur-md border border-slate-800 text-xs font-mono text-white flex items-center gap-2">
                    <span className="text-slate-400">TARGETS:</span>
                    <span
                      className={`font-black ${
                        pedestrianCount > 0 ? 'text-red-400' : 'text-emerald-400'
                      }`}
                    >
                      {pedestrianCount} PEDESTRIANS
                    </span>
                  </div>

                  <div className="px-3 py-1.5 rounded-lg bg-black/85 backdrop-blur-md border border-slate-800 text-xs font-mono text-slate-300 hidden sm:flex items-center gap-2">
                    <span className="text-slate-500">LATENCY:</span>
                    <span className="text-cyan-300 font-bold">{inferenceTimeMs.toFixed(1)} ms</span>
                    <span className="text-slate-600">|</span>
                    <span className="text-slate-400">{fps} FPS</span>
                  </div>
                </div>

                {pedestrianCount > 0 && (
                  <div className="px-3 py-1.5 rounded-lg bg-red-950/90 border border-red-500/60 text-red-300 text-xs font-mono font-bold animate-pulse">
                    PEDESTRIAN PRESENT
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Snapshot feedback pill */}
          {lastSnapshotSaved && (
            <div className="p-2.5 rounded-xl bg-emerald-950/50 border border-emerald-500/40 text-emerald-300 text-xs font-mono flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>{lastSnapshotSaved}</span>
            </div>
          )}
        </div>

        {/* Telemetry & Controls Sidebar (1 Col) */}
        <div className="space-y-4">
          {/* Telemetry Card */}
          <div className="p-5 rounded-2xl bg-[#0a0d18] border border-slate-800 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <span className="text-xs font-mono tracking-wider text-slate-400 uppercase font-semibold">
                LIVE TELEMETRY
              </span>
              <span
                className={`px-2 py-0.5 rounded text-[9px] font-mono font-bold ${
                  pedestrianCount > 0
                    ? 'bg-red-950 border border-red-500/50 text-red-400'
                    : 'bg-emerald-950 border border-emerald-500/50 text-emerald-400'
                }`}
              >
                {pedestrianCount > 0 ? 'ACTIVE TARGET' : 'NORMAL FLOW'}
              </span>
            </div>

            {/* Pedestrians Metric */}
            <div className="p-3.5 rounded-xl bg-[#06080e] border border-slate-800">
              <div className="text-[11px] text-slate-400 font-mono mb-1">Pedestrian Count</div>
              <div className="text-3xl font-black text-white font-mono flex items-center justify-between">
                <span>{pedestrianCount}</span>
                <ShieldAlert
                  className={`w-6 h-6 ${
                    pedestrianCount > 0 ? 'text-red-400' : 'text-slate-600'
                  }`}
                />
              </div>
            </div>

            {/* Inference Status */}
            <div className="p-3.5 rounded-xl bg-[#06080e] border border-slate-800 space-y-1.5 text-xs font-mono">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Stream Status:</span>
                <span className="text-white font-semibold">
                  {isMonitoring ? 'Processing' : 'Idle'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Inference Time:</span>
                <span className="text-cyan-400 font-bold">{inferenceTimeMs.toFixed(1)} ms</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Frames / Sec:</span>
                <span className="text-slate-200">{fps} FPS</span>
              </div>
            </div>

            {/* Confidence Slider */}
            <div>
              <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5">
                <span className="flex items-center gap-1.5 font-mono">
                  <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                  Min Confidence:
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

            {/* Quick Actions */}
            <div className="pt-2 border-t border-slate-800 space-y-2">
              <button
                onClick={handleCaptureSnapshot}
                disabled={!isWebcamConnected}
                className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-xs font-bold text-white transition-colors border border-slate-700 cursor-pointer"
              >
                <SnapshotIcon className="w-4 h-4 text-cyan-400" />
                Capture Evidentiary Snapshot
              </button>

              <button
                onClick={() => setAudioAlert(!audioAlert)}
                className="w-full flex items-center justify-center gap-2 py-2 rounded-xl bg-slate-900/60 hover:bg-slate-900 text-xs font-medium text-slate-400 hover:text-slate-200 transition-colors border border-slate-800 cursor-pointer"
              >
                {audioAlert ? (
                  <>
                    <Volume2 className="w-3.5 h-3.5 text-cyan-400" />
                    Audio Alert: Enabled
                  </>
                ) : (
                  <>
                    <VolumeX className="w-3.5 h-3.5 text-slate-500" />
                    Audio Alert: Muted
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
