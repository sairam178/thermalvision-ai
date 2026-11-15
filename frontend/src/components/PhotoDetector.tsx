import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Upload,
  Image as ImageIcon,
  Sliders,
  CheckCircle2,
  Download,
  FileText,
  Eye,
  BookmarkPlus,
  RefreshCw,
  Sparkles,
  Layers,
  Activity,
  Crosshair,
  AlertCircle,
  Flame,
} from 'lucide-react';
import { detectImageFrame, saveSnapshotEvent, getMediaUrl, fetchSampleImages, generateNLPNarrativeReport } from '../services/api';
import type { FrameDetectionResult } from '../types';

export const PhotoDetector: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [detectionResult, setDetectionResult] = useState<FrameDetectionResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [saveSuccessMsg, setSaveSuccessMsg] = useState<string | null>(null);
  const [samples, setSamples] = useState<string[]>([]);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [colorMode, setColorMode] = useState<'normal' | 'ironbow' | 'invert'>('normal');
  const [aiReport, setAiReport] = useState<{
    threat_level: string;
    verdict_headline: string;
    summary_text: string;
    detailed_narrative: string;
    recommendations: string[];
    markdown_report: string;
  } | null>(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Load sample thermal images
  useEffect(() => {
    fetchSampleImages().then((res) => {
      if (res && res.length > 0) setSamples(res);
    });
  }, []);

  // Run detection
  const runDetection = useCallback(async (imageInput: Blob | string, nameHint: string = 'thermal_photo.jpg') => {
    try {
      setIsLoading(true);
      setErrorMsg(null);
      setSaveSuccessMsg(null);

      const res = await detectImageFrame(imageInput, confThreshold);
      res.filename = nameHint;
      setDetectionResult(res);
      setAiReport(null);
      setIsLoading(false);
    } catch (err: unknown) {
      console.error('Detection error:', err);
      setErrorMsg(err instanceof Error ? err.message : 'Photo thermal detection failed');
      setIsLoading(false);
    }
  }, [confThreshold]);

  // Handle local file select
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      runDetection(file, file.name);
    }
  };

  // Drag and drop
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      runDetection(file, file.name);
    }
  };

  // Load sample image
  const handleLoadSample = async (samplePath: string) => {
    try {
      setIsLoading(true);
      const fullUrl = getMediaUrl(samplePath);
      setPreviewUrl(fullUrl);
      setSelectedFile(null);

      const resp = await fetch(fullUrl);
      const blob = await resp.blob();
      const filename = samplePath.split('/').pop() || 'sample_thermal.png';
      runDetection(blob, filename);
    } catch (err) {
      console.error('Failed to load sample:', err);
      setErrorMsg('Failed to fetch sample image');
      setIsLoading(false);
    }
  };

  // Generate AI Narrative Report
  const handleGenerateAIReport = async () => {
    if (!detectionResult) return;
    try {
      setIsGeneratingReport(true);
      setErrorMsg(null);
      const rep = await generateNLPNarrativeReport({
        filename: detectionResult.filename || 'Thermal Photo',
        targets_count: detectionResult.targets_count ?? detectionResult.pedestrians_count,
        peak_confidence: peakConfidence,
        boxes: detectionResult.boxes,
        conf_threshold: confThreshold,
      });
      setAiReport(rep);
    } catch (err) {
      console.error('NLP Report error:', err);
      setErrorMsg('Failed to generate AI narrative report');
    } finally {
      setIsGeneratingReport(false);
    }
  };

  // Save to history
  const handleSaveToHistory = async () => {
    if (!detectionResult) return;
    try {
      setSaveSuccessMsg('Saving to evidence vault...');
      let blobToSave: Blob;

      if (detectionResult.annotated_image) {
        const resp = await fetch(detectionResult.annotated_image);
        blobToSave = await resp.blob();
      } else if (selectedFile) {
        blobToSave = selectedFile;
      } else if (previewUrl) {
        const resp = await fetch(previewUrl);
        blobToSave = await resp.blob();
      } else {
        throw new Error('No image available to save');
      }

      const count = detectionResult.targets_count ?? detectionResult.pedestrians_count;
      const peakConf = detectionResult.boxes.length > 0
        ? Math.max(...detectionResult.boxes.map((b) => b.confidence))
        : 0;

      await saveSnapshotEvent(blobToSave, {
        filename: detectionResult.filename || 'Thermal Photo Capture',
        frame_number: 1,
        targets_count: count,
        confidence: peakConf,
      });

      setSaveSuccessMsg('Successfully saved to Detection History Vault!');
      setTimeout(() => setSaveSuccessMsg(null), 4000);
    } catch (err) {
      console.error('Save to history failed:', err);
      setErrorMsg('Failed to save event to history');
    }
  };

  // Download annotated image
  const handleDownloadAnnotated = () => {
    if (!detectionResult?.annotated_image) return;
    const link = document.createElement('a');
    link.href = detectionResult.annotated_image;
    link.download = `annotated_${detectionResult.filename || 'thermal_photo.jpg'}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Export JSON Report
  const handleExportJSON = () => {
    if (!detectionResult) return;
    const report = {
      image_name: detectionResult.filename || 'thermal_photo.jpg',
      detection_mode: 'Photo Thermal Detection',
      model: 'YOLOv8 1-Channel LWIR',
      targets_detected: detectionResult.targets_count ?? detectionResult.pedestrians_count,
      inference_time_ms: detectionResult.inference_time_ms,
      confidence_threshold: confThreshold,
      boxes: detectionResult.boxes,
      timestamp: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `report_${detectionResult.filename || 'thermal_photo'}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const targetsCount = detectionResult?.targets_count ?? detectionResult?.pedestrians_count ?? 0;
  const peakConfidence = detectionResult?.boxes && detectionResult.boxes.length > 0
    ? Math.max(...detectionResult.boxes.map((b) => b.confidence))
    : 0;
  const avgConfidence = detectionResult?.boxes && detectionResult.boxes.length > 0
    ? detectionResult.boxes.reduce((acc, b) => acc + b.confidence, 0) / detectionResult.boxes.length
    : 0;

  return (
    <div className="space-y-8 pb-16">
      {/* Top Banner */}
      <div className="p-6 rounded-3xl bg-[#080d19] border border-[#192338] shadow-2xl relative overflow-hidden flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-600/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 flex-shrink-0 shadow-lg shadow-cyan-950/40">
            <Flame className="w-6 h-6 text-cyan-400" />
          </div>
          <div>
            <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-[10px] font-mono font-bold text-cyan-400 uppercase tracking-widest mb-1">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
              1-CHANNEL LWIR PHOTO INFERENCE
            </div>
            <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight">
              Photo Thermal Detection
            </h1>
            <p className="text-xs text-slate-400 mt-1 max-w-xl">
              Upload long-wave infrared (LWIR) or thermal imagery. The YOLOv8 1-channel engine identifies heat signatures, isolates distinct target coordinates, and renders high-precision bounding telemetry.
            </p>
          </div>
        </div>

        {/* Confidence Threshold Quick Slider */}
        <div className="p-3.5 rounded-2xl bg-[#050811] border border-slate-800/80 flex items-center gap-3 self-start md:self-center">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <div>
            <div className="text-[10px] font-mono uppercase text-slate-400 font-semibold flex justify-between">
              <span>Confidence Threshold</span>
              <span className="text-cyan-400 font-bold ml-3">{Math.round(confThreshold * 100)}%</span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.85"
              step="0.05"
              value={confThreshold}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setConfThreshold(val);
                if (previewUrl) {
                  // Re-run detection with updated confidence
                  if (selectedFile) {
                    runDetection(selectedFile, selectedFile.name);
                  } else {
                    fetch(previewUrl)
                      .then((r) => r.blob())
                      .then((b) => runDetection(b, detectionResult?.filename || 'sample.png'));
                  }
                }
              }}
              className="w-36 accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* Main Grid: Upload & Visualizer on Left (7 cols), Telemetry & Results on Right (5 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Image Canvas & Uploader */}
        <div className="lg:col-span-7 space-y-4">
          {/* Main Display Box */}
          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            className="relative rounded-3xl bg-[#060912] border border-[#192338] overflow-hidden min-h-[380px] sm:min-h-[460px] flex flex-col justify-center items-center shadow-2xl group"
          >
            {previewUrl ? (
              <div className="relative w-full h-full flex items-center justify-center p-3">
                {/* Active Image */}
                <img
                  src={
                    showBoxes && detectionResult?.annotated_image
                      ? detectionResult.annotated_image
                      : previewUrl
                  }
                  alt="Thermal Target Detection"
                  className={`max-h-[520px] w-auto max-w-full rounded-2xl object-contain transition-all duration-300 ${
                    colorMode === 'ironbow' ? 'filter hue-rotate-180 contrast-125' : ''
                  } ${colorMode === 'invert' ? 'filter invert contrast-125' : ''}`}
                />

                {/* Floating Top Left Badge */}
                <div className="absolute top-5 left-5 px-3 py-1.5 rounded-xl bg-black/80 backdrop-blur-md border border-slate-800 text-[11px] font-mono text-cyan-300 flex items-center gap-2">
                  <Crosshair className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
                  <span>
                    {targetsCount > 0 ? `${targetsCount} THERMAL TARGETS` : 'ANALYSIS CLEAR'}
                  </span>
                </div>

                {/* Floating Action Controls on Image Bottom */}
                <div className="absolute bottom-5 inset-x-5 flex flex-wrap items-center justify-between gap-2 bg-[#080d19]/90 backdrop-blur-md p-2 rounded-2xl border border-slate-800/80">
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => setShowBoxes(!showBoxes)}
                      className={`px-3 py-1.5 rounded-xl text-xs font-mono font-semibold transition-all cursor-pointer ${
                        showBoxes
                          ? 'bg-cyan-500/20 border border-cyan-500/50 text-cyan-300'
                          : 'bg-slate-900 border border-slate-800 text-slate-400'
                      }`}
                    >
                      <Eye className="w-3.5 h-3.5 inline mr-1" />
                      Boxes: {showBoxes ? 'ON' : 'OFF'}
                    </button>

                    <button
                      onClick={() => {
                        const modes: Array<'normal' | 'ironbow' | 'invert'> = ['normal', 'ironbow', 'invert'];
                        const nextIdx = (modes.indexOf(colorMode) + 1) % modes.length;
                        setColorMode(modes[nextIdx]);
                      }}
                      className="px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-mono text-slate-300 transition-colors cursor-pointer"
                    >
                      <Layers className="w-3.5 h-3.5 inline mr-1 text-amber-400" />
                      Palette: {colorMode.toUpperCase()}
                    </button>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-semibold text-slate-300 transition-colors cursor-pointer"
                    >
                      Replace Photo
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              /* Drag & Drop Prompt Area */
              <div
                onClick={() => fileInputRef.current?.click()}
                className="cursor-pointer p-8 text-center flex flex-col items-center justify-center space-y-4 hover:scale-[1.01] transition-transform"
              >
                <div className="w-20 h-20 rounded-3xl bg-[#09152b] border border-cyan-500/40 flex items-center justify-center text-cyan-400 shadow-xl shadow-cyan-950/40">
                  <Upload className="w-10 h-10 animate-bounce" />
                </div>
                <div className="space-y-1">
                  <h3 className="text-lg font-bold text-white">
                    Drop your thermal photo here, or browse
                  </h3>
                  <p className="text-xs text-slate-400 max-w-sm">
                    Supports JPG, PNG, WEBP, and BMP thermal sensor images. Genuine 1-channel grayscale conversion runs automatically.
                  </p>
                </div>
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-black font-extrabold text-xs tracking-wider transition-colors">
                  <ImageIcon className="w-4 h-4" />
                  <span>SELECT THERMAL PHOTO</span>
                </div>
              </div>
            )}

            {/* Hidden Input */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
            />

            {/* Loading Overlay */}
            {isLoading && (
              <div className="absolute inset-0 bg-[#060912]/85 backdrop-blur-sm flex flex-col items-center justify-center gap-3 z-30">
                <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin" />
                <div className="text-xs font-mono font-bold tracking-widest text-cyan-300 uppercase">
                  Running YOLOv8 Thermal Inference...
                </div>
              </div>
            )}
          </div>

          {/* Quick Preset Samples Gallery */}
          <div className="p-4 rounded-2xl bg-[#080d19] border border-[#192338]">
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-mono tracking-widest text-slate-400 uppercase font-bold flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                PRELOADED FLIR THERMAL SAMPLES (1-CLICK TEST)
              </span>
              <span className="text-[10px] font-mono text-slate-500">26k FLIR dataset frames</span>
            </div>

            <div className="grid grid-cols-4 gap-3">
              {samples.map((s, idx) => (
                <button
                  key={s}
                  onClick={() => handleLoadSample(s)}
                  className="group relative rounded-xl overflow-hidden border border-slate-800 hover:border-cyan-500/60 transition-all aspect-video bg-black flex items-center justify-center cursor-pointer shadow-md"
                >
                  <img
                    src={getMediaUrl(s)}
                    alt={`Sample ${idx + 1}`}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300 opacity-80 group-hover:opacity-100"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent flex items-end p-1.5">
                    <span className="text-[9px] font-mono text-cyan-300 font-bold truncate">
                      Sample #{idx + 1}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Telemetry & Detection Dossier */}
        <div className="lg:col-span-5 space-y-4">
          {/* Target Detection Summary Card */}
          <div className="p-6 rounded-3xl bg-[#080d19] border border-[#192338] shadow-xl space-y-5">
            <div className="flex items-center justify-between pb-4 border-b border-slate-800/80">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                <h3 className="text-xs font-mono font-bold tracking-wider uppercase text-white">
                  DETECTION TELEMETRY
                </h3>
              </div>
              <span
                className={`px-3 py-1 rounded-full text-[10px] font-mono font-bold uppercase tracking-wider ${
                  targetsCount > 0
                    ? 'bg-red-950/50 border border-red-500/50 text-red-400'
                    : 'bg-emerald-950/50 border border-emerald-500/50 text-emerald-400'
                }`}
              >
                {targetsCount > 0 ? 'TARGET ACQUIRED' : 'ZERO ANOMALIES'}
              </span>
            </div>

            {/* 4 Stats Grid */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3.5 rounded-2xl bg-[#050811] border border-slate-800/80">
                <div className="text-[10px] font-mono uppercase text-slate-500 font-semibold">
                  TARGETS FOUND
                </div>
                <div className="text-2xl font-black text-white font-mono mt-0.5">
                  {targetsCount}
                </div>
                <div className="text-[10px] font-mono text-cyan-400">Heat Signatures</div>
              </div>

              <div className="p-3.5 rounded-2xl bg-[#050811] border border-slate-800/80">
                <div className="text-[10px] font-mono uppercase text-slate-500 font-semibold">
                  PEAK CONFIDENCE
                </div>
                <div className="text-2xl font-black text-white font-mono mt-0.5">
                  {(peakConfidence * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] font-mono text-slate-400">
                  Avg: {(avgConfidence * 100).toFixed(0)}%
                </div>
              </div>

              <div className="p-3.5 rounded-2xl bg-[#050811] border border-slate-800/80">
                <div className="text-[10px] font-mono uppercase text-slate-500 font-semibold">
                  INFERENCE SPEED
                </div>
                <div className="text-2xl font-black text-white font-mono mt-0.5">
                  {detectionResult?.inference_time_ms ? `${detectionResult.inference_time_ms} ms` : '--'}
                </div>
                <div className="text-[10px] font-mono text-slate-400">1-Ch Grayscale</div>
              </div>

              <div className="p-3.5 rounded-2xl bg-[#050811] border border-slate-800/80">
                <div className="text-[10px] font-mono uppercase text-slate-500 font-semibold">
                  RESOLUTION
                </div>
                <div className="text-2xl font-black text-white font-mono mt-0.5">
                  {detectionResult?.image_width
                    ? `${detectionResult.image_width}x${detectionResult.image_height}`
                    : '640x512'}
                </div>
                <div className="text-[10px] font-mono text-slate-400">Sensor Matrix</div>
              </div>
            </div>

            {/* List of Detected Bounding Boxes */}
            <div className="space-y-2">
              <div className="text-[11px] font-mono uppercase font-bold text-slate-400 flex items-center justify-between">
                <span>DETECTED TARGETS BREAKDOWN</span>
                <span className="text-slate-500">{detectionResult?.boxes.length || 0} items</span>
              </div>

              <div className="max-h-48 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
                {detectionResult?.boxes && detectionResult.boxes.length > 0 ? (
                  detectionResult.boxes.map((b, i) => (
                    <div
                      key={i}
                      className="p-2.5 rounded-xl bg-[#060912] border border-slate-800 flex items-center justify-between text-xs font-mono"
                    >
                      <div className="flex items-center gap-2">
                        <span className="w-5 h-5 rounded-lg bg-cyan-950 border border-cyan-500/40 text-cyan-300 flex items-center justify-center text-[10px] font-bold">
                          {i + 1}
                        </span>
                        <div>
                          <div className="font-bold text-white">Thermal Target #{i + 1}</div>
                          <div className="text-[10px] text-slate-500">
                            Box: [{Math.round(b.x1)}, {Math.round(b.y1)}] to [{Math.round(b.x2)}, {Math.round(b.y2)}]
                          </div>
                        </div>
                      </div>

                      <div className="text-right">
                        <span className="px-2 py-0.5 rounded-md bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 font-bold text-[11px]">
                          {(b.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="p-4 rounded-xl bg-[#050811] border border-dashed border-slate-800 text-center text-xs text-slate-500 font-mono">
                    No targets detected. Select a sample or upload a photo to evaluate.
                  </div>
                )}
              </div>
            </div>

            {/* Actions Toolbar */}
            <div className="pt-2 border-t border-slate-800/80 space-y-2">
              <button
                disabled={!detectionResult || isGeneratingReport}
                onClick={handleGenerateAIReport}
                className="w-full py-2.5 rounded-2xl bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-600 hover:from-indigo-500 hover:to-cyan-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-extrabold text-xs tracking-wider transition-all flex items-center justify-center gap-2 shadow-lg shadow-indigo-950/50 cursor-pointer"
              >
                <Sparkles className="w-4 h-4 text-cyan-300 animate-pulse" />
                <span>{isGeneratingReport ? 'SYNTHESIZING NLP NARRATIVE...' : 'GENERATE AI INCIDENT NARRATIVE (NLP)'}</span>
              </button>

              <button
                disabled={!detectionResult}
                onClick={handleSaveToHistory}
                className="w-full py-2.5 rounded-2xl bg-[#1689ea] hover:bg-[#1a94fe] disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold text-xs tracking-wider transition-all flex items-center justify-center gap-2 shadow-lg shadow-sky-950/50 cursor-pointer"
              >
                <BookmarkPlus className="w-4 h-4" />
                <span>SAVE CAPTURE TO INCIDENT HISTORY</span>
              </button>

              <div className="grid grid-cols-2 gap-2">
                <button
                  disabled={!detectionResult?.annotated_image}
                  onClick={handleDownloadAnnotated}
                  className="py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed border border-slate-800 text-slate-200 font-semibold text-xs transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Download Image</span>
                </button>

                <button
                  disabled={!detectionResult}
                  onClick={handleExportJSON}
                  className="py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed border border-slate-800 text-slate-200 font-semibold text-xs transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <FileText className="w-3.5 h-3.5 text-amber-400" />
                  <span>Export JSON</span>
                </button>
              </div>

              {/* AI Narrative Report Card */}
              {aiReport && (
                <div className="p-4 rounded-2xl bg-gradient-to-br from-[#0c142b] to-[#070b18] border border-cyan-500/40 shadow-xl space-y-3 animate-in fade-in">
                  <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                    <span className="text-[10px] font-mono uppercase font-bold text-cyan-300 flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                      NLP TACTICAL NARRATIVE
                    </span>
                    <span className="px-2 py-0.5 rounded text-[9px] font-mono font-bold bg-cyan-950 border border-cyan-500/50 text-cyan-300">
                      {aiReport.threat_level}
                    </span>
                  </div>

                  <div>
                    <h4 className="text-xs font-bold text-white mb-1">{aiReport.verdict_headline}</h4>
                    <p className="text-[11px] text-slate-300 leading-relaxed font-sans">
                      {aiReport.summary_text}
                    </p>
                  </div>

                  <div className="p-2.5 rounded-xl bg-black/40 border border-slate-800/80 text-[11px] text-slate-400 leading-relaxed font-mono">
                    {aiReport.detailed_narrative}
                  </div>

                  <div className="space-y-1">
                    <span className="text-[10px] font-mono uppercase text-slate-400 font-bold">TACTICAL DIRECTIVES:</span>
                    <ul className="space-y-1 text-[11px] text-slate-300">
                      {aiReport.recommendations.map((rec, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <span className="text-cyan-400 font-bold">•</span>
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {/* Feedback messages */}
              {saveSuccessMsg && (
                <div className="p-2.5 rounded-xl bg-emerald-950/60 border border-emerald-500/40 text-emerald-300 text-xs font-mono flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                  <span>{saveSuccessMsg}</span>
                </div>
              )}

              {errorMsg && (
                <div className="p-2.5 rounded-xl bg-red-950/60 border border-red-500/40 text-red-300 text-xs font-mono flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <span>{errorMsg}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
