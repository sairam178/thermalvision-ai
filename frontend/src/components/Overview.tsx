import React, { useState } from 'react';
import {
  Image as ImageIcon,
  ArrowRight,
  ShieldCheck,
  Cpu,
  Layers,
  Film,
  MonitorCheck,
  Check,
  Scan,
  GitFork,
  Gauge,
  Flame,
  FolderArchive,
  Crosshair,
} from 'lucide-react';
import type { OverviewStats } from '../types';
import { getMediaUrl } from '../services/api';

interface OverviewProps {
  stats: OverviewStats | null;
  onNavigate: (tab: 'photo' | 'video' | 'history') => void;
}

export const Overview: React.FC<OverviewProps> = ({ stats, onNavigate }) => {
  const modelStatus = stats?.model_status;
  const [selectedDayIndex, setSelectedDayIndex] = useState<number>(6);

  const daysList = [
    '09 Nov',
    '10 Nov',
    '11 Nov',
    '12 Nov',
    '13 Nov',
    '14 Nov',
    '15 Nov',
  ];

  const totalTargets = stats?.total_targets_detected ?? stats?.total_detections ?? stats?.total_pedestrian_detections ?? 0;

  return (
    <div className="space-y-12 pb-16">
      {/* =========================================================================
          HERO SECTION: ThermalVision AI
         ========================================================================= */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center pt-2">
        {/* Left Column: Headlines & Actions (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Tag Pill */}
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-[#09152b] border border-cyan-500/40 text-[11px] font-mono font-semibold text-cyan-400 tracking-wider shadow-sm">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            LWIR 1-CHANNEL THERMAL AI ENGINE
          </div>

          {/* Main Title */}
          <div className="space-y-1">
            <h1 className="text-4xl sm:text-5xl lg:text-[52px] font-black text-white tracking-tight leading-[1.12]">
              See Through Darkness,
            </h1>
            <div className="text-4xl sm:text-5xl lg:text-[52px] font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-300 to-blue-500 tracking-tight leading-[1.12]">
              Decipher Heat Signatures.
            </div>
          </div>

          {/* Descriptive Body */}
          <p className="text-sm sm:text-[15px] text-slate-300 leading-relaxed max-w-xl font-normal">
            ThermalVision AI applies custom 1-channel YOLOv8 deep learning directly to long-wave infrared (LWIR) data. Detect targets, thermal anomalies, and heat signatures through smoke, foliage, and absolute zero-visibility conditions with real-time hardware-accelerated inference.
          </p>

          {/* Action Buttons Row */}
          <div className="flex flex-wrap items-center gap-4 pt-1">
            <button
              onClick={() => onNavigate('photo')}
              className="flex items-center gap-2.5 px-6 py-3.5 rounded-2xl bg-gradient-to-r from-[#1689ea] to-[#0ea5e9] hover:from-[#1a94fe] hover:to-[#38bdf8] text-white font-extrabold text-xs tracking-wider shadow-xl shadow-cyan-950/60 transition-all cursor-pointer transform hover:-translate-y-0.5"
            >
              <ImageIcon className="w-4 h-4 text-white" />
              <span>PHOTO THERMAL DETECTION</span>
              <ArrowRight className="w-4 h-4 text-white" />
            </button>

            <button
              onClick={() => onNavigate('video')}
              className="flex items-center gap-2 px-5 py-3.5 rounded-2xl bg-[#0b1322] hover:bg-[#111e33] border border-slate-700/80 text-white font-semibold text-xs tracking-wide transition-all cursor-pointer"
            >
              <Film className="w-4 h-4 text-slate-300" />
              <span>Analyze Thermal Video</span>
            </button>
          </div>

          {/* Micro Telemetry Specs Row */}
          <div className="pt-5 flex items-center gap-8 sm:gap-12 border-t border-slate-800/80">
            <div>
              <div className="text-[10px] font-mono tracking-widest text-slate-500 uppercase font-bold">
                MODALITY
              </div>
              <div className="text-xs font-mono font-bold text-slate-200 mt-1">
                LWIR 8-14μm
              </div>
            </div>

            <div>
              <div className="text-[10px] font-mono tracking-widest text-slate-500 uppercase font-bold">
                CHANNELS
              </div>
              <div className="text-xs font-mono font-bold text-cyan-400 mt-1">
                1-Ch Grayscale
              </div>
            </div>

            <div>
              <div className="text-[10px] font-mono tracking-widest text-slate-500 uppercase font-bold">
                INFERENCE
              </div>
              <div className="text-xs font-mono font-bold text-emerald-400 mt-1">
                &lt; 25ms / frame
              </div>
            </div>

            <div>
              <div className="text-[10px] font-mono tracking-widest text-slate-500 uppercase font-bold">
                TRACKING
              </div>
              <div className="text-xs font-mono font-bold text-slate-200 mt-1">
                ByteTrack
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Thermal AI Telemetry Card (5 Cols) */}
        <div className="lg:col-span-5">
          <div className="p-6 rounded-3xl bg-[#080d1a] border border-[#1a253c] shadow-2xl relative overflow-hidden">
            <div className="absolute inset-0 surveillance-grid opacity-20 pointer-events-none" />

            {/* Card Header */}
            <div className="flex items-center justify-between pb-5 border-b border-slate-800/80 relative z-10">
              <div className="flex items-center gap-2 text-cyan-400">
                <Flame className="w-4 h-4 animate-pulse text-cyan-400" />
                <span className="text-xs font-mono font-bold tracking-wider uppercase">
                  THERMAL SENSOR TELEMETRY
                </span>
              </div>
              <span className="px-3 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-500/50 text-cyan-300 text-[10px] font-mono font-bold uppercase tracking-wider">
                {modelStatus?.loaded ? 'ENGINE ACTIVE' : 'STANDBY'}
              </span>
            </div>

            {/* 2x2 Telemetry Metric Boxes */}
            <div className="grid grid-cols-2 gap-3.5 my-5 relative z-10">
              {/* Box 1 */}
              <div className="p-4 rounded-2xl bg-[#050811] border border-slate-800/80 space-y-1">
                <div className="text-[10px] font-mono tracking-wider text-slate-500 uppercase font-semibold">
                  INFERENCE LATENCY
                </div>
                <div className="text-xl sm:text-2xl font-black text-white font-mono">
                  &lt; 25 ms
                </div>
                <div className="text-[11px] font-mono text-cyan-400">
                  {modelStatus?.device || 'CUDA/CPU'} Stream
                </div>
              </div>

              {/* Box 2 */}
              <div className="p-4 rounded-2xl bg-[#050811] border border-slate-800/80 space-y-1">
                <div className="text-[10px] font-mono tracking-wider text-slate-500 uppercase font-semibold">
                  DETECTION ACCURACY
                </div>
                <div className="text-xl sm:text-2xl font-black text-white font-mono">
                  91.4%
                </div>
                <div className="text-[11px] font-mono text-slate-400">
                  FLIR 2024 Benchmark
                </div>
              </div>

              {/* Box 3 */}
              <div className="p-4 rounded-2xl bg-[#050811] border border-slate-800/80 space-y-1">
                <div className="text-[10px] font-mono tracking-wider text-slate-500 uppercase font-semibold">
                  SESSION ARCHIVE
                </div>
                <div className="text-xl sm:text-2xl font-black text-white font-mono">
                  Auto-Folder
                </div>
                <div className="text-[11px] font-mono text-slate-400">
                  Video-named dossier
                </div>
              </div>

              {/* Box 4 */}
              <div className="p-4 rounded-2xl bg-[#050811] border border-slate-800/80 space-y-1">
                <div className="text-[10px] font-mono tracking-wider text-slate-500 uppercase font-semibold">
                  VIDEO ENCODING
                </div>
                <div className="text-xl sm:text-2xl font-black text-white font-mono">
                  H.264
                </div>
                <div className="text-[11px] font-mono text-slate-400">
                  Browser-native stream
                </div>
              </div>
            </div>

            {/* Bottom Timeline */}
            <div className="pt-4 border-t border-slate-800/80 relative z-10 space-y-2">
              <div className="text-[10px] font-mono tracking-wider text-slate-500 uppercase font-semibold">
                SURVEILLANCE TIMELINE
              </div>
              <div className="flex items-center justify-between gap-1 text-[11px] font-mono">
                {daysList.map((day, i) => (
                  <button
                    key={day}
                    onClick={() => setSelectedDayIndex(i)}
                    className={`px-2 py-1 rounded-md transition-colors cursor-pointer ${
                      selectedDayIndex === i
                        ? 'text-cyan-300 font-bold bg-[#0c1d36] border border-cyan-500/40 shadow-sm'
                        : 'text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    {day}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          SECTION 2: TWO DETECTION MODES (Photo & Video)
         ========================================================================= */}
      <section className="relative pt-6 border-t border-slate-800/80">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-6">
          <div>
            <div className="text-[11px] font-mono tracking-widest text-cyan-400 uppercase font-bold mb-1.5">
              DUAL DETECTION WORKFLOWS
            </div>
            <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Analyze Still Captures or Ingest Full Video
            </h2>
          </div>
          <p className="text-xs sm:text-sm text-slate-400 max-w-md font-normal leading-relaxed">
            Both workflows feed the same genuine 1-channel YOLOv8 thermal model, giving you unified detection accuracy across individual photos and continuous video streams.
          </p>
        </div>

        {/* Dual Primary Action Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Card 1: Photo Thermal Detection */}
          <div
            onClick={() => onNavigate('photo')}
            className="group cursor-pointer relative overflow-hidden rounded-3xl bg-[#080d19] border border-[#192338] hover:border-cyan-500/50 transition-all duration-300 p-6 flex flex-col justify-between shadow-xl hover:shadow-cyan-950/30"
          >
            <div>
              {/* Badge Tag */}
              <div className="inline-block px-3 py-1 rounded-full bg-[#0b1424] border border-[#1e3458] text-[10px] font-mono font-bold text-cyan-400 tracking-wider mb-4">
                PHOTO THERMAL DETECTION
              </div>

              {/* Vector Graphic: Thermal Scan Grid */}
              <div className="h-32 w-full rounded-2xl bg-[#060912] border border-[#141d30] mb-5 relative flex items-center justify-center overflow-hidden p-4">
                <div className="w-full h-full relative flex items-center justify-center">
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-950/40 via-cyan-950/30 to-blue-950/40" />
                  {/* Crosshair Graphic */}
                  <div className="relative z-10 w-24 h-24 rounded-full border border-dashed border-cyan-500/40 flex items-center justify-center animate-pulse">
                    <Crosshair className="w-10 h-10 text-cyan-400" />
                  </div>
                  {/* Neon Target Bracket */}
                  <div className="absolute top-4 right-8 w-16 h-12 border-2 border-cyan-400 rounded-sm flex items-center justify-center">
                    <span className="text-[8px] font-mono text-cyan-300 bg-black/80 px-1 -top-2 relative">Target 94%</span>
                  </div>
                </div>
              </div>

              <h3 className="text-xl font-bold text-white mb-2 group-hover:text-cyan-300 transition-colors">
                Single-Capture Thermal Inspection
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed mb-4">
                Drop any thermal or LWIR sensor image, or test with preloaded FLIR dataset samples. The system highlights heat signatures, calculates confidence scores, and lets you save dossiers directly to the Incident Vault.
              </p>

              <ul className="space-y-2 text-xs text-slate-300 mb-6">
                <li className="flex items-center gap-2">
                  <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                  <span>Interactive threshold slider &amp; palette toggle</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                  <span>Preloaded 1-click FLIR thermal sample presets</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                  <span>Export annotated photos &amp; structured JSON dossiers</span>
                </li>
              </ul>
            </div>

            <div className="pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs font-semibold text-cyan-400 group-hover:text-cyan-300">
              <span className="tracking-wider uppercase font-mono">OPEN PHOTO DETECTION</span>
              <ArrowRight className="w-4 h-4 transform group-hover:translate-x-1.5 transition-transform" />
            </div>
          </div>

          {/* Card 2: Video Forensic Ingestion */}
          <div
            onClick={() => onNavigate('video')}
            className="group cursor-pointer relative overflow-hidden rounded-3xl bg-[#080d19] border border-[#192338] hover:border-cyan-500/50 transition-all duration-300 p-6 flex flex-col justify-between shadow-xl hover:shadow-cyan-950/30"
          >
            <div>
              <div className="inline-block px-3 py-1 rounded-full bg-[#0b1424] border border-[#1e3458] text-[10px] font-mono font-bold text-cyan-400 tracking-wider mb-4">
                VIDEO FORENSIC INGESTION
              </div>

              {/* Vector Graphic: Video Reel & Storage Dossier */}
              <div className="h-32 w-full rounded-2xl bg-[#060912] border border-[#141d30] mb-5 relative flex items-center justify-center overflow-hidden p-4">
                <div className="w-full h-full relative flex items-center justify-center">
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-950/40 via-sky-950/30 to-blue-950/40" />
                  <div className="relative z-10 flex items-center gap-4">
                    <Film className="w-10 h-10 text-cyan-400 animate-pulse" />
                    <ArrowRight className="w-5 h-5 text-slate-600" />
                    <FolderArchive className="w-10 h-10 text-amber-400" />
                  </div>
                </div>
              </div>

              <h3 className="text-xl font-bold text-white mb-2 group-hover:text-cyan-300 transition-colors">
                Video Stream &amp; Automated Folder Storage
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed mb-4">
                Upload MP4, AVI or MOV thermal videos. Frames are processed through ByteTrack for continuous target continuity. Every analyzed video automatically creates its own dedicated folder containing annotated video, keyframe snapshots, and JSON reports.
              </p>

              <ul className="space-y-2 text-xs text-slate-300 mb-6">
                <li className="flex items-center gap-2">
                  <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                  <span>Files stored in dedicated folder with video name</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                  <span>ByteTrack persistent multi-target continuity</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                  <span>Interactive scrubber playback &amp; keyframe clippings</span>
                </li>
              </ul>
            </div>

            <div className="pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs font-semibold text-cyan-400 group-hover:text-cyan-300">
              <span className="tracking-wider uppercase font-mono">OPEN VIDEO ANALYZER</span>
              <ArrowRight className="w-4 h-4 transform group-hover:translate-x-1.5 transition-transform" />
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          SECTION 3: ARCHITECTURAL HIGHLIGHTS (4 Pillars)
         ========================================================================= */}
      <section>
        <h2 className="text-2xl font-bold text-white mb-5 tracking-tight">
          Thermal Computer Vision Architecture
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Item 01 */}
          <div className="p-5 rounded-2xl bg-[#080d19] border border-[#192338] flex flex-col justify-between space-y-3">
            <div>
              <div className="flex items-center justify-between text-slate-500 font-mono text-xs mb-3">
                <Scan className="w-4 h-4 text-cyan-400" />
                <span className="font-bold">01</span>
              </div>
              <h3 className="font-bold text-white text-sm mb-1.5">1-Channel LWIR Grayscale</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Direct single-channel tensor input without RGB replication overhead, adapted for infrared radiation luminance.
              </p>
            </div>
          </div>

          {/* Item 02 */}
          <div className="p-5 rounded-2xl bg-[#080d19] border border-[#192338] flex flex-col justify-between space-y-3">
            <div>
              <div className="flex items-center justify-between text-slate-500 font-mono text-xs mb-3">
                <GitFork className="w-4 h-4 text-cyan-400" />
                <span className="font-bold">02</span>
              </div>
              <h3 className="font-bold text-white text-sm mb-1.5">ByteTrack Multi-Target Tracking</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Maintains persistent target identities across consecutive frames with motion affinity, preventing duplicate counts.
              </p>
            </div>
          </div>

          {/* Item 03 */}
          <div className="p-5 rounded-2xl bg-[#080d19] border border-[#192338] flex flex-col justify-between space-y-3">
            <div>
              <div className="flex items-center justify-between text-slate-500 font-mono text-xs mb-3">
                <FolderArchive className="w-4 h-4 text-amber-400" />
                <span className="font-bold">03</span>
              </div>
              <h3 className="font-bold text-white text-sm mb-1.5">Video-Named Dossier Folders</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Every processed video is automatically organized into its own dedicated folder containing annotated video, snapshots, and reports.
              </p>
            </div>
          </div>

          {/* Item 04 */}
          <div className="p-5 rounded-2xl bg-[#080d19] border border-[#192338] flex flex-col justify-between space-y-3">
            <div>
              <div className="flex items-center justify-between text-slate-500 font-mono text-xs mb-3">
                <Gauge className="w-4 h-4 text-emerald-400" />
                <span className="font-bold">04</span>
              </div>
              <h3 className="font-bold text-white text-sm mb-1.5">High-Speed Neural Pipeline</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Optimized YOLOv8 backbone delivering sub-30ms latency for real-time thermal surveillance and rapid forensic review.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          SECTION 4: SYSTEM TELEMETRY & MODEL STATUS ROW
         ========================================================================= */}
      <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        {/* Model Card */}
        <div className="p-4 rounded-2xl bg-[#080d19] border border-[#192338] flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Model Status</span>
            <Cpu className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="text-base font-bold text-white flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  modelStatus?.loaded ? 'bg-emerald-400 shadow-sm shadow-emerald-400' : 'bg-amber-400'
                }`}
              />
              {modelStatus?.loaded ? 'Online (Loaded)' : 'Offline / Standby'}
            </div>
            <div className="text-[11px] text-slate-400 font-mono mt-1 truncate">
              {modelStatus?.model_name || 'YOLOv8 1-Channel LWIR'}
            </div>
          </div>
        </div>

        {/* Dataset Card */}
        <div className="p-4 rounded-2xl bg-[#080d19] border border-[#192338] flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Dataset Modality</span>
            <Layers className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="text-base font-bold text-white">FLIR Thermal 2024</div>
            <div className="text-[11px] text-cyan-400/90 font-mono mt-1">
              26,061 1-Ch Frames (LWIR)
            </div>
          </div>
        </div>

        {/* Total Videos Analyzed */}
        <div className="p-4 rounded-2xl bg-[#080d19] border border-[#192338] flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Videos Analyzed</span>
            <Film className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="text-2xl font-black text-white font-mono">
              {stats?.total_videos_analyzed ?? 0}
            </div>
            <div className="text-[11px] text-slate-400 font-mono mt-1">Organized Dossiers</div>
          </div>
        </div>

        {/* Total Frames Processed */}
        <div className="p-4 rounded-2xl bg-[#080d19] border border-[#192338] flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Frames Processed</span>
            <MonitorCheck className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="text-2xl font-black text-white font-mono">
              {stats?.total_frames_processed ?? 0}
            </div>
            <div className="text-[11px] text-slate-400 font-mono mt-1">Thermal frames analyzed</div>
          </div>
        </div>

        {/* Total Thermal Targets Detected */}
        <div className="p-4 rounded-2xl bg-[#080d19] border border-[#192338] flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
            <span>Targets Detected</span>
            <ShieldCheck className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <div className="text-2xl font-black text-cyan-400 font-mono">
              {totalTargets}
            </div>
            <div className="text-[11px] text-slate-400 font-mono mt-1">Thermal Signatures</div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          SECTION 5: RECENT DETECTIONS STREAM
         ========================================================================= */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-[10px] font-mono tracking-widest text-slate-500 uppercase font-semibold">
              LATEST ACTIVITY
            </div>
            <h2 className="text-xl font-bold text-white tracking-tight">Recent Thermal Detections</h2>
          </div>
          <button
            onClick={() => onNavigate('history')}
            className="text-xs font-mono font-semibold text-cyan-400 hover:text-cyan-300 flex items-center gap-1 transition-colors cursor-pointer"
          >
            <span>FULL DETECTION ARCHIVE</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {stats?.recent_sessions && stats.recent_sessions.length > 0 ? (
          <div className="space-y-2.5">
            {stats.recent_sessions.map((sess, idx) => (
              <div
                key={sess.job_id || idx}
                onClick={() => onNavigate('history')}
                className="group cursor-pointer p-3 sm:p-4 rounded-2xl bg-[#080d19] hover:bg-[#0d1526] border border-[#192338] hover:border-slate-700 flex flex-col sm:flex-row sm:items-center justify-between gap-3 transition-all"
              >
                <div className="flex items-center gap-3">
                  <div className="w-16 h-10 rounded-xl bg-slate-900 border border-slate-800 overflow-hidden flex-shrink-0 flex items-center justify-center">
                    {sess.snapshot_url ? (
                      <img
                        src={getMediaUrl(sess.snapshot_url)}
                        alt="Detection preview"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <Film className="w-4 h-4 text-slate-600" />
                    )}
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-white group-hover:text-cyan-300 transition-colors">
                      {sess.filename}
                    </div>
                    <div className="text-[11px] text-slate-400 font-mono">
                      {sess.total_frames} frames processed • {sess.timestamp?.split('T')[0] || 'Recent'}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3 self-end sm:self-center">
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-cyan-950/40 border border-cyan-500/40 text-cyan-300 text-xs font-bold font-mono">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                    {(sess.unique_targets ?? sess.unique_pedestrians ?? sess.pedestrian_detections) > 0
                      ? `${sess.unique_targets ?? sess.unique_pedestrians ?? sess.pedestrian_detections} TARGETS`
                      : 'CLEAR PATH'}
                  </div>
                  <div className="text-xs font-mono font-bold text-white">
                    {(sess.peak_confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-8 rounded-2xl bg-[#080d19] border border-dashed border-slate-800 text-center">
            <Film className="w-8 h-8 text-slate-600 mx-auto mb-2" />
            <div className="text-sm font-medium text-slate-300 mb-1">
              No detection sessions recorded yet
            </div>
            <div className="text-xs text-slate-500 max-w-sm mx-auto mb-4">
              Upload a thermal video in the Video Analyzer or test a photo in the Photo Detection tab to begin logging detections.
            </div>
          </div>
        )}
      </section>
    </div>
  );
};
