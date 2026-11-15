# ThermalVision AI – Infrared Object & Target Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-cyan.svg)](https://react.dev)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-purple.svg)](https://tailwindcss.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-1--Channel%20LWIR-orange.svg)](https://github.com/ultralytics/ultralytics)

ThermalVision AI is a high-performance computer vision and operational intelligence platform engineered specifically for **1-Channel Long-Wave Infrared (LWIR 8–14 μm) Thermal Imaging**. Trained on the **FLIR Thermal Dataset (26,061 annotated frames)**, the system replaces traditional RGB assumptions with native single-channel luminance tensor operations, ByteTrack multi-target spatial-temporal continuity, automated dossier folder archiving, and an integrated **Tactical NLP Intelligence Copilot**.

---

## 🚀 Live Application URL

- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Interactive OpenAPI Specification**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## 🌟 Key Capabilities

### 1. Cyber Telemetry Dashboard (`Overview.tsx`)
- **System Telemetry & Health**: Live model architecture indicators, hardware acceleration status (CPU / CUDA), and FLIR baseline dataset parameters.
- **Dynamic Surveillance Counters**: Session analytics tracking total videos analyzed, cumulative unique targets detected, and total processed frames.
- **Core Architecture Pillars**: Real-time breakdown of single-channel tensor extraction (`Conv2d(1, 16, 3, 2, 1)`), multi-scale anchor grids, and forensic evidentiary logging.
- **Activity Feed**: Interactive stream displaying recent thermal target encounters with timestamps and confidence scores.

### 2. Photo Thermal Detection (`PhotoDetector.tsx`)
- **Single Capture Ingestion**: Drag-and-drop or browse thermal captures (JPG, PNG, WEBP, BMP).
- **FLIR Sample Preset Gallery**: Preloaded 1-click test gallery with authentic FLIR thermal captures.
- **Interactive Confidence Threshold**: Dynamic sensitivity slider (10% to 85%) for fine-tuning detection thresholds.
- **Thermal False-Color Palettes**: Real-time rendering in **Standard Thermal**, **Ironbow / Heatmap**, or **High-Contrast Invert**.
- **Bounding Box Overlay**: Cyber-neon target brackets displaying target index, normalized coordinates, and confidence ratings.
- **Direct Export & Vault Storage**: Save detections directly to the Evidence History vault, download annotated images, or export raw JSON detection manifests.
- **AI Narrative Synthesis**: One-click generation of tactical natural language incident narratives from photo thermal telemetry.

### 3. Video Forensic Analyzer & Per-Video Folder Storage (`VideoAnalyzer.tsx`)
- **Thermal Video Ingestion**: Asynchronous multi-format video processing (MP4, AVI, MOV).
- **Multi-Target Tracking**: ByteTrack integration linking bounding boxes across frames to prevent duplicate target counts.
- **Automated Video-Named Folder Organization**: Every processed video is archived in a dedicated folder under `outputs/<video_name>/`:
  - `outputs/<video_name>/<video_name>_annotated.mp4` – Full-fidelity video with bounding boxes and target tracking IDs.
  - `outputs/<video_name>/snapshots/` – Peak detection frame captures and keyframe clippings.
  - `outputs/<video_name>/detection_report.json` – Comprehensive machine-readable forensic dossier.
  - `outputs/<video_name>/source_<video_name>.mp4` – Backup copy of original uncompressed input.
- **Interactive Video Player**: Integrated scrubber bar, playback rate controls, evidentiary snapshots, and frame stepper.
- **Tactical Dossier Generation**: Instant generation of AI NLP narrative operational reports directly from video analysis data.

### 4. Built-in Tactical NLP Engine (`nlp_engine.py` & `NLPCopilot.tsx`)
- **Floating AI Copilot Assistant**: Dedicated cyber-styled conversational interface in the bottom-right corner for conversational queries.
- **Domain Intelligence**: Pre-trained operational knowledge of LWIR 8–14 μm thermal physics, 1-channel tensor Conv2d adaptation, FLIR thermal dataset properties, and ByteTrack continuity.
- **Session-Aware Semantic Queries**: Ask queries such as:
  - *"Summarize recent thermal targets detected."*
  - *"How does 1-channel thermal processing work without RGB channels?"*
  - *"Explain the difference between Ironbow and White-Hot thermal palettes."*
  - *"How does ByteTrack resolve occlusion in night-vision scenarios?"*
- **Automated Tactical Narrative Reports**: Automatically converts raw bounding box coordinates, thermal activity indices, and target counts into structured operational briefing documents:
  - Executive Tactical Summary
  - Threat Assessment Level (Elevated, Moderate, Minimal)
  - Target Spatial & Behavioral Distribution Narrative
  - Recommended Operational Directives

### 5. Incident Evidence Vault (`DetectionHistory.tsx`)
- **Chronological Audit Trail**: Visual cards grouping detections by timestamp, source file, target count, and peak confidence.
- **High-Resolution Forensic Inspection**: Fullscreen modal for deep inspection of peak evidentiary frames.
- **Granular Management**: Delete individual records, re-download annotated outputs, or wipe the vault with modal confirmation.

---

## 🏛️ System Architecture

```
thermalvision-ai/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application server, CORS, routing & static mounts
│   │   ├── model_loader.py      # 1-Channel YOLOv8 inference wrapper with channel adaptation
│   │   ├── video_processor.py   # Asynchronous video pipeline with ByteTrack & folder isolation
│   │   ├── nlp_engine.py        # Tactical NLP engine for conversational copilot & report generation
│   │   └── database.py          # SQLite database schema, migrations, and CRUD operations
│   └── data/
│       └── surveillance.db      # SQLite database storing detection history & events
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.tsx       # Top navigation header with operational status
│   │   │   ├── Overview.tsx     # Futuristic Home dashboard & system telemetry HUD
│   │   │   ├── PhotoDetector.tsx# Dedicated photo thermal detection with palette filters
│   │   │   ├── VideoAnalyzer.tsx# Video ingestion, dossier folder viewer & analysis HUD
│   │   │   ├── DetectionHistory.tsx # Chronological evidence vault & snapshot viewer
│   │   │   └── NLPCopilot.tsx   # Floating interactive AI assistant & quick prompt menu
│   │   ├── services/api.ts      # Typed REST API client for backend communication
│   │   ├── types/index.ts       # Shared TypeScript schemas & interfaces
│   │   ├── App.tsx              # Root application router and layout wrapper
│   │   └── index.css            # Tailwind CSS styling and cyber-neon visual theme
│   ├── public/samples/          # Preloaded authentic FLIR thermal sample captures
│   ├── dist/                    # Compiled production assets served by FastAPI
│   └── vite.config.ts           # Vite bundler configuration & local API proxy
├── models/
│   └── best.pt                  # Pre-trained 1-channel YOLOv8 thermal checkpoint
├── uploads/                     # Temporary staging for incoming media files
├── outputs/                     # Dedicated folders per video: outputs/<video_name>/
│   └── <video_name>/
│       ├── <video_name>_annotated.mp4
│       ├── detection_report.json
│       ├── source_<video_name>.mp4
│       └── snapshots/
└── snapshots/                   # Captured keyframe evidentiary images
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | `GET` | Serves the compiled production React application |
| `GET /api/stats` | `GET` | Retrieve session telemetry (total targets, videos analyzed, frames) |
| `POST /api/detect/photo` | `POST` | Execute 1-channel thermal target detection on an uploaded image |
| `POST /api/detect/video` | `POST` | Upload video and queue asynchronous multi-target tracking job |
| `GET /api/detect/video/{job_id}/status` | `GET` | Poll progress, ETA, and target count for active video jobs |
| `GET /api/detect/video/{job_id}/folder` | `GET` | List all archived files within the video's dedicated output dossier |
| `GET /api/history` | `GET` | Retrieve all historical photo and video detection events |
| `DELETE /api/history/{id}` | `DELETE` | Delete a specific detection record from the forensic vault |
| `DELETE /api/history` | `DELETE` | Purge all detection records from the forensic vault |
| `GET /api/samples` | `GET` | List available preloaded FLIR sample images |
| `POST /api/nlp/chat` | `POST` | Query the conversational AI Tactical Copilot |
| `POST /api/nlp/generate-report` | `POST` | Generate an AI narrative tactical incident report from telemetry |

---

## ⚡ Quickstart Guide

### Prerequisites
- **Python**: Version 3.10+ (tested with Python 3.13)
- **Node.js**: Version 18+ (for building the frontend)

### 1. Unified Launch (Recommended)
Run the FastAPI production server (automatically serves the compiled React interface at root `/`):

```powershell
# Navigate into the project folder
cd thermalvision-ai-main

# Start the server
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open your browser at: **[http://localhost:8000](http://localhost:8000)**

### 2. Development Setup (Optional)
If you wish to make live modifications with Hot-Module Replacement (HMR):

**Terminal 1 (Backend API):**
```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (React Vite Dev Server):**
```powershell
cd frontend
npm run dev
```
Access the development environment at **`http://localhost:5173`**.

---

## 🛡️ Operational Directives & Testing

1. **Photo Detection**:
   - Navigate to the **Photo Detection** tab.
   - Click any of the **FLIR Thermal Sample Presets** or upload your own infrared capture.
   - Adjust the confidence slider, toggle between **Normal / Ironbow / Invert** palettes, and click **"Generate AI Incident Narrative (NLP)"** to see automatic tactical synthesis.

2. **Video Analysis & Dossier Storage**:
   - Navigate to the **Video Analysis** tab.
   - Upload any thermal MP4/AVI/MOV video or test clip.
   - Monitor real-time frame progress. Once complete, inspect the organized dossier folder at `outputs/<video_name>/`.
   - Click **"Generate AI Tactical Report (NLP)"** for an instant narrative report.

3. **NLP Tactical Copilot**:
   - Click the floating **AI Tactical Copilot** button at the bottom-right of any screen.
   - Select quick prompts or ask domain questions about infrared physics, target tracking, or current session statistics.
