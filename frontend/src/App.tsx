import { useState, useEffect, useCallback } from 'react';
import { Navbar } from './components/Navbar';
import { Overview } from './components/Overview';
import { PhotoDetector } from './components/PhotoDetector';
import { VideoAnalyzer } from './components/VideoAnalyzer';
import { DetectionHistory } from './components/DetectionHistory';
import { NLPCopilot } from './components/NLPCopilot';
import { fetchModelStatus, fetchOverviewStats, fetchHistory } from './services/api';
import type { ModelStatus, OverviewStats, DetectionEvent } from './types';
import { AlertCircle } from 'lucide-react';

export function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'photo' | 'video' | 'history'>('overview');
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [overviewStats, setOverviewStats] = useState<OverviewStats | null>(null);
  const [historyEvents, setHistoryEvents] = useState<DetectionEvent[]>([]);
  const [backendError, setBackendError] = useState<boolean>(false);

  const loadData = useCallback(async () => {
    try {
      const [mStatus, oStats, hEvents] = await Promise.all([
        fetchModelStatus().catch(() => null),
        fetchOverviewStats().catch(() => null),
        fetchHistory().catch(() => []),
      ]);

      if (mStatus) setModelStatus(mStatus);
      if (oStats) setOverviewStats(oStats);
      if (hEvents) setHistoryEvents(hEvents);
      setBackendError(!mStatus && !oStats);
    } catch {
      setBackendError(true);
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 8000);
    return () => clearInterval(interval);
  }, [loadData]);

  return (
    <div className="min-h-screen bg-[#06080e] text-slate-200 flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
      {/* Top Surveillance Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        modelStatus={modelStatus}
        historyCount={historyEvents.length}
      />

      {/* Backend connection warning banner if server is starting or offline */}
      {backendError && (
        <div className="bg-amber-950/60 border-b border-amber-500/40 px-4 py-2 text-xs font-mono text-amber-300 flex items-center justify-center gap-2">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
          <span>Backend service connecting on localhost:8000... Launching ThermalVision AI engine.</span>
        </div>
      )}

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 lg:px-8 pt-8">
        {activeTab === 'overview' && (
          <Overview stats={overviewStats} onNavigate={(tab) => setActiveTab(tab)} />
        )}
        {activeTab === 'photo' && <PhotoDetector />}
        {activeTab === 'video' && <VideoAnalyzer />}
        {activeTab === 'history' && (
          <DetectionHistory events={historyEvents} onRefresh={loadData} />
        )}
      </main>

      {/* Floating NLP AI Thermal Copilot */}
      <NLPCopilot />
    </div>
  );
}

export default App;
