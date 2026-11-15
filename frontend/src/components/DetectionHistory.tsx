import React, { useState } from 'react';
import {
  ShieldAlert,
  Trash2,
  Film,
  Download,
  Eye,
  AlertTriangle,
  X,
  FolderOpen,
} from 'lucide-react';
import type { DetectionEvent } from '../types';
import { deleteHistoryItem, clearAllHistory, getMediaUrl } from '../services/api';

interface DetectionHistoryProps {
  events: DetectionEvent[];
  onRefresh: () => void;
}

export const DetectionHistory: React.FC<DetectionHistoryProps> = ({ events, onRefresh }) => {
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [selectedSnapshot, setSelectedSnapshot] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Group events by video filename or job_id
  const groupedEvents: { [key: string]: DetectionEvent[] } = {};
  for (const ev of events) {
    const key = ev.filename || 'Direct Feed Captures';
    if (!groupedEvents[key]) groupedEvents[key] = [];
    groupedEvents[key].push(ev);
  }

  const handleDeleteItem = async (id: string) => {
    try {
      await deleteHistoryItem(id);
      onRefresh();
    } catch (err) {
      console.error('Delete failed:', err);
    }
  };

  const handleClearAll = async () => {
    try {
      setIsDeleting(true);
      await clearAllHistory();
      setIsDeleting(false);
      setShowClearConfirm(false);
      onRefresh();
    } catch (err) {
      console.error('Clear failed:', err);
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-8 pb-16">
      {/* Top Banner (Matches Screenshots 4 & 5) */}
      <div className="p-6 rounded-2xl bg-[#0a0d18] border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-red-950/40 border border-red-500/30 flex items-center justify-center text-red-400 flex-shrink-0">
            <ShieldAlert className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight">
              Incident Evidence Vault
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Archived thermal target captures and forensic dossiers organized upload-wise by video session.
            </p>
          </div>
        </div>

        {events.length > 0 && (
          <button
            onClick={() => setShowClearConfirm(true)}
            className="self-start sm:self-center flex items-center gap-2 px-4 py-2 rounded-xl bg-red-950/40 hover:bg-red-900/50 border border-red-500/40 text-red-300 font-semibold text-xs transition-colors cursor-pointer"
          >
            <Trash2 className="w-4 h-4" />
            Clear Incident History
          </button>
        )}
      </div>

      {/* Confirmation Modal */}
      {showClearConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="max-w-md w-full p-6 rounded-2xl bg-[#0c101c] border border-red-500/40 shadow-2xl space-y-4">
            <div className="flex items-center gap-3 text-red-400">
              <AlertTriangle className="w-6 h-6 flex-shrink-0" />
              <h3 className="text-lg font-bold text-white">Clear Forensic Evidence Vault?</h3>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              This will permanently delete all {events.length} archived detection records, snapshots, and session dossiers from the backend database. This action cannot be undone.
            </p>
            <div className="flex items-center justify-end gap-3 pt-3">
              <button
                onClick={() => setShowClearConfirm(false)}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-300 transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={handleClearAll}
                disabled={isDeleting}
                className="px-4 py-2 rounded-xl bg-red-600 hover:bg-red-500 text-xs font-bold text-white transition-colors cursor-pointer"
              >
                {isDeleting ? 'Clearing...' : 'Confirm Clear'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Snapshot Modal View */}
      {selectedSnapshot && (
        <div
          onClick={() => setSelectedSnapshot(null)}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-md cursor-pointer"
        >
          <div className="relative max-w-4xl max-h-[85vh] rounded-2xl overflow-hidden border border-slate-700 bg-black">
            <button
              onClick={() => setSelectedSnapshot(null)}
              className="absolute top-3 right-3 p-2 rounded-full bg-black/70 text-white hover:text-cyan-400 transition-colors cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>
            <img
              src={getMediaUrl(selectedSnapshot)}
              alt="Enlarged forensic capture"
              className="w-full h-full object-contain"
            />
          </div>
        </div>
      )}

      {/* Sessions Groups */}
      {Object.keys(groupedEvents).length > 0 ? (
        Object.entries(groupedEvents).map(([filename, group]) => {
          const folderName = group[0]?.folder_name || filename.replace(/\.[^/.]+$/, '').replace(/[^\w\-_\. ]/g, '_').trim() || 'video_folder';
          return (
            <div key={filename} className="rounded-2xl bg-[#070b16] border border-cyan-500/30 overflow-hidden shadow-xl space-y-0 mb-6">
              {/* Folder Titlebar with Video Name */}
              <div className="p-4 sm:p-5 bg-gradient-to-r from-[#0b1424] via-[#080e1a] to-[#0c1626] border-b border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-3.5">
                  <div className="w-10 h-10 rounded-xl bg-cyan-950/80 border border-cyan-500/50 flex items-center justify-center text-cyan-400 shadow-md">
                    <FolderOpen className="w-5 h-5 text-cyan-300" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono uppercase font-bold text-cyan-400 tracking-wider">
                        VIDEO DOSSIER FOLDER
                      </span>
                      <span className="px-2 py-0.5 rounded text-[9px] font-mono font-bold bg-cyan-950 border border-cyan-500/40 text-cyan-300">
                        {group.length} FRAMES
                      </span>
                    </div>
                    <div className="text-sm sm:text-base font-black text-white font-mono mt-0.5 flex items-center gap-2">
                      <span>📁 outputs/{folderName}/</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <div className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-300">
                    Source: <span className="text-white font-bold">{filename}</span>
                  </div>
                  {group[0]?.processed_video_url && (
                    <a
                      href={getMediaUrl(group[0].processed_video_url)}
                      download={`${folderName}_annotated.mp4`}
                      className="px-3 py-1 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs font-mono transition-colors flex items-center gap-1.5 cursor-pointer shadow-md shadow-cyan-950/40"
                    >
                      <Film className="w-3.5 h-3.5" />
                      <span>Video</span>
                    </a>
                  )}
                </div>
              </div>

              {/* 3-Column Event Cards Grid */}
              <div className="p-5">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                  {group.map((ev) => (
                    <div
                      key={ev.id}
                      className="rounded-2xl bg-[#0a0d18] border border-slate-800 hover:border-slate-700 transition-all overflow-hidden flex flex-col justify-between shadow-xl"
                    >
                      {/* Snapshot Thumbnail with Overlays */}
                      <div className="relative aspect-video bg-[#06080e] overflow-hidden group">
                        {ev.snapshot_url ? (
                          <img
                            src={getMediaUrl(ev.snapshot_url)}
                            alt={`Detection frame #${ev.frame_number}`}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-slate-600">
                            <Film className="w-8 h-8" />
                          </div>
                        )}

                        {/* Frame Number Tag (Top Left) */}
                        <div className="absolute top-2.5 left-2.5 px-2.5 py-1 rounded-md bg-black/85 backdrop-blur-md border border-slate-800 text-[10px] font-mono font-bold text-cyan-300">
                          Frame #{ev.frame_number || '001'}
                        </div>

                        {/* Timestamp Tag (Top Right) */}
                        <div className="absolute top-2.5 right-2.5 px-2 py-0.5 rounded-md bg-black/85 backdrop-blur-md border border-slate-800 text-[9px] font-mono text-slate-300">
                          {ev.timestamp.split('T')[0] || '19-01-2026'}
                        </div>
                      </div>

                      {/* Card Content Footer */}
                      <div className="p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-1.5 text-cyan-400 font-mono text-xs font-black">
                            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                            THERMAL TARGET DETECTED
                          </div>
                          <div className="text-xs font-mono font-black text-white">
                            {(ev.confidence * 100).toFixed(1)}%
                          </div>
                        </div>

                        <div className="text-[11px] text-slate-500 font-mono">
                          {ev.timestamp.replace('T', ' ').substring(0, 19)}
                        </div>

                        {/* Action Buttons */}
                        <div className="pt-2 flex items-center gap-2 border-t border-slate-800/80">
                          <button
                            onClick={() => setSelectedSnapshot(ev.snapshot_url)}
                            className="flex-1 flex items-center justify-center gap-2 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs transition-colors shadow-md shadow-blue-950/40 cursor-pointer"
                          >
                            <Eye className="w-3.5 h-3.5" />
                            View Evidence
                          </button>

                          {ev.processed_video_url && (
                            <a
                              href={getMediaUrl(ev.processed_video_url)}
                              download={`detection_${ev.filename}`}
                              className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-cyan-400 transition-colors cursor-pointer"
                              title="Download Annotated Clip"
                            >
                              <Download className="w-4 h-4" />
                            </a>
                          )}

                          <button
                            onClick={() => handleDeleteItem(ev.id)}
                            className="p-2 rounded-xl bg-slate-900 hover:bg-red-950/50 border border-slate-800 hover:border-red-500/40 text-slate-400 hover:text-red-400 transition-colors cursor-pointer"
                            title="Delete Record"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          );
        })
      ) : (
        <div className="p-12 rounded-2xl bg-[#090d16] border border-dashed border-slate-800 text-center space-y-3">
          <ShieldAlert className="w-10 h-10 text-slate-600 mx-auto" />
          <div className="text-base font-bold text-slate-300">
            No Forensic Evidence Filed Yet
          </div>
          <div className="text-xs text-slate-500 max-w-sm mx-auto">
            When you run video detection or capture snapshots in the Live Monitor, detection events with confidence scores and frame clippings will be logged here automatically.
          </div>
        </div>
      )}
    </div>
  );
};
