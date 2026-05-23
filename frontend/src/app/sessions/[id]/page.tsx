"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import { useParams, useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import { api } from "@/lib/api";
import { useWebSocket } from "@/hooks/useWebSocket";
import SyncedPlayer from "@/components/SyncedPlayer";
import { useToast } from "@/components/Toast";

const fetcher = (id: string) => api.sessions.get(id);
const syncReportFetcher = (id: string) => api.sessions.syncReport(id);

function formatSyncMethod(method?: string | null) {
  switch (method) {
    case "audio":
      return "Audio Cross-Correlation";
    case "feature_based":
    case "feature":
      return "MultiVidSynch";
    case "multividsynch":
    case "multividsync":
    case "multisyncvideo":
      return "MultiSyncVideo";
    case "visual_hybrid":
    case "hybrid":
      return "Hybrid Visual";
    case "visual_consensus":
      return "Hybrid Visual Consensus";
    case "auto_coarse_to_fine":
      return "Auto Coarse-to-Fine";
    case "sesyn_net":
    case "sesyn":
      return "SeSyn-Net";
    case "auto":
      return "Auto";
    default:
      return method || "Pending";
  }
}

function formatSeconds(value?: number) {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}s`;
}

function formatScore(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return value.toFixed(4);
}

export default function SessionDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const router = useRouter();
  const { addToast } = useToast();

  const { data: session, error, isLoading, mutate } = useSWR(id ? `session-${id}` : null, () => fetcher(id));
  const { data: syncReport, mutate: mutateSyncReport } = useSWR(id ? `sync-report-${id}` : null, () => syncReportFetcher(id));
  const { events, connected } = useWebSocket(id);

  const [currentVideoUrl, setCurrentVideoUrl] = useState<string | null>(null);
  const [progressStage, setProgressStage] = useState<string>("queued");
  const [progressMessage, setProgressMessage] = useState<string>("Waiting for backend sync to begin...");
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (!events || events.length === 0) return;

    const progressOrder = [
      "queued",
      "master_started",
      "concatenating",
      "computing_offsets",
      "aligning",
      "master_done",
      "master_error",
    ];

    let latestStage = "queued";
    let latestMessage = "Waiting for backend sync to begin...";
    let latestUrl: string | null = null;

    events.forEach((ev) => {
      switch (ev.type) {
        case "master_error":
          latestStage = "master_error";
          latestMessage = ev.message || "Full sync failed.";
          break;
        case "sync_warning":
          latestStage = "computing_offsets";
          latestMessage = ev.message || "Sync warning: zero offsets were produced.";
          break;
        case "master_done":
          latestStage = "master_done";
          latestMessage = ev.message || "Master video is ready.";
          latestUrl = ev.url || latestUrl;
          break;
        case "aligning":
          latestStage = "aligning";
          latestMessage = ev.message || "Trimming and aligning video streams...";
          break;
        case "computing_offsets":
          latestStage = "computing_offsets";
          latestMessage = ev.message || "Computing offsets...";
          break;
        case "concatenating":
          latestStage = "concatenating";
          latestMessage = ev.message || "Concatenating source videos...";
          break;
        case "master_started":
        case "processing_started":
          latestStage = "master_started";
          latestMessage = ev.message || "Building final synced video...";
          break;
        default:
          break;
      }
    });

    setProgressStage(latestStage);
    setProgressMessage(latestMessage);

    if (!currentVideoUrl && latestUrl) {
      setCurrentVideoUrl(latestUrl);
    }

    const lastEvent = events[events.length - 1];
    if (!lastEvent) return;

    if (lastEvent.type === "master_error") {
      addToast({ type: "error", title: "Sync Error", message: lastEvent.message || "Pipeline encountered an error." });
    } else if (lastEvent.type === "sync_warning") {
      addToast({ type: "info", title: "Sync Warning", message: lastEvent.message || "Sync produced zero offsets." });
    } else if (lastEvent.type === "master_done") {
      addToast({ type: "success", title: "Sync Completed", message: "Final master video is ready." });
      mutateSyncReport();
      mutate();
    } else if (lastEvent.type === "master_started") {
      addToast({ type: "info", title: "Sync Started", message: "Final master sync has started." });
    }
  }, [events, currentVideoUrl, addToast, mutateSyncReport, mutate]);

  useEffect(() => {
    if (session?.master_url && !currentVideoUrl) {
      setCurrentVideoUrl(session.master_url);
      setProgressStage("master_done");
      setProgressMessage("Final master video is ready.");
    }
  }, [session, currentVideoUrl]);

  const handleDelete = async () => {
    if (!session) return;
    if (!confirm(`Permanently delete "${session.name}"? All raw and synced files will be lost.`)) return;
    
    setDeleting(true);
    try {
      await api.sessions.delete(session.id);
      addToast({ type: "success", title: "Deleted", message: "Session removed successfully." });
      router.push("/sessions");
    } catch (err: any) {
      addToast({ type: "error", title: "Failed to delete", message: err.message });
      setDeleting(false);
    }
  };

  if (isLoading) {
    return (
      <>
        <Navbar />
        <main className="container" style={{ paddingTop: 60 }}>
          <div className="skeleton" style={{ height: 100, marginBottom: 32 }} />
          <div className="grid-2" style={{ gridTemplateColumns: "1fr 340px", gap: 32 }}>
            <div className="skeleton" style={{ height: 500 }} />
            <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
              <div className="skeleton" style={{ height: 200 }} />
              <div className="skeleton" style={{ height: 300 }} />
            </div>
          </div>
        </main>
      </>
    );
  }

  if (error || !session) {
    return (
      <>
        <Navbar />
        <main className="container" style={{ paddingTop: 100, textAlign: "center" }}>
          <div className="card badge-failed fade-in-up" style={{ padding: "40px", maxWidth: 400, margin: "0 auto", background: "rgba(239, 68, 68, 0.1)" }}>
            <div style={{ fontSize: 40, marginBottom: 16 }}>⚠️</div>
            <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 8 }}>Session Not Found</h2>
            <p style={{ color: "var(--text-secondary)", marginBottom: 24 }}>The session may have been deleted or the server is unreachable.</p>
            <button className="btn btn-ghost" onClick={() => router.push("/sessions")}>
              ← Back to Sessions
            </button>
          </div>
        </main>
      </>
    );
  }

  return (
    <>
      <Navbar />
      <main className="container" style={{ paddingTop: 40, paddingBottom: 80 }}>
        {/* Header Section */}
        <div className="fade-in-up" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", borderBottom: "1px solid var(--border)", paddingBottom: 24, marginBottom: 32 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
              <button 
                className="btn btn-ghost" 
                style={{ padding: "6px", width: 32, height: 32, borderRadius: "50%" }}
                onClick={() => router.push("/sessions")}
                title="Back to Sessions"
              >
                ←
              </button>
              <h1 className="page-title" style={{ fontSize: 28, margin: 0 }}>{session.name}</h1>
              <span className={`badge badge-${session.status}`} style={{ margin: 0, padding: "2px 8px", fontSize: 11 }}>
                {session.status}
              </span>
            </div>
            <p className="mono" style={{ fontSize: 13, color: "var(--text-muted)", margin: "8px 0 0 44px" }}>
              ID: {session.id} • {session.camera_count} CAMERAS CONFIGURED
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div 
              className={`badge ${connected ? "badge-completed" : "badge-failed"}`} 
              style={{ fontSize: 11, padding: "6px 12px", background: connected ? "rgba(16, 185, 129, 0.1)" : "rgba(239, 68, 68, 0.1)" }}
            >
              <span className={`badge-dot ${connected ? "pulse" : ""}`} />
              {connected ? "LIVE FEED ACTIVE" : "DISCONNECTED (RETRYING)"}
            </div>
            <button 
              className="btn btn-danger" 
              onClick={handleDelete}
              disabled={deleting}
              style={{ padding: "8px 16px", fontSize: 13 }}
            >
              {deleting ? "Deleting..." : "🗑️ Delete Session"}
            </button>
          </div>
        </div>

        <div className="grid-2 fade-in-up stagger-1" style={{ gridTemplateColumns: "1fr 340px", gap: 32 }}>
          {/* Left Column: Player & History */}
          <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
            <SyncedPlayer
              url={currentVideoUrl}
              title={currentVideoUrl ? "Final Master Video" : "Final Sync Monitor"}
            />

            <div className="card">
              <div className="card-header" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <h3 className="card-title"><span style={{ fontSize: 20, marginRight: 8 }}>🚧</span> Backend Sync Progress</h3>
                <span className={`badge badge-${session.status}`} style={{ background: "transparent", border: "none" }}>
                  {session.status.toUpperCase()}
                </span>
              </div>

              <div style={{ padding: "24px 0 0" }}>
                <div style={{ display: "grid", gap: 16 }}>
                  {[
                    { key: "master_started", label: "Begin full sync" },
                    { key: "concatenating", label: "Concatenate source videos" },
                    { key: "computing_offsets", label: "Compute camera offsets" },
                    { key: "aligning", label: "Align and trim streams" },
                    { key: "master_done", label: "Final master ready" },
                  ].map((step, index) => {
                    const stageOrder = ["queued", "master_started", "concatenating", "computing_offsets", "aligning", "master_done", "master_error"];
                    const currentIndex = stageOrder.indexOf(progressStage);
                    const stepIndex = stageOrder.indexOf(step.key);
                    const isCompleted = currentIndex > stepIndex || progressStage === "master_done";
                    const isCurrent = progressStage === step.key;
                    const isErrored = progressStage === "master_error" && step.key === "master_done";

                    return (
                      <div key={step.key} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
                          <span style={{ fontWeight: 700, color: isCurrent ? "var(--text-primary)" : "var(--text-secondary)" }}>{step.label}</span>
                          <span style={{ fontSize: 12, color: isCompleted ? "var(--accent-green)" : isCurrent ? "var(--accent-amber)" : "var(--text-muted)" }}>
                            {isCompleted ? "Completed" : isCurrent ? "In progress" : "Pending"}
                          </span>
                        </div>
                        <div style={{ width: "100%", height: 10, background: "rgba(255,255,255,0.1)", borderRadius: 999 }}>
                          <div style={{ width: isCompleted ? "100%" : isCurrent ? "50%" : "0%", height: "100%", transition: "width 0.3s ease", background: isErrored ? "var(--accent-red)" : "var(--accent-blue)" }} />
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div style={{ marginTop: 24, padding: 16, borderRadius: 12, background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)" }}>
                  <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>Current Step</div>
                  <div style={{ fontSize: 14, color: "var(--text-secondary)" }}>{progressMessage}</div>
                  {currentVideoUrl && (
                    <div style={{ marginTop: 12 }}>
                      <button className="btn btn-primary" style={{ padding: "10px 16px", fontSize: 13 }} onClick={() => window.open(currentVideoUrl, "_blank")}>Watch final video</button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Metadata & Offsets */}
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            <div className="card">
              <div className="card-header" style={{ marginBottom: 12, borderBottom: "1px solid var(--border)", paddingBottom: 16 }}>
                <h3 className="card-title">Sync Method</h3>
              </div>
              <div style={{ display: "grid", gap: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Requested</span>
                  <span className="mono" style={{ color: "var(--text-primary)", fontSize: 13, textAlign: "right" }}>
                    {formatSyncMethod(syncReport?.requested_strategy || session.sync_strategy)}
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
                  <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Actually used</span>
                  <span className="mono" style={{ color: "var(--accent-cyan)", fontSize: 13, textAlign: "right", fontWeight: 700 }}>
                    {formatSyncMethod(syncReport?.selected_method)}
                  </span>
                </div>
                {syncReport?.input_mode && (
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
                    <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Input type</span>
                    <span className="mono" style={{ color: "var(--text-primary)", fontSize: 13, textAlign: "right" }}>
                      {syncReport.input_mode === "chunks" ? "Recorded chunks" : "Uploaded videos"}
                    </span>
                  </div>
                )}
                {syncReport?.anchor_video && (
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 16 }}>
                    <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>Anchor</span>
                    <span className="mono" style={{ color: "var(--text-primary)", fontSize: 13, textAlign: "right" }}>
                      {syncReport.anchor_video}
                    </span>
                  </div>
                )}
                {syncReport?.strategy_details?.selection_reason && (
                  <div style={{ padding: 12, borderRadius: "var(--radius-sm)", background: "rgba(6, 182, 212, 0.08)", border: "1px solid rgba(6, 182, 212, 0.25)", fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                    {syncReport.strategy_details.selection_confidence && (
                      <div className="mono" style={{ color: syncReport.strategy_details.selection_confidence === "low" ? "var(--accent-amber)" : "var(--accent-cyan)", fontWeight: 700, marginBottom: 4 }}>
                        Confidence: {syncReport.strategy_details.selection_confidence}
                      </div>
                    )}
                    {syncReport.strategy_details.selection_reason}
                  </div>
                )}
                {syncReport?.strategy_details?.pipeline_stages && syncReport.strategy_details.pipeline_stages.length > 0 && (
                  <div style={{ display: "grid", gap: 8, paddingTop: 4 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text-secondary)" }}>Pipeline stages</div>
                    {syncReport.strategy_details.pipeline_stages.map((stage, index) => (
                      <div key={`${stage}-${index}`} className="mono" style={{ fontSize: 12, color: "var(--text-primary)" }}>
                        {index + 1}. {stage}
                      </div>
                    ))}
                  </div>
                )}
                {syncReport?.strategy_details?.coarse_offsets && Object.keys(syncReport.strategy_details.coarse_offsets).length > 0 && (
                  <div style={{ display: "grid", gap: 8, paddingTop: 4 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text-secondary)" }}>Coarse offsets</div>
                    {Object.entries(syncReport.strategy_details.coarse_offsets).map(([camId, offset]) => (
                      <div key={`coarse-${camId}`} style={{ display: "flex", justifyContent: "space-between", gap: 16, fontSize: 12 }}>
                        <span className="mono" style={{ color: "var(--text-muted)" }}>{camId}</span>
                        <span className="mono" style={{ color: "var(--text-primary)" }}>{formatSeconds(offset)}</span>
                      </div>
                    ))}
                  </div>
                )}
                {syncReport?.strategy_details?.fine_residual_offsets && Object.keys(syncReport.strategy_details.fine_residual_offsets).length > 0 && (
                  <div style={{ display: "grid", gap: 8, paddingTop: 4 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text-secondary)" }}>SeSyn fine residual</div>
                    {Object.entries(syncReport.strategy_details.fine_residual_offsets).map(([camId, offset]) => (
                      <div key={`fine-${camId}`} style={{ display: "flex", justifyContent: "space-between", gap: 16, fontSize: 12 }}>
                        <span className="mono" style={{ color: "var(--text-muted)" }}>{camId}</span>
                        <span className="mono" style={{ color: "var(--text-primary)" }}>{formatSeconds(offset)}</span>
                      </div>
                    ))}
                  </div>
                )}
                {syncReport?.strategy_details?.candidates && Object.keys(syncReport.strategy_details.candidates).length > 0 && (
                  <div style={{ display: "grid", gap: 10, paddingTop: 4 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text-secondary)" }}>Method candidates</div>
                    {Object.entries(syncReport.strategy_details.candidates).map(([method, candidate]) => (
                      <div key={method} style={{ display: "grid", gap: 6, padding: 10, borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "rgba(255,255,255,0.02)" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, fontSize: 12 }}>
                          <span className="mono" style={{ color: "var(--text-primary)", fontWeight: 700 }}>{formatSyncMethod(method)}</span>
                          <span className="mono" style={{ color: "var(--text-muted)" }}>
                            score {formatScore(syncReport.strategy_details?.candidate_scores?.[method])}
                          </span>
                        </div>
                        {Object.entries(candidate).map(([camId, offset]) => (
                          <div key={`${method}-${camId}`} style={{ display: "flex", justifyContent: "space-between", gap: 12, fontSize: 12 }}>
                            <span className="mono" style={{ color: "var(--text-muted)" }}>{camId}</span>
                            <span className="mono" style={{ color: "var(--text-primary)" }}>{formatSeconds(offset)}</span>
                          </div>
                        ))}
                      </div>
                    ))}
                    {typeof syncReport.strategy_details.max_disagreement_seconds === "number" && (
                      <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                        Max disagreement: {formatSeconds(syncReport.strategy_details.max_disagreement_seconds)}
                      </div>
                    )}
                    {typeof syncReport.strategy_details.score_margin === "number" && (
                      <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                        Score margin: {formatScore(syncReport.strategy_details.score_margin)}
                      </div>
                    )}
                  </div>
                )}
                {syncReport?.duration_hints && Object.keys(syncReport.duration_hints).length > 0 && (
                  <div style={{ padding: 12, borderRadius: "var(--radius-sm)", background: "rgba(245, 158, 11, 0.08)", border: "1px solid rgba(245, 158, 11, 0.25)" }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "var(--accent-amber)", marginBottom: 8 }}>Duration correction applied</div>
                    {Object.entries(syncReport.duration_hints).map(([camId, hint]) => (
                      <div key={camId} style={{ display: "grid", gap: 4, fontSize: 12, color: "var(--text-secondary)" }}>
                        <div className="mono" style={{ color: "var(--text-primary)" }}>{camId}</div>
                        <div>Raw: {formatSeconds(hint.original_offset)} → Final: {formatSeconds(hint.adjusted_offset)}</div>
                      </div>
                    ))}
                  </div>
                )}
                {syncReport?.final_offsets && Object.keys(syncReport.final_offsets).length > 0 && (
                  <div style={{ display: "grid", gap: 8, paddingTop: 4 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text-secondary)" }}>Final offsets</div>
                    {Object.entries(syncReport.final_offsets).map(([camId, offset]) => (
                      <div key={camId} style={{ display: "flex", justifyContent: "space-between", gap: 16, fontSize: 12 }}>
                        <span className="mono" style={{ color: "var(--text-muted)" }}>{camId}</span>
                        <span className="mono" style={{ color: "var(--text-primary)" }}>
                          {formatSeconds(offset)}
                          {typeof syncReport.frame_offsets?.[camId] === "number" ? ` (${syncReport.frame_offsets[camId]}f)` : ""}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
                {!syncReport && (
                  <div style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.5 }}>
                    Method details will appear after the next full sync finishes.
                  </div>
                )}
              </div>
            </div>

            {/* Event Feed */}
            <div className="card" style={{ flexGrow: 1, display: "flex", flexDirection: "column" }}>
              <div className="card-header" style={{ marginBottom: 12, borderBottom: "1px solid var(--border)", paddingBottom: 16 }}>
                <h3 className="card-title">🚀 Real-Time Logs</h3>
              </div>
              <div style={{ flexGrow: 1, maxHeight: 380, overflowY: "auto", display: "flex", flexDirection: "column", gap: 8, paddingRight: 8 }}>
                {events.slice().reverse().map((ev, i) => {
                  let accent = "var(--border)";
                  if (ev.type === "master_done" || ev.type === "chunk_done") accent = "var(--accent-green)";
                  if (ev.type === "master_error" || ev.type === "error" || ev.type === "sync_warning") accent = "var(--accent-red)";
                  if (ev.type === "master_started" || ev.type === "processing_started" || ev.type === "concatenating" || ev.type === "computing_offsets" || ev.type === "aligning") accent = "var(--accent-amber)";
                  if (ev.type === "chunk_uploaded") accent = "var(--accent-blue)";
                  if (ev.type === "processing_started") accent = "var(--accent-amber)";
                  if (ev.type === "chunk_uploaded") accent = "var(--accent-blue)";

                  return (
                    <div key={i} className="fade-in-up" style={{ padding: "10px 12px", background: "rgba(0,0,0,0.2)", borderRadius: "var(--radius-sm)", borderLeft: `3px solid ${accent}`, fontSize: 12 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                        <span className="mono" style={{ fontWeight: 700, color: "var(--text-primary)" }}>{ev.type.toUpperCase()}</span>
                        <span style={{ color: "var(--text-muted)", fontSize: 10 }}>Just now</span>
                      </div>
                      {ev.message && <div style={{ color: "var(--text-secondary)", marginTop: 4, lineHeight: 1.4 }}>{ev.message}</div>}
                      <div style={{ display: "flex", gap: 12, marginTop: 6 }}>
                        {ev.cam_id && <span className="badge" style={{ fontSize: 9, padding: "2px 6px", background: "rgba(255,255,255,0.05)", border: "1px solid var(--border)" }}>CAM: {ev.cam_id}</span>}
                        {ev.chunk_index !== undefined && <span className="badge" style={{ fontSize: 9, padding: "2px 6px", background: "rgba(255,255,255,0.05)", border: "1px solid var(--border)" }}>CHUNK: {ev.chunk_index}</span>}
                      </div>
                    </div>
                  );
                })}
                {events.length === 0 && (
                  <div style={{ fontSize: 13, color: "var(--text-muted)", textAlign: "center", margin: "auto", padding: 20 }}>
                    Listening for telemetry on <span className="mono" style={{ color: "var(--accent-cyan)" }}>ws://</span>...
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
