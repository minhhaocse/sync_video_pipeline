"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";

type MasterStatus = "idle" | "pending" | "processing" | "completed" | "failed";

export default function LivePage() {
  const router = useRouter();
  const [cameras, setCameras] = useState<{ id: string; image: string; lastSeen: number }[]>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);
  const [syncStrategy, setSyncStrategy] = useState("auto");
  const [finalizeError, setFinalizeError] = useState<string | null>(null);

  // Phase-1 preview chunks
  const [previewChunks, setPreviewChunks] = useState<{ index: number; url: string }[]>([]);

  // Phase-2 master video
  const [masterStatus, setMasterStatus] = useState<MasterStatus>("idle");
  const [masterUrl, setMasterUrl] = useState<string | null>(null);
  const [masterError, setMasterError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const masterPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Stable ref so camera IDs are never lost when the watchdog clears them from state
  const cameraIdsRef = useRef<Set<string>>(new Set());
  // Stable ref so sessionId is always readable inside async callbacks
  const sessionIdRef = useRef<string | null>(null);

  // Poll master status until completed or failed
  const startMasterPolling = useCallback((sid: string) => {
    if (masterPollRef.current) clearInterval(masterPollRef.current);
    setMasterStatus("processing");

    masterPollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/live/master-status/${sid}`);
        if (!res.ok) return;
        const data = await res.json();
        setMasterStatus(data.status as MasterStatus);
        if (data.status === "completed") {
          setMasterUrl(data.url);
          clearInterval(masterPollRef.current!);
        } else if (data.status === "failed") {
          setMasterError(data.error || "Master render failed.");
          clearInterval(masterPollRef.current!);
        }
      } catch { /* network hiccup — keep polling */ }
    }, 4000); // Poll every 4 seconds
  }, []);

  useEffect(() => {
    const connect = () => {
      const protocol = window.location.protocol === "https:" ? "wss://" : "ws://";
      const wsHost = window.location.host;
      const ws = new WebSocket(`${protocol}${wsHost}/ws/dashboard`);
      wsRef.current = ws;

      ws.onopen = () => setWsConnected(true);
      ws.onclose = () => {
        setWsConnected(false);
        setTimeout(connect, 3000); // auto-reconnect
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "preview") {
            setCameras((prev) => {
              const now = Date.now();
              const existing = prev.find((c) => c.id === data.id);
              // Always track camera in ref so it survives the watchdog
              cameraIdsRef.current.add(data.id);
              if (existing) {
                return prev.map((c) =>
                  c.id === data.id ? { ...c, image: data.image, lastSeen: now } : c
                );
              } else {
                return [...prev, { id: data.id, image: data.image, lastSeen: now }];
              }
            });
          } else if (data.type === "disconnect") {
            setCameras((prev) => prev.filter((c) => c.id !== data.id));
            // Do NOT remove from cameraIdsRef — we still want to process their chunks
          } else if (data.type === "info") {
            if (data.message === "ESP32_STARTED" || data.message === "STARTED") {
              setIsRecording(true);
              if (data.session_id) setSessionId(data.session_id);
            } else if (data.message === "STOPPED") {
              setIsRecording(false);
            }
          } else if (data.type === "status") {
            if (data.command === "stop" && data.captured_cameras) {
              const captured = data.captured_cameras as string[];
              setCameras((prev) => {
                const updated = [...prev];
                captured.forEach(id => {
                  if (!updated.find(c => c.id === id)) {
                    updated.push({ id, image: "", lastSeen: Date.now() });
                  }
                });
                return updated;
              });
            }
          } else if (data.type === "chunk_done") {
            // Phase-1 preview chunk arrived
            setPreviewChunks((prev) => {
              const exists = prev.find(c => c.index === data.chunk_index);
              if (exists) return prev;
              return [...prev, { index: data.chunk_index, url: data.url }].sort((a, b) => a.index - b.index);
            });
          } else if (data.type === "master_started") {
            setMasterStatus("processing");
          } else if (data.type === "master_done") {
            setMasterStatus("completed");
            setMasterUrl(data.url);
            if (masterPollRef.current) clearInterval(masterPollRef.current);
          } else if (data.type === "master_error") {
            setMasterStatus("failed");
            setMasterError(data.message);
            if (masterPollRef.current) clearInterval(masterPollRef.current);
          }
        } catch { }
      };
    };

    connect();

    // Watchdog timer to remove phantom cameras
    const interval = setInterval(() => {
      const now = Date.now();
      setCameras((prev) => prev.filter((c) => now - c.lastSeen < 2500));
    }, 1000);

    return () => {
      wsRef.current?.close();
      clearInterval(interval);
      if (masterPollRef.current) clearInterval(masterPollRef.current);
    };
  }, []);

  const handleStart = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      // Use a real UUID so it matches the backend uuid.UUID() parsing
      const sid = crypto.randomUUID();
      setSessionId(sid);
      sessionIdRef.current = sid;
      setIsRecording(true);
      setPreviewChunks([]);
      setMasterStatus("idle");
      setMasterUrl(null);
      setMasterError(null);
      // Reset camera tracking for the new session
      cameraIdsRef.current = new Set(cameras.map(c => c.id));
      wsRef.current.send(JSON.stringify({ command: "start", session_id: sid }));
    }
  };

  const handleStop = async () => {
    const currentSessionId = sessionIdRef.current;
    if (wsRef.current?.readyState === WebSocket.OPEN && currentSessionId) {
      setIsRecording(false);
      setIsFinalizing(true);
      setFinalizeError(null);
      wsRef.current.send(JSON.stringify({ command: "stop" }));

      // Use ref so we always have the full set of cameras that participated,
      // even if some disconnected and were removed by the watchdog.
      const camIds = Array.from(cameraIdsRef.current).join(",");
      if (camIds) {
        // Small delay to let WS "stop" propagate to the backend and update
        // the session status from "recording" to "stopped" before we call /finalize.
        await new Promise(resolve => setTimeout(resolve, 600));

        const formData = new URLSearchParams();
        formData.append("session_id", currentSessionId);
        formData.append("selected_cameras", camIds);
        formData.append("layout", "hstack");
        formData.append("sync_strategy", syncStrategy);
        try {
          const res = await fetch("/api/live/finalize", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: formData.toString()
          });
          if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Finalize failed: ${errText}`);
          }
          const json = await res.json();
          setIsFinalizing(false);
          // Always poll — works for both fresh start and already-running cases
          startMasterPolling(currentSessionId);
        } catch (err: any) {
          console.error("Failed to finalize session:", err);
          setFinalizeError(err.message || "Failed to finalize session. Please retry.");
          setIsFinalizing(false);
        }
      } else {
        // No cameras recorded — nothing to sync
        setIsFinalizing(false);
        setFinalizeError("No cameras were recorded. Cannot sync.");
      }
    } else if (wsRef.current?.readyState === WebSocket.OPEN) {
      setIsRecording(false);
      wsRef.current.send(JSON.stringify({ command: "stop" }));
    }
  };

  return (
    <>
      <Navbar />
      <main className="container" style={{ paddingTop: 40, minHeight: "100vh" }}>
        <div className="page-header" style={{ textAlign: "center", marginBottom: 20 }}>
          <h1 className="page-title">🎬 Master Control Dashboard</h1>
          <p className="page-subtitle">Real-time synchronized multi-camera control panel</p>
        </div>

        {/* Status bar */}
        <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 20, marginBottom: 24, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 10, height: 10, borderRadius: "50%", backgroundColor: wsConnected ? "#2ecc71" : "#e74c3c", display: "inline-block", boxShadow: wsConnected ? "0 0 8px #2ecc71" : "0 0 8px #e74c3c" }} />
            <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>{wsConnected ? "Server connected" : "Reconnecting..."}</span>
          </div>
          {sessionId && (
            <div style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "monospace" }}>
              Session: <strong style={{ color: isRecording ? "#e74c3c" : "var(--text-secondary)" }}>{sessionId}</strong>
              {isRecording && <span style={{ marginLeft: 8 }}>🔴 Recording</span>}
            </div>
          )}
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 20, marginBottom: 40, alignItems: "flex-end" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <label style={{ fontSize: 12, fontWeight: "bold", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Sync Strategy</label>
            <select
              value={syncStrategy}
              onChange={(e) => setSyncStrategy(e.target.value)}
              disabled={isFinalizing}
              style={{
                padding: "12px 16px",
                borderRadius: 8,
                border: "1px solid #444",
                background: "#1a1a1a",
                color: "white",
                minWidth: 200,
                fontSize: 14,
                cursor: isFinalizing ? "not-allowed" : "pointer"
              }}
            >
              <option value="auto">Auto (MultiSyncVideo, then SeSyn-Net)</option>
              <option value="multividsynch">MultiSyncVideo (Audio / Global)</option>
              <option value="sesyn_net">SeSyn-Net (AI)</option>
            </select>
          </div>

          <div style={{ display: "flex", gap: 15 }}>
            <button
              onClick={handleStart}
              disabled={isRecording || !wsConnected || isFinalizing}
              style={{
                padding: "12px 24px",
                fontSize: 16,
                cursor: isRecording || !wsConnected || isFinalizing ? "not-allowed" : "pointer",
                borderRadius: 8,
                border: "none",
                background: isRecording ? "#450a0a" : "#dc2626",
                color: "white",
                fontWeight: "bold",
                opacity: isRecording || isFinalizing ? 0.6 : 1,
                boxShadow: isRecording ? "none" : "0 4px 12px rgba(220, 38, 38, 0.3)",
                transition: "all 0.2s"
              }}
            >
              {isRecording ? "● RECORDING..." : "🔴 START RECORDING"}
            </button>
            <button
              onClick={handleStop}
              disabled={(!isRecording && !sessionId) || !wsConnected || isFinalizing}
              style={{
                padding: "12px 24px",
                fontSize: 16,
                cursor: (!isRecording && !sessionId) || !wsConnected || isFinalizing ? "not-allowed" : "pointer",
                borderRadius: 8,
                border: "none",
                background: "#4b5563",
                color: "white",
                fontWeight: "bold",
                opacity: !isRecording || isFinalizing ? 0.6 : 1,
                transition: "all 0.2s"
              }}
            >
              {isFinalizing ? "⏹ PROCESSING..." : "⏹ STOP & SYNC"}
            </button>
          </div>
        </div>

        {/* Finalize Error Banner */}
        {finalizeError && (
          <div style={{
            maxWidth: 600,
            margin: "0 auto 24px",
            padding: "14px 18px",
            background: "#450a0a",
            border: "1px solid #7f1d1d",
            borderRadius: 8,
            color: "#fca5a5",
            display: "flex",
            alignItems: "center",
            gap: 12,
            fontSize: 14
          }}>
            <span style={{ fontSize: 20 }}>⚠️</span>
            <span style={{ flex: 1 }}>{finalizeError}</span>
            <button
              onClick={handleStop}
              style={{ background: "#dc2626", color: "white", border: "none", padding: "6px 14px", borderRadius: 6, cursor: "pointer", fontSize: 13, fontWeight: "bold" }}
            >
              Retry
            </button>
          </div>
        )}

        {/* Phase-2 Master Video Banner */}
        {masterStatus !== "idle" && (
          <div style={{
            maxWidth: 800,
            margin: "0 auto 32px",
            padding: "20px 24px",
            background: masterStatus === "completed" ? "#052e16" : masterStatus === "failed" ? "#450a0a" : "#1e1b4b",
            border: `1px solid ${masterStatus === "completed" ? "#166534" : masterStatus === "failed" ? "#7f1d1d" : "#3730a3"}`,
            borderRadius: 12,
            color: "white",
          }}>
            {(masterStatus === "processing" || masterStatus === "pending") && (
              <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                <span style={{ fontSize: 28 }}>🎬</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: "bold", fontSize: 16, marginBottom: 4 }}>Building Final Master Video…</div>
                  <div style={{ fontSize: 13, color: "#a5b4fc" }}>
                    Concatenating all chunks, aligning audio, and rendering your high-quality export.
                    This runs in the background — you can still browse preview chunks below.
                  </div>
                  <div style={{ marginTop: 10, height: 4, background: "#312e81", borderRadius: 4, overflow: "hidden", position: "relative" }}>
                    <div style={{
                      height: "100%",
                      background: "linear-gradient(90deg, #6366f1, #818cf8)",
                      borderRadius: 4,
                      width: "40%",
                      animation: "masterProgress 1.8s ease-in-out infinite",
                      position: "absolute",
                    }} />
                  </div>
                </div>
              </div>
            )}
            {masterStatus === "completed" && masterUrl && (
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
                  <span style={{ fontSize: 28 }}>✅</span>
                  <div>
                    <div style={{ fontWeight: "bold", fontSize: 18 }}>Master Video Ready!</div>
                    <div style={{ fontSize: 13, color: "#86efac" }}>High-quality, seamlessly aligned final export</div>
                  </div>
                  <a
                    href={masterUrl}
                    download={`master_${sessionId}.mp4`}
                    style={{
                      marginLeft: "auto",
                      padding: "10px 20px",
                      background: "#16a34a",
                      color: "white",
                      borderRadius: 8,
                      textDecoration: "none",
                      fontWeight: "bold",
                      fontSize: 14,
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                    }}
                  >
                    ⬇️ Download Master
                  </a>
                </div>
                <video
                  src={masterUrl}
                  controls
                  style={{ width: "100%", borderRadius: 8, maxHeight: 400, background: "#000" }}
                />
              </div>
            )}
            {masterStatus === "failed" && (
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <span style={{ fontSize: 24 }}>❌</span>
                <div>
                  <div style={{ fontWeight: "bold" }}>Master Render Failed</div>
                  <div style={{ fontSize: 13, color: "#fca5a5", marginTop: 4 }}>{masterError}</div>
                  <div style={{ fontSize: 12, color: "#f87171", marginTop: 4 }}>
                    Preview chunks are still available below.
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Phase-1 Preview Chunks */}
        {previewChunks.length > 0 && masterStatus !== "completed" && (
          <div style={{ maxWidth: 800, margin: "0 auto 32px" }}>
            <h3 style={{ color: "var(--text-secondary)", fontSize: 14, marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              📼 Live Preview Chunks ({previewChunks.length})
            </h3>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {previewChunks.map((chunk) => (
                <a
                  key={chunk.index}
                  href={chunk.url}
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    padding: "6px 14px",
                    background: "#1e293b",
                    border: "1px solid #334155",
                    borderRadius: 6,
                    color: "#94a3b8",
                    fontSize: 13,
                    textDecoration: "none",
                    fontFamily: "monospace"
                  }}
                >
                  Chunk {chunk.index}
                </a>
              ))}
            </div>
          </div>
        )}

        {/* Camera Grid */}
        <div
          className="camera-grid"
          style={{
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "center",
            gap: 20,
          }}
        >
          {cameras.length === 0 ? (
            <div style={{ color: "var(--text-secondary)", marginTop: 40 }}>
              Waiting for cameras to connect...
            </div>
          ) : (
            cameras.map((cam) => (
              <div
                key={cam.id}
                className="cam-view card"
                style={{
                  background: "#333",
                  padding: 10,
                  borderRadius: 8,
                  border: "2px solid #555",
                  textAlign: "center",
                }}
              >
                <h3 style={{ margin: "0 0 10px 0", color: "#fff", fontSize: 16 }}>
                  Camera: {cam.id}
                </h3>
                {cam.image && (
                  <img
                    src={cam.image}
                    alt={`Cam ${cam.id}`}
                    style={{
                      width: 320,
                      height: 240,
                      backgroundColor: "black",
                      borderRadius: 4,
                      objectFit: "cover",
                    }}
                  />
                )}
                {!cam.image && (
                  <div style={{ width: 320, height: 240, background: "black", borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)" }}>
                    No Feed
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        <style>{`
          @keyframes masterProgress {
            0%   { left: -40%; }
            100% { left: 140%; }
          }
        `}</style>
      </main>
    </>
  );
}
