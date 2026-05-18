"use client";

import { useEffect, useRef, useState } from "react";

interface SyncedPlayerProps {
  url: string | null;
  title?: string;
  onEnded?: () => void;
}

/**
 * A sleek, custom video player for synced outputs.
 * In Phase 1, it plays static synced_chunk_N.mp4 files.
 * In Phase 2, this will be upgraded to an HLS.js player.
 */
export default function SyncedPlayer({ url, title, onEnded }: SyncedPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // Auto-play when the URL changes (e.g. next chunk is ready)
  useEffect(() => {
    if (videoRef.current && url) {
      videoRef.current.load();
      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {
        // Handle autoplay block by browser
        setIsPlaying(false);
        console.log("Autoplay blocked, user interaction required");
      });
    }
  }, [url]);

  return (
    <div className="card" style={{ padding: 0, overflow: "hidden", position: "relative", border: "1px solid var(--border)", boxShadow: "0 20px 40px rgba(0,0,0,0.6)" }}>
      {/* Player Header / Title Bar */}
      <div
        style={{
          padding: "16px 20px",
          background: "linear-gradient(to bottom, rgba(10, 11, 30, 0.9), rgba(10, 11, 30, 0.4))",
          borderBottom: "1px solid rgba(255,255,255,0.05)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 10,
          backdropFilter: "blur(12px)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 16 }}>📺</span>
          <span style={{ fontSize: 14, fontWeight: 700, letterSpacing: "0.02em", color: "var(--text-primary)" }}>
            {title || "Synced Output Preview"}
          </span>
        </div>
        <div className={`badge ${url && isPlaying ? "badge-processing" : "badge-uploaded"}`} style={{ fontSize: 10, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)" }}>
          {url && isPlaying ? (
            <><span className="badge-dot pulse" style={{ color: "var(--accent-red)" }}/> LIVE SYNC</>
          ) : (
            "READY"
          )}
        </div>
      </div>

      <video
        ref={videoRef}
        controls
        playsInline
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => { setIsPlaying(false); if (onEnded) onEnded(); }}
        style={{
          width: "100%",
          display: "block",
          aspectRatio: "16/9",
          background: "#000",
          objectFit: "contain",
        }}
      >
        {url && <source src={url} type="video/mp4" />}
        Your browser does not support the video tag.
      </video>

      {!url && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(10, 11, 30, 0.8)",
            backdropFilter: "blur(20px)",
            color: "var(--text-muted)",
            gap: 16,
          }}
        >
          <div className="skeleton" style={{ width: 64, height: 64, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(255,255,255,0.05)" }}>
            <span style={{ fontSize: 24, opacity: 0.5 }}>💤</span>
          </div>
          <div style={{ textAlign: "center" }}>
            <p style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", marginBottom: 4 }}>Awaiting Final Sync</p>
            <p style={{ fontSize: 13 }}>The final master sync output will appear here when ready.</p>
          </div>
        </div>
      )}
    </div>
  );
}
