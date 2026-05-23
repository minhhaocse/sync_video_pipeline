"use client";

import { useState } from "react";
import useSWR from "swr";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import { api, Session } from "@/lib/api";
import { useToast } from "@/components/Toast";

const fetcher = () => api.sessions.list();

const STATUS_LABELS: Record<string, string> = {
  recording:  "recording",
  processing: "processing",
  completed:  "completed",
  failed:     "failed",
};

function formatSyncStrategy(strategy: string) {
  switch (strategy) {
    case "auto":
      return "Auto Coarse-to-Fine";
    case "multisyncvideo":
    case "multividsync":
    case "multividsynch":
      return "MultiSyncVideo";
    case "visual_hybrid":
      return "Hybrid Visual";
    case "sesyn_net":
      return "SeSyn-Net";
    case "feature_based":
      return "MultiVidSynch";
    default:
      return strategy;
  }
}

export default function SessionsPage() {
  const { data: sessions, error, isLoading, mutate } = useSWR("sessions", fetcher, {
    refreshInterval: 5000,
  });
  const { addToast } = useToast();

  const [deletingId, setDeletingId] = useState<string | null>(null);

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Are you sure you want to delete session "${name}"? This cannot be undone.`)) return;
    
    setDeletingId(id);
    try {
      await api.sessions.delete(id);
      mutate();
      addToast({ type: "success", title: "Session Deleted", message: `Removed "${name}"` });
    } catch (err: any) {
      addToast({ type: "error", title: "Deletion Failed", message: err.message || "An error occurred" });
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <>
      <Navbar />
      <main className="container" style={{ paddingTop: 60, paddingBottom: 80 }}>
        {/* Header */}
        <div className="page-header fade-in-up">
          <h1 className="page-title">Recording Sessions</h1>
          <p className="page-subtitle">
            View and manage recorded sessions, monitor sync progress, and inspect final synced outputs.
          </p>
        </div>

        {/* Sessions List */}

        {/* Sessions List */}
        {isLoading && (
          <div className="skeleton" style={{ height: 200, width: "100%", borderRadius: "var(--radius-md)" }} />
        )}

        {error && (
          <div className="card badge-failed fade-in-up" style={{ padding: 24, textAlign: "center", background: "rgba(239, 68, 68, 0.1)" }}>
            <div style={{ fontSize: 24, marginBottom: 8 }}>⚠️</div>
            <div style={{ fontWeight: 700 }}>Failed to load sessions</div>
            <div style={{ color: "var(--text-secondary)", fontSize: 14 }}>Is the FastAPI backend running on port 8000?</div>
          </div>
        )}

        {sessions && sessions.length === 0 && (
          <div className="card fade-in-up stagger-2" style={{ textAlign: "center", padding: "80px 24px", background: "rgba(0,0,0,0.2)", borderStyle: "dashed" }}>
            <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.8 }}>📭</div>
            <h3 style={{ fontSize: 18, fontWeight: 600, color: "var(--text-primary)", marginBottom: 8 }}>No sessions active</h3>
            <p style={{ color: "var(--text-muted)", fontSize: 14 }}>Create your first recording session above to begin capturing.</p>
          </div>
        )}

        <div className="grid-2">
          {sessions?.map((session: Session, index: number) => (
            <div
              key={session.id}
              className={`card interactive fade-in-up`}
              style={{ padding: 0, animationDelay: `${(index % 4) * 50}ms` }}
            >
              <div style={{ padding: 24 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                  <div>
                    <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4, letterSpacing: "-0.01em" }}>
                      {session.name}
                    </h3>
                    <div className="mono" style={{ fontSize: 12, opacity: 0.7 }}>{session.id.slice(0, 8)}…</div>
                  </div>
                  <span className={`badge badge-${session.status}`}>
                    <span className={`badge-dot ${session.status === "recording" ? "pulse" : ""}`} />
                    {STATUS_LABELS[session.status]}
                  </span>
                </div>

                <div style={{ display: "flex", gap: 24, color: "var(--text-secondary)", fontSize: 14, marginBottom: 24 }}>
                  <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ color: "var(--accent-fuchsia)" }}>📱</span> {session.camera_count} Cameras
	                  </span>
	                  <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
	                    <span style={{ color: "var(--accent-blue)" }}>⚙️</span> {formatSyncStrategy(session.sync_strategy)}
	                  </span>
                  <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ color: "var(--accent-blue)" }}>🕒</span>
                    {new Date(session.created_at).toLocaleDateString("en-US", {
                      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
                    })}
                  </span>
                </div>
              </div>

              <div style={{ display: "flex", borderTop: "1px solid var(--border)", background: "rgba(0,0,0,0.2)" }}>
                <Link
                  href={`/sessions/${session.id}`}
                  className="btn btn-ghost"
                  style={{ flex: 1, border: "none", borderRadius: 0, padding: 16, borderRight: "1px solid var(--border)" }}
                >
                  View Detail
                </Link>
                <button
                  onClick={() => handleDelete(session.id, session.name)}
                  disabled={deletingId === session.id}
                  className="btn btn-ghost"
                  style={{ padding: "16px 24px", border: "none", borderRadius: 0, color: "var(--accent-red)" }}
                  title="Delete Session"
                >
                  {deletingId === session.id ? "..." : "🗑️"}
                </button>
              </div>
            </div>
          ))}
        </div>
      </main>
    </>
  );
}
