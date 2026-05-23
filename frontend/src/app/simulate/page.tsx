"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";

interface CameraInput {
  id: string;
  file: File | null;
}

export default function SimulatePage() {
  const router = useRouter();
  const [sessionName, setSessionName] = useState("");
  const [syncStrategy, setSyncStrategy] = useState("auto");
  const [cameraInputs, setCameraInputs] = useState<CameraInput[]>([
    { id: "cam1", file: null },
    { id: "cam2", file: null },
  ]);
  const [isUploading, setIsUploading] = useState(false);
  const [status, setStatus] = useState("");

  const addCamera = () => {
    const nextId = `cam${cameraInputs.length + 1}`;
    setCameraInputs([...cameraInputs, { id: nextId, file: null }]);
  };

  const removeCamera = (index: number) => {
    if (cameraInputs.length <= 1) return;
    const newInputs = [...cameraInputs];
    newInputs.splice(index, 1);
    // Optional: Re-index IDs to keep them sequential
    const reindexed = newInputs.map((input, idx) => ({
      ...input,
      id: `cam${idx + 1}`
    }));
    setCameraInputs(reindexed);
  };

  const handleFileChange = (index: number, file: File | null) => {
    const newInputs = [...cameraInputs];
    newInputs[index].file = file;
    setCameraInputs(newInputs);
  };

  const handleIdChange = (index: number, id: string) => {
    const newInputs = [...cameraInputs];
    newInputs[index].id = id;
    setCameraInputs(newInputs);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const validInputs = cameraInputs.filter(input => input.file !== null);
    
    if (!sessionName || validInputs.length === 0) {
      setStatus("Session Name and at least one Camera file are required.");
      return;
    }

    setIsUploading(true);
    setStatus("Creating session...");

    try {
      // Create session
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: sessionName,
          camera_count: validInputs.length,
          sync_strategy: syncStrategy,
        }),
      });

      if (!res.ok) {
        const rawText = await res.text();
        console.error('Session creation failed (Raw Response):', res.status, rawText);
        let errorDetail = res.statusText;
        try {
          const parsed = JSON.parse(rawText);
          if (parsed.detail) errorDetail = parsed.detail;
        } catch (e) {}
        throw new Error(`Failed to create session [Status ${res.status}]: ${errorDetail}`);
      }
      const session = await res.json();

      setStatus("Uploading videos for simulation (this may take a while)...");

      const formData = new FormData();
      formData.append("session_id", session.id);
      formData.append("sync_strategy", syncStrategy);
      formData.append("selected_cameras", validInputs.map(i => i.id).join(","));
      
      validInputs.forEach((input, idx) => {
        // Backend expects camN_id and camN_file or similar
        // Based on app/routers/simulate.py, it looks for keys starting with "cam"
        const prefix = `cam${idx + 1}`;
        formData.append(`${prefix}_id`, input.id);
        if (input.file) {
          formData.append(`${prefix}_file`, input.file);
        }
      });

      const uploadRes = await fetch("/api/simulate/upload", {
        method: "POST",
        body: formData,
      });

      if (!uploadRes.ok) {
        const errorText = await uploadRes.text();
        throw new Error(`Upload failed: ${errorText}`);
      }

      setStatus("Simulation started! Redirecting...");
      setTimeout(() => {
        router.push(`/sessions/${session.id}`);
      }, 1000);

    } catch (err: any) {
      console.error(err);
      setStatus(`Error: ${err.message}`);
      setIsUploading(false);
    }
  };

  return (
    <>
      <Navbar />
      <main className="container" style={{ paddingTop: 40, paddingBottom: 60 }}>
        <h2 className="page-title">🎬 Simulate Upload</h2>
        <p style={{ color: "var(--text-secondary)", marginBottom: 30 }}>
          Upload full pre-recorded videos to simulate a live multi-camera sync session.
        </p>

        <form onSubmit={handleSubmit} style={{ maxWidth: 700, display: "flex", flexDirection: "column", gap: 24 }}>
          <div className="card shadow-lg" style={{ border: "1px solid #333" }}>
            <h3 style={{ marginBottom: 16, display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontSize: "1.2rem" }}>⚙️</span> Session Configuration
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <label className="text-sm font-medium">Session Name</label>
                <input
                  type="text"
                  placeholder="e.g. Landscape Demo"
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value)}
                  required
                  style={{ padding: 12, borderRadius: 8, border: "1px solid #444", background: "#1a1a1a", color: "white" }}
                />
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <label className="text-sm font-medium">Sync Strategy</label>
                <select
                  value={syncStrategy}
                  onChange={(e) => setSyncStrategy(e.target.value)}
                  style={{ padding: 12, borderRadius: 8, border: "1px solid #444", background: "#1a1a1a", color: "white" }}
                >
                  <option value="auto">Auto (MultiSyncVideo, then SeSyn-Net)</option>
                  <option value="multividsynch">MultiSyncVideo (Audio / Global)</option>
                  <option value="sesyn_net">SeSyn-Net (Deep Learning)</option>
                </select>
              </div>
            </div>
          </div>

          <div className="card shadow-lg" style={{ border: "1px solid #333" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
              <h3 style={{ margin: 0, display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: "1.2rem" }}>🎥</span> Camera Inputs
              </h3>
              <button 
                type="button" 
                onClick={addCamera}
                className="btn"
                style={{ background: "#2563eb", color: "white", padding: "8px 16px", borderRadius: 6, fontSize: 14 }}
              >
                + Add Camera
              </button>
            </div>
            
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {cameraInputs.map((input, index) => (
                <div 
                  key={`camera-${input.id}-${index}`} 
                  style={{ 
                    display: "flex", 
                    flexDirection: "column", 
                    gap: 12, 
                    padding: 16, 
                    background: "#1a1a1a", 
                    borderRadius: 8,
                    border: "1px solid #333",
                    position: "relative"
                  }}
                >
                  <div style={{ display: "flex", gap: 16, alignItems: "flex-end" }}>
                    <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 6 }}>
                      <label className="text-xs font-semibold text-gray-400">CAMERA ID</label>
                      <input 
                        type="text"
                        value={input.id}
                        onChange={(e) => handleIdChange(index, e.target.value)}
                        placeholder="e.g. cam1"
                        style={{ padding: "8px 12px", borderRadius: 6, border: "1px solid #444", background: "#262626", color: "white" }}
                      />
                    </div>
                    <div style={{ flex: 2, display: "flex", flexDirection: "column", gap: 6 }}>
                      <label className="text-xs font-semibold text-gray-400">VIDEO FILE</label>
                      <input
                        type="file"
                        accept="video/mp4,video/webm,video/quicktime"
                        onChange={(e) => handleFileChange(index, e.target.files?.[0] || null)}
                        style={{ padding: "6px 0", color: "#ccc", fontSize: 13 }}
                      />
                    </div>
                    {cameraInputs.length > 1 && (
                      <button 
                        type="button" 
                        onClick={() => removeCamera(index)}
                        style={{ 
                          background: "#dc2626", 
                          color: "white", 
                          border: "none", 
                          padding: "8px 12px", 
                          borderRadius: 6, 
                          cursor: "pointer",
                          fontSize: 13
                        }}
                      >
                        🗑️
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={isUploading}
            style={{ 
              padding: "16px", 
              fontSize: 16, 
              marginTop: 10, 
              fontWeight: "bold",
              boxShadow: "0 4px 14px 0 rgba(37, 99, 235, 0.39)"
            }}
          >
            {isUploading ? (
              <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}>
                <span className="spinner" style={{ width: 20, height: 20, border: "2px solid rgba(255,255,255,0.3)", borderTop: "2px solid white", borderRadius: "50%" }}></span>
                Processing Simulation...
              </span>
            ) : "Start Simulation"}
          </button>

          {status && (
            <div style={{ 
              marginTop: 10, 
              padding: 16, 
              borderRadius: 8, 
              backgroundColor: status.includes("Error") ? "#450a0a" : "#064e3b",
              color: status.includes("Error") ? "#fca5a5" : "#6ee7b7",
              border: status.includes("Error") ? "1px solid #7f1d1d" : "1px solid #065f46",
              textAlign: "center"
            }}>
              {status}
            </div>
          )}
        </form>
      </main>
      <style>{`
        .spinner {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </>
  );
}
