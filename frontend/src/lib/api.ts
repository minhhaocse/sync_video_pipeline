// ── Types ─────────────────────────────────────────────────────────────────────

export interface Session {
  id: string;
  name: string;
  camera_count: number;
  status: "recording" | "processing" | "completed" | "failed";
  master_url?: string | null;
  sync_strategy: string;
  created_at: string;
  updated_at: string;
}

export interface Offset {
  cam_id: string;
  offset_seconds: number;
  computed_at: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
// All paths are relative — Next.js rewrites forward them to the FastAPI backend.
// This means the app works correctly both locally AND over Cloudflare tunnel
// without needing NEXT_PUBLIC_API_URL baked into the build.

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Sessions ──────────────────────────────────────────────────────────────────

export const api = {
  sessions: {
    list: (skip = 0, limit = 20) =>
      apiFetch<Session[]>(`/api/sessions?skip=${skip}&limit=${limit}`),

    get: (id: string) => apiFetch<Session>(`/api/sessions/${id}`),

    create: (name: string, cameraCount: number, syncStrategy: string = "auto") =>
      apiFetch<Session>("/api/sessions", {
        method: "POST",
        body: JSON.stringify({ name, camera_count: cameraCount, sync_strategy: syncStrategy }),
      }),

    delete: (id: string) =>
      apiFetch<void>(`/api/sessions/${id}`, { method: "DELETE" }),

    offsets: (id: string) => apiFetch<Offset[]>(`/api/sessions/${id}/offsets`),

    chunks: (id: string) => apiFetch<number[]>(`/api/sessions/${id}/chunks`),
  },

  health: () => apiFetch<{ status: string }>("/health"),
};
