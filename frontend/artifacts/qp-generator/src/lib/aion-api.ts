/**
 * AION API Configuration
 * All backend communication goes through this file.
 * Backend runs at http://127.0.0.1:8100
 * Vite proxy forwards /api/* to backend automatically.
 */

import { setBaseUrl } from "@workspace/api-client-react";

const API_URL = import.meta.env.VITE_API_URL ?? "";

export function initAPI() {
  setBaseUrl(API_URL);
  console.log("[AION] API connected →", API_URL || "http://127.0.0.1:8100 (via proxy)");
}

// ── Direct fetch helpers ──────────────────────────────────────────────────────

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error ?? `HTTP ${res.status}`);
  }
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error ?? `HTTP ${res.status}`);
  }
  return res.json();
}

// ── AION Backend Endpoints ────────────────────────────────────────────────────

export const aionAPI = {
  health: () => get("/api/health"),
  tags:   () => get("/api/tags"),

  upload: async (file: File, subject?: string, category?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (subject)  form.append("subject",  subject);
    if (category) form.append("category", category);
    const res = await fetch("/api/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
    return res.json();
  },

  files: {
    list:   ()       => get("/api/files"),
    delete: (id: string) => fetch(`/api/files/${id}`, { method: "DELETE" }),
  },

  documents: {
    status:  (docId: string) => get(`/api/documents/${docId}/status`),
    modules: (docId: string) => get(`/api/documents/${docId}/modules`),
  },

  generateStream: (payload: {
    file_id?:               string;
    fileId?:                string;
    subject?:               string;
    department?:            string;
    semester?:              number;
    exam_type?:             string;
    examType?:              string;
    mode?:                  string;
    difficulty?:            string;
    bloom_levels?:          string[];
    bloomsTaxonomy?:        string[];
    selected_modules?:      number[];
    modules?:               any[];
    question_types?:        string[];
    model?:                 string;
    include_visual?:        boolean;
    useImages?:             boolean;
    notes_text?:            string;
    notesText?:             string;
  }) => {
    console.log("[AION FRONTEND] Outgoing Payload:", payload);
    const body = JSON.stringify(payload);
    return fetch("/api/generate/stream", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
  },

  generateAsync: (payload: unknown) => post<{ jobId: string; status: string }>("/api/generate", payload),

  jobStatus: (jobId: string) => get(`/api/generate/status/${jobId}`),

  preview: (payload: unknown) => post("/api/preview", payload),
};
