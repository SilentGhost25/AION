// ── AION API Client ───────────────────────────────────────────────────────────
// Single source of truth for all backend communication.
// Never call the backend directly from components — always use this client.

const BASE_URL = import.meta.env.VITE_AION_API_URL ?? "http://127.0.0.1:8100";

// ── Core fetch wrapper ────────────────────────────────────────────────────────

async function aionFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${BASE_URL}${path}`;

  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(error.error ?? `HTTP ${res.status}`);
  }

  return res.json();
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  api_version: string;
  active_model: string;
  models_available: number;
  services: {
    aion_api: string;
    ollama: string;
  };
}

export interface ModelInfo {
  id: string;
  runtime: string;
  size_gb: number;
  loaded: boolean;
  healthy: boolean;
  context_window: number;
}

export interface ModelsResponse {
  active_model: string;
  runtime_status: string;
  models: ModelInfo[];
}

export interface UploadResponse {
  upload_id: string;
  filename: string;
  status: string;
  size_bytes: number;
}

export interface JobResponse {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  workspace: string;
  progress?: number;
  message?: string;
  result?: unknown;
  error?: string;
}

export interface QuestionGenerationRequest {
  subject_code: string;
  modules: number[];
  exam_type: "IA" | "SEE";
  marks: number;
  difficulty: "Easy" | "Medium" | "Hard" | "Mixed";
  bloom_levels: string[];
  question_types?: string[];
  professor_style_id?: string;
  diagram_generation?: boolean;
  numerical_generation?: boolean;
  novelty_target?: number;
  grounding_required?: boolean;
  model?: string;
}

export interface DiagnosticsResponse {
  overall: "healthy" | "degraded" | "unhealthy";
  services: Record<string, {
    status: "healthy" | "degraded" | "unhealthy";
    latency_ms?: number;
    message?: string;
  }>;
}

// ── API Methods ───────────────────────────────────────────────────────────────

export const aion = {

  // Health
  health: () =>
    aionFetch<HealthResponse>("/api/v1/health"),

  // Dashboard
  dashboard: {
    summary: () =>
      aionFetch("/api/v1/dashboard/summary"),
    activity: () =>
      aionFetch("/api/v1/dashboard/activity"),
  },

  // Models
  models: {
    list: () =>
      aionFetch<ModelsResponse>("/api/v1/models"),
    load: (modelId: string) =>
      aionFetch(`/api/v1/models/${modelId}/load`, { method: "POST" }),
    unload: (modelId: string) =>
      aionFetch(`/api/v1/models/${modelId}/unload`, { method: "POST" }),
    benchmark: (modelId: string) =>
      aionFetch(`/api/v1/models/${modelId}/benchmark`, { method: "POST" }),
  },

  // Uploads
  uploads: {
    upload: async (file: File, metadata?: Record<string, string>) => {
      const form = new FormData();
      form.append("file", file);
      if (metadata) {
        Object.entries(metadata).forEach(([k, v]) => form.append(k, v));
      }
      let res = await fetch(`${BASE_URL}/api/v1/uploads`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        res = await fetch(`${BASE_URL}/api/upload`, {
          method: "POST",
          body: form,
        });
      }
      if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
      return res.json() as Promise<UploadResponse>;
    },
    list: () =>
      aionFetch("/api/v1/uploads").catch(() => aionFetch("/api/files")),
  },

  // Training
  training: {
    analyze: (uploadId: string, options?: Record<string, unknown>) =>
      aionFetch<JobResponse>("/api/v1/training/analyze", {
        method: "POST",
        body: JSON.stringify({ upload_id: uploadId, ...options }),
      }),
    approveModules: (uploadId: string, modules: unknown[]) =>
      aionFetch(`/api/v1/training/module-map/${uploadId}/approve`, {
        method: "POST",
        body: JSON.stringify({ modules }),
      }),
    start: (uploadId: string, config?: Record<string, unknown>) =>
      aionFetch<JobResponse>("/api/v1/training/start", {
        method: "POST",
        body: JSON.stringify({ upload_id: uploadId, ...config }),
      }),
  },

  // Questions
  questions: {
    generate: (config: QuestionGenerationRequest) =>
      aionFetch<JobResponse>("/api/v1/questions/generate", {
        method: "POST",
        body: JSON.stringify(config),
      }),
    get: (questionId: string) =>
      aionFetch(`/api/v1/questions/${questionId}`),
    approve: (questionId: string) =>
      aionFetch(`/api/v1/review/${questionId}/accept`, { method: "POST" }),
    reject: (questionId: string, reason?: string) =>
      aionFetch(`/api/v1/review/${questionId}/reject`, {
        method: "POST",
        body: JSON.stringify({ reason }),
      }),
  },

  // Papers
  papers: {
    generate: (config: Record<string, unknown>) =>
      aionFetch<JobResponse>("/api/v1/papers/generate", {
        method: "POST",
        body: JSON.stringify(config),
      }),
    get: (paperId: string) =>
      aionFetch(`/api/v1/papers/${paperId}`),
    export: (paperId: string, format: string) =>
      aionFetch(`/api/v1/papers/${paperId}/export`, {
        method: "POST",
        body: JSON.stringify({ format }),
      }),
  },

  // Knowledge
  knowledge: {
    subjects: () =>
      aionFetch("/api/v1/knowledge/subjects"),
    graph: (subjectId: string) =>
      aionFetch(`/api/v1/knowledge/graph?subject_id=${subjectId}`),
    concept: (conceptId: string) =>
      aionFetch(`/api/v1/knowledge/concepts/${conceptId}`),
  },

  // Diagnostics
  diagnostics: {
    summary: () =>
      aionFetch<DiagnosticsResponse>("/api/v1/diagnostics"),
  },

  // Datasets
  datasets: {
    list: () =>
      aionFetch("/api/v1/datasets"),
    get: (datasetId: string) =>
      aionFetch(`/api/v1/datasets/${datasetId}`),
    export: (datasetId: string, format: string) =>
      aionFetch(`/api/v1/datasets/${datasetId}/export`, {
        method: "POST",
        body: JSON.stringify({ format }),
      }),
  },

  // Jobs (polling)
  jobs: {
    get: (jobId: string) =>
      aionFetch<JobResponse>(`/api/v1/jobs/${jobId}`),
    events: (jobId: string, onEvent: (event: JobResponse) => void): EventSource => {
      const es = new EventSource(`${BASE_URL}/api/v1/jobs/${jobId}/events`);
      es.onmessage = (e) => {
        try {
          onEvent(JSON.parse(e.data));
        } catch {
          // ignore parse errors
        }
      };
      return es;
    },
  },
};

export default aion;
