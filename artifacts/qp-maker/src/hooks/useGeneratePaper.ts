import { useState, useCallback, useRef } from "react";

// ── Types ────────────────────────────────────────────────────

export type SubQuestion = {
  letter: string | null;
  text:   string;
  marks:  number;
};

export type MainQuestion = {
  mqIndex:      number;
  totalMarks:   number;
  bloomLevel:   number;
  bloomName:    string;
  subQuestions: SubQuestion[];
};

export type ModuleQuestions = {
  moduleIndex: number;
  moduleTitle: string;
  questions:   MainQuestion[];
};

export type GeneratedPaper = {
  id:          string;
  subject:     string;
  examType:    string;
  mode:        string;
  modules:     ModuleQuestions[];
  generatedAt: string;
  totalMarks:  number;
};

export type UploadedFile = {
  id:          string;
  filename:    string;
  storedPath:  string;
  subject:     string;
  category:    string;
  uploadedAt:  string;
  sizeBytes:   number;
};

export type GenerateConfig = {
  fileId:      string;             // ← returned by uploadFile()
  subject:     string;
  examType:    "ia" | "see";
  mode:        "turbo" | "balanced" | "deep";
  maxConcepts: number;
  questionsPerModule: number;
  moduleFilter?: number[];
};

type Status = "idle" | "running" | "done" | "error";

// ── API URL ──────────────────────────────────────────────────
const API_URL = import.meta.env.VITE_AION_API_URL ?? "http://localhost:8100";

// ── Standalone Upload Function ───────────────────────────────

export async function uploadFile(
  file:     File,
  subject:  string,
  category: string = "notes",
  onProgress?: (pct: number) => void,
): Promise<UploadedFile> {
  const form = new FormData();
  form.append("file",     file);
  form.append("subject",  subject);
  form.append("category", category);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(e.loaded / e.total);
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status === 201) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(`Upload failed: ${xhr.status} ${xhr.responseText}`));
      }
    });

    xhr.addEventListener("error", () => {
      reject(new Error("Upload network error"));
    });

    xhr.open("POST", `${API_URL}/api/upload`);
    xhr.send(form);
  });
}

// ── Hook ─────────────────────────────────────────────────────

export function useGeneratePaper() {
  const [status,   setStatus]   = useState<Status>("idle");
  const [progress, setProgress] = useState(0);
  const [logs,     setLogs]     = useState<string[]>([]);
  const [paper,    setPaper]    = useState<GeneratedPaper | null>(null);
  const [error,    setError]    = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const generate = useCallback(async (config: GenerateConfig) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setStatus("running");
    setProgress(0);
    setLogs([]);
    setPaper(null);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/api/generate/stream`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(config),
        signal:  controller.signal,
      });

      if (!res.ok || !res.body) {
        throw new Error(`Server returned ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let   buffer  = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";

        for (const block of blocks) {
          if (!block.trim()) continue;

          const lines     = block.split("\n");
          const eventLine = lines.find((l) => l.startsWith("event:"));
          const dataLine  = lines.find((l) => l.startsWith("data:"));

          if (!eventLine || !dataLine) continue;

          const event = eventLine.replace("event:", "").trim();
          let data: any;
          try {
            data = JSON.parse(dataLine.replace("data:", "").trim());
          } catch {
            continue;
          }

          switch (event) {
            case "status":
              setLogs((prev) => [...prev, data.message ?? "Starting..."]);
              break;

            case "log":
              setLogs((prev) => [...prev, data.message]);
              const moduleMatch = data.message?.match(/\[MODULE (\d+)\]/);
              if (moduleMatch) {
                setProgress(parseInt(moduleMatch[1]) / 5);
              }
              break;

            case "result":
              setPaper(data);
              setProgress(1);
              setStatus("done");
              break;

            case "done":
              setStatus("done");
              break;

            case "error":
              throw new Error(data.message ?? "Unknown error");
          }
        }
      }

      // Functional status update after stream completion
      setStatus((prev) => (prev === "running" ? "done" : prev));

    } catch (err: any) {
      if (err.name === "AbortError") return;
      setError(err.message ?? "Generation failed");
      setStatus("error");
    }
  }, [paper]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setStatus("idle");
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setProgress(0);
    setLogs([]);
    setPaper(null);
    setError(null);
  }, []);

  return { generate, cancel, reset, status, progress, logs, paper, error };
}
