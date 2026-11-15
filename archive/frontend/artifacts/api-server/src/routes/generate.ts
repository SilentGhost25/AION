import { Router, Request, Response } from "express";
import { spawn } from "child_process";
import path from "path";
import { randomUUID } from "crypto";
import { materialRegistry } from "./materials";

const router = Router();

// Job store (replace with Redis/DB in production)
const jobStore = new Map<string, {
  status:   "queued" | "running" | "done" | "failed";
  progress: number;
  result:   object | null;
  error:    string | null;
  logs:     string[];
}>();

const AION_DIR = path.resolve(
  process.env.AION_DIR ?? "d:/AION"
);

// ── Helper: Run AION Python pipeline ─────────────────────────
function runAION(
  filePaths:   string[],
  subject:     string,
  examType:    string,
  mode:        string,
  maxConcepts: number,
  onLog:       (line: string) => void,
  onResult:    (data: object) => void,
  onError:     (err: string) => void,
) {
  const script = path.join(AION_DIR, "aion_server.py");

  const proc = spawn(
    process.env.PYTHON_BIN ?? "python",
    [script],
    {
      env: {
        ...process.env,
        AION_MODEL:    process.env.AION_MODEL ?? "qwen2.5:3b",
        AION_FILES:    JSON.stringify(filePaths),
        AION_SUBJECT:  subject,
        AION_EXAM:     examType,
        AION_MODE:     mode,
        AION_N:        String(maxConcepts),
      },
      cwd: AION_DIR,
    }
  );

  let stdout = "";
  let stderr = "";

  proc.stdout.on("data", (chunk: Buffer) => {
    const line = chunk.toString();
    stdout += line;
    onLog(line.trim());
  });

  proc.stderr.on("data", (chunk: Buffer) => {
    stderr += chunk.toString();
  });

  proc.on("close", (code) => {
    if (code !== 0) {
      onError(stderr || `AION exited with code ${code}`);
      return;
    }
    try {
      // aion_server.py prints JSON result as last line
      const lines  = stdout.trim().split("\n");
      const last   = lines[lines.length - 1];
      const result = JSON.parse(last);
      onResult(result);
    } catch (e) {
      onError(`Failed to parse AION output: ${e}. Raw output: ${stdout}`);
    }
  });

  return proc;
}

// ── POST /generate (async job) ───────────────────────────
router.post("/generate", (req: Request, res: Response) => {
  const {
    subject     = "Unknown",
    examType    = "see",
    mode        = "turbo",
    maxConcepts = 10,
    materialIds = [],
  } = req.body;

  // Resolve file paths from material registry
  const filePaths: string[] = materialIds
    .map((id: string) => (materialRegistry.get(id) as any)?.storedPath)
    .filter(Boolean);

  if (filePaths.length === 0) {
    res.status(400).json({ error: "No valid materials selected." });
    return;
  }

  const jobId = randomUUID();
  jobStore.set(jobId, {
    status:   "queued",
    progress: 0,
    result:   null,
    error:    null,
    logs:     [],
  });

  res.status(202).json({ jobId, status: "queued" });

  // Run AION async
  const job = jobStore.get(jobId)!;
  job.status = "running";

  runAION(
    filePaths, subject, examType, mode, maxConcepts,
    (line) => {
      job.logs.push(line);
      // Parse progress from AION log lines
      const match = line.match(/\[MODULE (\d+)\]/);
      if (match) {
        job.progress = parseInt(match[1]) / 5; // assumes 5 modules
      }
    },
    (result) => {
      job.status   = "done";
      job.progress = 1.0;
      job.result   = result;
    },
    (err) => {
      job.status = "failed";
      job.error  = err;
    }
  );
});

// ── POST /generate/stream (SSE) ──────────────────────────
router.post("/generate/stream", (req: Request, res: Response) => {
  const {
    subject     = "Unknown",
    examType    = "see",
    mode        = "turbo",
    maxConcepts = 10,
    materialIds = [],
  } = req.body;

  const filePaths: string[] = materialIds
    .map((id: string) => (materialRegistry.get(id) as any)?.storedPath)
    .filter(Boolean);

  if (filePaths.length === 0) {
    res.status(400).json({ error: "No valid materials selected." });
    return;
  }

  // SSE headers
  res.setHeader("Content-Type",  "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection",    "keep-alive");
  res.flushHeaders();

  const send = (event: string, data: object) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  send("status", { status: "started" });

  runAION(
    filePaths, subject, examType, mode, maxConcepts,
    (line) => {
      send("log", { message: line });
    },
    (result) => {
      send("result", { paper: result });
      send("done",   { status: "done" });
      res.end();
    },
    (err) => {
      send("error",  { message: err });
      res.end();
    }
  );

  req.on("close", () => res.end());
});

// ── GET /generate/status/:jobId ──────────────────────────
router.get("/generate/status/:jobId", (req: Request, res: Response) => {
  const job = jobStore.get(req.params.jobId as string);
  if (!job) {
    res.status(404).json({ error: "Job not found" });
    return;
  }
  res.json({
    jobId:    req.params.jobId,
    status:   job.status,
    progress: job.progress,
    result:   job.result,
    error:    job.error,
  });
});

export default router;
