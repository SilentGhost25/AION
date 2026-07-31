import { Router } from "express";

const router = Router();

router.get("/healthz", async (_req, res) => {
  let ollamaOk = false;
  let model = process.env.AION_MODEL ?? "qwen2.5:3b";

  try {
    const response = await fetch("http://localhost:11434/api/tags", {
      signal: AbortSignal.timeout(3000),
    });
    if (response.ok) {
      const data = await response.json() as { models: { name: string }[] };
      ollamaOk = data.models.some((m) => m.name.includes(model));
    }
  } catch {
    ollamaOk = false;
  }

  res.json({
    status: ollamaOk ? "ok" : "degraded",
    ollama: ollamaOk,
    model,
    version: "0.1.0",
  });
});

export default router;
