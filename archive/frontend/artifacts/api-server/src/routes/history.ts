import { Router } from "express";
import fs from "fs";
import path from "path";
import { randomUUID } from "crypto";

const router = Router();

const HISTORY_DIR = path.resolve(
  process.env.AION_HISTORY ?? "../../workspace/history"
);
fs.mkdirSync(HISTORY_DIR, { recursive: true });

// In-memory index (replace with DB in production)
const paperIndex = new Map<string, object>();

export function savePaper(paper: any) {
  const id   = randomUUID();
  const meta = {
    id,
    subject:     paper.subject     ?? "Unknown",
    examType:    paper.examType    ?? "see",
    generatedAt: new Date().toISOString(),
    totalMarks:  paper.totalMarks  ?? 0,
    mode:        paper.mode        ?? "turbo",
  };

  paperIndex.set(id, meta);
  fs.writeFileSync(
    path.join(HISTORY_DIR, `${id}.json`),
    JSON.stringify({ ...meta, ...paper }, null, 2),
    "utf-8"
  );

  return id;
}

router.get("/history", (req, res) => {
  const { subject, examType } = req.query;
  let results = [...paperIndex.values()] as any[];

  if (subject)  results = results.filter((p) => p.subject === subject);
  if (examType) results = results.filter((p) => p.examType === examType);

  results.sort((a, b) =>
    new Date(b.generatedAt).getTime() - new Date(a.generatedAt).getTime()
  );
  res.json(results);
});

router.get("/history/:id", (req, res) => {
  const file = path.join(HISTORY_DIR, `${req.params.id}.json`);
  if (!fs.existsSync(file)) {
    res.status(404).json({ error: "Not found" });
    return;
  }
  res.json(JSON.parse(fs.readFileSync(file, "utf-8")));
});

router.delete("/history/:id", (req, res) => {
  const file = path.join(HISTORY_DIR, `${req.params.id}.json`);
  paperIndex.delete(req.params.id);
  try { fs.unlinkSync(file); } catch { /* already gone */ }
  res.json({ ok: true });
});

export default router;
