import { Router } from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import { randomUUID } from "crypto";

const router = Router();

// Store uploads in AION workspace
const UPLOAD_DIR = path.resolve(
  process.env.AION_WORKSPACE ?? "../../workspace/uploads"
);
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `${randomUUID()}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB
  fileFilter: (_req, file, cb) => {
    const allowed = [".pdf", ".txt", ".docx", ".pptx", ".md"];
    const ext = path.extname(file.originalname).toLowerCase();
    cb(null, allowed.includes(ext));
  },
});

// In-memory registry (replace with DB in production)
const materialRegistry = new Map<string, object>();

router.get("/materials", (req, res) => {
  const { subject } = req.query;
  const all = [...materialRegistry.values()];
  const filtered = subject
    ? all.filter((m: any) => m.subject === subject)
    : all;
  res.json(filtered);
});

router.post(
  "/materials",
  upload.single("file"),
  (req, res) => {
    if (!req.file) {
      res.status(400).json({ error: "No file uploaded" });
      return;
    }

    const material = {
      id:         randomUUID(),
      filename:   req.file.originalname,
      storedName: req.file.filename,
      storedPath: req.file.path,
      subject:    req.body.subject ?? "unknown",
      category:   req.body.category ?? "notes",
      uploadedAt: new Date().toISOString(),
      sizeBytes:  req.file.size,
      wordCount:  0,
    };

    materialRegistry.set(material.id, material);
    res.status(201).json(material);
  }
);

router.delete("/materials/:id", (req, res) => {
  const mat = materialRegistry.get(req.params.id) as any;
  if (!mat) {
    res.status(404).json({ error: "Not found" });
    return;
  }

  try {
    fs.unlinkSync(mat.storedPath);
  } catch { /* file already gone */ }

  materialRegistry.delete(req.params.id);
  res.json({ ok: true });
});

export { materialRegistry };
export default router;
