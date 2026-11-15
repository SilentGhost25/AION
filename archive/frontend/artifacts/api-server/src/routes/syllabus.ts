import { Router } from "express";

const router = Router();

// Mirrors constants/co-bloom.ts from qp-maker frontend
const SYLLABUS_MAP: Record<string, object> = {
  default: {
    modules: [
      { moduleIndex: 1, co: "CO1", bloomLevels: [1, 2], topics: ["Introduction", "Fundamentals"] },
      { moduleIndex: 2, co: "CO2", bloomLevels: [3],    topics: ["Application", "Implementation"] },
      { moduleIndex: 3, co: "CO3", bloomLevels: [4],    topics: ["Analysis", "Comparison"] },
      { moduleIndex: 4, co: "CO3", bloomLevels: [4],    topics: ["Advanced Analysis"] },
      { moduleIndex: 5, co: "CO2", bloomLevels: [3],    topics: ["Applied Problems"] },
    ],
  },
};

router.get("/syllabus", (req, res) => {
  const subject = (req.query.subject as string) ?? "default";
  const map = SYLLABUS_MAP[subject] ?? SYLLABUS_MAP["default"];
  res.json({ subject, ...map });
});

export default router;
