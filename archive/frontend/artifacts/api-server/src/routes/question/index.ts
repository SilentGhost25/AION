import { Router, type IRouter } from "express";
import OpenAI from "openai";

const router: IRouter = Router();

function getOpenAI(): OpenAI {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error("OPENAI_API_KEY environment variable is not set.");
  return new OpenAI({ apiKey });
}

router.post("/question/preview", async (req, res): Promise<void> => {
  const body = req.body as {
    config?: {
      institutionDepartment: string;
      subjectName: string;
      subjectCode: string;
      examType: string;
      semester: number;
      maxMarks: number;
      duration: string;
    };
    section?: {
      sectionNumber: number;
      mainQuestions: number;
      subQuestionsPerQ: number;
      marks: number;
      notesText: string;
    };
  };

  if (!body.config || !body.section) {
    res.status(400).json({ error: "config and section are required" });
    return;
  }

  const { config, section } = body;
  const isOrQ = section.sectionNumber % 2 === 0;

  const prompt = `You are an expert academic question setter at DSATM (Dayananda Sagar Academy of Technology & Management) under VTU.

Subject: ${config.subjectName} (${config.subjectCode})
Department: ${config.institutionDepartment}
Exam: ${config.examType} | Semester: ${config.semester} | Marks: ${section.marks}
Question Number: ${section.sectionNumber} (${isOrQ ? "OR alternative — must be different from its pair" : "main question"})
Sub-questions required: ${section.subQuestionsPerQ}

Reference Material (use this to derive the question):
${section.notesText || "(No notes — generate a relevant question based on the subject)"}

Generate ONE academically rigorous question worth ${section.marks} marks with exactly ${section.subQuestionsPerQ} sub-part(s).
${section.subQuestionsPerQ > 1 ? `Label sub-parts as a), b), c)… and distribute marks so they sum to ${section.marks}.` : ""}

Assign an appropriate Course Outcome (CO1–CO5) and Bloom's Taxonomy RBT level (L1–L6).

Return ONLY valid JSON — no markdown fences, no extra text:
{
  "question": "Full question text here",
  "co": "CO2",
  "rbt": "L3"
}`;

  let rawContent = "";
  try {
    const client = getOpenAI();
    const response = await client.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 1024,
      messages: [{ role: "user", content: prompt }],
    });

    rawContent = response.choices[0]?.message?.content ?? "";
    const jsonMatch = rawContent.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      res.status(500).json({ error: "AI returned invalid response" });
      return;
    }

    const data = JSON.parse(jsonMatch[0]) as {
      question?: string;
      co?: string;
      rbt?: string;
    };

    res.json({
      question: data.question ?? "",
      co: data.co ?? "CO1",
      rbt: data.rbt ?? "L3",
    });
  } catch (err) {
    req.log.error({ err, rawContent }, "Error previewing question");
    res.status(500).json({ error: "Failed to generate question preview" });
  }
});

export default router;
