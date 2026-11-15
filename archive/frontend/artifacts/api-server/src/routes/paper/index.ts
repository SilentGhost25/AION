import { Router, type IRouter } from "express";
import { GeneratePaperBody, GeneratePaperResponse } from "@workspace/api-zod";
import OpenAI from "openai";

const router: IRouter = Router();

function getOpenAI(): OpenAI {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY environment variable is not set.");
  }
  return new OpenAI({ apiKey });
}

router.post("/paper/generate", async (req, res): Promise<void> => {
  const parsed = GeneratePaperBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }

  const { config, sections } = parsed.data;

  const examLabel =
    config.examType === "IAT1"
      ? "First Internal Assessment Test (IAT-1)"
      : config.examType === "IAT2"
        ? "Second Internal Assessment Test (IAT-2)"
        : "Semester End Examination";

  const systemPrompt = `You are an expert academic question paper setter for ${config.institutionDepartment} at Dayananda Sagar Academy of Technology & Management (DSATM), an autonomous institute under VTU.

You are generating questions for:
- Subject: ${config.subjectName} (Code: ${config.subjectCode})
- Exam: ${examLabel}
- Semester: ${config.semester}
- Max Marks: ${config.maxMarks}
- Duration: ${config.duration}

You must generate exactly 10 questions (2 per section, as OR alternatives) following Bloom's Taxonomy (RBT levels L1-L6) and map each to Course Outcomes (CO1-CO5).

CRITICAL: Return ONLY a valid JSON object, no markdown, no explanation, just the JSON.`;

  const sectionPrompts = sections
    .map((section) => {
      return `Question ${section.sectionNumber}:
Marks: ${section.marks}
Sub-questions required: ${section.subQuestionsPerQ ?? 1}
Notes/Reference Material:
${section.notesText || "(No notes provided — generate a relevant question based on the subject)"}`;
    })
    .join("\n\n");

  const userPrompt = `Generate exactly 10 questions for the following sections:

${sectionPrompts}

Return a JSON object with this exact structure:
{
  "questions": [
    {
      "qNo": 1,
      "text": "Question text here",
      "marks": 10,
      "co": "CO1",
      "rbt": "L3",
      "sectionNumber": 1,
      "isOrQuestion": false
    },
    {
      "qNo": 2,
      "text": "OR alternative question text here",
      "marks": 10,
      "co": "CO1",
      "rbt": "L2",
      "sectionNumber": 1,
      "isOrQuestion": true
    }
  ],
  "courseOutcomes": [
    "Understand fundamental concepts of ${config.subjectName}",
    "Apply core algorithms and techniques in ${config.subjectName}",
    "Analyze and evaluate solutions for problems in ${config.subjectName}",
    "Design and implement solutions using ${config.subjectName} concepts",
    "Synthesize knowledge to build real-world applications"
  ],
  "coCoverage": {
    "co1": 20,
    "co2": 20,
    "co3": 20,
    "co4": 20,
    "co5": 20
  },
  "syllabusCoverage": {
    "s1": 20,
    "s2": 20,
    "s3": 20,
    "s4": 20,
    "s5": 20
  }
}

Rules:
- Generate exactly 10 questions: questions 1-2 (section 1), 3-4 (section 2), 5-6 (section 3), 7-8 (section 4), 9-10 (section 5)
- Odd-numbered questions (1,3,5,7,9): isOrQuestion = false
- Even-numbered questions (2,4,6,8,10): isOrQuestion = true
- Marks for each question = the section marks value provided
- Distribute CO1-CO5 across questions appropriately
- Use RBT levels L1-L6 (Remember, Understand, Apply, Analyze, Evaluate, Create) — vary them
- Questions must be academically rigorous and relevant to the subject and notes content
- CO coverage percentages must sum to 100
- Syllabus coverage percentages must sum to 100
- Return ONLY valid JSON, no markdown fences, no extra text`;

  let rawContent = "";
  try {
    const client = getOpenAI();
    const response = await client.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 4096,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
    });

    rawContent = response.choices[0]?.message?.content ?? "";

    // Strip markdown fences if present
    const jsonMatch = rawContent.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      req.log.error({ rawContent }, "No JSON found in AI response");
      res.status(500).json({ error: "AI returned invalid response format" });
      return;
    }

    const aiData = JSON.parse(jsonMatch[0]) as {
      questions: unknown[];
      courseOutcomes: unknown[];
      coCoverage: unknown;
      syllabusCoverage: unknown;
    };

    const paper = {
      config,
      questions: aiData.questions,
      courseOutcomes: aiData.courseOutcomes,
      coCoverage: aiData.coCoverage,
      syllabusCoverage: aiData.syllabusCoverage,
    };

    const validated = GeneratePaperResponse.safeParse(paper);
    if (!validated.success) {
      req.log.error(
        { error: validated.error.message, paper },
        "Generated paper failed validation",
      );
      res.status(500).json({ error: "Generated paper structure is invalid" });
      return;
    }

    res.json(validated.data);
  } catch (err) {
    req.log.error({ err, rawContent }, "Error generating paper");
    res.status(500).json({ error: "Failed to generate question paper" });
  }
});

export default router;
