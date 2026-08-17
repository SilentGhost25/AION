import { useState, useRef } from "react"
import { PaperConfig, SectionInput, GeneratedPaper, GeneratedQuestion } from "@workspace/api-client-react"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import {
  UploadCloud, CheckCircle2, FileText, ArrowLeft, Wand2,
  Info, FileWarning, Sparkles, Loader2, RefreshCw, ImagePlus, X, ListPlus
} from "lucide-react"
import { toast } from "sonner"
import { Spinner } from "@/components/ui/spinner"
import { aionAPI } from "@/lib/aion-api"

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SubQuestion {
  label: string
  text: string
  marks: number
  co: string
  rbt: string
  parts?: string[]
  images?: string[]
}

interface QuestionPreview {
  subQuestions: SubQuestion[]
}

export type QuestionWithImages = GeneratedQuestion & { images?: string[] }
export type QuestionWithSubs = GeneratedQuestion & {
  subQuestions?: SubQuestion[]
  images?: string[]
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function splitMarks(total: number, count: number): number[] {
  const n = Math.max(1, count)
  const base = Math.floor(total / n)
  const remainder = total - base * n
  return Array.from({ length: n }, (_, i) => base + (i < remainder ? 1 : 0))
}

const RBT_LEVELS = ["L1", "L2", "L3", "L4", "L5"]

function extractTopic(notes: string): string {
  const cleaned = (notes ?? "").trim().replace(/\s+/g, " ")
  if (!cleaned) return "the given topic"
  const firstSentence = cleaned.split(/[.\n]/)[0].trim()
  const words = firstSentence.split(" ").slice(0, 8).join(" ")
  return words.length > 0 ? words : "the given topic"
}

function toRoman(n: number): string {
  const map: [number, string][] = [
    [10, "x"], [9, "ix"], [5, "v"], [4, "iv"], [1, "i"]
  ]
  let out = "", rem = n
  for (const [val, sym] of map) {
    while (rem >= val) { out += sym; rem -= val }
  }
  return out
}

function uniq(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)))
}

function flattenPreview(preview: QuestionPreview): {
  text: string; co: string; rbt: string; images: string[]
} {
  const subs = preview.subQuestions
  const multi = subs.length > 1
  const text = subs.map(s => {
    const head = multi ? `${s.label}) ${s.text}` : s.text
    const parts = (s.parts ?? []).map((p, i) => `   ${toRoman(i + 1)}) ${p}`).join("\n")
    return parts ? `${head}\n${parts}` : head
  }).join("\n")
  return {
    text,
    co: uniq(subs.map(s => s.co)).join(", "),
    rbt: uniq(subs.map(s => s.rbt)).join(", "),
    images: subs.flatMap(s => s.images ?? []),
  }
}

// ── Parse AION backend response into SubQuestion[] ────────────────────────────

function parseAIONResponse(
  raw: string,
  section: SectionInput,
  qNum: number,
): SubQuestion[] {
  const subCount  = Math.max(1, section.subQuestionsPerQ || 1)
  const marksSplit = splitMarks(section.marks, subCount)

  // Try JSON parse first
  try {
    const stripped = raw
      .replace(/```json/gi, "")
      .replace(/```/g, "")
      .trim()

    const parsed = JSON.parse(stripped)

    // Shape: { question, sub_questions: [{part,marks,text,bloom}] }
    if (parsed.sub_questions && Array.isArray(parsed.sub_questions)) {
      return parsed.sub_questions.map((sq: any, i: number) => ({
        label: sq.part ?? String.fromCharCode(97 + i),
        text:  sq.text ?? parsed.question ?? raw,
        marks: sq.marks ?? marksSplit[i] ?? section.marks,
        co:    `CO${Math.min(5, Math.ceil(qNum / 2))}`,
        rbt:   sq.bloom ?? RBT_LEVELS[(qNum - 1 + i) % RBT_LEVELS.length],
      }))
    }

    // Shape: { question } — single question, split into parts
    if (parsed.question) {
      return splitIntoSubs(parsed.question, subCount, marksSplit, qNum)
    }
  } catch {
    // Not JSON — treat as plain text question
  }

  // Plain text — split into sub-questions
  return splitIntoSubs(raw.trim(), subCount, marksSplit, qNum)
}

function splitIntoSubs(
  text: string,
  subCount: number,
  marksSplit: number[],
  qNum: number,
): SubQuestion[] {
  if (subCount === 1) {
    return [{
      label: "a",
      text:  text,
      marks: marksSplit[0] ?? 10,
      co:    `CO${Math.min(5, Math.ceil(qNum / 2))}`,
      rbt:   RBT_LEVELS[(qNum - 1) % RBT_LEVELS.length],
    }]
  }

  // Split on sentence boundaries
  const sentences = text.split(/(?<=[.?])\s+/).filter(s => s.trim().length > 10)
  const chunkSize = Math.ceil(sentences.length / subCount)

  return Array.from({ length: subCount }, (_, i) => {
    const chunk = sentences.slice(i * chunkSize, (i + 1) * chunkSize).join(" ")
    return {
      label: String.fromCharCode(97 + i),
      text:  chunk || text,
      marks: marksSplit[i] ?? Math.floor(10 / subCount),
      co:    `CO${Math.min(5, Math.ceil(qNum / 2))}`,
      rbt:   RBT_LEVELS[(qNum - 1 + i) % RBT_LEVELS.length],
    }
  })
}

// ── Upload file to AION backend ───────────────────────────────────────────────

async function uploadToAION(
  file: File,
  subject: string,
): Promise<string> {
  const result = await aionAPI.upload(file, subject, "notes")
  return result.id as string
}

// ── Generate one question via AION backend ────────────────────────────────────

async function generateFromAION(
  notesText: string,
  subject: string,
  examType: string,
  qNum: number,
  marks: number,
  subCount: number,
): Promise<string> {
  // Upload text as a .txt file
  const blob = new Blob([notesText], { type: "text/plain" })
  const file = new File([blob], `q${qNum}_notes.txt`, { type: "text/plain" })

  const fileId = await uploadToAION(file, subject)

  // Call generate stream and collect SSE events
  const examTypeNorm = examType.startsWith("IAT") ? "IA" : examType

  return new Promise((resolve, reject) => {
    const payload = {
      file_id:       fileId,
      fileId:        fileId,
      subject:       subject,
      exam_type:     examTypeNorm,
      examType:      examTypeNorm,
      mode:          "turbo",
      difficulty:    "mixed",
      maxConcepts:   10,
      includeVisual: false,
    }

    aionAPI.generateStream(payload)
      .then(async (response) => {
        if (!response.ok) {
          const err = await response.json().catch(() => ({}))
          reject(new Error(err.error ?? `HTTP ${response.status}`))
          return
        }

        const reader  = response.body!.getReader()
        const decoder = new TextDecoder()
        let   buffer  = ""
        let   result  = ""

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split("\n")
          buffer = lines.pop() ?? ""

          let currentEvent = ""
          for (const line of lines) {
            const trimmed = line.trim()

            if (trimmed.startsWith("event:")) {
              currentEvent = trimmed.slice(6).trim()
              continue
            }

            if (!trimmed.startsWith("data:")) continue
            const raw = trimmed.slice(5).trim()
            if (!raw || raw === "[DONE]") continue

            try {
              const data = JSON.parse(raw)

              if (currentEvent === "paper_ready" || currentEvent === "result" || (data.modules && Array.isArray(data.modules)) || (data.paper && data.paper.modules)) {
                const paperObj = data.paper || data
                ;(window as any).__aionLastPaper = paperObj
                const firstSub = paperObj.modules?.[0]?.questions?.[0]?.subQuestions?.[0] || paperObj.modules?.[0]?.questions?.[0]?.sub_questions?.[0]
                result = firstSub?.text || "Paper generated. Click Generate Paper to continue."
                currentEvent = ""
                continue
              }

              if (currentEvent === "done" || data.status === "done") {
                currentEvent = "done"
                break
              }

              if (currentEvent === "error" || (data.message && data.status === "error")) {
                throw new Error(data.message || "Generation failed")
              }

              if (data.chunk)    result += data.chunk
              if (data.text)     result  = data.text
              if (data.question) result  = data.question

            } catch (parseErr) {
              if (parseErr instanceof Error && !parseErr.message.includes("JSON")) {
                throw parseErr
              }
            }
          }
          if (currentEvent === "done") break
        }


        resolve(result || "Question generation completed. Please review and edit.")
      })
      .catch(reject)
  })
}

// ── Step2Rules Component ──────────────────────────────────────────────────────

interface Step2RulesProps {
  config: PaperConfig
  sections: SectionInput[]
  setSections: (sections: SectionInput[]) => void
  onBack: () => void
  onSuccess: (paper: GeneratedPaper) => void
}

export function Step2Rules({
  config, sections, setSections, onBack, onSuccess
}: Step2RulesProps) {
  const [previews, setPreviews] = useState<(QuestionPreview | null)[]>(
    () => sections.map(() => null)
  )
  const [isBuilding, setIsBuilding] = useState(false)
  const [buildStatus, setBuildStatus] = useState("")

  const pairMarksValid = [0, 1, 2, 3, 4].every(pair => {
    const a = sections[pair * 2]
    const b = sections[pair * 2 + 1]
    return a && b && a.marks === b.marks
  })
  const pairSum = [0, 2, 4, 6, 8].reduce(
    (acc, i) => acc + (sections[i]?.marks ?? 0), 0
  )
  const isTotalValid = pairSum === config.maxMarks && pairMarksValid

  const setPreview = (idx: number, preview: QuestionPreview | null) => {
    setPreviews(prev => {
      const next = [...prev]; next[idx] = preview; return next
    })
  }

  const handleGenerate = async () => {
    if (!isTotalValid) {
      toast.error("Marks mismatch", {
        description: `Each OR pair must have equal marks and sum to ${config.maxMarks}.`,
      })
      return
    }

    const missingNotes = sections.some(
      s => !s.notesText || s.notesText.trim() === ""
    )
    if (missingNotes) {
      toast.error("Missing reference material", {
        description: "Provide reference material for all 10 questions.",
      })
      return
    }

    setIsBuilding(true)

    // ── Use AION-generated paper if available ─────────────────────────────
    const aionPaper = (window as any).__aionLastPaper
    if (aionPaper?.modules?.length > 0) {
      try {
        const questions = aionPaper.modules.flatMap((mod: any, modIdx: number) =>
          (mod.questions ?? []).map((q: any) => ({
            qNo:           q.mqIndex ?? q.qNo ?? (modIdx * 2 + 1),
            text:          q.subQuestions?.[0]?.text ?? "",
            marks:         q.totalMarks ?? 10,
            co:            q.subQuestions?.[0]?.co ?? `CO${modIdx + 1}`,
            rbt:           `L${q.bloomLevel ?? 2}`,
            sectionNumber: modIdx + 1,
            isOrQuestion:  q.isOr ?? false,
            subQuestions:  (q.subQuestions ?? []).map((sq: any) => ({
              label: sq.letter ?? "a",
              text:  sq.text ?? "",
              marks: sq.marks ?? 5,
              co:    sq.co ?? `CO${modIdx + 1}`,
              rbt:   `L${sq.bloom ?? 2}`,
            })),
          }))
        )
        const paper: GeneratedPaper = {
          config,
          questions,
          courseOutcomes: [1,2,3,4,5].map(i =>
            `Understand and apply Module ${i} concepts of ${config.subjectName || "the subject"}.`
          ),
          coCoverage:       { co1: 20, co2: 20, co3: 20, co4: 20, co5: 20 },
          syllabusCoverage: { s1: 20, s2: 20, s3: 20, s4: 20, s5: 20 },
        }

        // DOM Count & Question ID Verification (🔴 11, 🔴 12 & Change 2)
        const totalSubQuestionsCount = questions.reduce((acc: number, q: any) => acc + (q.subQuestions?.length || 0), 0)
        const parsedQuestionIds = questions.flatMap((q: any) => 
          (q.subQuestions || []).map((sq: any) => `module_${q.sectionNumber}_Q${q.qNo}_${sq.label}`)
        )
        const backendQuestionIds = aionPaper.integrity?.question_ids || []
        const idsMatch = backendQuestionIds.length === parsedQuestionIds.length && 
          backendQuestionIds.every((id: string, idx: number) => id === parsedQuestionIds[idx])

        console.log(`[AION INTEGRITY] Backend Count: ${aionPaper.integrity?.question_count}, Frontend Assembled Count: ${totalSubQuestionsCount}`)
        console.log(`[AION INTEGRITY] Backend IDs:`, backendQuestionIds)
        console.log(`[AION INTEGRITY] Frontend IDs:`, parsedQuestionIds)
        console.log(`[AION INTEGRITY] Checksum Hash: ${aionPaper.integrity?.canonical_hash}`)
        
        if (aionPaper.integrity && aionPaper.integrity.question_count !== totalSubQuestionsCount) {
          console.error("[AION INTEGRITY FAILURE] Question count mismatch between backend and frontend!")
          toast.error("Integrity Mismatch Detected", {
            description: `Authoritative count (${aionPaper.integrity.question_count}) differs from parsed count (${totalSubQuestionsCount}).`
          })
        } else if (backendQuestionIds.length > 0 && !idsMatch) {
          console.error("[AION INTEGRITY FAILURE] Question ID list mismatch between backend and frontend!")
          toast.error("Integrity Mismatch Detected", {
            description: `Authoritative question IDs differ from assembled question IDs.`
          })
        } else {
          console.log("[AION INTEGRITY SUCCESS] All counts and IDs match perfectly.")
        }

        delete (window as any).__aionLastPaper
        setIsBuilding(false)
        setBuildStatus("")
        onSuccess(paper)
        return
      } catch (err) {
        console.warn("AION paper assembly failed, falling back:", err)
        delete (window as any).__aionLastPaper
      }
    }
    setBuildStatus("Assembling paper...")

    try {
      const questions: QuestionWithSubs[] = sections.map((section, idx) => {
        const qNum    = idx + 1
        const preview = previews[idx]

        if (preview) {
          const { text, co, rbt, images } = flattenPreview(preview)
          return {
            qNo:          qNum,
            text,
            marks:        section.marks,
            co,
            rbt,
            sectionNumber: section.sectionNumber,
            isOrQuestion:  idx % 2 === 1,
            subQuestions:  preview.subQuestions,
            images:        images.length > 0 ? images : undefined,
          }
        }

        // Fallback for blocks not yet previewed
        const fallbackText = extractTopic(section.notesText)
        return {
          qNo:          qNum,
          text:         `Explain ${fallbackText} with relevant examples.`,
          marks:        section.marks,
          co:           `CO${Math.min(5, Math.ceil(qNum / 2))}`,
          rbt:          RBT_LEVELS[(qNum - 1) % RBT_LEVELS.length],
          sectionNumber: section.sectionNumber,
          isOrQuestion:  idx % 2 === 1,
          subQuestions:  [{
            label: "a",
            text:  `Explain ${fallbackText} with relevant examples.`,
            marks: section.marks,
            co:    `CO${Math.min(5, Math.ceil(qNum / 2))}`,
            rbt:   RBT_LEVELS[(qNum - 1) % RBT_LEVELS.length],
          }],
        }
      })

      const courseOutcomes = [0, 1, 2, 3, 4].map(
        i => `Understand and apply the concepts of ${extractTopic(sections[i * 2]?.notesText ?? "")}.`
      )

      const paper: GeneratedPaper = {
        config,
        questions,
        courseOutcomes,
        coCoverage:       { co1: 20, co2: 20, co3: 20, co4: 20, co5: 20 },
        syllabusCoverage: { s1: 20,  s2: 20,  s3: 20,  s4: 20,  s5: 20  },
      }

      setBuildStatus("")
      setIsBuilding(false)
      onSuccess(paper)

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to build paper"
      toast.error("Paper generation failed", { description: msg })
      setIsBuilding(false)
      setBuildStatus("")
    }
  }

  const updateSection = (idx: number, updates: Partial<SectionInput>) => {
    const newSections = [...sections]
    newSections[idx]  = { ...newSections[idx], ...updates }
    setSections(newSections)
  }

  return (
    <div className="max-w-3xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500 pb-24">
      <div className="mb-8">
        <div className="flex items-center gap-2 text-primary mb-2">
          <Wand2 className="h-5 w-5" />
          <span className="font-semibold uppercase tracking-wider text-xs">
            Step 2 of 3
          </span>
        </div>
        <h2 className="text-3xl font-bold text-slate-800 mb-2">Scope & Rules</h2>
        <p className="text-slate-600">
          Upload or paste reference material for each question.
          Click <strong>Generate Questions</strong> to call AION and get AI-generated questions.
          Edit any result before building the final paper.
        </p>
      </div>

      <div className="space-y-4">
        {sections.map((section, idx) => {
          const isOrQuestion = idx % 2 === 1
          return (
            <div key={idx}>
              {isOrQuestion && (
                <div className="flex items-center gap-3 my-1 px-1">
                  <div className="h-px flex-1 bg-slate-200" />
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">
                    OR
                  </span>
                  <div className="h-px flex-1 bg-slate-200" />
                </div>
              )}
              <QuestionCard
                qNum={idx + 1}
                isOrQuestion={isOrQuestion}
                section={section}
                preview={previews[idx]}
                config={config}
                onPreviewChange={(p) => setPreview(idx, p)}
                updateSection={(u) => updateSection(idx, u)}
              />
              {isOrQuestion && idx < 9 && <div className="h-4" />}
            </div>
          )
        })}
      </div>

      {/* Floating Bottom Bar */}
      <div className="fixed bottom-0 left-0 right-0 bg-white/80 backdrop-blur-md border-t p-4 shadow-[0_-4px_20px_rgba(0,0,0,0.05)] z-10 print:hidden">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <Button variant="ghost" onClick={onBack} disabled={isBuilding}>
            <ArrowLeft className="mr-2 h-4 w-4" /> Back
          </Button>

          <div className="flex items-center gap-6">
            <div className={`flex items-center gap-2 text-sm font-medium ${isTotalValid ? "text-emerald-600" : "text-amber-600"}`}>
              <Info className="h-4 w-4" />
              {pairSum} / {config.maxMarks} marks
            </div>

            <Button
              size="lg"
              onClick={handleGenerate}
              disabled={isBuilding || !isTotalValid}
              className="bg-primary hover:bg-primary/90 text-white min-w-[220px]"
            >
              {isBuilding ? (
                <>
                  <Spinner className="mr-2 h-4 w-4" />
                  {buildStatus || "Building Paper..."}
                </>
              ) : (
                <>
                  Generate Question Paper
                  <Wand2 className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── QuestionCard ──────────────────────────────────────────────────────────────

function QuestionCard({
  qNum, isOrQuestion, section, preview, config,
  onPreviewChange, updateSection,
}: {
  qNum: number
  isOrQuestion: boolean
  section: SectionInput
  preview: QuestionPreview | null
  config: PaperConfig
  onPreviewChange: (preview: QuestionPreview | null) => void
  updateSection: (updates: Partial<SectionInput>) => void
}) {
  const [fileName, setFileName]       = useState<string | null>(null)
  const [showTextarea, setShowTextarea] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [genError, setGenError]         = useState<string | null>(null)
  const [genStatus, setGenStatus]       = useState("")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const hasContent = Boolean(section.notesText && section.notesText.trim() !== "")

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setFileName(file.name)
    const ext = file.name.split(".").pop()?.toLowerCase()

    if (ext === "docx") {
      setShowTextarea(true)
      updateSection({ notesText: "" })
    } else {
      const reader = new FileReader()
      reader.onload = (event) => {
        updateSection({ notesText: event.target?.result as string })
        setShowTextarea(true)
      }
      reader.readAsText(file)
    }
  }

  // ── Generate via AION backend ───────────────────────────────────────────────
  const handleGenerateAll = async () => {
    if (!hasContent) return
    setIsGenerating(true)
    setGenError(null)
    setGenStatus("Uploading to AION...")

    try {
      const subject     = config.subjectName || "General"
      const examType    = config.examType    || "IA"
      const subCount    = Math.max(1, section.subQuestionsPerQ || 1)

      setGenStatus("Generating with AI...")

      const rawResponse = await generateFromAION(
        section.notesText,
        subject,
        examType,
        qNum,
        section.marks,
        subCount,
      )

      const subs = parseAIONResponse(rawResponse, section, qNum)

      // Preserve existing images
      const merged: QuestionPreview = {
        subQuestions: subs.map((sq, i) => ({
          ...sq,
          images: preview?.subQuestions[i]?.images,
        })),
      }

      onPreviewChange(merged)
      setGenStatus("")
      toast.success(`Q${qNum} generated by AION`)

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Generation failed"
      setGenError(msg)
      setGenStatus("")
      toast.error("AION generation failed", {
        description: msg + " — check that aion_api.py is running on port 8100.",
      })
    } finally {
      setIsGenerating(false)
    }
  }

  const updateSub = (subIndex: number, updates: Partial<SubQuestion>) => {
    if (!preview) return
    const subQuestions = preview.subQuestions.map((sq, i) =>
      i === subIndex ? { ...sq, ...updates } : sq
    )
    onPreviewChange({ subQuestions })
  }

  const regenerateSub = async (subIndex: number) => {
    if (!preview || !hasContent) return
    setIsGenerating(true)
    try {
      const rawResponse = await generateFromAION(
        section.notesText,
        config.subjectName || "General",
        config.examType    || "IA",
        qNum,
        preview.subQuestions[subIndex]?.marks ?? section.marks,
        1,
      )
      const [fresh] = parseAIONResponse(rawResponse, section, qNum)
      if (fresh) updateSub(subIndex, { text: fresh.text, co: fresh.co, rbt: fresh.rbt })
    } catch {
      toast.error("Regeneration failed")
    } finally {
      setIsGenerating(false)
    }
  }

  const subMarksTotal  = preview?.subQuestions.reduce((acc, s) => acc + (s.marks || 0), 0) ?? 0
  const marksBalanced  = !preview || subMarksTotal === section.marks

  return (
    <Card className="shadow-sm border-slate-200 overflow-hidden">
      <div className="bg-slate-50 border-b px-5 py-3 flex items-center justify-between flex-wrap gap-2">
        <h3 className="font-semibold text-slate-800 flex items-center gap-2">
          Question {qNum}
          {isOrQuestion && (
            <Badge variant="outline" className="text-[10px] border-slate-300 text-slate-500 font-normal">
              OR alternative
            </Badge>
          )}
        </h3>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Label className="text-xs text-muted-foreground whitespace-nowrap">Sub-questions</Label>
            <Input type="number" min={1} max={3} className="w-16 h-7 text-sm"
              value={section.subQuestionsPerQ}
              onChange={e => updateSection({ subQuestionsPerQ: parseInt(e.target.value) || 1 })} />
          </div>
          <div className="flex items-center gap-2">
            <Label className="text-xs text-muted-foreground">Marks</Label>
            <Input type="number" min={1} className="w-20 h-7 text-sm"
              value={section.marks}
              onChange={e => updateSection({ marks: parseInt(e.target.value) || 0 })} />
          </div>
        </div>
      </div>

      <CardContent className="p-5 space-y-4">
        {/* Reference material */}
        <div className="space-y-3">
          <div className="flex items-center justify-between gap-3">
            <Label className="text-sm font-medium flex items-center gap-2">
              Reference Material
              {fileName && (
                <span className="text-emerald-600 flex items-center text-xs">
                  <CheckCircle2 className="w-3 h-3 mr-1" /> Uploaded
                </span>
              )}
            </Label>
            <label className="flex items-center gap-1.5 shrink-0">
              <span className="text-xs text-muted-foreground whitespace-nowrap">Sub-questions</span>
              <select
                className="h-7 rounded-md border border-input bg-background px-2 text-xs focus:outline-none focus:ring-1 focus:ring-ring"
                value={section.subQuestionsPerQ ?? 1}
                onChange={e => updateSection({ subQuestionsPerQ: parseInt(e.target.value) || 1 })}
              >
                {[1, 2, 3, 4, 5].map(n => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </label>
          </div>

          {!fileName && (
            <div onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-300 rounded-lg p-5 text-center cursor-pointer hover:bg-slate-50 transition-colors">
              <UploadCloud className="w-7 h-7 text-slate-400 mx-auto mb-2" />
              <p className="text-sm font-medium text-slate-700">Upload Notes & Reference Material</p>
              <p className="text-xs text-muted-foreground mt-1">Click to browse (PDF, DOCX, TXT)</p>
              <input type="file" ref={fileInputRef} className="hidden"
                accept=".txt,.pdf,.docx" onChange={handleFileUpload} />
            </div>
          )}

          {fileName && !showTextarea && (
            <div className="flex items-center justify-between p-3 border rounded-md bg-emerald-50/50 border-emerald-100">
              <div className="flex items-center gap-3">
                <FileText className="w-5 h-5 text-emerald-600" />
                <span className="text-sm font-medium text-slate-700">{fileName}</span>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setShowTextarea(true)} className="text-xs">
                View / Edit Text
              </Button>
            </div>
          )}

          {showTextarea && (
            <div className="space-y-2 animate-in fade-in slide-in-from-top-2">
              {fileName?.endsWith(".docx") && (
                <div className="flex items-start gap-2 text-amber-700 bg-amber-50 p-3 rounded-md text-xs border border-amber-200">
                  <FileWarning className="w-4 h-4 mt-0.5 shrink-0" />
                  <p>Word document parsing is limited in the browser. Paste your notes below.</p>
                </div>
              )}
              <Textarea placeholder="Paste reference text here..."
                className="min-h-[100px] font-mono text-xs bg-slate-50"
                value={section.notesText}
                onChange={e => updateSection({ notesText: e.target.value })} />
            </div>
          )}

          {!fileName && !showTextarea && (
            <button className="text-xs text-primary underline underline-offset-2 hover:opacity-70"
              onClick={() => setShowTextarea(true)}>
              Or type notes directly
            </button>
          )}
        </div>

        {/* Generate panel */}
        {hasContent && (
          <div className="border border-slate-200 rounded-lg overflow-hidden">
            <div className="bg-slate-50 border-b px-4 py-2 flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-medium text-slate-700">
                <Sparkles className="w-4 h-4 text-primary" />
                {genStatus || "Generate via AION AI"}
              </div>
              {preview && (
                <span className={`text-xs font-medium ${marksBalanced ? "text-emerald-600" : "text-amber-600"}`}>
                  {subMarksTotal} / {section.marks} marks
                </span>
              )}
              <Button size="sm" variant={preview ? "outline" : "default"}
                className="h-7 text-xs gap-1.5"
                onClick={handleGenerateAll} disabled={isGenerating}>
                {isGenerating ? (
                  <><Loader2 className="w-3 h-3 animate-spin" /> {genStatus || "Generating…"}</>
                ) : preview ? (
                  <><RefreshCw className="w-3 h-3" /> Regenerate all</>
                ) : (
                  <><Sparkles className="w-3 h-3" /> Generate Questions</>
                )}
              </Button>
            </div>

            <div className="p-4 space-y-3">
              {!isGenerating && !preview && !genError && (
                <p className="text-xs text-slate-400 italic">
                  Click <span className="font-semibold text-slate-600">Generate Questions</span> to
                  send your notes to AION and receive AI-generated exam questions.
                </p>
              )}
              {isGenerating && (
                <div className="flex items-center gap-3 text-sm text-slate-500 py-2">
                  <Loader2 className="w-4 h-4 animate-spin text-primary" />
                  {genStatus || "AION is generating your question..."}
                </div>
              )}
              {genError && !isGenerating && (
                <div className="text-xs text-red-500 bg-red-50 p-2 rounded border border-red-100">
                  {genError}
                </div>
              )}
              {preview && !isGenerating && preview.subQuestions.map((sub, i) => (
                <SubQuestionEditor key={i} sub={sub} qNum={qNum}
                  showLabel={preview.subQuestions.length > 1}
                  onChange={(u) => updateSub(i, u)}
                  onRegenerate={() => regenerateSub(i)} />
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ── SubQuestionEditor (unchanged from original) ───────────────────────────────

function SubQuestionEditor({ sub, qNum, showLabel, onChange, onRegenerate }: {
  sub: SubQuestion
  qNum: number
  showLabel: boolean
  onChange: (updates: Partial<SubQuestion>) => void
  onRegenerate: () => void
}) {
  const imageInputRef = useRef<HTMLInputElement>(null)

  const handleAddImages = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return
    const readers = Array.from(files).map(file =>
      new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload  = () => resolve(reader.result as string)
        reader.onerror = () => reject(new Error("Failed to read image"))
        reader.readAsDataURL(file)
      })
    )
    Promise.all(readers)
      .then(urls => onChange({ images: [...(sub.images ?? []), ...urls] }))
      .catch(() => toast.error("Couldn't add image"))
    e.target.value = ""
  }

  const handleRemoveImage = (imgIdx: number) =>
    onChange({ images: (sub.images ?? []).filter((_, i) => i !== imgIdx) })

  const addPart    = () => onChange({ parts: [...(sub.parts ?? []), ""] })
  const updatePart = (pi: number, value: string) =>
    onChange({ parts: (sub.parts ?? []).map((p, i) => i === pi ? value : p) })
  const removePart = (pi: number) =>
    onChange({ parts: (sub.parts ?? []).filter((_, i) => i !== pi) })

  return (
    <div className="border border-slate-200 rounded-md bg-white">
      <div className="flex items-center justify-between px-3 py-1.5 border-b bg-slate-50/70">
        <span className="text-xs font-semibold text-slate-600">
          {showLabel ? `Sub-question ${sub.label})` : "Question"}
        </span>
        <Button type="button" size="sm" variant="ghost"
          className="h-6 text-xs gap-1.5 text-slate-500 hover:text-primary"
          onClick={onRegenerate}>
          <RefreshCw className="w-3 h-3" /> Regenerate
        </Button>
      </div>

      <div className="p-3 space-y-3">
        <Textarea className="min-h-[64px] text-sm bg-white border-slate-200 resize-none"
          value={sub.text} onChange={e => onChange({ text: e.target.value })} />

        {sub.parts && sub.parts.length > 0 && (
          <div className="space-y-2 pl-3 border-l-2 border-slate-200">
            {sub.parts.map((part, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-xs font-medium text-slate-400 w-8 shrink-0 text-right">
                  ({toRoman(i + 1)})
                </span>
                <Input className="h-7 text-xs flex-1" placeholder={`Part ${toRoman(i + 1)}…`}
                  value={part} onChange={e => updatePart(i, e.target.value)} />
                <button type="button" onClick={() => removePart(i)}
                  className="text-slate-400 hover:text-red-500 transition-colors shrink-0">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}

        {sub.images && sub.images.length > 0 && (
          <div className="flex flex-wrap gap-3">
            {sub.images.map((src, i) => (
              <div key={i} className="relative group">
                <img src={src} alt={`Figure ${i + 1}`}
                  className="max-h-28 rounded border border-slate-200 object-contain bg-white" />
                <button type="button" onClick={() => handleRemoveImage(i)}
                  className="absolute -top-2 -right-2 bg-white border border-slate-300 rounded-full p-0.5 shadow-sm text-slate-500 hover:text-red-500 transition-colors">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
          <Button type="button" size="sm" variant="outline" className="h-7 text-xs gap-1.5"
            onClick={() => imageInputRef.current?.click()}>
            <ImagePlus className="w-3.5 h-3.5" /> Add image
          </Button>
          <input type="file" ref={imageInputRef} className="hidden"
            accept="image/*" multiple onChange={handleAddImages} />

          <Button type="button" size="sm" variant="outline" className="h-7 text-xs gap-1.5"
            onClick={addPart}>
            <ListPlus className="w-3.5 h-3.5" /> Add part (i, ii…)
          </Button>

          <label className="flex items-center gap-1.5">
            <span className="text-slate-400">Marks</span>
            <Input type="number" min={0} className="w-16 h-7 text-xs"
              value={sub.marks} onChange={e => onChange({ marks: parseInt(e.target.value) || 0 })} />
          </label>
          <label className="flex items-center gap-1.5">
            <span className="text-slate-400">CO</span>
            <Input className="w-16 h-7 text-xs" value={sub.co}
              onChange={e => onChange({ co: e.target.value })} />
          </label>
          <label className="flex items-center gap-1.5">
            <span className="text-slate-400">RBT</span>
            <Input className="w-16 h-7 text-xs" value={sub.rbt}
              onChange={e => onChange({ rbt: e.target.value })} />
          </label>
        </div>
      </div>
    </div>
  )
}

