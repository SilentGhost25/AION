import { useState, useRef, useEffect } from "react"
import { PaperConfig, SectionInput, GeneratedPaper, useGeneratePaper } from "@workspace/api-client-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { DatePicker } from "@/components/ui/date-picker"
import { Badge } from "@/components/ui/badge"
import { BookOpen, Clock, Hash, GraduationCap, UploadCloud, CheckCircle2, FileText, Wand2, Info, AlertCircle, ChevronDown, ChevronUp, RefreshCw } from "lucide-react"
import { format } from "date-fns"
import { toast } from "sonner"
import { Spinner } from "@/components/ui/spinner"
import { buildTestPaper } from "./mockPaper"
import { aionAPI } from "@/lib/aion-api"
import { QuestionWithSubs } from "./Step2Preview"

interface PipelineError {
  code: string
  stage: string
  message: string
  recoverable: boolean
  debug?: any
}

const STAGE_LABELS: Record<string, string> = {
  connecting: "Connecting to AION...",
  validation: "Validating request...",
  document_check: "Checking document in repository...",
  extraction: "Extracting document content...",
  generation: "Generating questions via Qwen LLM...",
  qa: "Running quality gate validation...",
}

const ERROR_MESSAGES: Record<string, { title: string; hint: string }> = {
  MISSING_DOCUMENT_ID: {
    title: "Upload not linked to request",
    hint: "Re-upload your PDF files and try again.",
  },
  DOCUMENT_NOT_FOUND: {
    title: "Uploaded document not found on server",
    hint: "Re-upload your PDF files. The upload may have expired.",
  },
  EXTRACTION_FAILED: {
    title: "PDF extraction failed",
    hint: "Ensure you uploaded a valid, non-password-protected PDF.",
  },
  INSUFFICIENT_EVIDENCE: {
    title: "Not enough usable content in uploaded PDF",
    hint: "Upload a complete module PDF, not a summary or slide deck.",
  },
  GENERATION_FAILED: {
    title: "Question generation failed",
    hint: "The AI model encountered an error. Try again or upload a different PDF.",
  },
  INCOMPLETE_PAPER: {
    title: "Generation produced an incomplete paper",
    hint: "Try again. If this persists, check server logs.",
  },
  GENERATION_TIMEOUT: {
    title: "Generation timed out",
    hint: "The server took too long. Try with fewer questions or a smaller PDF.",
  },
  FORMATTING_FAILED: {
    title: "Paper formatting failed",
    hint: "Generated content could not be structured into a paper.",
  },
  CONTRACT_VIOLATION: {
    title: "Paper structure contract violated",
    hint: "The generated paper has incorrect structure (e.g. missing OR questions or mismatched partitions).",
  },
  QUALITY_GATE_FAILURE: {
    title: "Quality gate rejected the paper",
    hint: "The paper did not pass Bloom taxonomy or CO mapping validation. Try regenerating.",
  },
  INTERNAL_PIPELINE_ERROR: {
    title: "Internal pipeline error",
    hint: "An internal error occurred. Check the server logs.",
  },
  INTERNAL_ERROR: {
    title: "Internal server error",
    hint: "An unexpected error occurred on the server. Check the server logs.",
  },
}

interface Step1ConfigAndUploadProps {
  registerRegenerate?: (fn: (varIdx: number) => Promise<GeneratedPaper | null>) => void
  config: PaperConfig
  setConfig: (c: PaperConfig) => void
  sections: SectionInput[]
  setSections: (s: SectionInput[]) => void
  onSuccess: (paper: GeneratedPaper) => void
}

const SPLIT_VARIATIONS: Record<number, number[][]> = {
  1: [[10]],
  2: [[6, 4], [5, 5], [7, 3], [8, 2]],
  3: [[4, 3, 3], [5, 3, 2], [6, 2, 2], [4, 4, 2]]
};

export function Step1ConfigAndUpload({ config, setConfig, sections, setSections, onSuccess, registerRegenerate }: Step1ConfigAndUploadProps) {
  const [fileNames, setFileNames] = useState<(string | null)[]>(Array(10).fill(null))
  const [rawFiles, setRawFiles] = useState<(File | null)[]>(Array(10).fill(null))
  const [markSplits, setMarkSplits] = useState<number[][]>(() => Array(10).fill(null).map(() => SPLIT_VARIATIONS[2][0]))

  useEffect(() => {
    if (registerRegenerate) {
      registerRegenerate(generateSinglePaper)
    }
  }, [registerRegenerate, sections, markSplits, config, rawFiles])

  const handleModuleFileUpload = (moduleIdx: number, e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const q1 = moduleIdx * 2, q2 = q1 + 1
    setFileNames(prev => { const n = [...prev]; n[q1] = file.name; n[q2] = file.name; return n })
    setRawFiles(prev => { const n = [...prev]; n[q1] = file; n[q2] = file; return n })
    const newSections = [...sections]
    newSections[q1] = { ...newSections[q1], notesText: `[File: ${file.name}]` }
    newSections[q2] = { ...newSections[q2], notesText: `[File: ${file.name}]` }
    setSections(newSections)
    toast.info(`${file.name} attached to Module ${moduleIdx + 1}`)
  }

  const updateSubCount = (qIdx: number, count: number) => {
    const newSections = [...sections]
    newSections[qIdx] = { ...newSections[qIdx], subQuestionsPerQ: count }
    setSections(newSections)
    const newSplits = [...markSplits]
    newSplits[qIdx] = SPLIT_VARIATIONS[count]?.[0] || [10]
    setMarkSplits(newSplits)
  }

  const updateMarkSplit = (qIdx: number, splitStr: string) => {
    const newSplits = [...markSplits]
    newSplits[qIdx] = splitStr.split(",").map(Number)
    setMarkSplits(newSplits)
  }

  const generateSinglePaper = async (varIdx: number = 1): Promise<GeneratedPaper | null> => {
    if (!canGenerate) return null
    setIsGenerating(true)
    setPipelineError(null)
    setCurrentStage("connecting")
    try {
      // 1. Collect distinct files mapped by module index
      const distinctFiles: { file: File; moduleIdx: number }[] = []
      const seenNames = new Set<string>()
      for (let i = 0; i < 5; i++) {
        const qIdxA = i * 2
        const qIdxB = i * 2 + 1
        const f = rawFiles[qIdxA] || rawFiles[qIdxB]
        if (f && !seenNames.has(f.name)) {
          seenNames.add(f.name)
          distinctFiles.push({ file: f, moduleIdx: i })
        }
      }

      // Build structured notes with confirmed Strategy 1 module headers
      const moduleTexts: string[] = []
      for (let i = 0; i < 5; i++) {
        const qIdxA = i * 2
        const qIdxB = i * 2 + 1
        const rawText = sections[qIdxA]?.notesText?.trim() || sections[qIdxB]?.notesText?.trim() || ""
        const header = `Module ${i + 1}: ${config.subjectName || "Untitled"} - Part ${i + 1}`
        moduleTexts.push(`${header}\n${rawText}`)
      }
      const combinedNotes = moduleTexts.join("\n\n")

      // 2. Upload reference files mapped to their explicit module indices (1..5)
      let fileIds: string[] = []
      let primaryFileId: string | undefined = undefined
      const moduleFileMap: Record<number, string> = {}

      if (distinctFiles.length > 0) {
        for (const { file, moduleIdx } of distinctFiles) {
          toast.info(`Uploading Module ${moduleIdx + 1}: ${file.name}...`)
          const uploadRes = await aionAPI.upload(file, config.subjectName || "Subject", "notes")
          const fid = uploadRes.id || uploadRes.document_id
          if (!fid) {
            setPipelineError({
              code: "UPLOAD_FAILED",
              stage: "upload",
              message: `Failed to upload Module ${moduleIdx + 1} (${file.name}).`,
              recoverable: true,
            })
            setIsGenerating(false)
            return null
          }
          fileIds.push(fid)
          moduleFileMap[moduleIdx + 1] = fid
        }
        primaryFileId = fileIds[0]
      } else {
        const blob = new Blob([combinedNotes], { type: "text/plain" })
        const file = new File(
          [blob],
          `${(config.subjectCode || "syllabus").replace(/\s+/g, "_")}_notes.txt`,
          { type: "text/plain" }
        )
        const uploadRes = await aionAPI.upload(file, config.subjectName || "Subject", "notes")
        primaryFileId = uploadRes.id || uploadRes.document_id
        if (primaryFileId) fileIds.push(primaryFileId)
      }

      const response = await aionAPI.generateStream({
        file_id: primaryFileId,
        fileId: primaryFileId,
        file_ids: fileIds.length > 1 ? fileIds : undefined,
        fileIds: fileIds.length > 1 ? fileIds : undefined,
        module_files: Object.keys(moduleFileMap).length > 0 ? moduleFileMap : undefined,
        moduleFiles: Object.keys(moduleFileMap).length > 0 ? moduleFileMap : undefined,
        subject: config.subjectName || "Subject",
        department: (config as any).department || "Computer Science & Engineering",
        semester: (config as any).semester || 5,
        exam_type: config.examType || "IA",
        examType: config.examType || "IA",
        difficulty: "mixed",
        notes_text: combinedNotes,
        model: "qwen2.5:14b",
        sub_question_counts: sections.map(s => s?.subQuestionsPerQ ?? 2),
        subQuestionCounts: sections.map(s => s?.subQuestionsPerQ ?? 2),
        mark_splits: markSplits,
        markSplits: markSplits,
        sub_question_count: sections[0]?.subQuestionsPerQ ?? 2,
        subQuestionCount: sections[0]?.subQuestionsPerQ ?? 2,
        variation_index: varIdx,
      } as any)

      if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`)

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let formattedResult: any = null
      let currentEvent = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""

        for (const line of lines) {
          const trimmed = line.trim()
          if (trimmed.startsWith("event:")) {
            currentEvent = trimmed.slice(6).trim()
            continue
          }
          if (!trimmed.startsWith("data:")) continue
          let data: any
          try { data = JSON.parse(trimmed.slice(5).trim()) } catch { continue }

          if (currentEvent === "stage_update") {
            setCurrentStage(data.stage)
            setStageMessage(data.message || STAGE_LABELS[data.stage] || data.stage)
          } else if (currentEvent === "paper_ready" || currentEvent === "result") {
            formattedResult = data.paper || data
          }
        }
      }

      if (formattedResult && formattedResult.modules && formattedResult.modules.length > 0) {
        const markPerQuestion = Math.floor(config.maxMarks / 5)
        const questions: any[] = []
        let globalQNo = 1
        for (let mIdx = 0; mIdx < formattedResult.modules.length; mIdx++) {
          const mod = formattedResult.modules[mIdx]
          const modQuestions = mod.questions || []
          for (let qIdx = 0; qIdx < modQuestions.length; qIdx++) {
            const q = modQuestions[qIdx]
            const isOr = q.isOr ?? q.is_or ?? (qIdx % 2 === 1)
            const co = `CO${mIdx + 1}`
            const rawSubs = q.subQuestions || q.sub_questions || []
            const subs = rawSubs.map((sq: any) => ({
              label: sq.letter || sq.label || "a",
              text: sq.text || "",
              marks: sq.marks,
              co: sq.co || co,
              rbt: `L${sq.bloom || q.bloom_level || q.bloomLevel || 2}`,
            }))
            questions.push({
              qNo: globalQNo,
              text: "",
              marks: q.totalMarks || q.total_marks || markPerQuestion,
              co,
              rbt: `L${q.bloom_level || q.bloomLevel || 2}`,
              sectionNumber: globalQNo,
              isOrQuestion: isOr,
              subQuestions: subs,
            })
            globalQNo++
          }
        }
        return {
          config,
          questions,
          courseOutcomes: [
            "Understand fundamental concepts and theoretical foundations",
            "Apply analytical methods and problem-solving techniques",
            "Implement algorithms and system architectures",
            "Analyze performance tradeoffs and design alternatives",
            "Evaluate solution quality and system specifications",
          ],
          coCoverage: formattedResult.coCoverage || { co1: 40, co2: 40, co3: 20, co4: 0, co5: 0 },
          syllabusCoverage: formattedResult.syllabusCoverage || { s1: 20, s2: 20, s3: 20, s4: 20, s5: 20 },
        } as GeneratedPaper
      }
    } catch (e: any) {
      console.error("[AION] Generation error:", e)
      setPipelineError({ code: "GENERATION_ERROR", stage: "generation", message: e.message || "Failed to generate paper.", recoverable: true })
    } finally {
      setIsGenerating(false)
    }
    return null
  }
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentStage, setCurrentStage] = useState<string | null>(null)
  const [stageMessage, setStageMessage] = useState<string | null>(null)
  const [pipelineError, setPipelineError] = useState<PipelineError | null>(null)
  const [showDebug, setShowDebug] = useState(false)
  const fileInputRefs = useRef<HTMLInputElement[]>([])

  const generateMutation = useGeneratePaper({
    mutation: {
      onSuccess: (data: any) => {
        toast.success("Question paper generated successfully")
        onSuccess(data)
      },
      onError: () => {
        toast.warning("Backend not connected — showing sample paper", {
          description: "Displaying test data with large questions so you can verify the layout."
        })
        onSuccess(buildTestPaper(config))
      }
    }
  })

  const isConfigValid = config.institutionDepartment && config.subjectName && config.subjectCode && config.batch && config.maxMarks > 0

  // Count how many questions have content uploaded
  const uploadedCount = sections.filter(s => s.notesText && s.notesText.trim() !== "").length

  // Each OR pair must have content
  const allPairsHaveContent = [0, 1, 2, 3, 4].every(pair => {
    const a = sections[pair * 2]
    const b = sections[pair * 2 + 1]
    return a && a.notesText.trim() !== "" && b && b.notesText.trim() !== ""
  })

  const canGenerate = isConfigValid && allPairsHaveContent

  const handleFileUpload = (idx: number, e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const newFileNames = [...fileNames]
    newFileNames[idx] = file.name
    setFileNames(newFileNames)

    const newRawFiles = [...rawFiles]
    newRawFiles[idx] = file
    setRawFiles(newRawFiles)

    const newSections = [...sections]
    newSections[idx] = { ...newSections[idx], notesText: `[File: ${file.name}]` }
    setSections(newSections)

    toast.info(`${file.name} attached`, { description: "Original file will be uploaded directly to AION backend." })
  }

  const handleGenerate = async () => {
    const paper = await generateSinglePaper(1)
    if (paper) {
      toast.success("Question paper generated successfully!")
      onSuccess(paper)
      return
    }
    setPipelineError(null)
    setShowDebug(false)
    setCurrentStage("connecting")
    setStageMessage("Establishing connection to AION...")
    toast.info("Sending reference material to AION AI...", { description: "Generating questions with Qwen LLM..." })

    try {
      const combinedNotes = sections
        .map((s, i) => `=== Module ${Math.floor(i / 2) + 1} Question ${i + 1} ===\n${s.notesText}`)
        .join("\n\n")

      const uploadedFile = rawFiles.find(f => f !== null)
      let uploadRes: any

      if (uploadedFile) {
        toast.info(`Uploading ${uploadedFile.name}...`)
        uploadRes = await aionAPI.upload(uploadedFile, config.subjectName || "Subject", "notes")
      } else {
        const blob = new Blob([combinedNotes], { type: "text/plain" })
        const file = new File([blob], `${config.subjectCode || 'syllabus'}_notes.txt`, { type: "text/plain" })
        uploadRes = await aionAPI.upload(file, config.subjectName || "Subject", "notes")
      }
      const fileId = uploadRes.id || uploadRes.document_id

      const response = await aionAPI.generateStream({
        file_id: fileId,
        fileId: fileId,
        subject: config.subjectName || "Subject",
        department: (config as any).department || "Computer Science & Engineering",
        semester: (config as any).semester || 5,
        exam_type: config.examType || "IA",
        examType: config.examType || "IA",
        difficulty: "mixed",
        notes_text: combinedNotes,
        model: "qwen2.5:14b",
        sub_question_count: sections[0]?.subQuestionsPerQ ?? 2,
        subQuestionCount: sections[0]?.subQuestionsPerQ ?? 2,
      })

      if (!response.ok) {
        const err: PipelineError = {
          code: "HTTP_ERROR",
          stage: "connection",
          message: `Server returned ${response.status}: ${response.statusText}`,
          recoverable: response.status >= 500,
        }
        setPipelineError(err)
        toast.error("Connection failed", { description: err.message })
        return
      }

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let formattedResult: any = null
      let terminalReceived = false
      let currentEvent = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""

        for (const line of lines) {
          const trimmed = line.trim()
          if (trimmed.startsWith("event:")) {
            currentEvent = trimmed.slice(6).trim()
            continue
          }
          if (!trimmed.startsWith("data:")) continue

          let data: any
          try {
            data = JSON.parse(trimmed.slice(5).trim())
          } catch {
            console.warn("[AION] Malformed SSE data:", trimmed)
            continue
          }

          if (currentEvent === "stage_update") {
            setCurrentStage(data.stage)
            setStageMessage(data.message || STAGE_LABELS[data.stage] || data.stage)

          } else if (currentEvent === "pipeline_error") {
            terminalReceived = true
            const errData = data.error || data
            const err: PipelineError = {
              code: errData.code || "UNKNOWN_ERROR",
              stage: errData.stage || "unknown",
              message: errData.message || "An error occurred during generation.",
              recoverable: errData.recoverable !== undefined ? errData.recoverable : true,
              debug: errData.debug,
            }
            console.error("[AION] pipeline_error:", err)
            setPipelineError(err)
            const errInfo = ERROR_MESSAGES[err.code] ?? { title: `Error: ${err.code}`, hint: err.message }
            toast.error(errInfo.title, { description: errInfo.hint })
            return

          } else if (currentEvent === "paper_ready") {
            const paper = data.paper || data
            formattedResult = paper

          } else if (currentEvent === "result") {
            // Legacy compat — accept both wrapped and flat
            formattedResult = data.paper || data

          } else if (currentEvent === "done") {
            terminalReceived = true
            if (data.status !== "SUCCESS") {
              // If done arrived without paper_ready, pipeline_error should have already fired
              return
            }
          }
        }
      }

      if (!terminalReceived) {
        const err: PipelineError = {
          code: "STREAM_ENDED_WITHOUT_TERMINAL",
          stage: currentStage || "unknown",
          message: "Server closed connection without sending a completion event.",
          recoverable: true,
        }
        setPipelineError(err)
        toast.error("Generation incomplete", { description: err.message })
        return
      }

      if (formattedResult && formattedResult.modules && formattedResult.modules.length > 0) {
        const markPerQuestion = Math.floor(config.maxMarks / 5)
        const questions: QuestionWithSubs[] = []

        let globalQNo = 1
        for (let mIdx = 0; mIdx < formattedResult.modules.length; mIdx++) {
          const mod = formattedResult.modules[mIdx]
          const modQuestions = mod.questions || []

          for (let qIdx = 0; qIdx < modQuestions.length; qIdx++) {
            const q = modQuestions[qIdx]
            const isOr = q.isOr ?? q.is_or ?? (qIdx % 2 === 1)
            const co = `CO${mIdx + 1}`
            const rawSubs = q.subQuestions || q.sub_questions || []
            const subs = rawSubs.map((sq: any) => ({
              label: sq.letter || sq.label || "a",
              text: sq.text || "",
              marks: sq.marks,
              co: sq.co || co,
              rbt: `L${sq.bloom || q.bloom_level || q.bloomLevel || 2}`,
            }))

            questions.push({
              qNo: globalQNo,
              text: "",   // Pure renderer — text is in subQuestions
              marks: q.totalMarks || q.total_marks || markPerQuestion,
              co,
              rbt: `L${q.bloom_level || q.bloomLevel || 2}`,
              sectionNumber: globalQNo,
              isOrQuestion: isOr,
              subQuestions: subs,
            })
            globalQNo++
          }
        }

        const paper: GeneratedPaper = {
          config,
          questions,
          courseOutcomes: [
            "Understand fundamental concepts and theoretical foundations",
            "Apply analytical methods and problem-solving techniques",
            "Implement algorithms and system architectures",
            "Analyze performance tradeoffs and design alternatives",
            "Evaluate solution quality and system specifications",
          ],
          coCoverage: { co1: 0, co2: 0, co3: 0, co4: 0, co5: 0 },
          syllabusCoverage: { s1: 0, s2: 0, s3: 0, s4: 0, s5: 0 },
        }

        setCurrentStage(null)
        setStageMessage(null)
        toast.success("Question paper generated successfully!")
        onSuccess(paper)
      } else {
        const err: PipelineError = {
          code: "INCOMPLETE_PAPER",
          stage: "qa",
          message: "Backend returned no modules. Generation may have failed silently.",
          recoverable: true,
        }
        setPipelineError(err)
        toast.error("Generation Failed", { description: err.message })
      }
    } catch (err: any) {
      console.error("[AION] Generation error:", err)
      const netErr: PipelineError = {
        code: "NETWORK_ERROR",
        stage: "connection",
        message: err?.message || "A network error occurred during generation.",
        recoverable: true,
      }
      setPipelineError(netErr)
      toast.error("AION Generation Failed", { description: netErr.message })
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500 pb-24">
      {/* Config Section */}
      <Card className="border-t-4 border-t-primary shadow-lg mb-8">
        <CardHeader className="bg-slate-50/50 border-b">
          <div className="flex items-center gap-2 text-primary mb-2">
            <BookOpen className="h-5 w-5" />
            <span className="font-semibold uppercase tracking-wider text-xs">Examination Details</span>
          </div>
          <CardTitle className="text-2xl text-slate-800">Configure Paper Metadata</CardTitle>
          <CardDescription>
            Enter the structural details for the question paper header.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-8 space-y-6">

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label htmlFor="dept">Institution Department <span className="text-red-500">*</span></Label>
              <Input
                id="dept"
                placeholder="e.g. Computer Science & Engineering"
                value={config.institutionDepartment}
                onChange={e => setConfig({ ...config, institutionDepartment: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch">Batch <span className="text-red-500">*</span></Label>
              <div className="relative">
                <GraduationCap className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  id="batch"
                  className="pl-9"
                  placeholder="e.g. 2023-2027"
                  value={config.batch}
                  onChange={e => setConfig({ ...config, batch: e.target.value })}
                />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label htmlFor="subjectName">Subject Name <span className="text-red-500">*</span></Label>
              <Input
                id="subjectName"
                placeholder="e.g. Data Structures"
                value={config.subjectName}
                onChange={e => setConfig({ ...config, subjectName: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="subjectCode">Subject Code <span className="text-red-500">*</span></Label>
              <div className="relative">
                <Hash className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  id="subjectCode"
                  className="pl-9"
                  placeholder="e.g. 21CS32"
                  value={config.subjectCode}
                  onChange={e => setConfig({ ...config, subjectCode: e.target.value })}
                />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-2">
              <Label>Exam Type</Label>
              <Select
                value={config.examType}
                onValueChange={(val: any) => {
                  setConfig({
                    ...config,
                    examType: val,
                    maxMarks: val === "SEE" ? 100 : 50,
                    duration: val === "SEE" ? "3 hrs" : "1.5 hrs"
                  })
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select Exam Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="IAT1">IAT-1</SelectItem>
                  <SelectItem value="IAT2">IAT-2</SelectItem>
                  <SelectItem value="SEE">SEE</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Semester</Label>
              <Select
                value={config.semester.toString()}
                onValueChange={(val) => setConfig({ ...config, semester: parseInt(val, 10) })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select Semester" />
                </SelectTrigger>
                <SelectContent>
                  {[1,2,3,4,5,6,7,8].map(sem => (
                    <SelectItem key={sem} value={sem.toString()}>Semester {sem}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Date of Exam</Label>
              <DatePicker
                date={config.dateOfExam ? new Date(config.dateOfExam) : undefined}
                setDate={(date) => setConfig({ ...config, dateOfExam: date ? format(date, "yyyy-MM-dd") : undefined })}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-50 p-4 rounded-lg border">
            <div className="space-y-2">
              <Label htmlFor="maxMarks">Max Marks</Label>
              <Input
                id="maxMarks"
                type="number"
                value={config.maxMarks}
                onChange={e => setConfig({ ...config, maxMarks: parseInt(e.target.value, 10) || 0 })}
              />
            </div>
            <div className="space-y-2">
              <Label>Duration</Label>
              <Select
                value={config.duration}
                onValueChange={(val) => setConfig({ ...config, duration: val })}
              >
                <SelectTrigger className="w-full">
                  <Clock className="mr-2 h-4 w-4 text-muted-foreground" />
                  <SelectValue placeholder="Select Duration" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1 hr">1 hr</SelectItem>
                  <SelectItem value="1.5 hrs">1.5 hrs</SelectItem>
                  <SelectItem value="2 hrs">2 hrs</SelectItem>
                  <SelectItem value="3 hrs">3 hrs</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

        </CardContent>
      </Card>

      {/* Upload Section */}
      <div className="mb-8">
        <div className="mb-4">
          <div className="flex items-center gap-2 text-primary mb-1">
            <UploadCloud className="h-5 w-5" />
            <span className="font-semibold uppercase tracking-wider text-xs">Reference Material Upload</span>
          </div>
          <h2 className="text-2xl font-bold text-slate-800">Upload Content for Each Question</h2>
        </div>
        <p className="text-slate-600 mb-6">
          Upload reference material for all 10 questions (5 OR pairs). The backend AI will generate sub-questions, assign marks, CO, and Bloom's levels.
        </p>

        <div className="space-y-4">
          {[0, 1, 2, 3, 4].map((mIdx) => {
            const q1 = mIdx * 2
            const q2 = q1 + 1
            const hasC = !!sections[q1]?.notesText && sections[q1].notesText.trim() !== ""
            const fileName = fileNames[q1]

            return (
              <Card key={mIdx} className="shadow-sm border-slate-200 overflow-hidden">
                <div className="bg-slate-50 border-b px-5 py-3 flex items-center justify-between">
                  <h3 className="font-semibold text-slate-800 flex items-center gap-2">
                    Module {mIdx + 1} Reference Material
                  </h3>
                  {hasC && (
                    <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200">
                      <CheckCircle2 className="w-3 h-3 mr-1" />
                      Uploaded
                    </Badge>
                  )}
                </div>
                <CardContent className="pt-4 space-y-4">
                  <div className="flex items-center gap-4">
                    <Button
                      variant="outline"
                      size="lg"
                      className="flex-1 h-20 border-dashed border-2 hover:border-primary hover:bg-primary/5"
                      onClick={() => fileInputRefs.current[q1]?.click()}
                    >
                      {hasC ? (
                        <>
                          <FileText className="mr-2 h-5 w-5 text-emerald-600" />
                          <div className="text-left flex-1">
                            <div className="font-medium text-slate-800">{fileName || "Content uploaded"}</div>
                            <div className="text-xs text-slate-500">Shared between Question {q1 + 1} & Question {q2 + 1} (OR alternative) · Click to replace</div>
                          </div>
                        </>
                      ) : (
                        <>
                          <UploadCloud className="mr-2 h-5 w-5" />
                          Upload PDF, TXT, or DOCX for Module {mIdx + 1}
                        </>
                      )}
                    </Button>
                    <input
                      ref={el => { if (el) fileInputRefs.current[q1] = el }}
                      type="file"
                      className="hidden"
                      accept=".pdf,.txt,.docx"
                      onChange={(e) => handleModuleFileUpload(mIdx, e)}
                    />
                  </div>

                  {/* Settings for Q1 and Q2 */}
                  {[q1, q2].map((qIdx) => {
                    const isAlt = qIdx % 2 === 1
                    const count = sections[qIdx]?.subQuestionsPerQ ?? 2
                    const curSplit = markSplits[qIdx] || SPLIT_VARIATIONS[count][0]
                    const curSplitStr = curSplit.join(",")

                    return (
                      <div key={qIdx} className={`pt-3 ${isAlt ? "border-t border-slate-200/80" : ""}`}>
                        <div className="flex items-center justify-between flex-wrap gap-2">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-slate-800 text-sm">
                              Question {qIdx + 1}
                            </span>
                            {isAlt && (
                              <Badge variant="outline" className="text-[10px] border-slate-300 text-slate-500 font-normal">
                                OR alternative
                              </Badge>
                            )}
                          </div>
                          <div className="flex items-center gap-3">
                            <label className="flex items-center gap-1.5">
                              <span className="text-xs text-muted-foreground whitespace-nowrap">Sub-questions</span>
                              <select
                                className="h-7 rounded-md border border-input bg-background px-2 text-xs focus:outline-none focus:ring-1 focus:ring-ring font-medium"
                                value={count}
                                onChange={(e) => updateSubCount(qIdx, parseInt(e.target.value) || 2)}
                              >
                                <option value={1}>1</option>
                                <option value={2}>2</option>
                                <option value={3}>3</option>
                              </select>
                            </label>
                            <label className="flex items-center gap-1.5">
                              <span className="text-xs text-muted-foreground whitespace-nowrap">Mark split</span>
                              <select
                                className="h-7 rounded-md border border-input bg-background px-2 text-xs focus:outline-none focus:ring-1 focus:ring-ring font-medium"
                                value={curSplitStr}
                                onChange={(e) => updateMarkSplit(qIdx, e.target.value)}
                              >
                                {SPLIT_VARIATIONS[count].map((sp) => (
                                  <option key={sp.join(",")} value={sp.join(",")}>
                                    {sp.join("+")}M
                                  </option>
                                ))}
                              </select>
                            </label>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      {/* ── Generation Status Panel ─────────────────────────────────── */}
      {isGenerating && currentStage && (
        <div className="mt-6 rounded-xl border border-blue-200 bg-gradient-to-r from-blue-50 to-indigo-50 p-5 shadow-sm">
          <div className="flex items-center gap-4">
            <div className="relative flex-shrink-0">
              <div className="h-10 w-10 rounded-full border-2 border-blue-200 bg-white flex items-center justify-center">
                <Spinner className="h-5 w-5 text-blue-600" />
              </div>
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold text-blue-800 capitalize">
                {STAGE_LABELS[currentStage] || currentStage}
              </p>
              {stageMessage && (
                <p className="text-xs text-blue-600 mt-0.5">{stageMessage}</p>
              )}
            </div>
            <Badge className="bg-blue-100 text-blue-700 border-blue-200 text-[11px]">
              {currentStage?.replace(/_/g, " ").toUpperCase()}
            </Badge>
          </div>
          <div className="mt-3 flex gap-1.5">
            {["validation", "document_check", "extraction", "generation", "qa"].map((stage) => (
              <div
                key={stage}
                className={`h-1 flex-1 rounded-full transition-all duration-500 ${
                  ["validation", "document_check", "extraction", "generation", "qa"].indexOf(currentStage || "") >=
                  ["validation", "document_check", "extraction", "generation", "qa"].indexOf(stage)
                    ? "bg-blue-500"
                    : "bg-blue-100"
                }`}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── Pipeline Error Panel ─────────────────────────────────────── */}
      {!isGenerating && pipelineError && (
        <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-5 shadow-sm">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="font-semibold text-red-800">
                {ERROR_MESSAGES[pipelineError.code]?.title ?? `Error: ${pipelineError.code}`}
              </h3>
              <p className="text-sm text-red-700 mt-1">{pipelineError.message}</p>
              <p className="text-sm text-red-500 mt-2">
                <span className="font-medium">Hint:</span>{" "}
                {ERROR_MESSAGES[pipelineError.code]?.hint ?? pipelineError.message}
              </p>
              <div className="flex flex-wrap gap-2 mt-3">
                <span className="text-[11px] text-red-400 bg-red-100 px-2 py-0.5 rounded">
                  Stage: {pipelineError.stage}
                </span>
                <span className="text-[11px] text-red-400 bg-red-100 px-2 py-0.5 rounded">
                  Code: {pipelineError.code}
                </span>
                {pipelineError.recoverable && (
                  <span className="text-[11px] text-emerald-600 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded">
                    Recoverable
                  </span>
                )}
              </div>
              <div className="mt-4 flex gap-2">
                {pipelineError.recoverable && (
                  <Button
                    size="sm"
                    className="bg-red-600 hover:bg-red-700 text-white gap-1.5"
                    onClick={() => { setPipelineError(null); handleGenerate() }}
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                    Retry Generation
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="outline"
                  className="gap-1.5 text-red-600 border-red-200 hover:bg-red-50"
                  onClick={() => setShowDebug(!showDebug)}
                >
                  {showDebug ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                  {showDebug ? "Hide" : "Show"} Debug Info
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-slate-400 hover:text-slate-600"
                  onClick={() => setPipelineError(null)}
                >
                  Dismiss
                </Button>
              </div>
              {showDebug && (
                <pre className="mt-4 text-[11px] bg-slate-900 text-slate-100 p-3 rounded-lg overflow-auto max-h-64 leading-relaxed">
                  {JSON.stringify(pipelineError, null, 2)}
                </pre>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Floating Bottom Bar */}
      <div className="fixed bottom-0 left-0 right-0 bg-white/80 backdrop-blur-md border-t p-4 shadow-[0_-4px_20px_rgba(0,0,0,0.05)] z-10 print:hidden">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className={`flex items-center gap-2 text-sm font-medium ${allPairsHaveContent ? "text-emerald-600" : "text-amber-600"}`}>
            <Info className="h-4 w-4" />
            {uploadedCount} / 10 questions have content
          </div>

          <Button
            size="lg"
            onClick={handleGenerate}
            disabled={!canGenerate || isGenerating || generateMutation.isPending}
            className="bg-primary hover:bg-primary/90 text-white min-w-[240px]"
          >
            {isGenerating || generateMutation.isPending ? (
              <>
                <Spinner className="mr-2 h-4 w-4" />
                Generating Paper via AION...
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
  )
}
