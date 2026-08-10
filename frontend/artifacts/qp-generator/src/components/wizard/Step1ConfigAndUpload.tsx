import { useState, useRef } from "react"
import { PaperConfig, SectionInput, GeneratedPaper, useGeneratePaper } from "@workspace/api-client-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { DatePicker } from "@/components/ui/date-picker"
import { Badge } from "@/components/ui/badge"
import { BookOpen, Clock, Hash, GraduationCap, UploadCloud, CheckCircle2, FileText, Wand2, Info } from "lucide-react"
import { format } from "date-fns"
import { toast } from "sonner"
import { Spinner } from "@/components/ui/spinner"
import { buildTestPaper } from "./mockPaper"
import { aionAPI } from "@/lib/aion-api"
import { QuestionWithSubs } from "./Step2Preview"

interface Step1ConfigAndUploadProps {
  config: PaperConfig
  setConfig: (c: PaperConfig) => void
  sections: SectionInput[]
  setSections: (s: SectionInput[]) => void
  onSuccess: (paper: GeneratedPaper) => void
}

export function Step1ConfigAndUpload({ config, setConfig, sections, setSections, onSuccess }: Step1ConfigAndUploadProps) {
  const [fileNames, setFileNames] = useState<(string | null)[]>(Array(10).fill(null))
  const [isGenerating, setIsGenerating] = useState(false)
  const fileInputRefs = useRef<HTMLInputElement[]>([])

  const generateMutation = useGeneratePaper({
    mutation: {
      onSuccess: (data) => {
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

    const ext = file.name.split(".").pop()?.toLowerCase()

    if (ext === "docx") {
      toast.info("DOCX uploaded", { description: "File uploaded. Text extraction will happen on the backend." })
      const newSections = [...sections]
      newSections[idx] = { ...newSections[idx], notesText: `[DOCX: ${file.name}]` }
      setSections(newSections)
    } else {
      const reader = new FileReader()
      reader.onload = (event) => {
        const text = event.target?.result as string
        const newSections = [...sections]
        newSections[idx] = { ...newSections[idx], notesText: text }
        setSections(newSections)
      }
      reader.readAsText(file)
    }
  }

  const handleGenerate = async () => {
    if (!canGenerate) return
    setIsGenerating(true)
    toast.info("Sending reference material to AION AI...", { description: "Generating questions with Ollama..." })

    try {
      const combinedNotes = sections
        .map((s, i) => `=== Module ${Math.floor(i / 2) + 1} Question ${i + 1} ===\n${s.notesText}`)
        .join("\n\n")

      const blob = new Blob([combinedNotes], { type: "text/plain" })
      const file = new File([blob], `${config.subjectCode || 'syllabus'}_notes.txt`, { type: "text/plain" })
      const uploadRes = await aionAPI.upload(file, config.subjectName || "Subject", "notes")
      const fileId = uploadRes.id

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
        include_visual: true,
        sub_question_count: sections[0]?.subQuestionsPerQ ?? 2,
      })

      if (!response.ok) {
        throw new Error(`Generation failed: ${response.statusText}`)
      }

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let formattedResult: any = null

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

          try {
            const data = JSON.parse(trimmed.slice(5).trim())
            if (currentEvent === "result" || data.modules) {
              formattedResult = data
            }
          } catch {}
        }
      }

      if (formattedResult && formattedResult.modules) {
        const markPerQuestion = Math.floor(config.maxMarks / 5)
        const questions: QuestionWithSubs[] = []

        let globalQNo = 1
        for (let mIdx = 0; mIdx < Math.min(5, formattedResult.modules.length); mIdx++) {
          const mod = formattedResult.modules[mIdx]
          const modQuestions = mod.questions || []

          for (let qIdx = 0; qIdx < Math.min(2, modQuestions.length); qIdx++) {
            const q = modQuestions[qIdx]
            const isOr = qIdx === 1
            const co = `CO${mIdx + 1}`
            const subs = (q.subQuestions || []).map((sq: any) => ({
              label: sq.letter || "a",
              text: sq.text || "Explain the concept.",
              marks: sq.marks || Math.floor(markPerQuestion / (q.subQuestions.length || 1)),
              co,
              rbt: `L${sq.bloom || q.bloomLevel || 2}`,
            }))

            questions.push({
              qNo: globalQNo,
              text: subs.map((s: any) => `${s.label}) ${s.text}`).join("\n"),
              marks: markPerQuestion,
              co,
              rbt: `L${q.bloomLevel || 2}`,
              sectionNumber: globalQNo,
              isOrQuestion: isOr,
              subQuestions: subs.length > 0 ? subs : [
                { label: "a", text: "Explain the given concept with neat diagrams.", marks: markPerQuestion, co, rbt: "L2" }
              ],
            })
            globalQNo++
          }
        }

        while (questions.length < 10) {
          const qNo = questions.length + 1
          const co = `CO${Math.ceil(qNo / 2)}`
          questions.push({
            qNo,
            text: "a) Explain the fundamental principles and applications of the topic.",
            marks: markPerQuestion,
            co,
            rbt: "L2",
            sectionNumber: qNo,
            isOrQuestion: qNo % 2 === 0,
            subQuestions: [
              { label: "a", text: "Explain the fundamental principles and applications of the topic.", marks: markPerQuestion, co, rbt: "L2" }
            ]
          })
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
          coCoverage: { co1: 20, co2: 20, co3: 20, co4: 20, co5: 20 },
          syllabusCoverage: { s1: 20, s2: 20, s3: 20, s4: 20, s5: 20 },
        }

        toast.success("Question paper generated successfully!")
        onSuccess(paper)
      } else {
        toast.warning("Fallback to sample paper — AION pipeline returned partial data")
        onSuccess(buildTestPaper(config))
      }
    } catch (err: any) {
      console.error("[AION] Generation error:", err)
      toast.warning("AION connection issue — displaying test paper", {
        description: err?.message || "Using test paper layout"
      })
      onSuccess(buildTestPaper(config))
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
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
          <div>
            <div className="flex items-center gap-2 text-primary mb-1">
              <UploadCloud className="h-5 w-5" />
              <span className="font-semibold uppercase tracking-wider text-xs">Reference Material Upload</span>
            </div>
            <h2 className="text-2xl font-bold text-slate-800">Upload Content for Each Question</h2>
          </div>
          <div className="flex items-center gap-3 bg-white p-3 rounded-lg border shadow-sm">
            <Label className="text-sm font-semibold text-slate-700 whitespace-nowrap">Sub-questions per Question:</Label>
            <select
              className="h-9 rounded-md border border-input bg-background px-3 text-xs font-semibold focus:outline-none focus:ring-2 focus:ring-primary"
              value={sections[0]?.subQuestionsPerQ ?? 2}
              onChange={e => {
                const val = parseInt(e.target.value) || 2
                const newSections = sections.map(s => ({ ...s, subQuestionsPerQ: val }))
                setSections(newSections)
              }}
            >
              <option value={1}>1 Sub-question (10 Marks)</option>
              <option value={2}>2 Sub-questions (6+4 Marks)</option>
              <option value={3}>3 Sub-questions (4+3+3 Marks)</option>
            </select>
          </div>
        </div>
        <p className="text-slate-600 mb-6">
          Upload reference material for all 10 questions (5 OR pairs). The backend AI will generate sub-questions, assign marks, CO, and Bloom's levels.
        </p>

        <div className="space-y-4">
          {sections.map((section, idx) => {
            const isOrQuestion = idx % 2 === 1
            const hasContent = section.notesText && section.notesText.trim() !== ""
            return (
              <div key={idx}>
                {isOrQuestion && (
                  <div className="flex items-center gap-3 my-1 px-1">
                    <div className="h-px flex-1 bg-slate-200" />
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">OR</span>
                    <div className="h-px flex-1 bg-slate-200" />
                  </div>
                )}
                <Card className="shadow-sm border-slate-200">
                  <div className="bg-slate-50 border-b px-5 py-3 flex items-center justify-between">
                    <h3 className="font-semibold text-slate-800 flex items-center gap-2">
                      Question {idx + 1}
                      {isOrQuestion && (
                        <Badge variant="outline" className="text-[10px] border-slate-300 text-slate-500 font-normal">
                          OR alternative
                        </Badge>
                      )}
                    </h3>
                    <div className="flex items-center gap-3">
                      <label className="flex items-center gap-1.5">
                        <span className="text-xs text-muted-foreground whitespace-nowrap">Sub-questions</span>
                        <select
                          className="h-7 rounded-md border border-input bg-background px-2 text-xs focus:outline-none focus:ring-1 focus:ring-ring font-medium"
                          value={sections[idx]?.subQuestionsPerQ ?? 2}
                          onChange={e => {
                            const val = parseInt(e.target.value) || 2
                            const newSections = sections.map(s => ({ ...s, subQuestionsPerQ: val }))
                            setSections(newSections)
                          }}
                        >
                          <option value={1}>1 (10M)</option>
                          <option value={2}>2 (6+4M)</option>
                          <option value={3}>3 (4+3+3M)</option>
                        </select>
                      </label>
                      {hasContent && (
                        <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200">
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          Uploaded
                        </Badge>
                      )}
                    </div>
                  </div>
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-4">
                      <Button
                        variant="outline"
                        size="lg"
                        className="flex-1 h-20 border-dashed border-2 hover:border-primary hover:bg-primary/5"
                        onClick={() => fileInputRefs.current[idx]?.click()}
                      >
                        {hasContent ? (
                          <>
                            <FileText className="mr-2 h-5 w-5 text-emerald-600" />
                            <div className="text-left flex-1">
                              <div className="font-medium text-slate-800">{fileNames[idx] || "Content uploaded"}</div>
                              <div className="text-xs text-slate-500">Click to replace</div>
                            </div>
                          </>
                        ) : (
                          <>
                            <UploadCloud className="mr-2 h-5 w-5" />
                            Upload PDF, TXT, or DOCX
                          </>
                        )}
                      </Button>
                      <input
                        ref={el => { if (el) fileInputRefs.current[idx] = el }}
                        type="file"
                        className="hidden"
                        accept=".pdf,.txt,.docx"
                        onChange={(e) => handleFileUpload(idx, e)}
                      />
                    </div>
                  </CardContent>
                </Card>
                {isOrQuestion && idx < 9 && <div className="h-4" />}
              </div>
            )
          })}
        </div>
      </div>

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
