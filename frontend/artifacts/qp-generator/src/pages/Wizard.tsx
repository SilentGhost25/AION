import { useState, useRef } from "react"
import { PaperConfig, SectionInput, GeneratedPaper } from "@workspace/api-client-react"
import { Step1ConfigAndUpload } from "@/components/wizard/Step1ConfigAndUpload"
import { Step2Preview } from "@/components/wizard/Step2Preview"
import { format } from "date-fns"

export default function Wizard() {
  const [step, setStep] = useState<1 | 2>(1)

  const [config, setConfig] = useState<PaperConfig>({
    institutionDepartment: "",
    subjectName: "",
    subjectCode: "",
    examType: "IAT1",
    semester: 1,
    batch: "",
    maxMarks: 50,
    duration: "1.5 hrs",
    dateOfExam: format(new Date(), "yyyy-MM-dd"),
  })

  // One content slot per question (10 questions = 5 OR pairs). The faculty only
  // uploads reference material here — the backend decides sub-questions, marks,
  // CO and BL for every question.
  const [sections, setSections] = useState<SectionInput[]>(
    Array.from({ length: 10 }).map((_, i) => ({
      sectionNumber: i + 1,
      mainQuestions: 1,
      subQuestionsPerQ: 2,
      marks: 0,
      notesText: "",
    }))
  )

  const [generatedPaper, setGeneratedPaper] = useState<GeneratedPaper | null>(null)
  const [allPapers, setAllPapers] = useState<GeneratedPaper[]>([])
  const regenerateRef = useRef<(varIdx: number) => Promise<GeneratedPaper | null>>(undefined)

  return (
    <div className="min-h-screen bg-slate-50 print:bg-white text-slate-900">
      {/* Header - hide when printing */}
      <header className="bg-primary text-primary-foreground print:hidden py-4 px-6 shadow-md border-b border-primary/20">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold tracking-tight">DSATM AI Question Paper Generator</h1>
            <p className="text-sm text-primary-foreground/80">Faculty Assessment Portal</p>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-xs font-medium text-primary-foreground/70">
              Step {step} of 2
            </span>
            <div className="flex gap-2">
              <div className={`h-2 w-12 rounded-full ${step >= 1 ? 'bg-white' : 'bg-white/20'}`} />
              <div className={`h-2 w-12 rounded-full ${step >= 2 ? 'bg-white' : 'bg-white/20'}`} />
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto p-6 print:p-0 print:m-0 print:max-w-none">
        {step === 1 && (
          <Step1ConfigAndUpload
            config={config}
            setConfig={setConfig}
            sections={sections}
            setSections={setSections}
            registerRegenerate={(fn) => { regenerateRef.current = fn }}
            onSuccess={(paper) => {
              setGeneratedPaper(paper)
              setAllPapers([paper])
              setStep(2)
            }}
          />
        )}
        {step === 2 && generatedPaper && (
          <Step2Preview
            paper={generatedPaper}
            setPaper={setGeneratedPaper}
            allPapers={allPapers}
            setAllPapers={setAllPapers}
            onBack={() => setStep(1)}
            onRegenerate={async () => {
              if (!regenerateRef.current) return null
              return await regenerateRef.current(allPapers.length + 1)
            }}
          />
        )}
      </main>
    </div>
  )
}
