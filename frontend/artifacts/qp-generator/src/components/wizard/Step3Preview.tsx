import { GeneratedPaper } from "@workspace/api-client-react"
import { QuestionWithSubs, SubQuestion } from "./Step2Rules"
import { Button } from "@/components/ui/button"
import { ArrowLeft, Printer, AlertCircle } from "lucide-react"

export function Step3Preview({ 
  paper, 
  setPaper,
  onBack 
}: { 
  paper: GeneratedPaper, 
  setPaper: (p: GeneratedPaper) => void,
  onBack: () => void 
}) {
  const { config, questions, courseOutcomes } = paper
  const isSEE = config.examType === 'SEE'

  const coCoverage = (paper as any).coCoverage ?? {
    co1: 20, co2: 20, co3: 20, co4: 20, co5: 20
  }

  const syllabusCoverage = (paper as any).syllabusCoverage ?? {
    s1: 20, s2: 20, s3: 20, s4: 20, s5: 20
  }

  const validationReport = (paper as any).validationReport

  const handlePrint = () => {
    window.print()
  }

  // Edit the text of a single sub-question within a question block.
  const updateSubText = (qNo: number, subIndex: number, newText: string) => {
    const newQuestions = (questions as QuestionWithSubs[]).map(q => {
      if (q.qNo !== qNo || !q.subQuestions) return q
      const subQuestions = q.subQuestions.map((s, i) =>
        i === subIndex ? { ...s, text: newText } : s
      )
      return { ...q, subQuestions }
    })
    setPaper({ ...paper, questions: newQuestions })
  }

  return (
    <div className="max-w-[21cm] mx-auto animate-in fade-in pb-20 print:pb-0">
      
      {/* Editor Controls - Hidden in print */}
      <div className="flex items-center justify-between mb-6 print:hidden">
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Preview & Review</h2>
          <p className="text-sm text-slate-600">Review the generated paper. Click any question text to edit before exporting.</p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline" onClick={onBack}>
            <ArrowLeft className="mr-2 h-4 w-4" /> Back to Editor
          </Button>
          <Button onClick={handlePrint}>
            <Printer className="mr-2 h-4 w-4" /> Export PDF
          </Button>
        </div>
      </div>

      {/* Validation Report Banner — shown if paper has issues */}
      {validationReport && !validationReport.passed && (
        <div className="mb-4 border border-amber-200 bg-amber-50 rounded-lg p-4 print:hidden">
          <div className="flex items-center gap-2 font-semibold text-amber-800 mb-2">
            <AlertCircle className="h-4 w-4" />
            Paper Quality Issues Found ({validationReport.summary})
          </div>
          <div className="space-y-1">
            {validationReport.errors?.map((err: any, i: number) => (
              <div key={i} className="text-sm text-red-700">
                <span className="font-mono text-xs bg-red-100 px-1 rounded mr-2">
                  {err.code}
                </span>
                {err.message}
                {err.fix && (
                  <span className="text-red-500 ml-2 text-xs">Fix: {err.fix}</span>
                )}
              </div>
            ))}
            {validationReport.warnings?.map((w: any, i: number) => (
              <div key={i} className="text-sm text-amber-700">
                <span className="font-mono text-xs bg-amber-100 px-1 rounded mr-2">
                  {w.code}
                </span>
                {w.message}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* PAPER PREVIEW - A4 SIZE APPROXIMATION */}
      <div className="bg-white p-8 md:p-12 shadow-sm border border-slate-200 print:shadow-none print:border-none print:p-0">
        
        {/* Header Section — college identity with logos on both sides */}
        <div className="border-b-2 border-black pb-3 mb-4">
          <div className="flex items-start justify-between gap-4">
            {/* Left: college logo + name */}
            <div className="flex items-start gap-3 flex-1">
              <img src="/dsatm-logo.png" alt="DSATM logo" className="w-16 h-16 object-contain shrink-0" />
              <div className="leading-tight">
                <h1 className="text-base font-bold text-black">Dayananda Sagar Academy of Technology &amp; Management</h1>
                <p className="text-xs text-black">(Autonomous Institute under VTU)</p>
              </div>
            </div>

            {/* Right: accreditation block + IQAC seal */}
            <div className="flex items-start gap-2 shrink-0">
              <div className="text-[0.6rem] text-black leading-snug border-l border-black pl-2">
                <p>Affiliated to <span className="text-red-600 font-semibold">VTU</span></p>
                <p>Approved by <span className="text-red-600 font-semibold">AICTE</span></p>
                <p>Accredited by <span className="text-red-600 font-semibold">NAAC</span> with <span className="text-red-600 font-semibold">A+</span> Grade</p>
                <p>6 Programs Accredited by <span className="text-red-600 font-semibold">NBA</span></p>
                <p>(CSE, ISE, ECE, EEE, MECH, CV)</p>
              </div>
              <div className="flex flex-col items-center">
                <img src="/iqac-seal.jpg" alt="IQAC seal" className="w-12 h-12 object-contain" />
                <span className="text-[0.55rem] font-bold text-black mt-0.5">IQAC, DSATM</span>
              </div>
            </div>
          </div>
        </div>

        {/* USN Row */}
        <div className="flex justify-end items-center mb-4">
          <div className="flex items-center gap-2">
            <span className="font-bold text-black text-sm">USN:</span>
            <div className="flex">
              {Array.from({length: 10}).map((_, i) => (
                <div key={i} className="w-6 h-6 border border-black" />
              ))}
            </div>
          </div>
        </div>

        <div className="text-center mb-4">
          <h2 className="text-lg font-bold text-black">Department of {config.institutionDepartment}</h2>
        </div>

        {/* Metadata Table */}
        <div className="mb-4">
          <div className="text-center font-bold text-black border border-black border-b-0 py-1 text-sm">
            {config.examType === 'SEE' ? 'Semester End Examination' :
             config.examType === 'IAT1' ? 'First Internal Assessment Test (IAT-1)' :
             'Second Internal Assessment Test (IAT-2)'}
          </div>
          <table className="w-full border-collapse border border-black text-xs text-black">
            <tbody>
              <tr>
                <td className="border border-black p-1 font-bold w-[15%]">Subject:</td>
                <td className="border border-black p-1 w-[35%]">{config.subjectName}</td>
                <td className="border border-black p-1 font-bold w-[15%]">Subject Code:</td>
                <td className="border border-black p-1 w-[35%]">{config.subjectCode}</td>
              </tr>
              <tr>
                <td className="border border-black p-1 font-bold">Semester:</td>
                <td className="border border-black p-1">{config.semester}</td>
                <td className="border border-black p-1 font-bold">Max. Marks:</td>
                <td className="border border-black p-1">{config.maxMarks}</td>
              </tr>
              <tr>
                <td className="border border-black p-1 font-bold">Batch:</td>
                <td className="border border-black p-1">{config.batch}</td>
                <td className="border border-black p-1 font-bold">Duration:</td>
                <td className="border border-black p-1">{config.duration}</td>
              </tr>
              <tr>
                <td className="border border-black p-1 font-bold">Date of IAT:</td>
                <td className="border border-black p-1">{config.dateOfExam || '____/____/______'}</td>
                <td className="border border-black p-1 font-bold leading-tight">Teaching Department:</td>
                <td className="border border-black p-1">{config.institutionDepartment}</td>
              </tr>
              <tr>
                <td className="border border-black p-1 font-bold">RBT Levels:</td>
                <td colSpan={3} className="border border-black p-1">L1-Remember, L2-Understand, L3-Apply, L4-Analyze, L5-Evaluate, L6-Create</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Instruction */}
        <div className="text-center italic font-semibold text-sm text-black mb-3">
          Instruction: {isSEE
            ? 'Answer any FIVE full questions, choosing at least ONE question from each MODULE.'
            : 'Answer the following questions'}
        </div>

        {/* Questions Table */}
        <table className="w-full border-collapse border border-black text-sm text-black mb-8 break-inside-auto">
          <thead>
            <tr>
              <th className="border border-black p-1 w-[8%]">Q No</th>
              <th className="border border-black p-1 w-[68%]">Questions</th>
              <th className="border border-black p-1 w-[8%]">Marks</th>
              <th className="border border-black p-1 w-[8%]">CO's</th>
              <th className="border border-black p-1 w-[8%]">BL</th>
            </tr>
          </thead>
          <tbody>
            {renderQuestions(questions as QuestionWithSubs[], updateSubText, isSEE)}
          </tbody>
        </table>

        {/* Course Outcomes */}
        <div className="mb-6 print-break-inside-avoid">
          <h3 className="font-bold text-sm text-black mb-2">Course Outcomes (COs): At the end of the Course, the Student will be able to:</h3>
          <table className="w-full border-collapse border border-black text-xs text-black">
            <tbody>
              {courseOutcomes.map((co, i) => (
                <tr key={i}>
                  <td className="border border-black p-1 font-bold w-[10%] text-center">CO{i+1}</td>
                  <td className="border border-black p-1">{co}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Coverage Tables side-by-side */}
        <div className="flex gap-4 print-break-inside-avoid">
          <div className="flex-1">
            <h3 className="font-bold text-sm text-black mb-2 text-center">Percentage of CO Coverage</h3>
            <table className="w-full border-collapse border border-black text-xs text-black text-center">
              <tbody>
                <tr className="font-bold">
                  <td className="border border-black p-1">Course Outcomes</td>
                  <td className="border border-black p-1">CO1</td>
                  <td className="border border-black p-1">CO2</td>
                  <td className="border border-black p-1">CO3</td>
                  <td className="border border-black p-1">CO4</td>
                  <td className="border border-black p-1">CO5</td>
                </tr>
                <tr>
                  <td className="border border-black p-1 font-bold">Percentage</td>
                  <td className="border border-black p-1">{coCoverage.co1}%</td>
                  <td className="border border-black p-1">{coCoverage.co2}%</td>
                  <td className="border border-black p-1">{coCoverage.co3}%</td>
                  <td className="border border-black p-1">{coCoverage.co4}%</td>
                  <td className="border border-black p-1">{coCoverage.co5}%</td>
                </tr>
              </tbody>
            </table>
          </div>
          
          <div className="flex-1">
            <h3 className="font-bold text-sm text-black mb-2 text-center">Percentage of Syllabus Coverage</h3>
            <table className="w-full border-collapse border border-black text-xs text-black text-center">
              <tbody>
                <tr className="font-bold">
                  <td className="border border-black p-1">Modules Covered</td>
                  <td className="border border-black p-1">1</td>
                  <td className="border border-black p-1">2</td>
                  <td className="border border-black p-1">3</td>
                  <td className="border border-black p-1">4</td>
                  <td className="border border-black p-1">5</td>
                </tr>
                <tr>
                  <td className="border border-black p-1 font-bold">Percentage</td>
                  <td className="border border-black p-1">{syllabusCoverage.s1}%</td>
                  <td className="border border-black p-1">{syllabusCoverage.s2}%</td>
                  <td className="border border-black p-1">{syllabusCoverage.s3}%</td>
                  <td className="border border-black p-1">{syllabusCoverage.s4}%</td>
                  <td className="border border-black p-1">{syllabusCoverage.s5}%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  )
}

function QuestionFigures({ images, qNo }: { images?: string[]; qNo: number }) {
  if (!images || images.length === 0) return null
  return (
    <div className="flex flex-wrap gap-3 mt-2">
      {images.map((src, i) => (
        <img
          key={i}
          src={src}
          alt={`Figure ${i + 1} for question ${qNo}`}
          className="max-h-40 border border-black object-contain"
        />
      ))}
    </div>
  )
}

/** Lowercase roman numerals for nested part labels (i, ii, iii …). */
function toRoman(n: number): string {
  const map: [number, string][] = [[10, "x"], [9, "ix"], [5, "v"], [4, "iv"], [1, "i"]]
  let out = ""
  let rem = n
  for (const [val, sym] of map) {
    while (rem >= val) {
      out += sym
      rem -= val
    }
  }
  return out
}

/** Fallback single sub-question built from the flattened top-level fields. */
function fallbackSub(q: QuestionWithSubs): SubQuestion {
  return { label: "a", text: q.text, marks: q.marks, co: q.co, rbt: q.rbt, images: q.images }
}

/** Render all table rows for one question block (one <tr> per sub-question). */
function renderQuestionRows(
  q: QuestionWithSubs,
  updateSubText: (qNo: number, subIndex: number, text: string) => void,
): React.ReactNode[] {
  const subs = q.subQuestions && q.subQuestions.length > 0 ? q.subQuestions : [fallbackSub(q)]
  const multi = subs.length > 1

  return subs.map((sub, i) => (
    <tr key={`${q.qNo}-${i}`} className="print-break-inside-avoid">
      {i === 0 && (
        <td rowSpan={subs.length} className="border border-black p-2 text-center align-top">
          {q.qNo}.
        </td>
      )}
      <td className="border border-black p-2 align-top">
        <div className="flex gap-1.5">
          {multi && <span className="font-medium shrink-0">{sub.label})</span>}
          <div className="flex-1">
            <textarea
              value={sub.text}
              onChange={e => updateSubText(q.qNo, i, e.target.value)}
              className="w-full bg-transparent resize-none focus:outline-none min-h-[40px] hover:bg-slate-50 print:hover:bg-transparent print:resize-none transition-colors"
            />
            {sub.parts && sub.parts.length > 0 && (
              <div className="mt-1 space-y-0.5">
                {sub.parts.map((part, pi) => (
                  <div key={pi} className="flex gap-2 pl-4">
                    <span className="shrink-0">({toRoman(pi + 1)})</span>
                    <span>{part}</span>
                  </div>
                ))}
              </div>
            )}
            <QuestionFigures images={sub.images} qNo={q.qNo} />
          </div>
        </div>
      </td>
      <td className="border border-black p-2 text-center align-top">{sub.marks}</td>
      <td className="border border-black p-2 text-center align-top">{sub.co}</td>
      <td className="border border-black p-2 text-center align-top">{sub.rbt}</td>
    </tr>
  ))
}

function renderQuestions(
  questions: QuestionWithSubs[],
  updateSubText: (qNo: number, subIndex: number, text: string) => void,
  isSEE: boolean,
) {
  const rows: React.ReactNode[] = []

  // Group by OR pair (Q1 & Q2, Q3 & Q4 …). In SEE each pair is a Module.
  for (let pair = 0; pair < 5; pair++) {
    const q1 = questions.find(q => q.qNo === pair * 2 + 1)
    const q2 = questions.find(q => q.qNo === pair * 2 + 2)

    // SEE: full-width Module header before each pair.
    if (isSEE && (q1 || q2)) {
      rows.push(
        <tr key={`mod-${pair}`} className="print-break-inside-avoid">
          <td colSpan={5} className="border border-black p-1 text-center font-bold bg-gray-100 print:bg-white">
            Module - {pair + 1}
          </td>
        </tr>
      )
    }

    if (q1) rows.push(...renderQuestionRows(q1, updateSubText))

    if (q1 && q2) {
      rows.push(
        <tr key={`or-${pair}`} className="print-break-inside-avoid bg-gray-100/50 print:bg-white">
          <td colSpan={5} className="border border-black p-1 text-center font-bold">OR</td>
        </tr>
      )
    }

    if (q2) rows.push(...renderQuestionRows(q2, updateSubText))

    // Visual gap between pairs (screen only); SEE modules already read as grouped.
    if ((q1 || q2) && pair < 4 && !isSEE) {
      rows.push(
        <tr key={`sep-${pair}`} className="print:hidden">
          <td colSpan={5} className="h-4"></td>
        </tr>
      )
    }
  }

  return rows
}
