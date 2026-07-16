import React, { useRef } from "react";
import { Trash2, ImagePlus, X, GripVertical, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export interface PaperQuestion {
  id: string;
  text: string;
  marks: number;
  co: string;
  bloomLevel: string;
  module: number;
  diagram?: string; // base64 data URL or empty
  diagramCaption?: string;
}

interface PaperPreviewProps {
  formData?: any;
  questions: PaperQuestion[];
  editable?: boolean;
  onQuestionsChange?: (questions: PaperQuestion[]) => void;
}

const COs = ["CO1", "CO2", "CO3", "CO4", "CO5"];
const Blooms = ["L1", "L2", "L3", "L4", "L5", "L6"];
const BloomLabels: Record<string, string> = {
  L1: "Remember", L2: "Understand", L3: "Apply",
  L4: "Analyze", L5: "Evaluate", L6: "Create"
};

export function PaperPreview({ formData, questions, editable = false, onQuestionsChange }: PaperPreviewProps) {
  const fileInputRefs = useRef<Record<string, HTMLInputElement | null>>({});

  const defaults = {
    examType: "IAT-1",
    department: "Computer Science & Engineering",
    subjectName: "Machine Learning",
    subjectCode: "21AI51",
    semester: "5",
    maxMarks: 50,
    batch: "2022-26",
    duration: "1.5 hrs",
    dateOfIat: "2023-10-12",
    teachingDept: "AIML Dept"
  };
  const data = formData ? { ...defaults, ...formData } : defaults;

  const update = (idx: number, field: keyof PaperQuestion, value: any) => {
    if (!onQuestionsChange) return;
    const next = questions.map((q, i) => i === idx ? { ...q, [field]: value } : q);
    onQuestionsChange(next);
  };

  const deleteQuestion = (idx: number) => {
    if (!onQuestionsChange) return;
    onQuestionsChange(questions.filter((_, i) => i !== idx));
  };

  const addQuestion = () => {
    if (!onQuestionsChange) return;
    const newQ: PaperQuestion = {
      id: `q${Date.now()}`,
      text: "New question — click to edit",
      marks: 10,
      co: "CO1",
      bloomLevel: "L1",
      module: 1,
    };
    onQuestionsChange([...questions, newQ]);
  };

  const handleDiagramUpload = (idx: number, e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !onQuestionsChange) return;
    const reader = new FileReader();
    reader.onload = () => {
      update(idx, "diagram", reader.result as string);
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  const removeDiagram = (idx: number) => {
    if (!onQuestionsChange) return;
    update(idx, "diagram", "");
    update(idx, "diagramCaption", "");
  };

  return (
    <div className={`font-serif text-[11pt] leading-tight space-y-4 text-black ${editable ? "select-text" : ""}`}>

      {/* ── DSATM Header ── */}
      <div className="text-center border-b-2 border-black pb-4 space-y-1">
        <h1 className="font-bold text-lg">Dayananda Sagar Academy of Technology &amp; Management</h1>
        <p className="text-sm">(Autonomous Institute under VTU) Affiliated to VTU | Approved by AICTE</p>
        <p className="text-sm">Accredited by NAAC with A+ Grade | 6 Programs Accredited by NBA</p>
        <p className="font-bold pt-2">Department of {data.department}</p>
        <p className="font-bold underline">{data.examType}</p>
      </div>

      {/* ── Meta Info Table ── */}
      <table className="w-full text-sm border-collapse border border-black mb-4">
        <tbody>
          <tr>
            <td className="border border-black p-1 font-bold w-1/4">Subject / Course:</td>
            <td className="border border-black p-1 w-1/4">{data.subjectName}</td>
            <td className="border border-black p-1 font-bold w-1/4">Subject Code:</td>
            <td className="border border-black p-1 w-1/4">{data.subjectCode}</td>
          </tr>
          <tr>
            <td className="border border-black p-1 font-bold">Semester:</td>
            <td className="border border-black p-1">{data.semester}</td>
            <td className="border border-black p-1 font-bold">Max. Marks:</td>
            <td className="border border-black p-1">{data.maxMarks}</td>
          </tr>
          <tr>
            <td className="border border-black p-1 font-bold">Batch:</td>
            <td className="border border-black p-1">{data.batch}</td>
            <td className="border border-black p-1 font-bold">Duration:</td>
            <td className="border border-black p-1">{data.duration}</td>
          </tr>
          <tr>
            <td className="border border-black p-1 font-bold">Date of IAT:</td>
            <td className="border border-black p-1">{data.dateOfIat}</td>
            <td className="border border-black p-1 font-bold">Teaching Dept:</td>
            <td className="border border-black p-1">{data.teachingDept}</td>
          </tr>
        </tbody>
      </table>

      {/* ── RBT Legend ── */}
      <div className="text-xs space-y-1 my-4 bg-gray-50 p-2 border border-gray-300">
        <p><span className="font-bold">Course Outcomes (COs):</span> CO1, CO2, CO3, CO4, CO5</p>
        <p><span className="font-bold">Revised Bloom's Taxonomy (RBT) Levels:</span> L1-Remember, L2-Understand, L3-Apply, L4-Analyze, L5-Evaluate, L6-Create</p>
      </div>

      {/* ── Instruction ── */}
      <div className="font-bold text-center my-4">
        Instruction: Answer the following questions
      </div>

      {/* ── Questions Table ── */}
      <table className="w-full text-sm border-collapse border border-black mb-4">
        <thead>
          <tr className="bg-gray-100">
            <th className="border border-black p-2 w-10 text-center">Q.No</th>
            <th className="border border-black p-2 text-left">Questions</th>
            <th className="border border-black p-2 w-16 text-center">Marks</th>
            <th className="border border-black p-2 w-14 text-center">COs</th>
            <th className="border border-black p-2 w-14 text-center">RBTL</th>
            {editable && <th className="border border-black p-2 w-10 text-center bg-blue-50 text-blue-700 text-xs font-semibold">Edit</th>}
          </tr>
        </thead>
        <tbody>
          {questions.map((q, idx) => {
            const isOr = idx % 2 === 1;
            return (
              <React.Fragment key={q.id}>
                {isOr && (
                  <tr>
                    <td
                      colSpan={editable ? 6 : 5}
                      className="border-x border-black p-1 text-center font-bold bg-gray-50"
                    >
                      OR
                    </td>
                  </tr>
                )}
                <tr className={editable ? "bg-blue-50/30 hover:bg-blue-50/60 transition-colors" : ""}>
                  <td className="border border-black p-2 text-center align-top font-bold">{idx + 1}</td>

                  {/* Question Text Cell */}
                  <td className="border border-black p-2 align-top">
                    {editable ? (
                      <div className="space-y-2">
                        <Textarea
                          value={q.text}
                          onChange={e => update(idx, "text", e.target.value)}
                          className="text-sm font-serif border-dashed border-blue-300 focus:border-blue-500 bg-white min-h-[80px] resize-none leading-snug p-1.5"
                          rows={3}
                        />
                        {/* Diagram area */}
                        {q.diagram ? (
                          <div className="border border-dashed border-gray-400 rounded p-2 space-y-2 bg-white">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-gray-600 flex items-center gap-1">
                                <ImagePlus className="h-3 w-3" /> Diagram
                              </span>
                              <button
                                onClick={() => removeDiagram(idx)}
                                className="text-red-500 hover:text-red-700 p-0.5 rounded"
                                title="Remove diagram"
                              >
                                <X className="h-3.5 w-3.5" />
                              </button>
                            </div>
                            <img
                              src={q.diagram}
                              alt="Question diagram"
                              className="max-h-48 max-w-full object-contain mx-auto border border-gray-200 rounded"
                            />
                            <input
                              type="text"
                              value={q.diagramCaption || ""}
                              onChange={e => update(idx, "diagramCaption", e.target.value)}
                              placeholder="Add a caption for this diagram..."
                              className="w-full text-xs border border-dashed border-gray-300 rounded px-2 py-1 bg-white focus:outline-none focus:border-blue-400"
                            />
                          </div>
                        ) : (
                          <div>
                            <input
                              type="file"
                              accept="image/*"
                              ref={el => { fileInputRefs.current[q.id] = el; }}
                              onChange={e => handleDiagramUpload(idx, e)}
                              className="hidden"
                            />
                            <button
                              onClick={() => fileInputRefs.current[q.id]?.click()}
                              className="flex items-center gap-1.5 text-xs text-blue-600 hover:text-blue-800 border border-dashed border-blue-300 hover:border-blue-500 rounded px-2.5 py-1.5 bg-white hover:bg-blue-50 transition-colors"
                            >
                              <ImagePlus className="h-3.5 w-3.5" />
                              Add diagram / figure
                            </button>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div>
                        <p>{q.text}</p>
                        {q.diagram && (
                          <div className="mt-3 space-y-1">
                            <img
                              src={q.diagram}
                              alt="Question diagram"
                              className="max-h-56 max-w-full object-contain mx-auto border border-gray-300"
                            />
                            {q.diagramCaption && (
                              <p className="text-center text-xs text-gray-600 italic">{q.diagramCaption}</p>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </td>

                  {/* Marks Cell */}
                  <td className="border border-black p-2 text-center align-top">
                    {editable ? (
                      <input
                        type="number"
                        value={q.marks}
                        onChange={e => update(idx, "marks", Number(e.target.value))}
                        className="w-12 text-center border border-dashed border-blue-300 focus:border-blue-500 rounded p-1 bg-white text-sm focus:outline-none"
                        min={1}
                        max={100}
                      />
                    ) : (
                      q.marks
                    )}
                  </td>

                  {/* CO Cell */}
                  <td className="border border-black p-2 text-center align-top">
                    {editable ? (
                      <Select value={q.co} onValueChange={v => update(idx, "co", v)}>
                        <SelectTrigger className="h-8 w-16 text-xs border-dashed border-blue-300 bg-white px-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {COs.map(co => <SelectItem key={co} value={co} className="text-xs">{co}</SelectItem>)}
                        </SelectContent>
                      </Select>
                    ) : (
                      q.co
                    )}
                  </td>

                  {/* Bloom Level Cell */}
                  <td className="border border-black p-2 text-center align-top">
                    {editable ? (
                      <Select value={q.bloomLevel} onValueChange={v => update(idx, "bloomLevel", v)}>
                        <SelectTrigger className="h-8 w-16 text-xs border-dashed border-blue-300 bg-white px-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {Blooms.map(b => <SelectItem key={b} value={b} className="text-xs">{b} — {BloomLabels[b]}</SelectItem>)}
                        </SelectContent>
                      </Select>
                    ) : (
                      q.bloomLevel
                    )}
                  </td>

                  {/* Edit Actions Cell */}
                  {editable && (
                    <td className="border border-black p-2 text-center align-top">
                      <button
                        onClick={() => deleteQuestion(idx)}
                        className="text-red-500 hover:text-red-700 p-1 rounded hover:bg-red-50 transition-colors"
                        title="Remove this question"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </td>
                  )}
                </tr>
              </React.Fragment>
            );
          })}
        </tbody>
      </table>

      {/* ── Add Question button (edit mode only) ── */}
      {editable && (
        <div className="flex justify-center my-2">
          <button
            onClick={addQuestion}
            className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-800 border border-dashed border-blue-400 hover:border-blue-600 rounded-md px-4 py-2 bg-blue-50/50 hover:bg-blue-50 transition-colors font-medium"
          >
            <Plus className="h-4 w-4" />
            Add another question
          </button>
        </div>
      )}

      {/* ── CO Outcomes Table ── */}
      <div className="mt-8">
        <h3 className="font-bold mb-2">Course Outcomes:</h3>
        <table className="w-full text-sm border-collapse border border-black">
          <thead>
            <tr className="bg-gray-100">
              <th className="border border-black p-2 w-16 text-center">COs</th>
              <th className="border border-black p-2 text-left">Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-black p-2 text-center font-bold">CO1</td>
              <td className="border border-black p-2">Understand the fundamental concepts of the subject.</td>
            </tr>
            <tr>
              <td className="border border-black p-2 text-center font-bold">CO2</td>
              <td className="border border-black p-2">Apply algorithms and methodologies to solve real-world problems.</td>
            </tr>
            <tr>
              <td className="border border-black p-2 text-center font-bold">CO3</td>
              <td className="border border-black p-2">Analyze and evaluate the performance and trade-offs of different approaches.</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* ── Signature Block ── */}
      <div className="mt-8 flex justify-between text-sm font-bold">
        <div className="text-center">
          <div className="h-12 border-b border-black w-40" />
          <p className="mt-1">Faculty Signature</p>
        </div>
        <div className="text-center">
          <div className="h-12 border-b border-black w-40" />
          <p className="mt-1">HOD Signature</p>
        </div>
      </div>
    </div>
  );
}
