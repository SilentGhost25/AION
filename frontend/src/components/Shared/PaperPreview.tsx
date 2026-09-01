import React, { useRef } from "react";
import { Trash2, ImagePlus, X, Plus } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import MathText from "./MathText";

// ── Sub-part shape (for 6+4 or 5+5 splits) ────────────────────────────────────
export interface QuestionSubPart {
  label: string;   // "a", "b"
  text: string;
  marks: number;
}

// ── Question shape ─────────────────────────────────────────────────────────────
export interface PaperQuestion {
  id: string;
  text: string;
  marks: number;
  co: string;
  bloom: string;       // "L1" … "L6"
  module: number;
  diagram?: string;    // base64 data URL
  diagramCaption?: string;
  // legacy alias still accepted
  bloomLevel?: string;
  // optional sub-parts: if present the question is split (6+4 or 5+5)
  subParts?: QuestionSubPart[];
}

// ── Course Outcome description ─────────────────────────────────────────────────
export interface CourseOutcome {
  code: string;   // "CO1"
  text: string;
}

// ── Props ──────────────────────────────────────────────────────────────────────
interface PaperPreviewProps {
  formData?: {
    examType?: string;
    department?: string;
    subjectName?: string;
    subjectCode?: string;
    semester?: string;
    maxMarks?: number | string;
    batch?: string;
    duration?: string;
    dateOfIat?: string;
    teachingDept?: string;
  };
  questions: PaperQuestion[];
  courseOutcomes?: CourseOutcome[];
  editable?: boolean;
  onQuestionsChange?: (questions: PaperQuestion[]) => void;
}

const COs    = ["CO1","CO2","CO3","CO4","CO5"];
const Blooms = ["L1","L2","L3","L4","L5","L6"];
const BloomLabels: Record<string,string> = {
  L1:"Remember", L2:"Understand", L3:"Apply",
  L4:"Analyze",  L5:"Evaluate",   L6:"Create",
};

const EXAM_FULL_NAMES: Record<string,string> = {
  "IAT-1": "First Internal Assessment Test (IAT-1)",
  "IAT-2": "Second Internal Assessment Test (IAT-2)",
  "SEE":   "Semester End Examination (SEE)",
};

const DEFAULT_COS: CourseOutcome[] = [
  { code:"CO1", text:"" }, { code:"CO2", text:"" },
  { code:"CO3", text:"" }, { code:"CO4", text:"" },
  { code:"CO5", text:"" },
];

const USN_BOXES = 10;

// ── Main component ─────────────────────────────────────────────────────────────
export function PaperPreview({
  formData,
  questions,
  courseOutcomes,
  editable = false,
  onQuestionsChange,
}: PaperPreviewProps) {
  const fileInputRefs = useRef<Record<string, HTMLInputElement | null>>({});

  const d = {
    examType:    "IAT-1",
    department:  "Artificial Intelligence and Machine Learning",
    subjectName: "Machine Learning",
    subjectCode: "21AI51",
    semester:    "5",
    maxMarks:    50,
    batch:       "2022-26",
    duration:    "1.5 hrs",
    dateOfIat:   "",
    teachingDept:"AI & ML",
    ...formData,
  };

  const [cos, setCos] = React.useState<CourseOutcome[]>(courseOutcomes ?? DEFAULT_COS);

  React.useEffect(() => {
    if (courseOutcomes) {
      setCos(courseOutcomes);
    }
  }, [courseOutcomes]);

  const updateCO = (code: string, newText: string) => {
    setCos((prev: CourseOutcome[]) => prev.map((c: CourseOutcome) => c.code === code ? { ...c, text: newText } : c));
  };

  // Normalise bloom — support both `bloom` and legacy `bloomLevel`
  const qs: PaperQuestion[] = questions.map(q => ({
    ...q,
    bloom: q.bloom || q.bloomLevel || "L1",
  }));

  // ── helpers ────────────────────────────────────────────────────────────────
  const update = (idx: number, field: keyof PaperQuestion, value: unknown) => {
    if (!onQuestionsChange) return;
    onQuestionsChange(qs.map((q, i) => i === idx ? { ...q, [field]: value } : q));
  };

  const deleteQ = (idx: number) => {
    if (!onQuestionsChange) return;
    onQuestionsChange(qs.filter((_, i) => i !== idx));
  };

  const addQ = () => {
    if (!onQuestionsChange) return;
    onQuestionsChange([...qs, {
      id: `q${Date.now()}`, text: "", marks: 10,
      co: "CO1", bloom: "L1", module: 1,
    }]);
  };

  const onDiagramChange = (idx: number, e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !onQuestionsChange) return;
    const reader = new FileReader();
    reader.onload = () => update(idx, "diagram", reader.result as string);
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  // ── CO coverage % - computed from questions ─────────────────────────────────
  const coTotals: Record<string, number> = {};
  let totalMarks = 0;
  qs.forEach(q => {
    coTotals[q.co] = (coTotals[q.co] ?? 0) + q.marks;
    totalMarks += q.marks;
  });

  const modTotals: Record<number, number> = {};
  qs.forEach(q => {
    modTotals[q.module] = (modTotals[q.module] ?? 0) + q.marks;
  });

  // ── render ─────────────────────────────────────────────────────────────────
  return (
    <div className="font-sans text-[10pt] leading-snug text-black bg-white w-full"
         style={{ fontFamily: "'Times New Roman', serif" }}>

      {/* ════════════════════════════════════════════════════════════════════
          HEADER — logo · institution name · accreditations · IQAC logo
      ════════════════════════════════════════════════════════════════════ */}
      <table className="w-full mb-0" style={{ borderCollapse: "collapse" }}>
        <tbody>
          <tr>
            {/* Left logo — DSATM circular badge */}
            <td style={{ width: 72, verticalAlign: "middle", padding: "4px 8px 4px 4px" }}>
              <img src="/logo-dsatm.png" alt="DSATM Logo"
                   style={{ width: 64, height: 64, objectFit: "contain", display: "block" }} />
            </td>

            {/* Institution name */}
            <td style={{ verticalAlign: "middle", padding: "4px 8px" }}>
              <div style={{ fontWeight: "bold", fontSize: 13 }}>
                Dayananda Sagar Academy of Technology &amp; Management
              </div>
              <div style={{ fontSize: 10 }}>(Autonomous Institute under VTU)</div>
            </td>

            {/* Accreditations — bordered cell */}
            <td style={{
              verticalAlign: "middle", padding: "4px 8px",
              borderLeft: "1px solid #333", fontSize: 9, lineHeight: 1.5,
              minWidth: 190,
            }}>
              <div>Affiliated to <span style={{ color: "#c00", fontWeight: "bold" }}>VTU</span></div>
              <div>Approved by <span style={{ color: "#c00", fontWeight: "bold" }}>AICTE</span></div>
              <div>Accredited by <span style={{ color: "#c00", fontWeight: "bold" }}>NAAC</span> with A+ Grade</div>
              <div>4 Programs Accredited by <span style={{ color: "#c00", fontWeight: "bold" }}>NBA</span></div>
              <div>(CSE, ISE, ECE, MECH)</div>
            </td>

            {/* IQAC logo */}
            <td style={{ width: 70, verticalAlign: "middle", padding: "4px 4px 4px 8px", textAlign: "center" }}>
              <img src="/logo-iqac.png" alt="IQAC Logo"
                   style={{ width: 60, height: 60, objectFit: "contain", display: "block", margin: "0 auto" }} />
            </td>
          </tr>
        </tbody>
      </table>

      {/* Thick bottom border under header */}
      <div style={{ borderBottom: "3px solid black", marginBottom: 6 }} />

      {/* ════════════════════════════════════════════════════════════════════
          USN ROW
      ════════════════════════════════════════════════════════════════════ */}
      <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center",
                    padding: "2px 4px 4px", gap: 4 }}>
        <span style={{ fontWeight: "bold", fontSize: 10 }}>USN:</span>
        {Array.from({ length: USN_BOXES }).map((_, i) => (
          <div key={i} style={{
            width: 22, height: 22, border: "1px solid black",
            display: "inline-block",
          }} />
        ))}
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          DEPARTMENT + EXAM TYPE
      ════════════════════════════════════════════════════════════════════ */}
      <div style={{ textAlign: "center", fontWeight: "bold", fontSize: 12,
                    padding: "4px 0 2px" }}>
        Department of {d.department}
      </div>

      {/* Exam type in a bordered box */}
      <table className="w-full" style={{ borderCollapse: "collapse", marginBottom: 0 }}>
        <tbody>
          <tr>
            <td style={{ border: "1px solid black", textAlign: "center",
                         fontWeight: "bold", fontSize: 11, padding: "3px 8px" }}>
              {EXAM_FULL_NAMES[d.examType] ?? d.examType}
            </td>
          </tr>
        </tbody>
      </table>

      {/* ════════════════════════════════════════════════════════════════════
          META INFO TABLE
      ════════════════════════════════════════════════════════════════════ */}
      <table className="w-full" style={{ borderCollapse: "collapse", borderTop: "none" }}>
        <tbody>
          {/* Row 1 */}
          <tr>
            <td style={cellL}>Subject:</td>
            <td style={cellV}>{d.subjectName}</td>
            <td style={cellL}>Subject Code:</td>
            <td style={cellV}>{d.subjectCode}</td>
          </tr>
          {/* Row 2 */}
          <tr>
            <td style={cellL}>Semester:</td>
            <td style={cellV}>{d.semester}</td>
            <td style={cellL}>Max. Marks:</td>
            <td style={cellV}>{d.maxMarks}</td>
          </tr>
          {/* Row 3 */}
          <tr>
            <td style={cellL}>Batch:</td>
            <td style={cellV}>{d.batch}</td>
            <td style={cellL}>Duration:</td>
            <td style={cellV}>{d.duration}</td>
          </tr>
          {/* Row 4 */}
          <tr>
            <td style={cellL}>Date of IAT:</td>
            <td style={cellV}>{d.dateOfIat}</td>
            <td style={cellL}>Teaching Department:</td>
            <td style={cellV}>{d.teachingDept}</td>
          </tr>
          {/* Row 5 — RBT Levels */}
          <tr>
            <td style={cellL}>RBT Levels:</td>
            <td colSpan={3} style={{ ...cellV, fontStyle: "normal" }}>
              L1-Remember, L2-Understand, L3-Apply, L4-Analyze, L5-Evaluate, L6-Create
            </td>
          </tr>
        </tbody>
      </table>

      {/* ════════════════════════════════════════════════════════════════════
          INSTRUCTION
      ════════════════════════════════════════════════════════════════════ */}
      <div style={{ textAlign: "center", fontStyle: "italic", fontSize: 10,
                    padding: "5px 0 4px", fontWeight: "normal" }}>
        Instruction: Answer the following questions
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          QUESTIONS TABLE
      ════════════════════════════════════════════════════════════════════ */}
      <table className="w-full" style={{ borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={thStyle} className="w-10 text-center">Q No</th>
            <th style={thStyle} className="text-center">Questions</th>
            <th style={{ ...thStyle, width: 52, textAlign: "center" }}>Marks</th>
            <th style={{ ...thStyle, width: 42, textAlign: "center" }}>COs</th>
            <th style={{ ...thStyle, width: 42, textAlign: "center" }}>RBTL</th>
            {editable && <th style={{ ...thStyle, width: 36, backgroundColor: "#e8f0fe", textAlign: "center" }}>✎</th>}
          </tr>
        </thead>
        <tbody>
          {qs.map((q, idx) => {
            const showOr = idx > 0 && idx % 2 === 1;
            const qNum  = idx + 1;

            return (
              <React.Fragment key={q.id}>
                {/* OR spacer row */}
                {showOr && (
                  <tr>
                    <td colSpan={editable ? 6 : 5}
                        style={{ border: "1px solid black", textAlign: "center",
                                 fontWeight: "bold", fontSize: 10, padding: "1px 0",
                                 backgroundColor: "#fafafa" }}>
                      OR
                    </td>
                  </tr>
                )}

                {/* Question row */}
                <tr style={{ backgroundColor: editable ? "rgba(219,234,254,0.15)" : "white" }}>
                  {/* Q No */}
                  <td style={{ ...tdBase, textAlign: "center", fontWeight: "bold",
                                verticalAlign: "top", paddingTop: 6, width: 36 }}>
                    {qNum}.
                  </td>

                  {/* Question text */}
                  <td style={{ ...tdBase, verticalAlign: "top", minHeight: 36 }}>
                    {editable ? (
                      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        <Textarea
                          value=<MathText text={q.text} />
                          onChange={e => update(idx, "text", e.target.value)}
                          placeholder="Enter question text…"
                          className="text-sm border-dashed border-blue-300 focus:border-blue-500 bg-white resize-none leading-snug"
                          rows={3}
                          style={{ fontFamily: "'Times New Roman', serif", fontSize: 10 }}
                        />
                        {/* Diagram */}
                        {q.diagram ? (
                          <div style={{ border: "1px dashed #aaa", borderRadius: 4,
                                        padding: 6, background: "#fafafa" }}>
                            <div style={{ display: "flex", justifyContent: "space-between",
                                          marginBottom: 4, alignItems: "center" }}>
                              <span style={{ fontSize: 9, fontWeight: 600, color: "#555" }}>
                                📷 Diagram
                              </span>
                              <button onClick={() => { update(idx,"diagram",""); update(idx,"diagramCaption",""); }}
                                      style={{ background: "none", border: "none", cursor: "pointer",
                                               color: "#c00", padding: 2 }} title="Remove">
                                <X className="h-3.5 w-3.5" />
                              </button>
                            </div>
                            <img src={q.diagram} alt="diagram"
                                 style={{ maxHeight: 160, maxWidth: "100%", objectFit: "contain",
                                          display: "block", margin: "0 auto", border: "1px solid #ddd" }} />
                            <input type="text" value={q.diagramCaption || ""}
                                   onChange={e => update(idx,"diagramCaption",e.target.value)}
                                   placeholder="Caption (optional)…"
                                   style={{ marginTop: 4, width: "100%", fontSize: 9,
                                            border: "1px dashed #ccc", borderRadius: 3,
                                            padding: "2px 6px", background: "white" }} />
                          </div>
                        ) : (
                          <div>
                            <input type="file" accept="image/*"
                                   ref={el => { fileInputRefs.current[q.id] = el; }}
                                   onChange={e => onDiagramChange(idx, e)}
                                   style={{ display: "none" }} />
                            <button
                              onClick={() => fileInputRefs.current[q.id]?.click()}
                              style={{ display: "flex", alignItems: "center", gap: 5,
                                       fontSize: 9, color: "#2563eb", cursor: "pointer",
                                       border: "1px dashed #93c5fd", borderRadius: 3,
                                       padding: "3px 8px", background: "#eff6ff" }}>
                              <ImagePlus className="h-3 w-3" />
                              Attach diagram
                            </button>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div>
                        {/* ── Sub-part split (6+4 or 5+5) ── */}
                        {q.subParts && q.subParts.length > 0 ? (
                          <div style={{ fontSize: 10 }}>
                            {q.subParts.map((sp, si) => (
                              <div key={sp.label} style={{ marginBottom: si < q.subParts!.length - 1 ? 5 : 0, display: "flex", gap: 4 }}>
                                <span style={{ fontWeight: "bold", minWidth: 14, flexShrink: 0 }}>{sp.label}.</span>
                                <span style={{ flex: 1 }}>{sp.text}</span>
                                <span style={{ marginLeft: 6, whiteSpace: "nowrap", color: "#333",
                                               fontStyle: "italic", flexShrink: 0 }}>
                                  ({sp.marks} M)
                                </span>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <span style={{ fontSize: 10 }}><MathText text={q.text} /></span>
                        )}
                        {q.diagram && (
                          <div style={{ marginTop: 8, textAlign: "center" }}>
                            <img src={q.diagram} alt="diagram"
                                 style={{ maxHeight: 180, maxWidth: "100%", objectFit: "contain",
                                          border: "1px solid #ccc" }} />
                            {q.diagramCaption && (
                              <div style={{ fontSize: 8, color: "#555", marginTop: 3, fontStyle: "italic" }}>
                                {q.diagramCaption}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </td>

                  {/* Marks */}
                  <td style={{ ...tdBase, textAlign: "center", verticalAlign: "top",
                                paddingTop: 6, width: 52 }}>
                    {editable ? (
                      <input type="number" value={q.marks}
                             onChange={e => update(idx,"marks",Number(e.target.value))}
                             style={{ width: 40, textAlign: "center", border: "1px dashed #93c5fd",
                                      borderRadius: 3, padding: "2px 4px", fontSize: 10,
                                      background: "white" }}
                             min={1} max={100} />
                    ) : (
                      <span style={{ fontSize: 10 }}>{q.marks}</span>
                    )}
                  </td>

                  {/* COs */}
                  <td style={{ ...tdBase, textAlign: "center", verticalAlign: "top",
                                paddingTop: 6, width: 42 }}>
                    {editable ? (
                      <Select value={q.co} onValueChange={v => update(idx,"co",v)}>
                        <SelectTrigger className="h-7 w-14 text-xs border-dashed border-blue-300 bg-white px-1 mx-auto">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {COs.map(c => <SelectItem key={c} value={c} className="text-xs">{c}</SelectItem>)}
                        </SelectContent>
                      </Select>
                    ) : (
                      <span style={{ fontSize: 10 }}>{q.co}</span>
                    )}
                  </td>

                  {/* RBTL */}
                  <td style={{ ...tdBase, textAlign: "center", verticalAlign: "top",
                                paddingTop: 6, width: 42 }}>
                    {editable ? (
                      <Select value={q.bloom} onValueChange={v => update(idx,"bloom",v)}>
                        <SelectTrigger className="h-7 w-14 text-xs border-dashed border-blue-300 bg-white px-1 mx-auto">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {Blooms.map(b => (
                            <SelectItem key={b} value={b} className="text-xs">
                              {b} — {BloomLabels[b]}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    ) : (
                      <span style={{ fontSize: 10 }}>{q.bloom}</span>
                    )}
                  </td>

                  {/* Edit action */}
                  {editable && (
                    <td style={{ ...tdBase, textAlign: "center", verticalAlign: "top",
                                  paddingTop: 6, width: 36 }}>
                      <button onClick={() => deleteQ(idx)}
                              style={{ background: "none", border: "none", cursor: "pointer",
                                       color: "#dc2626", padding: 3 }} title="Delete question">
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

      {/* Add question button — edit mode only */}
      {editable && (
        <div style={{ display: "flex", justifyContent: "center", padding: "6px 0" }}>
          <button onClick={addQ}
                  style={{ display: "flex", alignItems: "center", gap: 6,
                           fontSize: 11, color: "#2563eb", border: "1px dashed #93c5fd",
                           borderRadius: 4, padding: "4px 14px", background: "#eff6ff",
                           cursor: "pointer", fontWeight: 500 }}>
            <Plus className="h-3.5 w-3.5" />
            Add question
          </button>
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════
          COURSE OUTCOMES TABLE
      ════════════════════════════════════════════════════════════════════ */}
      <div style={{ marginTop: 14 }}>
        <div style={{ fontWeight: "bold", fontSize: 10, marginBottom: 2 }}>
          Course Outcomes (COs):&nbsp; At the end of the Course, the Student will be able to:
        </div>
        <table className="w-full" style={{ borderCollapse: "collapse" }}>
          <tbody>
            {cos.map(co => (
              <tr key={co.code}>
                <td style={{ border: "1px solid black", fontWeight: "bold",
                             fontSize: 10, padding: "3px 8px", width: 44,
                             textAlign: "left", whiteSpace: "nowrap" }}>
                  {co.code}
                </td>
                <td style={{ border: "1px solid black", fontSize: 10,
                             padding: "3px 8px", minHeight: 20 }}>
                  {editable ? (
                    <input
                      type="text"
                      value={co.text}
                      onChange={e => updateCO(co.code, e.target.value)}
                      style={{ width: "100%", background: "transparent", border: "none", outline: "none" }}
                      placeholder="Click to edit course outcome..."
                    />
                  ) : (
                    co.text
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          PERCENTAGE OF CO COVERAGE (preview only, hidden in print/download)
      ════════════════════════════════════════════════════════════════════ */}
      <div className="print:hidden no-print" style={{ marginTop: 10 }}>
        <div style={{ fontWeight: "bold", fontSize: 10, marginBottom: 2, paddingLeft: 8 }}>
          Percentage of CO Coverage
        </div>
        <table className="w-full" style={{ borderCollapse: "collapse" }}>
          <tbody>
            <tr>
              <td style={{ ...covLabelCell, width: 130 }}>Course Outcomes</td>
              {["CO1","CO2","CO3","CO4","CO5"].map(co => (
                <td key={co} style={{ ...covHeaderCell }}>{co}</td>
              ))}
            </tr>
            <tr>
              <td style={covLabelCell}>Percentage</td>
              {["CO1","CO2","CO3","CO4","CO5"].map(co => {
                const pct = totalMarks > 0
                  ? Math.round(((coTotals[co] ?? 0) / totalMarks) * 100)
                  : 0;
                return (
                  <td key={co} style={{ ...covValueCell }}>
                    {pct > 0 ? `${pct}%` : ""}
                  </td>
                );
              })}
            </tr>
          </tbody>
        </table>
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          PERCENTAGE OF SYLLABUS COVERAGE (preview only, hidden in print/download)
      ════════════════════════════════════════════════════════════════════ */}
      <div className="print:hidden no-print" style={{ marginTop: 10, marginBottom: 6 }}>
        <div style={{ fontWeight: "bold", fontSize: 10, marginBottom: 2 }}>
          Percentage of Syllabus coverage
        </div>
        <table className="w-full" style={{ borderCollapse: "collapse" }}>
          <tbody>
            <tr>
              <td style={{ ...covLabelCell, width: 130 }}>Modules Covered</td>
              {[1,2,3,4,5].map(m => (
                <td key={m} style={covHeaderCell}>{m}</td>
              ))}
            </tr>
            <tr>
              <td style={covLabelCell}>Percentage</td>
              {[1,2,3,4,5].map(m => {
                const pct = totalMarks > 0
                  ? Math.round(((modTotals[m] ?? 0) / totalMarks) * 100)
                  : 0;
                return (
                  <td key={m} style={covValueCell}>
                    {pct > 0 ? `${pct}%` : ""}
                  </td>
                );
              })}
            </tr>
          </tbody>
        </table>
      </div>

    </div>
  );
}

// ── Shared cell styles (inline — survives print/export) ────────────────────────
const cellBorder = "1px solid black";

const cellL: React.CSSProperties = {
  border: cellBorder, fontWeight: "bold", fontSize: 10,
  padding: "3px 8px", whiteSpace: "nowrap", width: "18%",
};

const cellV: React.CSSProperties = {
  border: cellBorder, fontSize: 10, padding: "3px 8px",
};

const thStyle: React.CSSProperties = {
  border: cellBorder, fontWeight: "bold", fontSize: 10,
  padding: "4px 6px", backgroundColor: "#f0f0f0", textAlign: "center",
};

const tdBase: React.CSSProperties = {
  border: cellBorder, padding: "4px 6px", minHeight: 28,
};

const covLabelCell: React.CSSProperties = {
  border: cellBorder, fontWeight: "bold", fontSize: 10,
  padding: "3px 8px", textAlign: "left",
};

const covHeaderCell: React.CSSProperties = {
  border: cellBorder, fontWeight: "bold", fontSize: 10,
  padding: "3px 6px", textAlign: "center",
};

const covValueCell: React.CSSProperties = {
  border: cellBorder, fontSize: 10,
  padding: "3px 6px", textAlign: "center", minHeight: 18,
};
