import { useState, useRef, useCallback } from "react";
import {
  Building2, Save, BookOpen, Upload, FileText, Presentation, FileImage,
  File, Plus, Trash2, Check, X, ChevronDown, ChevronRight, Layers,
  Brain, ImageIcon, BarChart3, GripVertical, AlertTriangle,
  CheckCircle2, Clock, Sparkles, CloudUpload, Info
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";

// ────────────────────────────────────────────────────────────
//  Types
// ────────────────────────────────────────────────────────────
interface TemplateBlock {
  id: string;
  label: string;
  instruction: string;
  questions: number;
  attemptAll: boolean;
  attemptCount: number;
  marksPerQuestion: number;
  marksMin: number;        // lower bound when randomization is on
  marksMax: number;        // upper bound when randomization is on
  hasOrPattern: boolean;
}

interface ExamTemplate {
  examType: string;
  totalMarks: number;
  duration: string;
  blocks: TemplateBlock[];
  referenceFile: File | null;
  referenceFileName: string;
  randomizationEnabled: boolean; // use learned distributions instead of fixed marks
}

interface TrainingFile {
  id: string;
  name: string;
  size: string;
  uploadedOn: string;
  learnType: "framing" | "marks" | "diagrams" | "full-paper";
  examType: string;
  status: "processing" | "trained" | "failed";
  questionsLearned: number;
}

// ────────────────────────────────────────────────────────────
//  Constants
// ────────────────────────────────────────────────────────────
const EXAM_TYPES = ["IAT-1", "IAT-2", "SEE"];

const LEARN_TYPE_CONFIG: Record<TrainingFile["learnType"], { label: string; color: string; icon: typeof Brain }> = {
  framing:    { label: "Question Framing",   color: "bg-blue-100 text-blue-700 border-blue-200",   icon: Brain },
  marks:      { label: "Marks Distribution", color: "bg-emerald-100 text-emerald-700 border-emerald-200", icon: BarChart3 },
  diagrams:   { label: "Diagram Patterns",   color: "bg-violet-100 text-violet-700 border-violet-200",   icon: ImageIcon },
  "full-paper": { label: "Full Paper",       color: "bg-amber-100 text-amber-700 border-amber-200",   icon: FileText },
};

const DEFAULT_TEMPLATES: Record<string, ExamTemplate> = {
  "IAT-1": {
    examType: "IAT-1", totalMarks: 50, duration: "1.5 hrs",
    referenceFile: null, referenceFileName: "", randomizationEnabled: false,
    blocks: [
      { id: "b1", label: "Part A", instruction: "Answer ALL questions", questions: 5, attemptAll: true, attemptCount: 5, marksPerQuestion: 2, marksMin: 1, marksMax: 3, hasOrPattern: false },
      { id: "b2", label: "Part B", instruction: "Answer any FOUR of FIVE questions (with OR)", questions: 5, attemptAll: false, attemptCount: 4, marksPerQuestion: 10, marksMin: 8, marksMax: 12, hasOrPattern: true },
    ],
  },
  "IAT-2": {
    examType: "IAT-2", totalMarks: 50, duration: "1.5 hrs",
    referenceFile: null, referenceFileName: "", randomizationEnabled: false,
    blocks: [
      { id: "b1", label: "Part A", instruction: "Answer ALL questions", questions: 5, attemptAll: true, attemptCount: 5, marksPerQuestion: 2, marksMin: 1, marksMax: 3, hasOrPattern: false },
      { id: "b2", label: "Part B", instruction: "Answer any FOUR of FIVE questions (with OR)", questions: 5, attemptAll: false, attemptCount: 4, marksPerQuestion: 10, marksMin: 8, marksMax: 12, hasOrPattern: true },
    ],
  },
  "SEE": {
    examType: "SEE", totalMarks: 100, duration: "3 hrs",
    referenceFile: null, referenceFileName: "", randomizationEnabled: false,
    blocks: [
      { id: "b1", label: "Part A", instruction: "Answer TEN questions (2 marks each)", questions: 10, attemptAll: true, attemptCount: 10, marksPerQuestion: 2, marksMin: 1, marksMax: 3, hasOrPattern: false },
      { id: "b2", label: "Part B", instruction: "Answer ONE full question from each unit (10 marks × 5 units)", questions: 10, attemptAll: false, attemptCount: 5, marksPerQuestion: 10, marksMin: 8, marksMax: 12, hasOrPattern: true },
      { id: "b3", label: "Part C", instruction: "Answer ONE full question from each unit (10 marks × 5 units)", questions: 10, attemptAll: false, attemptCount: 5, marksPerQuestion: 10, marksMin: 8, marksMax: 12, hasOrPattern: true },
    ],
  },
};

const MOCK_TRAINING_FILES: TrainingFile[] = [
  { id: "tr1", name: "IAT1_ML_2023_Jan.pdf", size: "1.2 MB", uploadedOn: "2024-07-01", learnType: "full-paper", examType: "IAT-1", status: "trained", questionsLearned: 9 },
  { id: "tr2", name: "IAT2_ML_2023_Sep.pdf", size: "0.9 MB", uploadedOn: "2024-07-01", learnType: "full-paper", examType: "IAT-2", status: "trained", questionsLearned: 9 },
  { id: "tr3", name: "SEE_ML_2022_Nov.pdf",  size: "2.1 MB", uploadedOn: "2024-07-02", learnType: "full-paper", examType: "SEE",   status: "trained", questionsLearned: 30 },
  { id: "tr4", name: "diagram_questions_annotated.pdf", size: "3.8 MB", uploadedOn: "2024-07-05", learnType: "diagrams", examType: "IAT-1", status: "trained", questionsLearned: 14 },
  { id: "tr5", name: "marks_distribution_examples.docx", size: "0.4 MB", uploadedOn: "2024-07-08", learnType: "marks", examType: "SEE", status: "trained", questionsLearned: 22 },
];

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

const ACCEPTED = ".pdf,.doc,.docx,.ppt,.pptx,.png,.jpg,.jpeg";

// ────────────────────────────────────────────────────────────
//  Component
// ────────────────────────────────────────────────────────────
export default function Settings() {
  const [activeExamTab, setActiveExamTab] = useState("IAT-1");
  const [templates, setTemplates] = useState<Record<string, ExamTemplate>>(DEFAULT_TEMPLATES);
  const [expandedBlock, setExpandedBlock] = useState<string | null>("b1");
  const [trainingFiles, setTrainingFiles] = useState<TrainingFile[]>(MOCK_TRAINING_FILES);
  const [isDragOverTemplate, setIsDragOverTemplate] = useState(false);
  const [isDragOverTraining, setIsDragOverTraining] = useState(false);
  const [newLearnType, setNewLearnType] = useState<TrainingFile["learnType"]>("full-paper");
  const [newExamType, setNewExamType] = useState("IAT-1");
  const [showTrainingNote, setShowTrainingNote] = useState(true);

  const templateFileRef = useRef<HTMLInputElement>(null);
  const trainingFileRef = useRef<HTMLInputElement>(null);

  const tpl = templates[activeExamTab];

  // Computed totals
  const computedMaxMarks = tpl.blocks.reduce((sum, b) => {
    return sum + (b.hasOrPattern ? b.attemptCount * b.marksPerQuestion : b.questions * b.marksPerQuestion);
  }, 0);

  const updateBlock = (blockId: string, field: keyof TemplateBlock, value: any) => {
    setTemplates(prev => ({
      ...prev,
      [activeExamTab]: {
        ...prev[activeExamTab],
        blocks: prev[activeExamTab].blocks.map(b => b.id === blockId ? { ...b, [field]: value } : b),
      },
    }));
  };

  const toggleRandomization = (enabled: boolean) => {
    setTemplates(prev => ({
      ...prev,
      [activeExamTab]: { ...prev[activeExamTab], randomizationEnabled: enabled },
    }));
    if (enabled) {
      toast.info("Randomization on — AI will draw from training data to vary mark distributions each generation.");
    }
  };

  const addBlock = () => {
    const newId = `b${Date.now()}`;
    setTemplates(prev => ({
      ...prev,
      [activeExamTab]: {
        ...prev[activeExamTab],
        blocks: [...prev[activeExamTab].blocks, {
          id: newId,
          label: `Part ${String.fromCharCode(65 + prev[activeExamTab].blocks.length)}`,
          instruction: "Answer all questions",
          questions: 5, attemptAll: true, attemptCount: 5,
          marksPerQuestion: 10, marksMin: 8, marksMax: 12, hasOrPattern: false,
        }],
      },
    }));
    setExpandedBlock(newId);
  };

  const removeBlock = (blockId: string) => {
    setTemplates(prev => ({
      ...prev,
      [activeExamTab]: {
        ...prev[activeExamTab],
        blocks: prev[activeExamTab].blocks.filter(b => b.id !== blockId),
      },
    }));
  };

  const handleTemplateFile = (file: File) => {
    setTemplates(prev => ({
      ...prev,
      [activeExamTab]: { ...prev[activeExamTab], referenceFile: file, referenceFileName: file.name },
    }));
    toast.success(`"${file.name}" uploaded as reference template for ${activeExamTab}.`);
  };

  const handleTrainingUpload = (file: File) => {
    const newFile: TrainingFile = {
      id: `tr${Date.now()}`,
      name: file.name,
      size: formatBytes(file.size),
      uploadedOn: new Date().toISOString().slice(0, 10),
      learnType: newLearnType,
      examType: newExamType,
      status: "processing",
      questionsLearned: 0,
    };
    setTrainingFiles(prev => [newFile, ...prev]);
    toast.info(`"${file.name}" is being processed for AI training...`);
    // Simulate processing
    setTimeout(() => {
      const learned = Math.floor(Math.random() * 15) + 5;
      setTrainingFiles(prev => prev.map(f => f.id === newFile.id
        ? { ...f, status: "trained", questionsLearned: learned }
        : f
      ));
      toast.success(`"${file.name}" training complete — ${learned} patterns learned.`);
    }, 3000);
  };

  const handleSave = () => {
    toast.success("Settings saved. AI will use these templates and training data when generating papers.");
  };

  const totalTrained = trainingFiles.filter(f => f.status === "trained").reduce((s, f) => s + f.questionsLearned, 0);

  return (
    <div className="space-y-8 max-w-5xl pb-12">
      {/* Page header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-serif tracking-tight">Settings</h1>
          <p className="text-muted-foreground">Configure institution details, paper templates, and AI training data.</p>
        </div>
        <Button onClick={handleSave} className="gap-2">
          <Save className="h-4 w-4" /> Save Changes
        </Button>
      </div>

      {/* ── 1. Institution Details ─────────────────────────────── */}
      <Card className="border-muted shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Building2 className="h-5 w-5 text-primary" /> Institution Details
          </CardTitle>
          <CardDescription>Header information printed on every generated question paper.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1.5 md:col-span-2">
              <Label>Institution Name</Label>
              <Input defaultValue="Dayananda Sagar Academy of Technology & Management" />
            </div>
            <div className="space-y-1.5 md:col-span-2">
              <Label>Affiliations & Accreditations</Label>
              <Input defaultValue="(Autonomous Institute under VTU) Affiliated to VTU | Approved by AICTE | Accredited by NAAC with A+ Grade | 6 Programs Accredited by NBA" />
            </div>
            <div className="space-y-1.5">
              <Label>City</Label>
              <Input defaultValue="Bengaluru" />
            </div>
            <div className="space-y-1.5">
              <Label>University</Label>
              <Input defaultValue="Visvesvaraya Technological University (VTU)" />
            </div>
          </div>
          <Separator />
          <div className="flex items-center justify-between">
            <div>
              <Label>Include RBT & CO Legend</Label>
              <p className="text-xs text-muted-foreground mt-0.5">Print the Bloom's taxonomy definitions before the questions.</p>
            </div>
            <Switch defaultChecked />
          </div>
          <Separator />
          <div className="flex items-center justify-between">
            <div>
              <Label>Include Course Outcomes Table</Label>
              <p className="text-xs text-muted-foreground mt-0.5">Append CO mapping definitions at the end of the paper.</p>
            </div>
            <Switch defaultChecked />
          </div>
        </CardContent>
      </Card>

      {/* ── 2. Paper Templates ─────────────────────────────────── */}
      <Card className="border-muted shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <BookOpen className="h-5 w-5 text-primary" /> Paper Templates
          </CardTitle>
          <CardDescription>
            Define the structure (blocks, marks, question count) for each exam type. Upload a reference document and the AI will strictly follow this layout when generating papers.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Exam type tabs */}
          <div className="flex gap-1 border-b">
            {EXAM_TYPES.map(et => (
              <button
                key={et}
                onClick={() => { setActiveExamTab(et); setExpandedBlock(null); }}
                className={`px-5 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
                  activeExamTab === et
                    ? "border-primary text-primary"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                {et}
                <span className={`ml-1.5 text-[10px] px-1.5 py-0.5 rounded-full ${activeExamTab === et ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"}`}>
                  {templates[et].totalMarks}M
                </span>
              </button>
            ))}
          </div>

          {/* Template meta + reference upload */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-1.5">
              <Label>Total Marks</Label>
              <Input
                type="number"
                value={tpl.totalMarks}
                onChange={e => setTemplates(prev => ({ ...prev, [activeExamTab]: { ...prev[activeExamTab], totalMarks: Number(e.target.value) } }))}
                className="h-9"
              />
            </div>
            <div className="space-y-1.5">
              <Label>Duration</Label>
              <Select
                value={tpl.duration}
                onValueChange={v => setTemplates(prev => ({ ...prev, [activeExamTab]: { ...prev[activeExamTab], duration: v } }))}
              >
                <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {["1 hr", "1.5 hrs", "2 hrs", "2.5 hrs", "3 hrs"].map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label>Computed Max Marks</Label>
              <div className={`h-9 flex items-center px-3 rounded-md border text-sm font-semibold ${computedMaxMarks === tpl.totalMarks ? "bg-emerald-50 border-emerald-200 text-emerald-700" : "bg-amber-50 border-amber-200 text-amber-700"}`}>
                {computedMaxMarks === tpl.totalMarks
                  ? <><CheckCircle2 className="h-4 w-4 mr-1.5" />{computedMaxMarks} marks ✓</>
                  : <><AlertTriangle className="h-4 w-4 mr-1.5" />{computedMaxMarks} / {tpl.totalMarks} — mismatch</>
                }
              </div>
            </div>
          </div>

          {/* Reference template upload */}
          <div className="space-y-2">
            <Label className="flex items-center gap-1.5">
              Reference Template Document
              <span className="text-xs font-normal text-muted-foreground">(optional — AI will visually follow this layout)</span>
            </Label>
            <input
              ref={templateFileRef}
              type="file"
              accept={ACCEPTED}
              className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) handleTemplateFile(f); e.target.value = ""; }}
            />
            {tpl.referenceFileName ? (
              <div className="flex items-center gap-3 p-3 rounded-lg border bg-muted/30">
                <FileText className="h-5 w-5 text-primary shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{tpl.referenceFileName}</p>
                  <p className="text-xs text-muted-foreground">Used as visual reference for {activeExamTab} generation</p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground hover:text-destructive h-7"
                  onClick={() => setTemplates(prev => ({ ...prev, [activeExamTab]: { ...prev[activeExamTab], referenceFileName: "", referenceFile: null } }))}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ) : (
              <div
                onDragOver={e => { e.preventDefault(); setIsDragOverTemplate(true); }}
                onDragLeave={() => setIsDragOverTemplate(false)}
                onDrop={e => { e.preventDefault(); setIsDragOverTemplate(false); const f = e.dataTransfer.files[0]; if (f) handleTemplateFile(f); }}
                onClick={() => templateFileRef.current?.click()}
                className={`flex items-center gap-4 p-4 rounded-lg border-2 border-dashed cursor-pointer transition-all ${isDragOverTemplate ? "border-primary bg-primary/5" : "border-muted hover:border-primary/40 hover:bg-muted/20"}`}
              >
                <CloudUpload className={`h-6 w-6 ${isDragOverTemplate ? "text-primary" : "text-muted-foreground"}`} />
                <div>
                  <p className="text-sm font-medium">Drop or click to upload a reference paper</p>
                  <p className="text-xs text-muted-foreground">PDF, DOCX, PPT, or image · AI learns layout, marks, and structure from this</p>
                </div>
              </div>
            )}
          </div>

          {/* Randomization Mode toggle */}
          <div className={`rounded-xl border-2 p-4 transition-all ${tpl.randomizationEnabled ? "border-primary/40 bg-primary/5" : "border-muted"}`}>
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-start gap-3">
                <div className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${tpl.randomizationEnabled ? "bg-primary text-primary-foreground" : "bg-muted"}`}>
                  <Sparkles className="h-5 w-5" />
                </div>
                <div>
                  <p className="font-semibold text-sm">AI Randomization Mode</p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {tpl.randomizationEnabled
                      ? "Active — AI will learn from your uploaded training papers and vary mark distributions, question depth, and difficulty each time. No two generated papers will be identical."
                      : "Off — AI follows the fixed marks-per-block you set below. Turn on to let training data drive variation."}
                  </p>
                  {tpl.randomizationEnabled && (
                    <div className="flex flex-wrap gap-2 mt-2">
                      {[
                        "Marks drawn from learned range per block",
                        "Question depth varied by CO level",
                        "Diagram assignment from training patterns",
                        "Different every generation",
                      ].map(tag => (
                        <span key={tag} className="text-[10px] px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20 font-medium">{tag}</span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <Switch checked={tpl.randomizationEnabled} onCheckedChange={toggleRandomization} className="shrink-0 mt-0.5" />
            </div>
            {tpl.randomizationEnabled && trainingFiles.filter(f => f.status === "trained" && f.examType === activeExamTab).length === 0 && (
              <div className="mt-3 flex items-center gap-2 p-2.5 rounded-md bg-amber-50 border border-amber-200 text-xs text-amber-700">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                <span>No training files for <strong>{activeExamTab}</strong> yet. Upload past papers in the AI Training Data section below to power randomization.</span>
              </div>
            )}
            {tpl.randomizationEnabled && trainingFiles.filter(f => f.status === "trained" && f.examType === activeExamTab).length > 0 && (
              <div className="mt-3 flex items-center gap-2 p-2.5 rounded-md bg-emerald-50 border border-emerald-200 text-xs text-emerald-700">
                <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                <span>
                  <strong>{trainingFiles.filter(f => f.status === "trained" && f.examType === activeExamTab).length}</strong> trained {activeExamTab} file{trainingFiles.filter(f => f.status === "trained" && f.examType === activeExamTab).length > 1 ? "s" : ""} active — AI will randomize within learned boundaries.
                </span>
              </div>
            )}
          </div>

          <Separator />

          {/* Block Editor */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-semibold text-sm">Question Blocks / Parts</p>
                <p className="text-xs text-muted-foreground">
                  {tpl.randomizationEnabled
                    ? `Define mark ranges per block — AI will pick values within these bounds using learned patterns.`
                    : `Define each part of the ${activeExamTab} paper. AI strictly follows fixed marks.`}
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={addBlock} className="gap-1.5 h-8">
                <Plus className="h-3.5 w-3.5" /> Add Block
              </Button>
            </div>

            <div className="space-y-2">
              {tpl.blocks.map((block, idx) => {
                const isExpanded = expandedBlock === block.id;
                const blockMarks = block.hasOrPattern
                  ? block.attemptCount * block.marksPerQuestion
                  : block.questions * block.marksPerQuestion;

                return (
                  <div key={block.id} className="border rounded-lg overflow-hidden">
                    <div
                      role="button"
                      tabIndex={0}
                      className="w-full text-left px-4 py-3 hover:bg-muted/30 transition-colors flex items-center justify-between cursor-pointer"
                      onClick={() => setExpandedBlock(isExpanded ? null : block.id)}
                      onKeyDown={e => e.key === "Enter" && setExpandedBlock(isExpanded ? null : block.id)}
                    >
                      <div className="flex items-center gap-3">
                        <GripVertical className="h-4 w-4 text-muted-foreground/40" />
                        {isExpanded ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
                        <div>
                          <span className="font-semibold text-sm">{block.label}</span>
                          <span className="text-muted-foreground text-xs ml-2">{block.instruction}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">{block.questions} Qs</Badge>
                        <Badge variant="secondary" className="text-xs">{blockMarks} marks</Badge>
                        {block.hasOrPattern && <Badge className="text-[10px] bg-primary/10 text-primary border-0">OR pattern</Badge>}
                        <div
                          role="button"
                          tabIndex={0}
                          onClick={e => { e.stopPropagation(); removeBlock(block.id); }}
                          onKeyDown={e => { if (e.key === "Enter") { e.stopPropagation(); removeBlock(block.id); }}}
                          className="ml-2 p-1 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive cursor-pointer"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </div>
                      </div>
                    </div>

                    <AnimatePresence initial={false}>
                      {isExpanded && (
                        <motion.div initial={{ height: 0 }} animate={{ height: "auto" }} exit={{ height: 0 }} className="overflow-hidden">
                          <Separator />
                          <div className="p-4 bg-muted/10 space-y-4">
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                              <div className="space-y-1.5">
                                <Label className="text-xs">Block Label</Label>
                                <Input value={block.label} onChange={e => updateBlock(block.id, "label", e.target.value)} className="h-8 text-sm" />
                              </div>
                              <div className="space-y-1.5 md:col-span-2">
                                <Label className="text-xs">Instruction Text</Label>
                                <Input value={block.instruction} onChange={e => updateBlock(block.id, "instruction", e.target.value)} className="h-8 text-sm" />
                              </div>
                              <div className="space-y-1.5">
                                <Label className="text-xs">Total Questions in Block</Label>
                                <Input type="number" min={1} value={block.questions} onChange={e => updateBlock(block.id, "questions", Number(e.target.value))} className="h-8 text-sm" />
                              </div>
                              <div className="space-y-1.5">
                                <Label className="text-xs">Questions to Attempt</Label>
                                <Input type="number" min={1} max={block.questions} value={block.attemptCount} onChange={e => updateBlock(block.id, "attemptCount", Number(e.target.value))} className="h-8 text-sm" disabled={block.attemptAll} />
                              </div>

                              {/* Fixed vs Range marks based on randomization mode */}
                              {!tpl.randomizationEnabled ? (
                                <div className="space-y-1.5">
                                  <Label className="text-xs">Marks per Question (fixed)</Label>
                                  <Input type="number" min={1} value={block.marksPerQuestion} onChange={e => updateBlock(block.id, "marksPerQuestion", Number(e.target.value))} className="h-8 text-sm" />
                                </div>
                              ) : (
                                <>
                                  <div className="space-y-1.5">
                                    <Label className="text-xs text-primary">Min Marks / Question</Label>
                                    <Input type="number" min={1} max={block.marksMax - 1} value={block.marksMin} onChange={e => updateBlock(block.id, "marksMin", Number(e.target.value))} className="h-8 text-sm border-primary/40" />
                                  </div>
                                  <div className="space-y-1.5">
                                    <Label className="text-xs text-primary">Max Marks / Question</Label>
                                    <Input type="number" min={block.marksMin + 1} value={block.marksMax} onChange={e => updateBlock(block.id, "marksMax", Number(e.target.value))} className="h-8 text-sm border-primary/40" />
                                  </div>
                                </>
                              )}
                            </div>
                            <div className="flex items-center gap-6">
                              <div className="flex items-center gap-2">
                                <Switch checked={block.attemptAll} onCheckedChange={v => { updateBlock(block.id, "attemptAll", v); if (v) updateBlock(block.id, "attemptCount", block.questions); }} />
                                <Label className="text-xs">Answer ALL questions</Label>
                              </div>
                              <div className="flex items-center gap-2">
                                <Switch checked={block.hasOrPattern} onCheckedChange={v => updateBlock(block.id, "hasOrPattern", v)} />
                                <Label className="text-xs">Use OR pattern (Q1a OR Q1b)</Label>
                              </div>
                            </div>
                            {/* Preview pill */}
                            <div className={`flex items-center gap-2 p-2.5 rounded-md border text-xs ${tpl.randomizationEnabled ? "bg-primary/5 border-primary/20 text-primary" : "bg-muted/40 border-muted text-muted-foreground"}`}>
                              <Info className="h-3.5 w-3.5 shrink-0" />
                              {tpl.randomizationEnabled ? (
                                <span>
                                  AI generates <strong>{block.hasOrPattern ? block.attemptCount * 2 : block.questions} questions</strong>
                                  {block.hasOrPattern ? ` (${block.attemptCount} OR pairs)` : ""}, marks will be <strong>randomized {block.marksMin}–{block.marksMax} each</strong> using training data patterns → total varies per generation
                                </span>
                              ) : (
                                <span>
                                  AI generates <strong>{block.hasOrPattern ? block.attemptCount * 2 : block.questions} questions</strong>
                                  {block.hasOrPattern ? ` (${block.attemptCount} OR pairs)` : ""}, each fixed at <strong>{block.marksPerQuestion} marks</strong> → block total: <strong>{blockMarks} marks</strong>
                                </span>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                );
              })}
            </div>

            {/* Template visual summary */}
            <div className="mt-4 rounded-lg border bg-muted/20 overflow-hidden">
              <div className="px-4 py-2.5 border-b bg-muted/40 flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Template Preview — {activeExamTab}</span>
                <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${computedMaxMarks === tpl.totalMarks ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
                  {computedMaxMarks} / {tpl.totalMarks} marks
                </span>
              </div>
              <div className="p-4 font-mono text-xs space-y-1.5">
                <p className="font-bold text-center text-sm mb-2 font-serif">
                  {activeExamTab} · Max Marks: {tpl.totalMarks} · Duration: {tpl.duration}
                </p>
                {tpl.blocks.map((b, i) => (
                  <div key={b.id} className="space-y-1">
                    <p className="font-bold text-foreground">{b.label} — {b.instruction}</p>
                    {b.hasOrPattern
                      ? Array.from({ length: b.attemptCount }).map((_, qi) => (
                          <div key={qi} className="pl-4 space-y-0.5">
                            <p className="text-muted-foreground">Q{i * b.attemptCount + qi + 1}a. _________________________ ({b.marksPerQuestion} marks)</p>
                            <p className="text-muted-foreground/60 pl-4">OR</p>
                            <p className="text-muted-foreground">Q{i * b.attemptCount + qi + 1}b. _________________________ ({b.marksPerQuestion} marks)</p>
                          </div>
                        ))
                      : Array.from({ length: Math.min(b.questions, 3) }).map((_, qi) => (
                          <p key={qi} className="pl-4 text-muted-foreground">Q{qi + 1}. _________________________ ({b.marksPerQuestion} marks)</p>
                        ))
                    }
                    {!b.hasOrPattern && b.questions > 3 && <p className="pl-4 text-muted-foreground/50">… +{b.questions - 3} more</p>}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ── 3. AI Training Data ────────────────────────────────── */}
      <Card className="border-muted shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Brain className="h-5 w-5 text-primary" /> AI Training Data
          </CardTitle>
          <CardDescription>
            Upload past question papers and annotated examples so the AI learns how to frame questions, distribute marks, and assign diagrams for your institution's style.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">

          {/* Info banner */}
          {showTrainingNote && (
            <div className="flex items-start gap-3 p-3.5 rounded-lg border border-blue-200 bg-blue-50 dark:bg-blue-950/30 dark:border-blue-800">
              <Sparkles className="h-4 w-4 text-blue-600 shrink-0 mt-0.5" />
              <div className="flex-1 text-sm text-blue-700 dark:text-blue-300">
                <p className="font-semibold">How AI training works</p>
                <p className="mt-0.5 text-xs">Each uploaded file teaches the AI a specific pattern. Upload real past papers to improve question framing accuracy. Tag diagram-heavy papers separately so the AI learns which question types typically require figures or diagrams.</p>
              </div>
              <button onClick={() => setShowTrainingNote(false)} className="text-blue-400 hover:text-blue-600 shrink-0"><X className="h-4 w-4" /></button>
            </div>
          )}

          {/* Stats row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: "Files Trained", value: trainingFiles.filter(f => f.status === "trained").length, icon: FileText, color: "text-blue-600" },
              { label: "Patterns Learned", value: totalTrained, icon: Brain, color: "text-emerald-600" },
              { label: "Diagram Samples", value: trainingFiles.filter(f => f.learnType === "diagrams").length, icon: ImageIcon, color: "text-violet-600" },
              { label: "Mark Samples", value: trainingFiles.filter(f => f.learnType === "marks").length, icon: BarChart3, color: "text-amber-600" },
            ].map(stat => (
              <div key={stat.label} className="bg-muted/30 border rounded-lg px-3 py-2.5 flex items-center gap-2.5">
                <stat.icon className={`h-5 w-5 ${stat.color} shrink-0`} />
                <div>
                  <p className="text-lg font-bold leading-tight">{stat.value}</p>
                  <p className="text-[10px] text-muted-foreground">{stat.label}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Upload controls */}
          <div className="space-y-3">
            <p className="text-sm font-semibold">Upload Training File</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs">What should AI learn from this file?</Label>
                <Select value={newLearnType} onValueChange={v => setNewLearnType(v as TrainingFile["learnType"])}>
                  <SelectTrigger className="h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="full-paper">Full Paper — learns everything (framing, marks, diagrams)</SelectItem>
                    <SelectItem value="framing">Question Framing — how questions are worded</SelectItem>
                    <SelectItem value="marks">Marks Distribution — how marks are split per section</SelectItem>
                    <SelectItem value="diagrams">Diagram Patterns — which questions include figures</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Exam Type this file belongs to</Label>
                <Select value={newExamType} onValueChange={setNewExamType}>
                  <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {EXAM_TYPES.map(et => <SelectItem key={et} value={et}>{et}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <input
              ref={trainingFileRef}
              type="file"
              accept={ACCEPTED}
              className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) handleTrainingUpload(f); e.target.value = ""; }}
            />
            <div
              onDragOver={e => { e.preventDefault(); setIsDragOverTraining(true); }}
              onDragLeave={() => setIsDragOverTraining(false)}
              onDrop={e => { e.preventDefault(); setIsDragOverTraining(false); const f = e.dataTransfer.files[0]; if (f) handleTrainingUpload(f); }}
              onClick={() => trainingFileRef.current?.click()}
              className={`flex flex-col items-center justify-center gap-2.5 p-8 rounded-xl border-2 border-dashed cursor-pointer transition-all ${isDragOverTraining ? "border-primary bg-primary/5" : "border-muted hover:border-primary/40 hover:bg-muted/20"}`}
            >
              <div className={`w-12 h-12 rounded-full flex items-center justify-center ${isDragOverTraining ? "bg-primary/15" : "bg-muted"}`}>
                <CloudUpload className={`h-6 w-6 ${isDragOverTraining ? "text-primary" : "text-muted-foreground"}`} />
              </div>
              <div className="text-center">
                <p className="text-sm font-medium">{isDragOverTraining ? "Drop to upload" : "Drag & drop or click to upload training file"}</p>
                <p className="text-xs text-muted-foreground mt-0.5">PDF, DOCX, PPT, or image · Tagged as <strong>{LEARN_TYPE_CONFIG[newLearnType].label}</strong> · {newExamType}</p>
              </div>
            </div>
          </div>

          <Separator />

          {/* Training file list */}
          <div className="space-y-2">
            <p className="text-sm font-semibold">Uploaded Training Files ({trainingFiles.length})</p>
            <div className="space-y-2">
              {trainingFiles.map(file => {
                const cfg = LEARN_TYPE_CONFIG[file.learnType];
                const Icon = cfg.icon;
                return (
                  <motion.div
                    key={file.id}
                    layout
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-3 p-3 rounded-lg border bg-card hover:border-primary/30 transition-colors"
                  >
                    <div className="w-9 h-9 rounded-md bg-muted flex items-center justify-center shrink-0">
                      <FileText className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <p className="text-sm font-medium truncate max-w-[200px]">{file.name}</p>
                        <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${cfg.color}`}>
                          {cfg.label}
                        </span>
                        <Badge variant="outline" className="text-[10px]">{file.examType}</Badge>
                      </div>
                      <div className="flex items-center gap-3 mt-0.5 text-xs text-muted-foreground">
                        <span>{file.size}</span>
                        <span>·</span>
                        <span>{file.uploadedOn}</span>
                        {file.status === "trained" && (
                          <><span>·</span><span className="text-emerald-600 font-medium">{file.questionsLearned} patterns learned</span></>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {file.status === "processing" && (
                        <span className="flex items-center gap-1 text-xs text-amber-600 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full">
                          <Clock className="h-3 w-3 animate-pulse" /> Processing
                        </span>
                      )}
                      {file.status === "trained" && (
                        <span className="flex items-center gap-1 text-xs text-emerald-600 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">
                          <CheckCircle2 className="h-3 w-3" /> Trained
                        </span>
                      )}
                      {file.status === "failed" && (
                        <span className="flex items-center gap-1 text-xs text-red-600 bg-red-50 border border-red-200 px-2 py-0.5 rounded-full">
                          <AlertTriangle className="h-3 w-3" /> Failed
                        </span>
                      )}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-muted-foreground hover:text-destructive"
                        onClick={() => {
                          setTrainingFiles(prev => prev.filter(f => f.id !== file.id));
                          toast.success(`"${file.name}" removed from training data.`);
                        }}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
