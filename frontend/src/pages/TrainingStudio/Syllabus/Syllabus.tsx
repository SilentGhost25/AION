import { useState, useRef, useCallback } from "react";
import {
  Plus, Trash2, Save, BookOpen, GripVertical, ChevronDown, ChevronRight,
  Check, X, Upload, FileText, FileImage, Presentation, File,
  Loader2, ScanSearch, Brain, GitBranch, CheckCircle2, AlertCircle,
  CloudUpload, Sparkles, Eye, ArrowRight
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { departments, subjects, syllabusData } from "@/lib/mock-data";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";

interface SyllabusModule {
  id: string;
  number: number;
  title: string;
  topics: string[];
  coMapping: string;
  bloomLevel: string;
  hours: number;
}

interface SubjectSyllabus {
  subjectCode: string;
  subjectName: string;
  department: string;
  modules: SyllabusModule[];
}

const CO_BLOOM_DEFAULTS: Record<number, { co: string; bloom: string }> = {
  1: { co: "CO1", bloom: "L1/L2" },
  2: { co: "CO2", bloom: "L3" },
  3: { co: "CO3", bloom: "L4" },
  4: { co: "CO3", bloom: "L4" },
  5: { co: "CO2", bloom: "L3" },
};

const ACCEPTED = ".pdf,.doc,.docx,.ppt,.pptx,.png,.jpg,.jpeg,.webp,.bmp";

const FILE_ICONS: Record<string, React.ElementType> = {
  pdf: FileText,
  doc: FileText, docx: FileText,
  ppt: Presentation, pptx: Presentation,
  png: FileImage, jpg: FileImage, jpeg: FileImage, webp: FileImage, bmp: FileImage,
};

const SCAN_STEPS = [
  { icon: CloudUpload, label: "Uploading document", detail: "Transferring to processing server..." },
  { icon: ScanSearch, label: "Extracting content", detail: "OCR + text extraction from all pages..." },
  { icon: Brain, label: "Understanding structure", detail: "AI is identifying module headings and topic lists..." },
  { icon: GitBranch, label: "Mapping CO-Bloom levels", detail: "Assigning Course Outcomes and Bloom levels to each module..." },
  { icon: Sparkles, label: "Generating outline", detail: "Finalizing structured syllabus..." },
];

// Mock parsed syllabus returned after "AI scan"
const MOCK_PARSED_SYLLABUS: SyllabusModule[] = [
  {
    id: "pm1", number: 1,
    title: "Introduction to Machine Learning",
    topics: [
      "Definition, scope and limitations of Machine Learning",
      "Types of Machine Learning: Supervised, Unsupervised, Reinforcement",
      "Key concepts: Features, Labels, Training and Test sets",
      "Evaluation metrics: Accuracy, Precision, Recall, F1-score",
      "Bias-Variance Tradeoff and Overfitting / Underfitting",
    ],
    coMapping: "CO1", bloomLevel: "L1/L2", hours: 10,
  },
  {
    id: "pm2", number: 2,
    title: "Regression and Classification Algorithms",
    topics: [
      "Linear Regression and Gradient Descent",
      "Logistic Regression",
      "Decision Trees and Random Forests",
      "Support Vector Machines (SVM) with kernels",
      "k-Nearest Neighbors (k-NN)",
      "Naïve Bayes Classifier",
    ],
    coMapping: "CO2", bloomLevel: "L3", hours: 12,
  },
  {
    id: "pm3", number: 3,
    title: "Unsupervised Learning",
    topics: [
      "K-Means Clustering",
      "Hierarchical Clustering",
      "DBSCAN",
      "Principal Component Analysis (PCA)",
      "Singular Value Decomposition (SVD)",
      "Autoencoders for dimensionality reduction",
    ],
    coMapping: "CO3", bloomLevel: "L4", hours: 10,
  },
  {
    id: "pm4", number: 4,
    title: "Neural Networks and Deep Learning Basics",
    topics: [
      "Perceptron and Multi-layer Perceptron",
      "Backpropagation Algorithm",
      "Activation functions: ReLU, Sigmoid, Tanh, Softmax",
      "Convolutional Neural Networks (CNN)",
      "Recurrent Neural Networks (RNN) — LSTM basics",
      "Regularization: Dropout, Batch Normalization",
    ],
    coMapping: "CO3", bloomLevel: "L4", hours: 12,
  },
  {
    id: "pm5", number: 5,
    title: "Model Evaluation and Ensemble Methods",
    topics: [
      "Cross-validation techniques",
      "Hyperparameter tuning: Grid Search, Random Search",
      "Bagging and Boosting — AdaBoost, Gradient Boosting, XGBoost",
      "Model interpretability (LIME, SHAP)",
      "Deployment considerations and ML pipelines",
    ],
    coMapping: "CO2", bloomLevel: "L3", hours: 8,
  },
];

function getExt(name: string) {
  return name.split(".").pop()?.toLowerCase() ?? "";
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

export default function Syllabus() {
  const [selectedDept, setSelectedDept] = useState<string>("");
  const [selectedSubject, setSelectedSubject] = useState<string>("");
  const [expandedModule, setExpandedModule] = useState<number | null>(1);
  const [syllabi, setSyllabi] = useState<SubjectSyllabus[]>(syllabusData);
  const [editingTopic, setEditingTopic] = useState<{ moduleId: string; index: number } | null>(null);
  const [newTopicText, setNewTopicText] = useState("");

  // Upload / scan state
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [scanStepIndex, setScanStepIndex] = useState(-1); // -1 = idle, 0-4 = scanning, 5 = done
  const [scanProgress, setScanProgress] = useState(0);
  const [showReviewDialog, setShowReviewDialog] = useState(false);
  const [reviewExpandedMod, setReviewExpandedMod] = useState<number | null>(1);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const subjectList = selectedDept && (subjects as any)[selectedDept]
    ? (subjects as any)[selectedDept]
    : Object.values(subjects).flat();

  const currentSyllabus = syllabi.find(s => s.subjectName === selectedSubject) || syllabi[0];

  // ── Upload & scan simulation ──
  const startScan = useCallback((file: File) => {
    setUploadedFile(file);
    setScanStepIndex(0);
    setScanProgress(0);

    let step = 0;
    const totalSteps = SCAN_STEPS.length;

    const advance = () => {
      step++;
      if (step < totalSteps) {
        setScanStepIndex(step);
        setScanProgress(Math.round((step / totalSteps) * 100));
        setTimeout(advance, 900 + Math.random() * 500);
      } else {
        setScanStepIndex(totalSteps); // done
        setScanProgress(100);
        setTimeout(() => setShowReviewDialog(true), 400);
      }
    };
    setTimeout(advance, 900 + Math.random() * 400);
  }, []);

  const handleFileSelect = (file: File) => {
    const ext = getExt(file.name);
    const allowed = ACCEPTED.replace(/\./g, "").split(",");
    if (!allowed.includes(ext)) {
      toast.error(`Unsupported file type ".${ext}". Please upload PDF, DOCX, PPT, or an image.`);
      return;
    }
    startScan(file);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  }, []);

  const handleApplyParsed = () => {
    if (!currentSyllabus) return;
    setSyllabi(prev => prev.map(s =>
      s.subjectCode === currentSyllabus.subjectCode
        ? { ...s, modules: MOCK_PARSED_SYLLABUS.map(m => ({ ...m })) }
        : s
    ));
    setShowReviewDialog(false);
    setScanStepIndex(-1);
    setUploadedFile(null);
    setScanProgress(0);
    setExpandedModule(1);
    toast.success("Syllabus imported and applied successfully. AI will now use this syllabus for question generation.");
  };

  const handleCancelImport = () => {
    setShowReviewDialog(false);
    setScanStepIndex(-1);
    setUploadedFile(null);
    setScanProgress(0);
  };

  const handleSaveSyllabus = () => {
    toast.success("Syllabus saved. AI will now reference this syllabus strictly during question generation.");
  };

  const handleAddTopic = (moduleId: string) => {
    if (!newTopicText.trim()) return;
    setSyllabi(prev => prev.map(s => ({
      ...s,
      modules: s.modules.map(m => m.id === moduleId
        ? { ...m, topics: [...m.topics, newTopicText.trim()] }
        : m
      )
    })));
    setNewTopicText("");
    setEditingTopic(null);
  };

  const handleDeleteTopic = (moduleId: string, index: number) => {
    setSyllabi(prev => prev.map(s => ({
      ...s,
      modules: s.modules.map(m => m.id === moduleId
        ? { ...m, topics: m.topics.filter((_, i) => i !== index) }
        : m
      )
    })));
  };

  const handleModuleFieldChange = (moduleId: string, field: keyof SyllabusModule, value: string | number) => {
    setSyllabi(prev => prev.map(s => ({
      ...s,
      modules: s.modules.map(m => m.id === moduleId ? { ...m, [field]: value } : m)
    })));
  };

  if (!currentSyllabus) return null;

  const scanDone = scanStepIndex === SCAN_STEPS.length;
  const scanning = scanStepIndex >= 0 && !scanDone;
  const idle = scanStepIndex === -1;

  const FileIcon = uploadedFile ? (FILE_ICONS[getExt(uploadedFile.name)] ?? File) : CloudUpload;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground font-serif tracking-tight">Syllabus Manager</h1>
        <p className="text-muted-foreground mt-1">
          Import or manually define the syllabus per subject. The AI strictly adheres to this when generating questions.
        </p>
      </div>

      {/* ── Compliance Banner ── */}
      <div className="flex items-start gap-3 p-4 rounded-lg border border-primary/30 bg-primary/5">
        <BookOpen className="h-5 w-5 text-primary mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-semibold text-primary">Strict Syllabus Compliance</p>
          <p className="text-sm text-muted-foreground mt-0.5">
            Every question generated will be validated against the topics defined here. Each module has a fixed CO and Bloom's level that cannot be overridden during paper generation.
          </p>
        </div>
      </div>

      {/* ── Import Syllabus Card ── */}
      <Card className="border-dashed border-2 border-primary/30 bg-primary/[0.02] overflow-hidden">
        <CardHeader className="pb-2 pt-4 px-5">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-md bg-primary/10 flex items-center justify-center">
              <Upload className="h-4 w-4 text-primary" />
            </div>
            <div>
              <CardTitle className="text-base">Import Syllabus from Document</CardTitle>
              <p className="text-xs text-muted-foreground mt-0.5">
                Upload any syllabus document — PDF, DOCX, PPT, or image. The AI scans and extracts modules and topics automatically.
              </p>
            </div>
          </div>
        </CardHeader>
        <CardContent className="px-5 pb-5">
          <AnimatePresence mode="wait">

            {/* ── Idle: Drop zone ── */}
            {idle && (
              <motion.div key="dropzone" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={ACCEPTED}
                  className="hidden"
                  onChange={e => { const f = e.target.files?.[0]; if (f) handleFileSelect(f); e.target.value = ""; }}
                />
                <div
                  onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
                  onDragLeave={() => setIsDragOver(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`relative flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed cursor-pointer transition-all py-10 px-6 ${
                    isDragOver
                      ? "border-primary bg-primary/10 scale-[1.01]"
                      : "border-muted hover:border-primary/50 hover:bg-muted/30"
                  }`}
                >
                  <div className={`w-14 h-14 rounded-full flex items-center justify-center transition-all ${isDragOver ? "bg-primary/20" : "bg-muted"}`}>
                    <CloudUpload className={`h-7 w-7 transition-colors ${isDragOver ? "text-primary" : "text-muted-foreground"}`} />
                  </div>
                  <div className="text-center">
                    <p className="font-semibold text-sm text-foreground">
                      {isDragOver ? "Drop to upload" : "Drag & drop your syllabus document"}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">or <span className="text-primary font-medium">click to browse files</span></p>
                  </div>
                  <div className="flex flex-wrap justify-center gap-2 mt-1">
                    {[
                      { label: "PDF", icon: FileText },
                      { label: "DOCX", icon: FileText },
                      { label: "PPT / PPTX", icon: Presentation },
                      { label: "Image (PNG / JPG)", icon: FileImage },
                    ].map(({ label, icon: Icon }) => (
                      <span key={label} className="flex items-center gap-1 text-[10px] text-muted-foreground border border-muted/70 rounded px-2 py-0.5 bg-background">
                        <Icon className="h-3 w-3" /> {label}
                      </span>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {/* ── Scanning / Done ── */}
            {!idle && (
              <motion.div key="scanning" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-4">
                {/* File pill */}
                {uploadedFile && (
                  <div className="flex items-center gap-3 p-3 rounded-lg border bg-muted/30">
                    <div className="w-9 h-9 rounded-md bg-primary/10 flex items-center justify-center shrink-0">
                      <FileIcon className="h-5 w-5 text-primary" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{uploadedFile.name}</p>
                      <p className="text-xs text-muted-foreground">{formatBytes(uploadedFile.size)} · {getExt(uploadedFile.name).toUpperCase()}</p>
                    </div>
                    {scanDone && <CheckCircle2 className="h-5 w-5 text-emerald-500 shrink-0" />}
                    {scanning && <Loader2 className="h-5 w-5 text-primary animate-spin shrink-0" />}
                  </div>
                )}

                {/* Progress bar */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>{scanDone ? "Scan complete" : SCAN_STEPS[scanStepIndex]?.label}</span>
                    <span>{scanProgress}%</span>
                  </div>
                  <Progress value={scanProgress} className="h-2" />
                </div>

                {/* Step list */}
                <div className="space-y-2">
                  {SCAN_STEPS.map((step, i) => {
                    const done = i < scanStepIndex || scanDone;
                    const active = i === scanStepIndex && !scanDone;
                    const StepIcon = step.icon;
                    return (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0.3 }}
                        animate={{ opacity: done || active ? 1 : 0.35 }}
                        className={`flex items-center gap-3 p-2.5 rounded-lg border transition-all ${
                          active ? "border-primary/40 bg-primary/5" :
                          done ? "border-emerald-200 bg-emerald-50/50 dark:border-emerald-800 dark:bg-emerald-950/20" :
                          "border-muted bg-muted/10"
                        }`}
                      >
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${
                          active ? "bg-primary/15" : done ? "bg-emerald-100 dark:bg-emerald-900" : "bg-muted"
                        }`}>
                          {done
                            ? <Check className="h-3.5 w-3.5 text-emerald-600" />
                            : active
                              ? <Loader2 className="h-3.5 w-3.5 text-primary animate-spin" />
                              : <StepIcon className="h-3.5 w-3.5 text-muted-foreground" />
                          }
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className={`text-xs font-semibold ${done ? "text-emerald-700 dark:text-emerald-400" : active ? "text-primary" : "text-muted-foreground"}`}>
                            {step.label}
                          </p>
                          {active && <p className="text-[10px] text-muted-foreground mt-0.5">{step.detail}</p>}
                        </div>
                      </motion.div>
                    );
                  })}
                </div>

                {/* CTA after scan */}
                {scanDone && (
                  <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-3">
                    <div className="flex items-center gap-2 flex-1 p-3 rounded-lg bg-emerald-50 border border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-800">
                      <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0" />
                      <p className="text-sm text-emerald-700 dark:text-emerald-400 font-medium">
                        5 modules · 27 topics extracted successfully
                      </p>
                    </div>
                    <Button variant="outline" size="sm" onClick={handleCancelImport}>Cancel</Button>
                    <Button size="sm" className="gap-2" onClick={() => setShowReviewDialog(true)}>
                      <Eye className="h-4 w-4" /> Review &amp; Apply
                    </Button>
                  </motion.div>
                )}

                {/* Cancel while scanning */}
                {scanning && (
                  <div className="flex justify-end">
                    <Button variant="ghost" size="sm" className="text-muted-foreground text-xs" onClick={handleCancelImport}>
                      Cancel import
                    </Button>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>

      {/* ── Manual Editor ── */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Subject Selector */}
        <div className="lg:col-span-1">
          <Card className="border-muted">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Select Subject</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <Label>Department</Label>
                <Select onValueChange={v => { setSelectedDept(v); setSelectedSubject(""); }}>
                  <SelectTrigger><SelectValue placeholder="Select department" /></SelectTrigger>
                  <SelectContent>{departments.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}</SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Subject</Label>
                <Select onValueChange={setSelectedSubject} disabled={!selectedDept}>
                  <SelectTrigger><SelectValue placeholder="Select subject" /></SelectTrigger>
                  <SelectContent>{subjectList.map((s: any) => <SelectItem key={s.code} value={s.name}>{s.name}</SelectItem>)}</SelectContent>
                </Select>
              </div>
              <Separator />
              <div className="space-y-2">
                <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Fixed CO-Bloom Mapping</p>
                <div className="space-y-1.5">
                  {[1, 2, 3, 4, 5].map(mod => {
                    const d = CO_BLOOM_DEFAULTS[mod];
                    return (
                      <div key={mod} className="flex items-center justify-between text-xs px-2 py-1.5 rounded bg-muted/50">
                        <span className="font-medium">Module {mod}</span>
                        <div className="flex gap-1">
                          <Badge variant="secondary" className="text-[10px] px-1.5">{d.co}</Badge>
                          <Badge variant="outline" className="text-[10px] px-1.5">{d.bloom}</Badge>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <p className="text-[10px] text-muted-foreground">These assignments are locked during all question generation.</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Module Editor */}
        <div className="lg:col-span-3 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-semibold">{currentSyllabus.subjectName}</h2>
              <p className="text-sm text-muted-foreground">{currentSyllabus.subjectCode} · {currentSyllabus.department}</p>
            </div>
            <Button onClick={handleSaveSyllabus}>
              <Save className="mr-2 h-4 w-4" /> Save Syllabus
            </Button>
          </div>

          {currentSyllabus.modules.map((mod) => {
            const isExpanded = expandedModule === mod.number;
            const defaults = CO_BLOOM_DEFAULTS[mod.number];
            return (
              <Card key={mod.id} className="border-muted overflow-hidden">
                <button
                  className="w-full text-left"
                  onClick={() => setExpandedModule(isExpanded ? null : mod.number)}
                >
                  <div className="flex items-center justify-between px-5 py-3.5 hover:bg-muted/30 transition-colors">
                    <div className="flex items-center gap-3">
                      {isExpanded ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
                      <div>
                        <span className="font-semibold text-sm">Module {mod.number}: </span>
                        <span className="text-sm text-foreground">{mod.title}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="text-xs">{defaults.co}</Badge>
                      <Badge variant="outline" className="text-xs">{defaults.bloom}</Badge>
                      <span className="text-xs text-muted-foreground ml-1">{mod.topics.length} topics · {mod.hours}h</span>
                    </div>
                  </div>
                </button>

                <AnimatePresence initial={false}>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0 }}
                      animate={{ height: "auto" }}
                      exit={{ height: 0 }}
                      className="overflow-hidden"
                    >
                      <Separator />
                      <div className="px-5 py-4 space-y-5">
                        <div className="grid grid-cols-3 gap-4">
                          <div className="col-span-2 space-y-1.5">
                            <Label className="text-xs">Module Title</Label>
                            <Input value={mod.title} onChange={e => handleModuleFieldChange(mod.id, "title", e.target.value)} className="h-8 text-sm" />
                          </div>
                          <div className="space-y-1.5">
                            <Label className="text-xs">Teaching Hours</Label>
                            <Input type="number" value={mod.hours} onChange={e => handleModuleFieldChange(mod.id, "hours", parseInt(e.target.value))} className="h-8 text-sm" />
                          </div>
                        </div>

                        <div className="flex items-center gap-3 p-3 rounded-lg bg-primary/5 border border-primary/20">
                          <div className="flex items-center gap-2 flex-1">
                            <span className="text-xs font-semibold text-primary">Locked Assignment:</span>
                            <Badge className="text-xs bg-primary text-primary-foreground">{defaults.co}</Badge>
                            <span className="text-xs text-muted-foreground">·</span>
                            <Badge variant="outline" className="text-xs border-primary/40 text-primary">{defaults.bloom}</Badge>
                          </div>
                          <span className="text-[10px] text-muted-foreground">Cannot be changed — contact admin to modify</span>
                        </div>

                        <div className="space-y-2">
                          <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Syllabus Topics</Label>
                          <div className="space-y-1.5">
                            {mod.topics.map((topic, idx) => (
                              <div key={idx} className="flex items-center gap-2 p-2 rounded-md bg-muted/30 group">
                                <GripVertical className="h-3.5 w-3.5 text-muted-foreground opacity-40 cursor-grab" />
                                <span className="text-sm flex-1">{topic}</span>
                                <Button variant="ghost" size="icon" className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive" onClick={() => handleDeleteTopic(mod.id, idx)}>
                                  <Trash2 className="h-3 w-3" />
                                </Button>
                              </div>
                            ))}
                          </div>

                          {editingTopic?.moduleId === mod.id ? (
                            <div className="flex gap-2">
                              <Input
                                autoFocus
                                value={newTopicText}
                                onChange={e => setNewTopicText(e.target.value)}
                                placeholder="Enter topic name..."
                                className="h-8 text-sm flex-1"
                                onKeyDown={e => {
                                  if (e.key === "Enter") handleAddTopic(mod.id);
                                  if (e.key === "Escape") { setEditingTopic(null); setNewTopicText(""); }
                                }}
                              />
                              <Button size="sm" className="h-8" onClick={() => handleAddTopic(mod.id)}><Check className="h-3.5 w-3.5" /></Button>
                              <Button size="sm" variant="ghost" className="h-8" onClick={() => { setEditingTopic(null); setNewTopicText(""); }}><X className="h-3.5 w-3.5" /></Button>
                            </div>
                          ) : (
                            <Button variant="outline" size="sm" className="w-full h-8 border-dashed text-muted-foreground hover:text-foreground" onClick={() => setEditingTopic({ moduleId: mod.id, index: -1 })}>
                              <Plus className="h-3.5 w-3.5 mr-1" /> Add Topic
                            </Button>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </Card>
            );
          })}
        </div>
      </div>

      {/* ── Review & Apply Dialog ── */}
      <Dialog open={showReviewDialog} onOpenChange={open => { if (!open) handleCancelImport(); }}>
        <DialogContent className="max-w-3xl max-h-[85vh] flex flex-col p-0">
          <DialogHeader className="px-6 pt-6 pb-4 border-b shrink-0">
            <DialogTitle className="flex items-center gap-2 text-xl">
              <Sparkles className="h-5 w-5 text-primary" />
              Review Extracted Syllabus
            </DialogTitle>
            <DialogDescription>
              The AI has parsed <strong>{uploadedFile?.name}</strong> and extracted the following syllabus structure. Review each module's topics before applying. You can still edit after applying.
            </DialogDescription>
          </DialogHeader>

          {/* Stats Row */}
          <div className="px-6 py-3 border-b bg-muted/20 flex items-center gap-6 shrink-0">
            <div className="flex items-center gap-2 text-sm">
              <CheckCircle2 className="h-4 w-4 text-emerald-500" />
              <span className="font-semibold text-emerald-700 dark:text-emerald-400">5 modules detected</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <FileText className="h-4 w-4" />
              <span>27 topics extracted</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Brain className="h-4 w-4" />
              <span>CO-Bloom mapping applied automatically</span>
            </div>
          </div>

          {/* Module list */}
          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-3">
            {MOCK_PARSED_SYLLABUS.map(mod => {
              const isExpanded = reviewExpandedMod === mod.number;
              const defaults = CO_BLOOM_DEFAULTS[mod.number];
              return (
                <div key={mod.id} className="border rounded-lg overflow-hidden">
                  <button
                    className="w-full text-left px-4 py-3 hover:bg-muted/30 transition-colors flex items-center justify-between"
                    onClick={() => setReviewExpandedMod(isExpanded ? null : mod.number)}
                  >
                    <div className="flex items-center gap-3">
                      {isExpanded ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
                      <div>
                        <span className="font-semibold text-sm">Module {mod.number}: </span>
                        <span className="text-sm">{mod.title}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="text-[10px]">{defaults.co}</Badge>
                      <Badge variant="outline" className="text-[10px]">{defaults.bloom}</Badge>
                      <span className="text-xs text-muted-foreground">{mod.topics.length} topics · {mod.hours}h</span>
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 ml-1" />
                    </div>
                  </button>
                  <AnimatePresence initial={false}>
                    {isExpanded && (
                      <motion.div initial={{ height: 0 }} animate={{ height: "auto" }} exit={{ height: 0 }} className="overflow-hidden">
                        <Separator />
                        <div className="px-4 py-3 space-y-1.5 bg-muted/10">
                          {mod.topics.map((t, i) => (
                            <div key={i} className="flex items-start gap-2 text-sm">
                              <ArrowRight className="h-3.5 w-3.5 text-muted-foreground mt-0.5 shrink-0" />
                              <span>{t}</span>
                            </div>
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
            })}
          </div>

          <DialogFooter className="px-6 py-4 border-t bg-muted/20 shrink-0">
            <Button variant="outline" onClick={handleCancelImport}>Cancel</Button>
            <Button className="bg-primary gap-2" onClick={handleApplyParsed}>
              <Check className="h-4 w-4" /> Apply Syllabus
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
