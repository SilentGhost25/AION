import { useState, useRef } from "react";
import {
  useGeneratePaper,
  uploadFile,
  type UploadedFile,
  type GeneratedPaper,
  type MainQuestion,
  type ModuleQuestions,
} from "@/hooks/useGeneratePaper";

import { Button }       from "@/components/ui/button";
import { Input }        from "@/components/ui/input";
import { Label }        from "@/components/ui/label";
import { Progress }     from "@/components/ui/progress";
import { Badge }        from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const BLOOM_COLORS: Record<number, string> = {
  1: "bg-slate-500",
  2: "bg-blue-500",
  3: "bg-green-500",
  4: "bg-amber-500",
  5: "bg-red-500",
  6: "bg-purple-500",
};

const BLOOM_NAMES: Record<number, string> = {
  1: "Remember",
  2: "Understand",
  3: "Apply",
  4: "Analyze",
  5: "Evaluate",
  6: "Create",
};

export default function GeneratePage() {
  // ── File upload state ──────────────────────────────────────
  const fileInputRef                    = useRef<HTMLInputElement>(null);
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading]       = useState(false);
  const [uploadError, setUploadError]   = useState<string | null>(null);
  const [isDragging, setIsDragging]     = useState(false);

  // ── Config state ───────────────────────────────────────────
  const [subject,  setSubject]  = useState("AIML");
  const [examType, setExamType] = useState<"ia" | "see">("see");
  const [mode,     setMode]     = useState<"turbo" | "balanced" | "deep">("turbo");
  const [qpm,      setQpm]      = useState(4);

  // ── Generation hook ────────────────────────────────────────
  const { generate, cancel, reset, status, progress, logs, paper, error } =
    useGeneratePaper();

  // ── Selected questions ─────────────────────────────────────
  const [selectedQuestions, setSelectedQuestions] = useState<
    Map<string, MainQuestion>
  >(new Map());

  const toggleQuestion = (key: string, mq: MainQuestion) => {
    setSelectedQuestions((prev) => {
      const next = new Map(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.set(key, mq);
      }
      return next;
    });
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (!file) return;

    setUploading(true);
    setUploadError(null);
    setUploadedFile(null);
    setUploadProgress(0);

    try {
      const result = await uploadFile(file, subject, "notes", setUploadProgress);
      setUploadedFile(result);
    } catch (err: any) {
      setUploadError(err.message ?? "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  // ── File picker handler ────────────────────────────────────
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadError(null);
    setUploadedFile(null);
    setUploadProgress(0);

    try {
      const result = await uploadFile(
        file,
        subject,
        "notes",
        setUploadProgress,
      );
      setUploadedFile(result);
    } catch (err: any) {
      setUploadError(err.message ?? "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  // ── Generate handler ───────────────────────────────────────
  const handleGenerate = () => {
    if (!uploadedFile) return;
    generate({
      fileId:             uploadedFile.id,
      subject,
      examType,
      mode,
      maxConcepts:        qpm * 5,
      questionsPerModule: qpm,
    });
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-6">
      <h1 className="text-2xl font-bold">Generate Question Paper</h1>

      {/* ── Configuration ───────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>Configuration</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">

          {/* File picker with Drag and Drop */}
          <div
            className={`sm:col-span-2 lg:col-span-3 space-y-2 rounded-lg border-2 border-dashed p-6 transition-colors ${
              isDragging
                ? "border-primary bg-primary/5"
                : "border-muted-foreground/25"
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <Label>Academic Material (PDF, TXT, DOCX)</Label>

            <div className="flex flex-col items-center gap-3 py-4">
              <p className="text-sm text-muted-foreground">
                {isDragging
                  ? "Drop file here..."
                  : "Drag & drop a file here, or click below to browse"}
              </p>

              <Button
                type="button"
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
              >
                {uploading ? "Uploading..." : "📂 Choose File"}
              </Button>

              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.docx,.pptx,.md"
                className="hidden"
                onChange={handleFileChange}
              />
            </div>

            {/* Upload progress */}
            {uploading && (
              <div className="space-y-1">
                <Progress value={uploadProgress * 100} />
                <p className="text-xs text-muted-foreground">
                  Uploading... {Math.round(uploadProgress * 100)}%
                </p>
              </div>
            )}

            {/* Success */}
            {uploadedFile && !uploading && (
              <div className="flex items-center gap-2 text-sm text-green-600">
                <span>✓</span>
                <span>
                  {uploadedFile.filename} uploaded (
                  {(uploadedFile.sizeBytes / 1024).toFixed(0)} KB)
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setUploadedFile(null);
                    if (fileInputRef.current) fileInputRef.current.value = "";
                  }}
                >
                  ✕ Remove
                </Button>
              </div>
            )}

            {/* Error */}
            {uploadError && (
              <p className="text-sm text-destructive">{uploadError}</p>
            )}
          </div>

          {/* Subject */}
          <div>
            <Label>Subject</Label>
            <Input value={subject} onChange={(e) => setSubject(e.target.value)} />
          </div>

          {/* Exam type */}
          <div>
            <Label>Exam Type</Label>
            <Select value={examType} onValueChange={(v) => setExamType(v as any)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="ia">Internal Assessment (10M)</SelectItem>
                <SelectItem value="see">Semester End Exam (20M)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Mode */}
          <div>
            <Label>Mode</Label>
            <Select value={mode} onValueChange={(v) => setMode(v as any)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="turbo">⚡ Turbo (Fast)</SelectItem>
                <SelectItem value="balanced">⚖️ Balanced</SelectItem>
                <SelectItem value="deep">🎯 Deep</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Questions per module */}
          <div>
            <Label>Questions per Module</Label>
            <Input
              type="number"
              min={2}
              max={8}
              value={qpm}
              onChange={(e) => setQpm(Number(e.target.value))}
            />
          </div>

        </CardContent>
      </Card>

      {/* ── Action buttons ───────────────────────────────────── */}
      <div className="flex gap-3">
        <Button
          onClick={handleGenerate}
          disabled={
            status === "running" ||
            !uploadedFile        ||
            uploading
          }
        >
          {status === "running" ? "Generating..." : "🚀 Generate Paper"}
        </Button>

        {status === "running" && (
          <Button variant="outline" onClick={cancel}>
            Cancel
          </Button>
        )}

        {(status === "done" || status === "error") && (
          <Button variant="ghost" onClick={() => { reset(); setUploadedFile(null); }}>
            Reset
          </Button>
        )}
      </div>

      {/* ── Progress ─────────────────────────────────────────── */}
      {status === "running" && (
        <Card>
          <CardContent className="pt-6 space-y-3">
            <Progress value={progress * 100} />
            <div className="h-40 overflow-y-auto rounded bg-black p-3 font-mono text-xs text-green-400">
              {logs.map((log, i) => (
                <div key={i}>{log}</div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Error ────────────────────────────────────────────── */}
      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-6 space-y-1">
            <p className="font-medium text-destructive">{error}</p>
            <p className="text-sm text-muted-foreground">
              Make sure AION API is running:{" "}
              <code className="text-xs">python aion_api.py</code>
            </p>
          </CardContent>
        </Card>
      )}

      {/* ── Generated paper ──────────────────────────────────── */}
      {paper && (
        <PaperView
          paper={paper}
          selected={selectedQuestions}
          onToggle={toggleQuestion}
        />
      )}

      {/* ── Selected summary ─────────────────────────────────── */}
      {selectedQuestions.size > 0 && (
        <SelectedSummary
          selected={selectedQuestions}
          examType={examType}
        />
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Paper View — shows questions in OR format like terminal
// ─────────────────────────────────────────────────────────────

function PaperView({
  paper,
  selected,
  onToggle,
}: {
  paper:    GeneratedPaper;
  selected: Map<string, MainQuestion>;
  onToggle: (key: string, mq: MainQuestion) => void;
}) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">
          {paper.subject} — {paper.examType.toUpperCase()} Paper
        </h2>
        <Badge variant="outline">
          Total: {paper.totalMarks} Marks | {paper.modules.length} Modules
        </Badge>
      </div>

      {paper.modules.map((mod) => (
        <ModuleCard
          key={mod.moduleIndex}
          mod={mod}
          examType={paper.examType}
          selected={selected}
          onToggle={onToggle}
        />
      ))}
    </div>
  );
}


function ModuleCard({
  mod,
  examType,
  selected,
  onToggle,
}: {
  mod:      ModuleQuestions;
  examType: string;
  selected: Map<string, MainQuestion>;
  onToggle: (key: string, mq: MainQuestion) => void;
}) {
  // Group questions into OR pairs: [Q1 OR Q2], [Q3 OR Q4]
  const pairs: MainQuestion[][] = [];
  for (let i = 0; i < mod.questions.length; i += 2) {
    const pair = [mod.questions[i]];
    if (mod.questions[i + 1]) pair.push(mod.questions[i + 1]);
    pairs.push(pair);
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-3">
          <span>Module {mod.moduleIndex}</span>
          <span className="text-base font-normal text-muted-foreground">
            {mod.moduleTitle}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {pairs.map((pair, pairIdx) => (
          <div key={pairIdx} className="space-y-1">
            {/* Choice label */}
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              Choice {pairIdx + 1}
            </p>

            {pair.map((mq, qIdx) => {
              const key      = `m${mod.moduleIndex}-q${mq.mqIndex}`;
              const isSelected = selected.has(key);

              return (
                <div key={qIdx}>
                  {/* OR separator */}
                  {qIdx > 0 && (
                    <div className="flex items-center gap-2 py-1">
                      <div className="flex-1 border-t border-dashed" />
                      <span className="text-xs font-bold text-muted-foreground">OR</span>
                      <div className="flex-1 border-t border-dashed" />
                    </div>
                  )}

                  <div
                    className={`rounded-lg border p-4 cursor-pointer transition-all ${
                      isSelected
                        ? "border-primary bg-primary/5 ring-2 ring-primary"
                        : "hover:border-primary/50"
                    }`}
                    onClick={() => onToggle(key, mq)}
                  >
                    {/* Header row */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="font-bold">Q{mq.mqIndex}</span>
                        <Badge className={BLOOM_COLORS[mq.bloomLevel] ?? "bg-gray-500"}>
                          L{mq.bloomLevel}: {BLOOM_NAMES[mq.bloomLevel] || mq.bloomName}
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          {mq.totalMarks} Marks
                        </span>
                      </div>
                      {isSelected && (
                        <Badge variant="default" className="bg-green-600">
                          ✓ Added to Paper
                        </Badge>
                      )}
                    </div>

                    {/* Sub-questions */}
                    {mq.subQuestions.length === 1 ? (
                      <p className="text-sm leading-relaxed">
                        {mq.subQuestions[0].text}
                        <span className="ml-2 text-xs text-muted-foreground">
                          [{mq.subQuestions[0].marks}M]
                        </span>
                      </p>
                    ) : (
                      <div className="space-y-2">
                        {mq.subQuestions.map((sq, si) => (
                          <div key={si} className="flex gap-2 text-sm">
                            <span className="font-medium text-muted-foreground min-w-[24px]">
                              ({sq.letter})
                            </span>
                            <span className="flex-1 leading-relaxed">{sq.text}</span>
                            <span className="text-xs text-muted-foreground whitespace-nowrap">
                              [{sq.marks}M]
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}


// ─────────────────────────────────────────────────────────────
// Selected Questions Summary — "Add to Paper" panel
// ─────────────────────────────────────────────────────────────

function SelectedSummary({
  selected,
  examType,
}: {
  selected: Map<string, MainQuestion>;
  examType: string;
}) {
  const questions   = [...selected.values()];
  const totalMarks  = questions.reduce((sum, q) => sum + q.totalMarks, 0);
  const targetMarks = examType === "see" ? 100 : 50;

  return (
    <Card className="sticky bottom-4 border-primary bg-primary/5">
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-bold text-lg">
              📝 Selected Questions: {questions.length}
            </h3>
            <p className="text-sm text-muted-foreground">
              Total: {totalMarks} / {targetMarks} Marks
              {totalMarks > targetMarks && (
                <span className="ml-2 text-destructive font-medium">
                  (Exceeds target by {totalMarks - targetMarks})
                </span>
              )}
              {totalMarks === targetMarks && (
                <span className="ml-2 text-green-600 font-medium">
                  ✓ Perfect match
                </span>
              )}
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => window.print()}>
              🖨️ Print
            </Button>
            <Button
              size="sm"
              disabled={totalMarks !== targetMarks}
            >
              ✅ Finalize Paper
            </Button>
          </div>
        </div>

        {/* Quick preview of selected */}
        <div className="mt-4 space-y-1 max-h-40 overflow-y-auto">
          {questions.map((q, i) => (
            <div key={i} className="flex justify-between text-sm">
              <span>Q{q.mqIndex}: {q.subQuestions[0]?.text?.slice(0, 60)}...</span>
              <span className="text-muted-foreground">[{q.totalMarks}M]</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
