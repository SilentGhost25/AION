import { useState, useEffect } from "react";
import { Download, Trash2, Calendar, FileText, ShieldCheck, BookOpen, Clock, ChevronRight, Search, Edit3, Eye, CheckCircle2, Send, RotateCcw, AlertTriangle, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { historyData, MODULE_CO_BLOOM } from "@/lib/mock-data";
import { toast } from "sonner";
import { PaperPreview, PaperQuestion } from "@/components/paper-preview";
import { motion, AnimatePresence } from "framer-motion";

type PaperRecord = typeof historyData[number] & {
  questions: PaperQuestion[];
  finalized?: boolean;
};

export default function History() {
  const [papers, setPapers] = useState<PaperRecord[]>(
    historyData.map(p => ({ ...p, finalized: p.status === "Downloaded" }))
  );
  const [selectedId, setSelectedId] = useState<string>(historyData[0]?.id ?? "");
  const [search, setSearch] = useState("");
  const [editMode, setEditMode] = useState(false);
  const [editingQuestions, setEditingQuestions] = useState<PaperQuestion[]>([]);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [showFinalizeDialog, setShowFinalizeDialog] = useState(false);
  const [showDiscardDialog, setShowDiscardDialog] = useState(false);
  const [pendingSelectId, setPendingSelectId] = useState<string | null>(null);

  const filtered = papers.filter(p =>
    p.subject.toLowerCase().includes(search.toLowerCase()) ||
    p.examType.toLowerCase().includes(search.toLowerCase()) ||
    p.id.toLowerCase().includes(search.toLowerCase())
  );

  const selectedPaper = papers.find(p => p.id === selectedId);

  // When selected paper changes, load its questions into edit buffer
  useEffect(() => {
    if (selectedPaper) {
      setEditingQuestions(selectedPaper.questions.map(q => ({ ...q })));
      setEditMode(false);
      setHasUnsavedChanges(false);
    }
  }, [selectedId]);

  const handleSelectPaper = (id: string) => {
    if (hasUnsavedChanges && id !== selectedId) {
      setPendingSelectId(id);
      setShowDiscardDialog(true);
    } else {
      setSelectedId(id);
    }
  };

  const handleQuestionsChange = (questions: PaperQuestion[]) => {
    setEditingQuestions(questions);
    setHasUnsavedChanges(true);
  };

  const handleSaveEdits = () => {
    setPapers(prev =>
      prev.map(p => p.id === selectedId ? { ...p, questions: editingQuestions } : p)
    );
    setHasUnsavedChanges(false);
    toast.success("Changes saved to the paper.");
  };

  const handleDiscardEdits = () => {
    if (selectedPaper) {
      setEditingQuestions(selectedPaper.questions.map(q => ({ ...q })));
    }
    setHasUnsavedChanges(false);
    setEditMode(false);
    if (pendingSelectId) {
      setSelectedId(pendingSelectId);
      setPendingSelectId(null);
    }
    setShowDiscardDialog(false);
  };

  const handleFinalize = () => {
    if (hasUnsavedChanges) handleSaveEdits();
    setPapers(prev =>
      prev.map(p => p.id === selectedId ? { ...p, finalized: true, status: "Downloaded" } : p)
    );
    setEditMode(false);
    setShowFinalizeDialog(false);
    toast.success("Paper finalized and marked ready for HOD submission.", { duration: 4000 });
  };

  const handleDownload = (id: string) => {
    toast.success(`Paper ${id}.docx downloaded successfully`);
  };

  const handleDelete = (id: string) => {
    setPapers(prev => prev.filter(p => p.id !== id));
    const remaining = papers.filter(p => p.id !== id);
    setSelectedId(remaining[0]?.id ?? "");
    toast.success(`Paper ${id} deleted`);
  };

  const displayQuestions = editMode ? editingQuestions : (selectedPaper?.questions ?? []);

  return (
    <div className="space-y-4 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-start justify-between flex-shrink-0">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-serif tracking-tight">Paper Review</h1>
          <p className="text-muted-foreground">Review, edit, and finalize papers before submission to HOD.</p>
        </div>
        <div className="flex items-center gap-3 text-sm text-muted-foreground">
          <div className="flex items-center gap-1.5 bg-card border rounded-md px-3 py-1.5">
            <FileText className="h-4 w-4 text-primary" />
            <span className="font-medium text-foreground">{papers.length}</span>
            <span>papers</span>
          </div>
        </div>
      </div>

      {/* Split Panel */}
      <div className="flex gap-4 flex-1 min-h-0" style={{ height: "calc(100vh - 220px)" }}>

        {/* ── Left: Paper List ── */}
        <div className="w-80 flex-shrink-0 flex flex-col gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input placeholder="Search papers..." className="pl-9" value={search} onChange={e => setSearch(e.target.value)} />
          </div>
          <div className="flex-1 overflow-y-auto space-y-2 pr-1">
            {filtered.length === 0 && (
              <div className="text-center py-12 text-muted-foreground text-sm">No papers found.</div>
            )}
            {filtered.map(paper => {
              const isSelected = paper.id === selectedId;
              return (
                <button
                  key={paper.id}
                  onClick={() => handleSelectPaper(paper.id)}
                  className={`w-full text-left rounded-lg border p-4 transition-all duration-150 ${
                    isSelected ? "border-primary bg-primary/5 shadow-sm" : "border-muted bg-card hover:border-primary/40 hover:bg-muted/30"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <div className={`w-8 h-8 rounded-md flex items-center justify-center shrink-0 ${isSelected ? "bg-primary" : "bg-muted"}`}>
                        <FileText className={`h-4 w-4 ${isSelected ? "text-primary-foreground" : "text-muted-foreground"}`} />
                      </div>
                      <div className="min-w-0">
                        <p className="font-semibold text-sm leading-tight truncate">{paper.subject}</p>
                        <p className="text-xs text-muted-foreground font-mono">{paper.subjectCode}</p>
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-1 shrink-0">
                      <Badge variant={paper.finalized ? "default" : "secondary"} className="text-[10px]">
                        {paper.finalized ? "Finalized" : paper.status}
                      </Badge>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1"><BookOpen className="h-3 w-3" />{paper.examType}</span>
                    <span className="flex items-center gap-1"><Calendar className="h-3 w-3" />{paper.generatedOn}</span>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {paper.modulesIncluded.map(mod => (
                      <span key={mod} className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${isSelected ? "bg-primary/15 text-primary" : "bg-muted text-muted-foreground"}`}>
                        M{mod}
                      </span>
                    ))}
                  </div>
                  <div className="mt-2 flex items-center justify-between">
                    <span className="text-[10px] text-muted-foreground font-mono">{paper.id}</span>
                    <ChevronRight className={`h-3.5 w-3.5 ${isSelected ? "text-primary" : "text-muted-foreground/50"}`} />
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* ── Right: Paper Review / Edit ── */}
        <div className="flex-1 min-w-0 flex flex-col rounded-lg border bg-card overflow-hidden">
          <AnimatePresence mode="wait">
            {!selectedPaper ? (
              <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex-1 flex items-center justify-center text-center p-12">
                <div className="space-y-3">
                  <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto">
                    <FileText className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <p className="font-medium text-foreground">Select a paper to review</p>
                  <p className="text-sm text-muted-foreground">Click any paper from the list on the left.</p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key={selectedPaper.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.15 }}
                className="flex flex-col h-full"
              >
                {/* Review Header */}
                <div className="px-6 py-4 border-b flex items-start justify-between gap-4 flex-shrink-0">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <h2 className="font-bold text-lg">{selectedPaper.subject}</h2>
                      <Badge variant="outline" className="font-mono text-xs">{selectedPaper.subjectCode}</Badge>
                      {selectedPaper.finalized && (
                        <Badge className="bg-emerald-100 text-emerald-800 border border-emerald-200 dark:bg-emerald-900 dark:text-emerald-200">
                          <CheckCircle2 className="h-3 w-3 mr-1" /> Finalized
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      <span className="flex items-center gap-1.5"><BookOpen className="h-3.5 w-3.5" />{selectedPaper.examType} · Sem {selectedPaper.semester}</span>
                      <span className="flex items-center gap-1.5"><Calendar className="h-3.5 w-3.5" />Generated {selectedPaper.generatedOn}</span>
                      <span className="flex items-center gap-1.5"><Clock className="h-3.5 w-3.5" />{selectedPaper.duration}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 shrink-0 flex-wrap justify-end">
                    {editMode ? (
                      <>
                        {hasUnsavedChanges && (
                          <span className="text-xs text-amber-600 font-medium flex items-center gap-1">
                            <AlertTriangle className="h-3.5 w-3.5" /> Unsaved changes
                          </span>
                        )}
                        <Button variant="outline" size="sm" onClick={() => { if (hasUnsavedChanges) { setShowDiscardDialog(true); } else { setEditMode(false); } }}>
                          <RotateCcw className="mr-2 h-3.5 w-3.5" /> Discard
                        </Button>
                        <Button variant="outline" size="sm" onClick={handleSaveEdits} disabled={!hasUnsavedChanges}>
                          Save Edits
                        </Button>
                        <Button size="sm" className="bg-emerald-600 hover:bg-emerald-700 text-white" onClick={() => setShowFinalizeDialog(true)}>
                          <Send className="mr-2 h-3.5 w-3.5" /> Finalize for HOD
                        </Button>
                      </>
                    ) : (
                      <>
                        <Button
                          variant="outline"
                          size="sm"
                          className="text-destructive hover:text-destructive border-destructive/30 hover:border-destructive/60"
                          onClick={() => handleDelete(selectedPaper.id)}
                        >
                          <Trash2 className="mr-2 h-3.5 w-3.5" /> Delete
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => setEditMode(true)}>
                          <Edit3 className="mr-2 h-3.5 w-3.5" /> Edit Paper
                        </Button>
                        <Button size="sm" onClick={() => handleDownload(selectedPaper.id)}>
                          <Download className="mr-2 h-3.5 w-3.5" /> Download .docx
                        </Button>
                      </>
                    )}
                  </div>
                </div>

                {/* Edit Mode Banner */}
                <AnimatePresence>
                  {editMode && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="px-6 py-2.5 bg-blue-50 dark:bg-blue-950/40 border-b border-blue-200 dark:border-blue-800 flex items-center gap-2">
                        <Edit3 className="h-4 w-4 text-blue-600 shrink-0" />
                        <p className="text-xs text-blue-700 dark:text-blue-300 font-medium">
                          Edit mode is active — click any question to edit its text, marks, CO, or Bloom level. Use the <span className="font-bold">Add diagram / figure</span> button below a question to attach an image inline.
                        </p>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* CO Compliance bar */}
                <div className="px-6 py-3 border-b bg-muted/20 flex-shrink-0">
                  <div className="flex items-center gap-4 flex-wrap">
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="h-4 w-4 text-emerald-600" />
                      <span className="text-xs font-semibold text-emerald-700 dark:text-emerald-400">
                        {displayQuestions.length} questions · CO-Bloom verified
                      </span>
                    </div>
                    <Separator orientation="vertical" className="h-4" />
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <span className="font-medium">Coverage:</span>
                      <span className="italic">{selectedPaper.coverageScope}</span>
                    </div>
                    <Separator orientation="vertical" className="h-4" />
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground font-medium">Modules:</span>
                      {selectedPaper.modulesIncluded.map(mod => {
                        const mapping = MODULE_CO_BLOOM[mod];
                        return (
                          <span key={mod} className="inline-flex items-center gap-1 text-[10px] bg-primary/10 text-primary border border-primary/20 px-1.5 py-0.5 rounded font-semibold">
                            M{mod} · {mapping.co} · {mapping.bloom}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Paper Content */}
                <div className="flex-1 overflow-y-auto bg-slate-100 dark:bg-slate-900 p-6">
                  <div className="max-w-3xl mx-auto bg-white shadow-md border rounded-sm">
                    <div className="p-8">
                      <PaperPreview
                        formData={{
                          examType: selectedPaper.examType,
                          department: selectedPaper.department,
                          subjectName: selectedPaper.subject,
                          subjectCode: selectedPaper.subjectCode,
                          semester: selectedPaper.semester,
                          maxMarks: selectedPaper.maxMarks,
                          batch: selectedPaper.batch,
                          duration: selectedPaper.duration,
                          dateOfIat: selectedPaper.dateOfIat,
                          teachingDept: selectedPaper.teachingDept,
                        }}
                        questions={displayQuestions}
                        editable={editMode}
                        onQuestionsChange={editMode ? handleQuestionsChange : undefined}
                      />
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── Finalize for HOD Dialog ── */}
      <Dialog open={showFinalizeDialog} onOpenChange={setShowFinalizeDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Send className="h-5 w-5 text-emerald-600" />
              Finalize Paper for HOD
            </DialogTitle>
            <DialogDescription className="space-y-2 pt-2">
              <p>This will lock the paper and mark it as ready for HOD submission. You can still download or view it, but it will be tagged as finalized.</p>
              {hasUnsavedChanges && (
                <div className="flex items-start gap-2 p-3 rounded-md bg-amber-50 border border-amber-200 mt-3">
                  <AlertTriangle className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
                  <p className="text-sm text-amber-700">You have unsaved edits. They will be saved automatically when you finalize.</p>
                </div>
              )}
            </DialogDescription>
          </DialogHeader>
          <div className="py-2 space-y-2">
            <div className="flex items-center gap-3 p-3 rounded-md bg-muted/50 text-sm">
              <FileText className="h-4 w-4 text-primary shrink-0" />
              <div>
                <p className="font-medium">{selectedPaper?.subject}</p>
                <p className="text-muted-foreground text-xs">{selectedPaper?.examType} · {selectedPaper?.id} · {displayQuestions.length} questions</p>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowFinalizeDialog(false)}>Cancel</Button>
            <Button className="bg-emerald-600 hover:bg-emerald-700 text-white" onClick={handleFinalize}>
              <CheckCircle2 className="mr-2 h-4 w-4" /> Confirm &amp; Finalize
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── Discard Edits Dialog ── */}
      <Dialog open={showDiscardDialog} onOpenChange={setShowDiscardDialog}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-500" />
              Discard unsaved changes?
            </DialogTitle>
            <DialogDescription>
              You have unsaved edits to this paper. Leaving will discard all changes made in this session.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => { setShowDiscardDialog(false); setPendingSelectId(null); }}>Keep editing</Button>
            <Button variant="destructive" onClick={handleDiscardEdits}>Discard changes</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
