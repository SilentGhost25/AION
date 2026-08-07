import { useState } from "react";
import { Upload, FileText, BookOpen, Trash2, CheckCircle2, Clock, AlertCircle, ChevronDown, ChevronRight, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { departments, subjects, studyMaterials } from "@/lib/mock-data";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";

type ProcessingStatus = "processed" | "processing" | "failed";

interface Material {
  id: string;
  name: string;
  type: "notes" | "textbook";
  module: number | "all";
  subject: string;
  subjectCode: string;
  size: string;
  uploadedOn: string;
  status: ProcessingStatus;
  pages: number;
  images: number;
  chunks: number;
}

const statusConfig: Record<ProcessingStatus, { label: string; icon: typeof CheckCircle2; className: string }> = {
  processed: { label: "Processed", icon: CheckCircle2, className: "text-emerald-600 bg-emerald-50 border-emerald-200 dark:bg-emerald-950 dark:border-emerald-800" },
  processing: { label: "Processing...", icon: Clock, className: "text-amber-600 bg-amber-50 border-amber-200 dark:bg-amberald-950 dark:border-amber-800" },
  failed: { label: "Failed", icon: AlertCircle, className: "text-red-600 bg-red-50 border-red-200 dark:bg-red-950 dark:border-red-800" },
};

export default function Materials() {
  const [selectedDept, setSelectedDept] = useState<string>("");
  const [selectedSubject, setSelectedSubject] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const [expandedModule, setExpandedModule] = useState<number | null>(null);
  const [materials, setMaterials] = useState<Material[]>(studyMaterials);
  const [uploadModule, setUploadModule] = useState<string>("");
  const [uploadType, setUploadType] = useState<string>("");

  const filteredMaterials = materials.filter(m =>
    (!selectedSubject || m.subject === selectedSubject)
  );

  const grouped = [1, 2, 3, 4, 5, "all" as const].reduce<Record<string, Material[]>>((acc, mod) => {
    const key = mod === "all" ? "all" : `${mod}`;
    acc[key] = filteredMaterials.filter(m => m.module === mod || (mod === "all" && m.module === "all"));
    return acc;
  }, {});

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  };

  const handleFiles = (files: File[]) => {
    if (!selectedSubject) {
      toast.error("Please select a subject before uploading.");
      return;
    }
    if (!uploadModule) {
      toast.error("Please select which module this material belongs to.");
      return;
    }
    if (!uploadType) {
      toast.error("Please select material type (Notes or Textbook).");
      return;
    }

    const newMaterials: Material[] = files.map((file, i) => ({
      id: `new-${Date.now()}-${i}`,
      name: file.name,
      type: uploadType as "notes" | "textbook",
      module: uploadModule === "all" ? "all" : parseInt(uploadModule),
      subject: selectedSubject,
      subjectCode: "21AI51",
      size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
      uploadedOn: new Date().toISOString().slice(0, 10),
      status: "processing",
      pages: 0,
      images: 0,
      chunks: 0,
    }));

    setMaterials(prev => [...prev, ...newMaterials]);
    toast.success(`${files.length} file(s) uploaded. AI is processing the content...`);

    // Simulate processing completion
    setTimeout(() => {
      setMaterials(prev =>
        prev.map(m =>
          newMaterials.find(n => n.id === m.id)
            ? { ...m, status: "processed", pages: Math.floor(Math.random() * 80) + 20, images: Math.floor(Math.random() * 15) + 1, chunks: Math.floor(Math.random() * 120) + 30 }
            : m
        )
      );
      toast.success("Processing complete. Content is now available for AI question generation.");
    }, 3000);
  };

  const handleDelete = (id: string) => {
    setMaterials(prev => prev.filter(m => m.id !== id));
    toast.success("Material removed.");
  };

  const subjectList = selectedDept && (subjects as any)[selectedDept]
    ? (subjects as any)[selectedDept]
    : Object.values(subjects).flat();

  const totalProcessed = filteredMaterials.filter(m => m.status === "processed").length;
  const totalChunks = filteredMaterials.filter(m => m.status === "processed").reduce((a, m) => a + m.chunks, 0);
  const totalImages = filteredMaterials.filter(m => m.status === "processed").reduce((a, m) => a + m.images, 0);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground font-serif tracking-tight">Study Materials</h1>
        <p className="text-muted-foreground mt-1">Upload module notes and textbooks. The AI processes and indexes all content — including images and diagrams — for intelligent question generation.</p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="border-muted">
          <CardContent className="pt-5 pb-4">
            <div className="text-2xl font-bold text-foreground">{totalProcessed}</div>
            <div className="text-sm text-muted-foreground mt-1">Documents Indexed</div>
          </CardContent>
        </Card>
        <Card className="border-muted">
          <CardContent className="pt-5 pb-4">
            <div className="text-2xl font-bold text-foreground">{totalChunks.toLocaleString()}</div>
            <div className="text-sm text-muted-foreground mt-1">Content Segments Stored</div>
          </CardContent>
        </Card>
        <Card className="border-muted">
          <CardContent className="pt-5 pb-4">
            <div className="text-2xl font-bold text-foreground">{totalImages}</div>
            <div className="text-sm text-muted-foreground mt-1">Figures & Diagrams Indexed</div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Panel */}
        <div className="lg:col-span-1 space-y-4">
          <Card className="border-muted">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Upload Material</CardTitle>
              <CardDescription className="text-sm">PDFs, DOCx, and image files are supported.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Department</Label>
                <Select onValueChange={v => { setSelectedDept(v); setSelectedSubject(""); }}>
                  <SelectTrigger><SelectValue placeholder="Select department" /></SelectTrigger>
                  <SelectContent>
                    {departments.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Subject</Label>
                <Select onValueChange={setSelectedSubject} disabled={!selectedDept}>
                  <SelectTrigger><SelectValue placeholder="Select subject" /></SelectTrigger>
                  <SelectContent>
                    {subjectList.map((s: any) => <SelectItem key={s.code} value={s.name}>{s.name}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Module</Label>
                <Select onValueChange={setUploadModule} disabled={!selectedSubject}>
                  <SelectTrigger><SelectValue placeholder="Select module" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">Module 1</SelectItem>
                    <SelectItem value="2">Module 2</SelectItem>
                    <SelectItem value="3">Module 3</SelectItem>
                    <SelectItem value="4">Module 4</SelectItem>
                    <SelectItem value="5">Module 5</SelectItem>
                    <SelectItem value="all">Entire Subject (All Modules)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Material Type</Label>
                <Select onValueChange={setUploadType} disabled={!uploadModule}>
                  <SelectTrigger><SelectValue placeholder="Notes or Textbook?" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="notes">Module Notes</SelectItem>
                    <SelectItem value="textbook">Textbook / Reference</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Separator />

              {/* Drop Zone */}
              <div
                data-testid="upload-dropzone"
                onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer ${
                  isDragging ? "border-primary bg-primary/5" : "border-muted hover:border-primary/50 hover:bg-muted/30"
                } ${!uploadType ? "opacity-50 pointer-events-none" : ""}`}
                onClick={() => {
                  if (!uploadType) return;
                  const input = document.createElement("input");
                  input.type = "file";
                  input.multiple = true;
                  input.accept = ".pdf,.docx,.doc,.png,.jpg,.jpeg";
                  input.onchange = (e) => {
                    const files = Array.from((e.target as HTMLInputElement).files || []);
                    handleFiles(files);
                  };
                  input.click();
                }}
              >
                <Upload className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                <p className="text-sm font-medium">Drop files here or click to browse</p>
                <p className="text-xs text-muted-foreground mt-1">PDF, DOCX, PNG, JPG supported</p>
              </div>

              <div className="rounded-lg border bg-muted/30 p-3 text-xs text-muted-foreground space-y-1">
                <p className="font-medium text-foreground text-xs">What the AI extracts:</p>
                <p>All text content, chapter structure, definitions, formulas, tables, and embedded images are processed and stored as searchable segments that the AI uses during question generation.</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Materials List */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-foreground">
              {selectedSubject ? selectedSubject : "All Subjects"} — Uploaded Materials
            </h2>
            <Select onValueChange={setSelectedSubject} defaultValue="">
              <SelectTrigger className="w-[220px]">
                <SelectValue placeholder="Filter by subject" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">All subjects</SelectItem>
                {Object.values(subjects).flat().map((s: any) => (
                  <SelectItem key={s.code} value={s.name}>{s.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {[1, 2, 3, 4, 5].map(mod => {
            const mods = grouped[`${mod}`] || [];
            const isExpanded = expandedModule === mod;
            if (mods.length === 0 && !isExpanded) return null;

            return (
              <Card key={mod} className="border-muted overflow-hidden">
                <button
                  className="w-full text-left"
                  onClick={() => setExpandedModule(isExpanded ? null : mod)}
                  data-testid={`module-${mod}-toggle`}
                >
                  <div className="flex items-center justify-between px-5 py-3 hover:bg-muted/30 transition-colors">
                    <div className="flex items-center gap-3">
                      {isExpanded ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
                      <span className="font-semibold text-sm">Module {mod}</span>
                      <Badge variant="secondary" className="text-xs">{mods.length} file{mods.length !== 1 ? "s" : ""}</Badge>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      {mods.filter(m => m.status === "processed").length} processed
                    </div>
                  </div>
                </button>

                <AnimatePresence initial={false}>
                  {(isExpanded || mods.length > 0) && (
                    <motion.div
                      initial={{ height: 0 }}
                      animate={{ height: "auto" }}
                      exit={{ height: 0 }}
                      className="overflow-hidden"
                    >
                      <Separator />
                      {mods.length === 0 ? (
                        <div className="px-5 py-6 text-center text-sm text-muted-foreground">
                          No materials uploaded for Module {mod} yet.
                        </div>
                      ) : (
                        <div className="divide-y divide-muted/50">
                          {mods.map(mat => {
                            const status = statusConfig[mat.status];
                            return (
                              <div key={mat.id} className="px-5 py-3 flex items-center gap-4" data-testid={`material-${mat.id}`}>
                                <div className="h-9 w-9 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                                  {mat.type === "textbook" ? <BookOpen className="h-4 w-4 text-primary" /> : <FileText className="h-4 w-4 text-primary" />}
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2 flex-wrap">
                                    <span className="text-sm font-medium truncate">{mat.name}</span>
                                    <Badge variant="outline" className="text-[10px] shrink-0">
                                      {mat.type === "textbook" ? "Textbook" : "Notes"}
                                    </Badge>
                                  </div>
                                  <div className="flex items-center gap-3 mt-0.5">
                                    {mat.status === "processing" ? (
                                      <div className="flex items-center gap-2 flex-1">
                                        <Progress value={45} className="h-1.5 flex-1 max-w-[120px]" />
                                        <span className="text-xs text-amber-600">Processing content...</span>
                                      </div>
                                    ) : mat.status === "processed" ? (
                                      <span className="text-xs text-muted-foreground">
                                        {mat.pages}p · {mat.images} figures · {mat.chunks} segments · {mat.size}
                                      </span>
                                    ) : (
                                      <span className="text-xs text-red-500">Processing failed — try re-uploading</span>
                                    )}
                                  </div>
                                </div>
                                <div className="flex items-center gap-2 shrink-0">
                                  <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border font-medium ${status.className}`}>
                                    <status.icon className="h-3 w-3" />
                                    {status.label}
                                  </span>
                                  <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground hover:text-destructive" onClick={() => handleDelete(mat.id)}>
                                    <Trash2 className="h-3.5 w-3.5" />
                                  </Button>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </Card>
            );
          })}

          {filteredMaterials.length === 0 && (
            <div className="flex flex-col items-center justify-center py-16 text-center border-2 border-dashed border-muted rounded-xl">
              <BookOpen className="h-10 w-10 text-muted-foreground mb-3 opacity-50" />
              <p className="font-medium text-muted-foreground">No materials uploaded yet</p>
              <p className="text-sm text-muted-foreground mt-1">Select a subject and upload notes or textbooks to get started.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
