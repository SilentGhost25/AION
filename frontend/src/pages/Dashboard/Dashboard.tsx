import { useState } from "react";
import { motion } from "framer-motion";
import { Link } from "wouter";
import {
  ArrowRight, BookOpen, Brain, CheckCircle2, FileText, ShieldCheck, Zap,
  Plus, Trash2, GraduationCap, BookMarked, Sparkles, X, Building2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { useProfile } from "@/context/profile-context";
import { departments, subjects as SUBJECT_LIST } from "@/lib/mock-data";
import { toast } from "sonner";

const SEMESTERS = ["1", "2", "3", "4", "5", "6", "7", "8"];
const SEM_SUFFIX = ["st", "nd", "rd", "th", "th", "th", "th", "th"];

const SUBJECT_STATS: Record<string, { materials: number; papers: number; chunks: number }> = {
  "21AI51": { materials: 4, papers: 2, chunks: 2161 },
  "21AI52": { materials: 1, papers: 1, chunks: 320 },
  "21AI55": { materials: 2, papers: 1, chunks: 870 },
};

export default function Home() {
  const { profile, addSubject, removeSubject } = useProfile();
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [newDept, setNewDept] = useState("");
  const [newSubjectName, setNewSubjectName] = useState("");
  const [newSubjectCode, setNewSubjectCode] = useState("");
  const [newSemester, setNewSemester] = useState("");

  const subjectList = newDept && (SUBJECT_LIST as any)[newDept]
    ? (SUBJECT_LIST as any)[newDept]
    : [];

  const handleAdd = () => {
    if (!newSubjectName || !newSubjectCode || !newDept || !newSemester) {
      toast.error("Please fill in all fields before adding.");
      return;
    }
    addSubject({ code: newSubjectCode, name: newSubjectName, department: newDept, semester: newSemester });
    toast.success(`${newSubjectName} added to your subjects.`);
    setShowAddDialog(false);
    setNewDept(""); setNewSubjectName(""); setNewSubjectCode(""); setNewSemester("");
  };

  const handleRemove = (id: string, name: string) => {
    removeSubject(id);
    toast.success(`${name} removed from your subjects.`);
  };

  return (
    <div className="space-y-10 pb-10">

      {/* ── Hero ── */}
      <section className="relative overflow-hidden rounded-3xl bg-primary text-primary-foreground shadow-lg">
        <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1541339907198-e08756dedf3f?auto=format&fit=crop&q=80')] bg-cover bg-center opacity-10 mix-blend-overlay" />
        <div className="absolute inset-0 bg-gradient-to-r from-primary to-primary/80" />
        <div className="relative z-10 p-8 md:p-14 flex flex-col md:flex-row items-center gap-10">
          <div className="flex-1 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-foreground/10 border border-primary-foreground/20 text-sm font-medium backdrop-blur-sm">
              <ShieldCheck className="h-4 w-4" />
              <span>DSATM Official Tool</span>
            </div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight leading-tight font-serif">
              Institutional-Grade <br /> Question Papers in Minutes
            </h1>
            <p className="text-lg text-primary-foreground/80 max-w-xl leading-relaxed">
              Perfect CO-PO mapping, NAAC compliance, and NBA-ready reports — built exclusively for DSATM faculty.
            </p>
            <div className="flex flex-wrap gap-4 pt-2">
              <Link href="/generate">
                <Button size="lg" variant="secondary" className="font-semibold px-8">
                  Generate Paper <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link href="/materials">
                <Button size="lg" variant="outline" className="bg-transparent border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10 hover:text-primary-foreground">
                  Upload Materials
                </Button>
              </Link>
            </div>
          </div>
          <div className="hidden lg:block w-1/3">
            <div className="relative aspect-[3/4] w-full max-w-sm ml-auto rounded-xl border border-primary-foreground/20 bg-primary-foreground/5 shadow-2xl backdrop-blur-sm p-6 overflow-hidden">
              <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-transparent via-primary-foreground/50 to-transparent" />
              <div className="space-y-4">
                <div className="h-6 w-1/2 bg-primary-foreground/20 rounded animate-pulse" />
                <div className="h-4 w-1/3 bg-primary-foreground/10 rounded animate-pulse" />
                <div className="space-y-2 mt-8">
                  {[1, 2, 3, 4, 5].map(i => (
                    <div key={i} className="flex gap-4">
                      <div className="h-4 w-4 rounded-full bg-primary-foreground/20 shrink-0 mt-1" />
                      <div className="space-y-2 flex-1">
                        <div className="h-4 w-full bg-primary-foreground/10 rounded" />
                        <div className="h-4 w-4/5 bg-primary-foreground/10 rounded" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Teacher Profile + My Subjects ── */}
      <section className="space-y-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
              <span className="text-lg font-bold text-primary">
                {profile.name.split(" ").map(n => n[0]).slice(0, 2).join("")}
              </span>
            </div>
            <div>
              <h2 className="text-xl font-bold font-serif">{profile.name}</h2>
              <p className="text-sm text-muted-foreground">{profile.designation} · {profile.department.split(" ").slice(0, 3).join(" ")}</p>
            </div>
          </div>
          <Button onClick={() => setShowAddDialog(true)} className="gap-2">
            <Plus className="h-4 w-4" /> Add Subject
          </Button>
        </div>

        {/* Subject Cards */}
        {profile.subjects.length === 0 ? (
          <div className="border border-dashed rounded-xl p-12 text-center space-y-3">
            <div className="w-14 h-14 bg-muted rounded-full flex items-center justify-center mx-auto">
              <BookOpen className="h-7 w-7 text-muted-foreground" />
            </div>
            <p className="font-medium text-foreground">No subjects added yet</p>
            <p className="text-sm text-muted-foreground">Click "Add Subject" to assign subjects to your profile. They will be available across all features.</p>
            <Button variant="outline" className="mt-2 gap-2" onClick={() => setShowAddDialog(true)}>
              <Plus className="h-4 w-4" /> Add your first subject
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {profile.subjects.map((subject, idx) => {
              const stats = SUBJECT_STATS[subject.code];
              return (
                <motion.div
                  key={subject.id}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.05 }}
                >
                  <Card className="border-muted hover:border-primary/40 hover:shadow-sm transition-all relative group">
                    <button
                      onClick={() => handleRemove(subject.id, subject.name)}
                      className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-md hover:bg-destructive/10 text-muted-foreground hover:text-destructive"
                      title="Remove subject"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                    <CardHeader className="pb-2 pt-4">
                      <div className="flex items-start gap-3">
                        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                          <BookMarked className="h-5 w-5 text-primary" />
                        </div>
                        <div className="min-w-0 pr-6">
                          <p className="font-semibold text-sm leading-tight">{subject.name}</p>
                          <div className="flex items-center gap-1.5 mt-1">
                            <Badge variant="outline" className="text-[10px] font-mono">{subject.code}</Badge>
                            <span className="text-[10px] text-muted-foreground">Sem {subject.semester}</span>
                          </div>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0 pb-4">
                      <p className="text-xs text-muted-foreground mb-3 truncate">{subject.department}</p>
                      {stats ? (
                        <div className="grid grid-cols-3 gap-2 text-center">
                          {[
                            { v: stats.materials, l: "Materials" },
                            { v: stats.papers, l: "Papers" },
                            { v: stats.chunks.toLocaleString(), l: "Segments" },
                          ].map(s => (
                            <div key={s.l} className="bg-muted/50 rounded-md py-1.5">
                              <p className="text-sm font-bold text-foreground">{s.v}</p>
                              <p className="text-[10px] text-muted-foreground">{s.l}</p>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="flex items-center gap-1.5 text-xs text-muted-foreground border border-dashed rounded-md px-3 py-2">
                          <Sparkles className="h-3.5 w-3.5 text-primary" />
                          Upload notes to get started
                        </div>
                      )}
                      <div className="flex gap-2 mt-3">
                        <Link href="/materials" className="flex-1">
                          <Button variant="outline" size="sm" className="w-full h-7 text-xs">Upload Notes</Button>
                        </Link>
                        <Link href="/generate" className="flex-1">
                          <Button size="sm" className="w-full h-7 text-xs">Generate</Button>
                        </Link>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}

            {/* Add Subject CTA card */}
            <button
              onClick={() => setShowAddDialog(true)}
              className="border border-dashed rounded-xl p-6 flex flex-col items-center justify-center gap-2 text-muted-foreground hover:border-primary/50 hover:text-primary hover:bg-primary/5 transition-all min-h-[180px]"
            >
              <div className="w-10 h-10 rounded-full border-2 border-dashed border-current flex items-center justify-center">
                <Plus className="h-5 w-5" />
              </div>
              <span className="text-sm font-medium">Add Another Subject</span>
            </button>
          </div>
        )}
      </section>

      <Separator />

      {/* ── 4-Step Model ── */}
      <section className="space-y-6">
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <h2 className="text-3xl font-bold text-foreground font-serif">The 4-Step AI Model</h2>
          <p className="text-muted-foreground text-lg">A systematic approach to standardized evaluations, reducing faculty workload while increasing quality.</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            { step: "01", title: "Upload Materials", desc: "Upload notes, textbooks, and syllabi. AI indexes all content.", icon: BookOpen, color: "text-blue-500", bg: "bg-blue-500/10" },
            { step: "02", title: "Define Syllabus", desc: "Set syllabus per module with locked CO-Bloom mapping.", icon: Brain, color: "text-emerald-500", bg: "bg-emerald-500/10" },
            { step: "03", title: "Generate Paper", desc: "Define scope in plain English. AI generates CO-PO compliant questions.", icon: Zap, color: "text-amber-500", bg: "bg-amber-500/10" },
            { step: "04", title: "Review & Submit", desc: "Live-edit questions, attach diagrams, finalize for HOD.", icon: FileText, color: "text-purple-500", bg: "bg-purple-500/10" },
          ].map((item, idx) => (
            <motion.div key={idx} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: idx * 0.1 }}>
              <Card className="relative h-full border-muted/50 hover:border-primary/50 transition-colors">
                <CardHeader>
                  <div className={`w-12 h-12 rounded-lg ${item.bg} flex items-center justify-center mb-4`}>
                    <item.icon className={`h-6 w-6 ${item.color}`} />
                  </div>
                  <p className="text-sm font-bold text-muted-foreground">Step {item.step}</p>
                  <CardTitle className="text-xl">{item.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground leading-relaxed text-sm">{item.desc}</p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── Add Subject Dialog ── */}
      <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <GraduationCap className="h-5 w-5 text-primary" />
              Add Subject to Profile
            </DialogTitle>
            <DialogDescription>
              This subject will be available for uploading notes, generating papers, and browsing the knowledge base.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-1.5">
              <Label>Department</Label>
              <Select onValueChange={v => { setNewDept(v); setNewSubjectName(""); setNewSubjectCode(""); }}>
                <SelectTrigger><SelectValue placeholder="Select department" /></SelectTrigger>
                <SelectContent>
                  {departments.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label>Subject</Label>
              {subjectList.length > 0 ? (
                <Select onValueChange={v => {
                  setNewSubjectName(v);
                  const found = subjectList.find((s: any) => s.name === v);
                  if (found) setNewSubjectCode(found.code);
                }}>
                  <SelectTrigger><SelectValue placeholder="Select subject" /></SelectTrigger>
                  <SelectContent>
                    {subjectList.map((s: any) => <SelectItem key={s.code} value={s.name}>{s.name}</SelectItem>)}
                  </SelectContent>
                </Select>
              ) : (
                <Input placeholder="e.g. Machine Learning" value={newSubjectName} onChange={e => setNewSubjectName(e.target.value)} />
              )}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label>Subject Code</Label>
                <Input placeholder="e.g. 21AI51" value={newSubjectCode} onChange={e => setNewSubjectCode(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label>Semester</Label>
                <Select onValueChange={setNewSemester}>
                  <SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger>
                  <SelectContent>
                    {SEMESTERS.map((s, i) => <SelectItem key={s} value={s}>{s}{SEM_SUFFIX[i]} Sem</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAddDialog(false)}>Cancel</Button>
            <Button onClick={handleAdd} className="gap-2">
              <Plus className="h-4 w-4" /> Add Subject
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
