import { useState } from "react";
import { Plus, Trash2, Save, BookOpen, GripVertical, ChevronDown, ChevronRight, Edit2, Check, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
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

export default function Syllabus() {
  const [selectedDept, setSelectedDept] = useState<string>("");
  const [selectedSubject, setSelectedSubject] = useState<string>("");
  const [expandedModule, setExpandedModule] = useState<number | null>(1);
  const [editingModule, setEditingModule] = useState<string | null>(null);
  const [syllabi, setSyllabi] = useState<SubjectSyllabus[]>(syllabusData);
  const [editingTopic, setEditingTopic] = useState<{ moduleId: string; index: number } | null>(null);
  const [newTopicText, setNewTopicText] = useState("");

  const subjectList = selectedDept && (subjects as any)[selectedDept]
    ? (subjects as any)[selectedDept]
    : Object.values(subjects).flat();

  const currentSyllabus = syllabi.find(s => s.subjectName === selectedSubject) || syllabi[0];

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

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground font-serif tracking-tight">Syllabus Manager</h1>
        <p className="text-muted-foreground mt-1">Define the syllabus per subject. The AI strictly adheres to this syllabus when generating questions — no out-of-scope content.</p>
      </div>

      {/* Info Banner */}
      <div className="flex items-start gap-3 p-4 rounded-lg border border-primary/30 bg-primary/5">
        <BookOpen className="h-5 w-5 text-primary mt-0.5 shrink-0" />
        <div>
          <p className="text-sm font-semibold text-primary">Strict Syllabus Compliance</p>
          <p className="text-sm text-muted-foreground mt-0.5">
            Every question generated will be validated against the syllabus topics defined here. The AI will not generate questions on topics not listed in the syllabus. Each module has a fixed CO and Bloom's level assignment that cannot be overridden during paper generation.
          </p>
        </div>
      </div>

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

              <Separator />

              {/* Module CO mapping summary */}
              <div className="space-y-2">
                <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Fixed CO-Bloom Mapping</p>
                <div className="space-y-1.5">
                  {[1, 2, 3, 4, 5].map(mod => {
                    const defaults = CO_BLOOM_DEFAULTS[mod];
                    return (
                      <div key={mod} className="flex items-center justify-between text-xs px-2 py-1.5 rounded bg-muted/50">
                        <span className="font-medium">Module {mod}</span>
                        <div className="flex gap-1">
                          <Badge variant="secondary" className="text-[10px] px-1.5">{defaults.co}</Badge>
                          <Badge variant="outline" className="text-[10px] px-1.5">{defaults.bloom}</Badge>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <p className="text-[10px] text-muted-foreground">These assignments are locked and enforced during all question generation.</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Syllabus Editor */}
        <div className="lg:col-span-3 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-semibold">{currentSyllabus.subjectName}</h2>
              <p className="text-sm text-muted-foreground">{currentSyllabus.subjectCode} · {currentSyllabus.department}</p>
            </div>
            <Button onClick={handleSaveSyllabus} data-testid="save-syllabus">
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
                  data-testid={`syllabus-module-${mod.number}`}
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
                        {/* Module title & hours */}
                        <div className="grid grid-cols-3 gap-4">
                          <div className="col-span-2 space-y-1.5">
                            <Label className="text-xs">Module Title</Label>
                            <Input
                              value={mod.title}
                              onChange={e => handleModuleFieldChange(mod.id, "title", e.target.value)}
                              className="h-8 text-sm"
                              data-testid={`module-${mod.number}-title`}
                            />
                          </div>
                          <div className="space-y-1.5">
                            <Label className="text-xs">Teaching Hours</Label>
                            <Input
                              type="number"
                              value={mod.hours}
                              onChange={e => handleModuleFieldChange(mod.id, "hours", parseInt(e.target.value))}
                              className="h-8 text-sm"
                            />
                          </div>
                        </div>

                        {/* CO & Bloom (read-only) */}
                        <div className="flex items-center gap-3 p-3 rounded-lg bg-primary/5 border border-primary/20">
                          <div className="flex items-center gap-2 flex-1">
                            <span className="text-xs font-semibold text-primary">Locked Assignment:</span>
                            <Badge className="text-xs bg-primary text-primary-foreground">{defaults.co}</Badge>
                            <span className="text-xs text-muted-foreground">·</span>
                            <Badge variant="outline" className="text-xs border-primary/40 text-primary">{defaults.bloom}</Badge>
                          </div>
                          <span className="text-[10px] text-muted-foreground">Cannot be changed — contact admin to modify CO-Bloom mappings</span>
                        </div>

                        {/* Topics */}
                        <div className="space-y-2">
                          <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Syllabus Topics</Label>
                          <div className="space-y-1.5">
                            {mod.topics.map((topic, idx) => (
                              <div key={idx} className="flex items-center gap-2 p-2 rounded-md bg-muted/30 group">
                                <GripVertical className="h-3.5 w-3.5 text-muted-foreground opacity-40 cursor-grab" />
                                <span className="text-sm flex-1">{topic}</span>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive"
                                  onClick={() => handleDeleteTopic(mod.id, idx)}
                                >
                                  <Trash2 className="h-3 w-3" />
                                </Button>
                              </div>
                            ))}
                          </div>

                          {/* Add topic */}
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
                                data-testid="new-topic-input"
                              />
                              <Button size="sm" className="h-8" onClick={() => handleAddTopic(mod.id)}>
                                <Check className="h-3.5 w-3.5" />
                              </Button>
                              <Button size="sm" variant="ghost" className="h-8" onClick={() => { setEditingTopic(null); setNewTopicText(""); }}>
                                <X className="h-3.5 w-3.5" />
                              </Button>
                            </div>
                          ) : (
                            <Button
                              variant="outline"
                              size="sm"
                              className="w-full h-8 border-dashed text-muted-foreground hover:text-foreground"
                              onClick={() => setEditingTopic({ moduleId: mod.id, index: -1 })}
                              data-testid={`add-topic-module-${mod.number}`}
                            >
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
    </div>
  );
}
