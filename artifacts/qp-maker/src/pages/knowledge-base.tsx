import { useState, useMemo } from "react";
import { Search, FileText, ImageIcon, LayoutGrid, List, Filter, BookOpen, Hash, Layers, ChevronDown, ChevronRight, Copy, ExternalLink } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useProfile } from "@/context/profile-context";
import { KB_CHUNKS, KB_IMAGES, IMAGE_TYPE_COLORS, type ImageType } from "@/lib/knowledge-mock";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";

type TabType = "all" | "chunks" | "images";
type ViewMode = "grid" | "list";

const MODULE_COLORS: Record<number, string> = {
  1: "bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-900 dark:text-blue-300",
  2: "bg-emerald-100 text-emerald-700 border-emerald-200 dark:bg-emerald-900 dark:text-emerald-300",
  3: "bg-amber-100 text-amber-700 border-amber-200 dark:bg-amber-900 dark:text-amber-300",
  4: "bg-rose-100 text-rose-700 border-rose-200 dark:bg-rose-900 dark:text-rose-300",
  5: "bg-violet-100 text-violet-700 border-violet-200 dark:bg-violet-900 dark:text-violet-300",
};

export default function KnowledgeBase() {
  const { profile } = useProfile();
  const [selectedSubjectId, setSelectedSubjectId] = useState<string>(profile.subjects[0]?.id ?? "");
  const [selectedModule, setSelectedModule] = useState<string>("all");
  const [activeTab, setActiveTab] = useState<TabType>("all");
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [search, setSearch] = useState("");
  const [expandedChunk, setExpandedChunk] = useState<string | null>(null);

  const selectedSubject = profile.subjects.find(s => s.id === selectedSubjectId);
  const subjectCode = selectedSubject?.code ?? "";

  const chunks = useMemo(() => KB_CHUNKS.filter(c => {
    if (c.subjectCode !== subjectCode) return false;
    if (selectedModule !== "all" && c.module !== parseInt(selectedModule)) return false;
    if (search && !c.text.toLowerCase().includes(search.toLowerCase()) && !c.source.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  }), [subjectCode, selectedModule, search]);

  const images = useMemo(() => KB_IMAGES.filter(img => {
    if (img.subjectCode !== subjectCode) return false;
    if (selectedModule !== "all" && img.module !== parseInt(selectedModule)) return false;
    if (search && !img.caption.toLowerCase().includes(search.toLowerCase()) && !img.source.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  }), [subjectCode, selectedModule, search]);

  const totalChunks = KB_CHUNKS.filter(c => c.subjectCode === subjectCode).length;
  const totalImages = KB_IMAGES.filter(i => i.subjectCode === subjectCode).length;
  const totalTokens = KB_CHUNKS.filter(c => c.subjectCode === subjectCode).reduce((s, c) => s + c.tokens, 0);

  const handleCopyChunk = (text: string) => {
    navigator.clipboard?.writeText(text).catch(() => {});
    toast.success("Text chunk copied to clipboard");
  };

  const tabs: { id: TabType; label: string; icon: typeof FileText; count: number }[] = [
    { id: "all", label: "All Content", icon: Layers, count: chunks.length + images.length },
    { id: "chunks", label: "Text Chunks", icon: FileText, count: chunks.length },
    { id: "images", label: "Images & Figures", icon: ImageIcon, count: images.length },
  ];

  const showChunks = activeTab === "all" || activeTab === "chunks";
  const showImages = activeTab === "all" || activeTab === "images";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-serif tracking-tight">Knowledge Base</h1>
          <p className="text-muted-foreground mt-1">Browse all text segments and images extracted from uploaded notes and textbooks.</p>
        </div>
      </div>

      {/* Subject + Module Filter Bar */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <BookOpen className="h-4 w-4 text-muted-foreground" />
          <Select value={selectedSubjectId} onValueChange={setSelectedSubjectId}>
            <SelectTrigger className="w-64 h-9">
              <SelectValue placeholder="Select subject" />
            </SelectTrigger>
            <SelectContent>
              {profile.subjects.map(s => (
                <SelectItem key={s.id} value={s.id}>
                  <span className="font-medium">{s.name}</span>
                  <span className="text-muted-foreground ml-1 text-xs">({s.code})</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-muted-foreground" />
          <Select value={selectedModule} onValueChange={setSelectedModule}>
            <SelectTrigger className="w-40 h-9">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Modules</SelectItem>
              {[1, 2, 3, 4, 5].map(m => (
                <SelectItem key={m} value={String(m)}>Module {m}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="relative flex-1 min-w-48">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            className="pl-8 h-9"
            placeholder="Search content..."
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>

        <div className="flex items-center border rounded-md overflow-hidden">
          <button
            onClick={() => setViewMode("grid")}
            className={`p-2 transition-colors ${viewMode === "grid" ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}
            title="Grid view"
          >
            <LayoutGrid className="h-4 w-4" />
          </button>
          <button
            onClick={() => setViewMode("list")}
            className={`p-2 transition-colors ${viewMode === "list" ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}
            title="List view"
          >
            <List className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Stats Row */}
      {selectedSubject && (
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: "Text Chunks", value: totalChunks, icon: FileText, color: "text-blue-600" },
            { label: "Images & Figures", value: totalImages, icon: ImageIcon, color: "text-violet-600" },
            { label: "Total Tokens Indexed", value: totalTokens.toLocaleString(), icon: Hash, color: "text-emerald-600" },
          ].map(stat => (
            <div key={stat.label} className="bg-card border rounded-lg px-4 py-3 flex items-center gap-3">
              <div className="w-9 h-9 rounded-md bg-muted flex items-center justify-center">
                <stat.icon className={`h-5 w-5 ${stat.color}`} />
              </div>
              <div>
                <p className="text-2xl font-bold text-foreground">{stat.value}</p>
                <p className="text-xs text-muted-foreground">{stat.label}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="flex items-center gap-1 border-b">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors -mb-px ${
              activeTab === tab.id
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            <tab.icon className="h-4 w-4" />
            {tab.label}
            <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-semibold ${
              activeTab === tab.id ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"
            }`}>
              {tab.count}
            </span>
          </button>
        ))}
      </div>

      {/* No subject selected */}
      {!selectedSubject && (
        <div className="text-center py-20 text-muted-foreground">
          <BookOpen className="h-12 w-12 mx-auto mb-3 opacity-30" />
          <p className="font-medium">Select a subject to browse its knowledge base</p>
          <p className="text-sm mt-1">Add subjects from your Home dashboard if none appear here.</p>
        </div>
      )}

      {/* Content */}
      {selectedSubject && (
        <div className="space-y-8">

          {/* ── Text Chunks ── */}
          {showChunks && (
            <section>
              {activeTab === "all" && (
                <div className="flex items-center gap-2 mb-4">
                  <FileText className="h-4 w-4 text-primary" />
                  <h2 className="font-semibold text-sm">Text Chunks <span className="text-muted-foreground font-normal">({chunks.length})</span></h2>
                </div>
              )}
              {chunks.length === 0 ? (
                <div className="text-center py-10 text-muted-foreground text-sm border border-dashed rounded-lg">
                  No text chunks found for the selected filters.
                </div>
              ) : viewMode === "grid" ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {chunks.map(chunk => (
                    <motion.div
                      key={chunk.id}
                      layout
                      className="border rounded-lg bg-card overflow-hidden hover:border-primary/40 transition-colors"
                    >
                      <div
                        className="flex items-center justify-between px-4 py-2.5 bg-muted/30 cursor-pointer"
                        onClick={() => setExpandedChunk(expandedChunk === chunk.id ? null : chunk.id)}
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${MODULE_COLORS[chunk.module]}`}>
                            M{chunk.module}
                          </span>
                          <span className="text-xs text-muted-foreground truncate">{chunk.source} · p.{chunk.page}</span>
                          <Badge variant="outline" className="text-[10px] shrink-0">{chunk.sourceType}</Badge>
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                          <span className="text-[10px] text-muted-foreground">{chunk.tokens} tokens</span>
                          {expandedChunk === chunk.id ? <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" /> : <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />}
                        </div>
                      </div>
                      <div className="px-4 py-3">
                        <p className={`text-sm text-foreground leading-relaxed ${expandedChunk === chunk.id ? "" : "line-clamp-3"}`}>
                          {chunk.text}
                        </p>
                        {expandedChunk === chunk.id && (
                          <div className="flex justify-end mt-2">
                            <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => handleCopyChunk(chunk.text)}>
                              <Copy className="h-3 w-3 mr-1" /> Copy
                            </Button>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="space-y-2">
                  {chunks.map(chunk => (
                    <div key={chunk.id} className="border rounded-lg bg-card px-4 py-3 flex items-start gap-4 hover:border-primary/40 transition-colors">
                      <div className="flex flex-col items-center gap-1 shrink-0 pt-0.5">
                        <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${MODULE_COLORS[chunk.module]}`}>M{chunk.module}</span>
                        <span className="text-[10px] text-muted-foreground">{chunk.tokens}t</span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs text-muted-foreground mb-1">{chunk.source} · page {chunk.page} · <Badge variant="outline" className="text-[10px]">{chunk.sourceType}</Badge></p>
                        <p className="text-sm text-foreground leading-relaxed line-clamp-2">{chunk.text}</p>
                      </div>
                      <Button variant="ghost" size="icon" className="h-7 w-7 shrink-0" onClick={() => handleCopyChunk(chunk.text)}>
                        <Copy className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}

          {activeTab === "all" && showChunks && showImages && chunks.length > 0 && images.length > 0 && (
            <Separator />
          )}

          {/* ── Images & Figures ── */}
          {showImages && (
            <section>
              {activeTab === "all" && (
                <div className="flex items-center gap-2 mb-4">
                  <ImageIcon className="h-4 w-4 text-primary" />
                  <h2 className="font-semibold text-sm">Images & Figures <span className="text-muted-foreground font-normal">({images.length})</span></h2>
                </div>
              )}
              {images.length === 0 ? (
                <div className="text-center py-10 text-muted-foreground text-sm border border-dashed rounded-lg">
                  No images found for the selected filters.
                </div>
              ) : (
                <div className={viewMode === "grid" ? "grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4" : "space-y-2"}>
                  {images.map(img => {
                    const typeStyle = IMAGE_TYPE_COLORS[img.type as ImageType];
                    return viewMode === "grid" ? (
                      <motion.div key={img.id} layout className="border rounded-lg overflow-hidden bg-card hover:border-primary/50 hover:shadow-sm transition-all group">
                        {/* Placeholder visual */}
                        <div className={`${img.color} aspect-[4/3] flex items-center justify-center relative`}>
                          <div className="text-center space-y-2 opacity-60">
                            <ImageIcon className="h-8 w-8 mx-auto" />
                            <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${typeStyle}`}>
                              {img.type}
                            </span>
                          </div>
                          <div className="absolute top-2 left-2">
                            <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border bg-white/80 dark:bg-black/50 ${MODULE_COLORS[img.module]}`}>
                              M{img.module}
                            </span>
                          </div>
                        </div>
                        <div className="p-3 space-y-1.5">
                          <p className="text-xs font-medium text-foreground leading-snug line-clamp-2">{img.caption}</p>
                          <p className="text-[10px] text-muted-foreground truncate">{img.source} · p.{img.page}</p>
                        </div>
                      </motion.div>
                    ) : (
                      <div key={img.id} className="border rounded-lg bg-card flex items-center gap-4 px-4 py-3 hover:border-primary/40 transition-colors">
                        <div className={`${img.color} w-12 h-12 rounded-md flex items-center justify-center shrink-0`}>
                          <ImageIcon className="h-6 w-6 opacity-50" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-foreground truncate">{img.caption}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border ${MODULE_COLORS[img.module]}`}>M{img.module}</span>
                            <span className={`text-[10px] px-1.5 py-0.5 rounded border ${typeStyle}`}>{img.type}</span>
                            <span className="text-[10px] text-muted-foreground">{img.source} · p.{img.page}</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          )}
        </div>
      )}
    </div>
  );
}
