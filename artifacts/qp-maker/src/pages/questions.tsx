import { useState } from "react";
import { Plus, Search, Filter, Edit2, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { questionBank, subjects } from "@/lib/mock-data";
import { toast } from "sonner";

export default function QuestionBank() {
  const [search, setSearch] = useState("");
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

  const getDifficultyColor = (diff: string) => {
    switch (diff.toLowerCase()) {
      case 'easy': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300';
      case 'medium': return 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300';
      case 'hard': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getBloomColor = (level: string) => {
    return 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900 dark:text-blue-300 dark:border-blue-800';
  };

  const handleDelete = (id: string) => {
    toast.success("Question deleted successfully");
  };

  const filteredQuestions = questionBank.filter((q: any) => 
    q.text.toLowerCase().includes(search.toLowerCase()) ||
    q.subject.toLowerCase().includes(search.toLowerCase()) ||
    q.co.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground font-serif tracking-tight">Question Bank</h1>
          <p className="text-muted-foreground">Manage and organize your repository of questions.</p>
        </div>
        
        <Dialog open={isAddModalOpen} onOpenChange={setIsAddModalOpen}>
          <DialogTrigger asChild>
            <Button className="hover-elevate">
              <Plus className="mr-2 h-4 w-4" /> Add Question
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[600px]">
            <DialogHeader>
              <DialogTitle>Add New Question</DialogTitle>
              <DialogDescription>Add a new question to the central repository.</DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="question">Question Text</Label>
                <Textarea id="question" placeholder="Enter the full question text..." className="min-h-[100px]" />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Subject</Label>
                  <Select>
                    <SelectTrigger><SelectValue placeholder="Select subject" /></SelectTrigger>
                    <SelectContent>
                      {subjects["Artificial Intelligence & Machine Learning"].map(s => (
                        <SelectItem key={s.code} value={s.code}>{s.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Module Number</Label>
                  <Select>
                    <SelectTrigger><SelectValue placeholder="Select module" /></SelectTrigger>
                    <SelectContent>
                      {[1,2,3,4,5].map(m => <SelectItem key={m} value={`${m}`}>Module {m}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid grid-cols-4 gap-4">
                <div className="space-y-2">
                  <Label>CO Tag</Label>
                  <Select>
                    <SelectTrigger><SelectValue placeholder="CO" /></SelectTrigger>
                    <SelectContent>
                      {["CO1","CO2","CO3","CO4","CO5"].map(co => <SelectItem key={co} value={co}>{co}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Bloom's</Label>
                  <Select>
                    <SelectTrigger><SelectValue placeholder="L" /></SelectTrigger>
                    <SelectContent>
                      {["L1","L2","L3","L4","L5","L6"].map(l => <SelectItem key={l} value={l}>{l}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Difficulty</Label>
                  <Select>
                    <SelectTrigger><SelectValue placeholder="Diff" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Easy">Easy</SelectItem>
                      <SelectItem value="Medium">Medium</SelectItem>
                      <SelectItem value="Hard">Hard</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Marks</Label>
                  <Input type="number" defaultValue={5} />
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsAddModalOpen(false)}>Cancel</Button>
              <Button onClick={() => {
                toast.success("Question added successfully");
                setIsAddModalOpen(false);
              }}>Save Question</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <div className="flex flex-col sm:flex-row gap-4 items-center bg-card p-4 rounded-lg border shadow-sm">
        <div className="relative flex-1 w-full">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input 
            placeholder="Search questions, subjects, or tags..." 
            className="pl-9 w-full bg-background"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div className="flex gap-2 w-full sm:w-auto">
          <Select>
            <SelectTrigger className="w-[130px] bg-background">
              <SelectValue placeholder="Bloom Level" />
            </SelectTrigger>
            <SelectContent>
              {["All", "L1", "L2", "L3", "L4", "L5", "L6"].map(l => <SelectItem key={l} value={l}>{l}</SelectItem>)}
            </SelectContent>
          </Select>
          <Select>
            <SelectTrigger className="w-[130px] bg-background">
              <SelectValue placeholder="Difficulty" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="All">All</SelectItem>
              <SelectItem value="Easy">Easy</SelectItem>
              <SelectItem value="Medium">Medium</SelectItem>
              <SelectItem value="Hard">Hard</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="bg-card border rounded-lg shadow-sm overflow-hidden">
        <Table>
          <TableHeader className="bg-muted/50">
            <TableRow>
              <TableHead className="w-16">No.</TableHead>
              <TableHead className="w-1/2">Question Text</TableHead>
              <TableHead>Subject</TableHead>
              <TableHead className="text-center">Tags</TableHead>
              <TableHead className="text-center">Marks</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredQuestions.length > 0 ? (
              filteredQuestions.map((q: any, idx: number) => (
                <TableRow key={q.id} className="hover:bg-muted/20">
                  <TableCell className="font-medium text-muted-foreground">{idx + 1}</TableCell>
                  <TableCell>
                    <p className="line-clamp-2 text-sm leading-relaxed">{q.text}</p>
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">{q.subject}</TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1 justify-center">
                      <Badge variant="outline" className="text-xs">{q.co}</Badge>
                      <Badge variant="outline" className={`text-xs ${getBloomColor(q.bloomLevel)}`}>{q.bloomLevel}</Badge>
                      <Badge variant="secondary" className={`text-xs ${getDifficultyColor(q.difficulty)} border-transparent`}>{q.difficulty}</Badge>
                    </div>
                  </TableCell>
                  <TableCell className="text-center font-medium">{q.marks}</TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-2">
                      <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-primary">
                        <Edit2 className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-destructive" onClick={() => handleDelete(q.id)}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={6} className="h-32 text-center text-muted-foreground">
                  No questions found matching your search criteria.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
