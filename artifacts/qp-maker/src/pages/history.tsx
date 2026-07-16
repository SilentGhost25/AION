import { useState } from "react";
import { Eye, Download, Trash2, Calendar, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { historyData } from "@/lib/mock-data";
import { toast } from "sonner";
import { PaperPreview } from "@/components/paper-preview";

const MOCK_PREVIEW_QUESTIONS = [
  { id: "q1", text: "Define Machine Learning. Explain the different types with examples.", marks: 10, co: "CO1", bloomLevel: "L1", module: 1 },
  { id: "q2", text: "What is Bias-Variance Tradeoff? Explain with diagrams.", marks: 10, co: "CO1", bloomLevel: "L2", module: 1 },
  { id: "q3", text: "Apply Gradient Descent to optimize Linear Regression parameters.", marks: 10, co: "CO2", bloomLevel: "L3", module: 2 },
  { id: "q4", text: "Using SVM, apply the kernel trick to classify a non-linearly separable dataset.", marks: 10, co: "CO2", bloomLevel: "L3", module: 2 },
  { id: "q5", text: "Analyze K-Means clustering. Explain the Elbow method for choosing K.", marks: 10, co: "CO3", bloomLevel: "L4", module: 3 },
  { id: "q6", text: "Analyze PCA and demonstrate dimensionality reduction while preserving variance.", marks: 10, co: "CO3", bloomLevel: "L4", module: 3 },
  { id: "q7", text: "Analyze the Backpropagation algorithm and demonstrate weight updates for a 3-layer network.", marks: 10, co: "CO3", bloomLevel: "L4", module: 4 },
  { id: "q8", text: "Compare and analyze ReLU, Sigmoid, and Softmax activation functions.", marks: 10, co: "CO3", bloomLevel: "L4", module: 4 },
  { id: "q9", text: "Apply cross-validation techniques to evaluate and compare ML models.", marks: 10, co: "CO2", bloomLevel: "L3", module: 5 },
  { id: "q10", text: "Apply AdaBoost and compare results with a single Decision Tree classifier.", marks: 10, co: "CO2", bloomLevel: "L3", module: 5 },
];

export default function History() {
  const [selectedPaper, setSelectedPaper] = useState<any>(null);

  const handleDelete = (id: string) => {
    toast.success(`Paper ${id} deleted permanently`);
  };

  const handleDownload = (id: string) => {
    toast.success(`Downloading paper ${id}.docx...`);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground font-serif tracking-tight">Generated Papers</h1>
        <p className="text-muted-foreground">View and manage previously generated question papers.</p>
      </div>

      <div className="bg-card border rounded-lg shadow-sm overflow-hidden">
        <Table>
          <TableHeader className="bg-muted/50">
            <TableRow>
              <TableHead>Paper ID</TableHead>
              <TableHead>Subject</TableHead>
              <TableHead>Exam Type</TableHead>
              <TableHead>Semester</TableHead>
              <TableHead>Generated On</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {historyData.map((paper) => (
              <TableRow key={paper.id}>
                <TableCell className="font-medium font-mono text-xs">{paper.id}</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <FileText className="h-4 w-4 text-primary opacity-70" />
                    {paper.subject}
                  </div>
                </TableCell>
                <TableCell>{paper.examType}</TableCell>
                <TableCell>{paper.semester}</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2 text-muted-foreground text-sm">
                    <Calendar className="h-3 w-3" />
                    {paper.generatedOn}
                  </div>
                </TableCell>
                <TableCell>
                  <Badge variant={paper.status === 'Downloaded' ? 'default' : 'secondary'} className="font-normal">
                    {paper.status}
                  </Badge>
                </TableCell>
                <TableCell className="text-right">
                  <div className="flex justify-end gap-2">
                    <Button variant="ghost" size="icon" className="h-8 w-8 hover:text-primary" onClick={() => setSelectedPaper(paper)}>
                      <Eye className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-8 w-8 hover:text-primary" onClick={() => handleDownload(paper.id)}>
                      <Download className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-destructive" onClick={() => handleDelete(paper.id)}>
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <Dialog open={!!selectedPaper} onOpenChange={(open) => !open && setSelectedPaper(null)}>
        <DialogContent className="max-w-4xl max-h-[90vh] flex flex-col p-0">
          <DialogHeader className="p-6 border-b pb-4">
            <div className="flex items-center justify-between pr-8">
              <div>
                <DialogTitle className="text-xl">{selectedPaper?.id}</DialogTitle>
                <p className="text-sm text-muted-foreground mt-1">Generated on {selectedPaper?.generatedOn}</p>
              </div>
              <Button size="sm" onClick={() => selectedPaper && handleDownload(selectedPaper.id)}>
                <Download className="mr-2 h-4 w-4" /> Download .docx
              </Button>
            </div>
          </DialogHeader>
          <div className="flex-1 overflow-y-auto p-6 bg-muted/30">
            <div className="bg-white p-8 shadow-sm border max-w-3xl mx-auto">
              <PaperPreview 
                formData={{
                  subjectName: selectedPaper?.subject,
                  examType: selectedPaper?.examType,
                  semester: selectedPaper?.semester
                }} 
                questions={MOCK_PREVIEW_QUESTIONS} 
              />
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
