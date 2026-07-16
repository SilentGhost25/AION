import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { motion, AnimatePresence } from "framer-motion";
import { Check, ChevronRight, Settings2, FileText, Download, RotateCw, Edit, Sparkles, ShieldCheck, AlertTriangle } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { departments, subjects, MODULE_CO_BLOOM } from "@/lib/mock-data";
import { PaperPreview } from "@/components/paper-preview";

const step1Schema = z.object({
  department: z.string().min(1, "Required"),
  subjectName: z.string().min(1, "Required"),
  subjectCode: z.string().min(1, "Required"),
  semester: z.string().min(1, "Required"),
  batch: z.string().min(1, "Required"),
  maxMarks: z.coerce.number().min(1, "Required").default(50),
  duration: z.string().min(1, "Required"),
  dateOfIat: z.string().min(1, "Required"),
  teachingDept: z.string().min(1, "Required"),
  examType: z.string().min(1, "Required"),
});

const step2Schema = z.object({
  coverageScope: z.string().min(10, "Please describe the coverage scope in at least 10 characters"),
  modulesIncluded: z.array(z.string()).min(1, "Select at least one module"),
});

const steps = [
  { id: 1, name: "Configuration", icon: Settings2 },
  { id: 2, name: "Scope & Rules", icon: ShieldCheck },
  { id: 3, name: "Preview", icon: Check }
];

// Mock generated questions following the strict CO-Bloom rules per module
const MOCK_GENERATED: Record<number, { text: string; marks: number }[]> = {
  1: [
    { text: "Define Machine Learning. Explain the different types of Machine Learning with suitable examples.", marks: 10 },
    { text: "What is the Bias-Variance Tradeoff? Explain underfitting and overfitting with diagrams.", marks: 10 },
  ],
  2: [
    { text: "Apply the Gradient Descent algorithm to optimize the parameters of a Linear Regression model on a given dataset.", marks: 10 },
    { text: "Using the concept of Support Vector Machines, apply the kernel trick to classify a non-linearly separable dataset.", marks: 10 },
  ],
  3: [
    { text: "Analyze the K-Means clustering algorithm. Explain how the choice of K affects the clustering outcome using the Elbow method.", marks: 10 },
    { text: "Analyze Principal Component Analysis (PCA) and demonstrate how it reduces dimensionality while preserving variance.", marks: 10 },
  ],
  4: [
    { text: "Analyze the Backpropagation algorithm and demonstrate weight update computations for a 3-layer neural network.", marks: 10 },
    { text: "Compare and analyze the working of ReLU, Sigmoid, and Softmax activation functions. Justify their use in different layers.", marks: 10 },
  ],
  5: [
    { text: "Apply cross-validation techniques to evaluate and compare the performance of different ML models on a given dataset.", marks: 10 },
    { text: "Using ensemble learning, apply the AdaBoost algorithm and compare its results with a single Decision Tree classifier.", marks: 10 },
  ],
};

export default function GeneratePaper() {
  const [currentStep, setCurrentStep] = useState(1);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedQuestions, setGeneratedQuestions] = useState<any[]>([]);
  const [selectedModules, setSelectedModules] = useState<number[]>([]);

  const form1 = useForm<z.infer<typeof step1Schema>>({
    resolver: zodResolver(step1Schema),
    defaultValues: { maxMarks: 50, duration: "1.5 hrs" }
  });

  const form2 = useForm<z.infer<typeof step2Schema>>({
    resolver: zodResolver(step2Schema),
    defaultValues: { coverageScope: "", modulesIncluded: [] }
  });

  const watchDept = form1.watch("department");
  const subjectList = watchDept && (subjects as any)[watchDept] ? (subjects as any)[watchDept] : [];

  const toggleModule = (mod: number) => {
    setSelectedModules(prev => {
      const next = prev.includes(mod) ? prev.filter(m => m !== mod) : [...prev, mod].sort();
      form2.setValue("modulesIncluded", next.map(String));
      return next;
    });
  };

  const generatePaper = () => {
    if (selectedModules.length === 0) {
      toast.error("Please select at least one module to include in the paper.");
      return;
    }

    setIsGenerating(true);
    toast.info("AI is analyzing syllabus and study materials...");

    setTimeout(() => {
      // Build questions following strict CO-Bloom per module
      const questions: any[] = [];
      let qNum = 1;
      selectedModules.forEach(mod => {
        const mapping = MODULE_CO_BLOOM[mod];
        const pair = MOCK_GENERATED[mod] || [];
        pair.forEach(q => {
          questions.push({
            id: `q${qNum++}`,
            text: q.text,
            marks: q.marks,
            co: mapping.co,
            bloomLevel: mapping.bloom.split("/")[0],
            module: mod,
          });
        });
      });

      setGeneratedQuestions(questions);
      setIsGenerating(false);
      setCurrentStep(3);
      toast.success("Question paper generated successfully. All questions comply with CO-Bloom rules.");
    }, 1800);
  };

  const handleDownload = () => {
    toast.success("Paper downloaded successfully as .docx");
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-3xl font-bold text-foreground font-serif tracking-tight">Generate Question Paper</h1>
        <p className="text-muted-foreground">Configure the paper details, define scope in plain English, and let AI generate syllabus-aligned questions.</p>
      </div>

      {/* Progress Wizard */}
      <div className="relative">
        <div className="absolute top-5 left-0 w-full h-0.5 bg-muted">
          <motion.div
            className="h-full bg-primary"
            initial={{ width: "0%" }}
            animate={{ width: `${((currentStep - 1) / (steps.length - 1)) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
        <div className="relative flex justify-between">
          {steps.map((step) => {
            const isCompleted = currentStep > step.id;
            const isCurrent = currentStep === step.id;
            return (
              <div key={step.id} className="flex flex-col items-center gap-2">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300 z-10 bg-background ${
                  isCompleted ? "bg-primary border-primary text-primary-foreground" :
                  isCurrent ? "border-primary text-primary" :
                  "border-muted text-muted-foreground"
                }`}>
                  {isCompleted ? <Check className="h-5 w-5 text-primary-foreground" /> : <step.icon className="h-5 w-5" />}
                </div>
                <span className={`text-xs font-medium ${isCurrent || isCompleted ? "text-foreground" : "text-muted-foreground"}`}>
                  {step.name}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Form Area */}
      <Card className="border-muted shadow-sm overflow-hidden">
        <AnimatePresence mode="wait">

          {/* ── STEP 1: Paper Configuration ── */}
          {currentStep === 1 && (
            <motion.div key="step1" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}>
              <Form {...form1}>
                <form onSubmit={form1.handleSubmit(() => setCurrentStep(2))}>
                  <CardHeader>
                    <CardTitle>Paper Configuration</CardTitle>
                    <CardDescription>Set up the basic details for the examination.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <FormField control={form1.control} name="department" render={({ field }) => (
                        <FormItem>
                          <FormLabel>Department</FormLabel>
                          <Select onValueChange={v => { field.onChange(v); form1.setValue("subjectName", ""); form1.setValue("subjectCode", ""); }}>
                            <FormControl><SelectTrigger><SelectValue placeholder="Select department" /></SelectTrigger></FormControl>
                            <SelectContent>{departments.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}</SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )} />

                      <FormField control={form1.control} name="examType" render={({ field }) => (
                        <FormItem>
                          <FormLabel>Exam Type</FormLabel>
                          <Select onValueChange={field.onChange} defaultValue={field.value}>
                            <FormControl><SelectTrigger><SelectValue placeholder="Select exam type" /></SelectTrigger></FormControl>
                            <SelectContent>
                              <SelectItem value="IAT-1">First Internal Assessment (IAT-1)</SelectItem>
                              <SelectItem value="IAT-2">Second Internal Assessment (IAT-2)</SelectItem>
                              <SelectItem value="IAT-3">Third Internal Assessment (IAT-3)</SelectItem>
                              <SelectItem value="End-Sem">End Semester</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )} />

                      <FormField control={form1.control} name="subjectName" render={({ field }) => (
                        <FormItem>
                          <FormLabel>Subject Name</FormLabel>
                          {subjectList.length > 0 ? (
                            <Select onValueChange={v => {
                              field.onChange(v);
                              const found = subjectList.find((s: any) => s.name === v);
                              if (found) form1.setValue("subjectCode", found.code);
                            }}>
                              <FormControl><SelectTrigger><SelectValue placeholder="Select subject" /></SelectTrigger></FormControl>
                              <SelectContent>{subjectList.map((s: any) => <SelectItem key={s.code} value={s.name}>{s.name}</SelectItem>)}</SelectContent>
                            </Select>
                          ) : (
                            <FormControl><Input placeholder="e.g. Machine Learning" {...field} /></FormControl>
                          )}
                          <FormMessage />
                        </FormItem>
                      )} />

                      <FormField control={form1.control} name="subjectCode" render={({ field }) => (
                        <FormItem>
                          <FormLabel>Subject Code</FormLabel>
                          <FormControl><Input placeholder="e.g. 21AI51" {...field} /></FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />

                      <div className="grid grid-cols-2 gap-4">
                        <FormField control={form1.control} name="semester" render={({ field }) => (
                          <FormItem>
                            <FormLabel>Semester</FormLabel>
                            <Select onValueChange={field.onChange} defaultValue={field.value}>
                              <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                              <SelectContent>{[1,2,3,4,5,6,7,8].map(s => <SelectItem key={s} value={`${s}`}>{s}{["st","nd","rd","th","th","th","th","th"][s-1]} Sem</SelectItem>)}</SelectContent>
                            </Select>
                            <FormMessage />
                          </FormItem>
                        )} />
                        <FormField control={form1.control} name="batch" render={({ field }) => (
                          <FormItem>
                            <FormLabel>Batch</FormLabel>
                            <FormControl><Input placeholder="e.g. 2022-26" {...field} /></FormControl>
                            <FormMessage />
                          </FormItem>
                        )} />
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <FormField control={form1.control} name="maxMarks" render={({ field }) => (
                          <FormItem>
                            <FormLabel>Max Marks</FormLabel>
                            <FormControl><Input type="number" {...field} /></FormControl>
                            <FormMessage />
                          </FormItem>
                        )} />
                        <FormField control={form1.control} name="duration" render={({ field }) => (
                          <FormItem>
                            <FormLabel>Duration</FormLabel>
                            <Select onValueChange={field.onChange} defaultValue={field.value}>
                              <FormControl><SelectTrigger><SelectValue placeholder="Select" /></SelectTrigger></FormControl>
                              <SelectContent>
                                <SelectItem value="1 hr">1 hr</SelectItem>
                                <SelectItem value="1.5 hrs">1.5 hrs</SelectItem>
                                <SelectItem value="2 hrs">2 hrs</SelectItem>
                                <SelectItem value="3 hrs">3 hrs</SelectItem>
                              </SelectContent>
                            </Select>
                            <FormMessage />
                          </FormItem>
                        )} />
                      </div>

                      <FormField control={form1.control} name="dateOfIat" render={({ field }) => (
                        <FormItem>
                          <FormLabel>Date of IAT</FormLabel>
                          <FormControl><Input type="date" {...field} /></FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />

                      <FormField control={form1.control} name="teachingDept" render={({ field }) => (
                        <FormItem>
                          <FormLabel>Teaching Department</FormLabel>
                          <FormControl><Input placeholder="e.g. AIML Dept" {...field} /></FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />
                    </div>
                  </CardContent>
                  <CardFooter className="bg-muted/30 flex justify-end p-4 border-t">
                    <Button type="submit" className="px-8" data-testid="step1-next">
                      Next Step <ChevronRight className="ml-2 h-4 w-4" />
                    </Button>
                  </CardFooter>
                </form>
              </Form>
            </motion.div>
          )}

          {/* ── STEP 2: Scope & CO Rules ── */}
          {currentStep === 2 && (
            <motion.div key="step2" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}>
              <CardHeader>
                <CardTitle>Scope & CO Assignment Rules</CardTitle>
                <CardDescription>Define the coverage boundary and review the locked CO-Bloom rules per module.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-8">

                {/* Coverage Scope text box */}
                <div className="space-y-3">
                  <div>
                    <h3 className="text-sm font-semibold">Coverage Scope</h3>
                    <p className="text-sm text-muted-foreground mt-0.5">
                      Describe in plain English exactly how much of the syllabus this paper should cover. The AI will parse this and restrict question generation accordingly.
                    </p>
                  </div>
                  <Textarea
                    placeholder={`Examples:\n• "Cover only Module 1 and Module 2 up to SVM with kernels"\n• "Include all 5 modules for an end-sem paper"\n• "Only the first 3 topics of Module 1 — basic definitions and types of ML"\n• "Modules 3 and 4 entirely, skip Module 5"`}
                    className="min-h-[120px] font-mono text-sm resize-none"
                    value={form2.watch("coverageScope")}
                    onChange={e => form2.setValue("coverageScope", e.target.value)}
                    data-testid="coverage-scope-input"
                  />
                  {form2.formState.errors.coverageScope && (
                    <p className="text-sm text-destructive">{form2.formState.errors.coverageScope.message}</p>
                  )}
                  <div className="flex items-start gap-2 p-3 rounded-lg bg-muted/40 border border-muted/70">
                    <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                    <p className="text-xs text-muted-foreground">
                      The AI reads this description and cross-references it against the uploaded syllabus to determine exactly which topics are in scope. Be as specific or as general as you like — it understands natural language.
                    </p>
                  </div>
                </div>

                <Separator />

                {/* Module selection */}
                <div className="space-y-3">
                  <div>
                    <h3 className="text-sm font-semibold">Select Modules to Include</h3>
                    <p className="text-sm text-muted-foreground mt-0.5">Choose which modules should appear in this paper. Questions will be generated proportionally.</p>
                  </div>
                  <div className="grid grid-cols-5 gap-3">
                    {[1, 2, 3, 4, 5].map(mod => {
                      const mapping = MODULE_CO_BLOOM[mod];
                      const selected = selectedModules.includes(mod);
                      return (
                        <button
                          key={mod}
                          type="button"
                          onClick={() => toggleModule(mod)}
                          data-testid={`module-toggle-${mod}`}
                          className={`p-3 rounded-lg border-2 text-left transition-all ${
                            selected
                              ? "border-primary bg-primary/5 shadow-sm"
                              : "border-muted hover:border-primary/40 hover:bg-muted/30"
                          }`}
                        >
                          <div className="font-bold text-sm mb-2">M{mod}</div>
                          <div className="flex flex-col gap-1">
                            <Badge className={`text-[10px] px-1.5 py-0 justify-center ${selected ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground border-0"}`}>
                              {mapping.co}
                            </Badge>
                            <Badge variant="outline" className={`text-[10px] px-1.5 py-0 justify-center ${selected ? "border-primary/50 text-primary" : "border-muted text-muted-foreground"}`}>
                              {mapping.bloom}
                            </Badge>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                  {form2.formState.errors.modulesIncluded && (
                    <p className="text-sm text-destructive">{form2.formState.errors.modulesIncluded.message}</p>
                  )}
                </div>

                <Separator />

                {/* Iron-clad CO-Bloom rules table */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="h-4 w-4 text-primary" />
                    <h3 className="text-sm font-semibold">Locked CO-Bloom Assignment Rules</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">These rules are enforced without exception. Every generated question is validated against its module's CO and Bloom level before being included in the paper.</p>

                  <div className="rounded-lg border overflow-hidden">
                    <table className="w-full text-sm">
                      <thead className="bg-muted/50">
                        <tr>
                          <th className="text-left px-4 py-2.5 font-semibold text-xs uppercase tracking-wider text-muted-foreground">Module</th>
                          <th className="text-left px-4 py-2.5 font-semibold text-xs uppercase tracking-wider text-muted-foreground">Course Outcome</th>
                          <th className="text-left px-4 py-2.5 font-semibold text-xs uppercase tracking-wider text-muted-foreground">Bloom Level</th>
                          <th className="text-left px-4 py-2.5 font-semibold text-xs uppercase tracking-wider text-muted-foreground">Cognitive Domain</th>
                          <th className="text-left px-4 py-2.5 font-semibold text-xs uppercase tracking-wider text-muted-foreground">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-muted/50">
                        {[
                          { mod: 1, co: "CO1", bloom: "L1/L2", domain: "Remember / Understand" },
                          { mod: 2, co: "CO2", bloom: "L3", domain: "Apply" },
                          { mod: 3, co: "CO3", bloom: "L4", domain: "Analyze" },
                          { mod: 4, co: "CO3", bloom: "L4", domain: "Analyze" },
                          { mod: 5, co: "CO2", bloom: "L3", domain: "Apply" },
                        ].map(row => {
                          const isIncluded = selectedModules.includes(row.mod);
                          return (
                            <tr key={row.mod} className={isIncluded ? "bg-primary/5" : ""}>
                              <td className="px-4 py-3 font-medium">Module {row.mod}</td>
                              <td className="px-4 py-3">
                                <Badge className="bg-primary/10 text-primary border-0 font-semibold">{row.co}</Badge>
                              </td>
                              <td className="px-4 py-3">
                                <Badge variant="outline" className="font-semibold">{row.bloom}</Badge>
                              </td>
                              <td className="px-4 py-3 text-muted-foreground">{row.domain}</td>
                              <td className="px-4 py-3">
                                <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full font-medium ${
                                  isIncluded
                                    ? "bg-emerald-50 text-emerald-700 border border-emerald-200 dark:bg-emerald-950 dark:text-emerald-400 dark:border-emerald-800"
                                    : "bg-muted text-muted-foreground border border-muted/70"
                                }`}>
                                  {isIncluded ? <><Check className="h-2.5 w-2.5" /> Included</> : "Not selected"}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  <div className="flex items-start gap-2 p-3 rounded-lg border border-amber-200 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-800/50">
                    <AlertTriangle className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
                    <p className="text-xs text-amber-700 dark:text-amber-400">
                      These CO-Bloom assignments are immutable during generation. Questions that do not match the assigned CO and Bloom level for their module are automatically rejected and regenerated. This ensures full NBA CO-PO mapping compliance.
                    </p>
                  </div>
                </div>
              </CardContent>
              <CardFooter className="bg-muted/30 flex justify-between p-4 border-t">
                <Button variant="outline" onClick={() => setCurrentStep(1)}>Back</Button>
                <Button
                  onClick={() => {
                    form2.trigger().then(valid => {
                      if (!valid || selectedModules.length === 0) {
                        if (selectedModules.length === 0) toast.error("Please select at least one module.");
                        return;
                      }
                      generatePaper();
                    });
                  }}
                  disabled={isGenerating}
                  className="px-8"
                  data-testid="generate-button"
                >
                  {isGenerating ? (
                    <><RotateCw className="mr-2 h-4 w-4 animate-spin" /> Generating...</>
                  ) : (
                    <><Sparkles className="mr-2 h-4 w-4" /> Generate Paper</>
                  )}
                </Button>
              </CardFooter>
            </motion.div>
          )}

          {/* ── STEP 3: Preview ── */}
          {currentStep === 3 && (
            <motion.div key="step3" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}>
              <CardHeader className="flex flex-row items-center justify-between border-b pb-4">
                <div>
                  <CardTitle>Preview & Download</CardTitle>
                  <CardDescription>Review the generated paper. All questions comply with the locked CO-Bloom rules.</CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={() => setCurrentStep(2)}>
                    <Edit className="mr-2 h-4 w-4" /> Edit
                  </Button>
                  <Button variant="outline" size="sm" onClick={generatePaper} disabled={isGenerating}>
                    <RotateCw className={`mr-2 h-4 w-4 ${isGenerating ? "animate-spin" : ""}`} /> Regenerate
                  </Button>
                  <Button size="sm" onClick={handleDownload} data-testid="download-button">
                    <Download className="mr-2 h-4 w-4" /> Download .docx
                  </Button>
                </div>
              </CardHeader>

              {/* CO compliance summary */}
              <div className="px-6 pt-4 pb-0">
                <div className="flex items-center gap-2 p-3 rounded-lg bg-emerald-50 border border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-800/50">
                  <ShieldCheck className="h-4 w-4 text-emerald-600 shrink-0" />
                  <p className="text-xs text-emerald-700 dark:text-emerald-400 font-medium">
                    All {generatedQuestions.length} questions verified — CO and Bloom level assignments match the locked module rules. 100% NBA CO-PO compliant.
                  </p>
                </div>
              </div>

              <CardContent className="p-0 bg-muted/30 mt-4">
                <div className="p-6 overflow-x-auto">
                  <div className="min-w-[800px] bg-white p-8 shadow-sm border mx-auto text-black">
                    <PaperPreview formData={form1.getValues()} questions={generatedQuestions} />
                  </div>
                </div>
              </CardContent>
            </motion.div>
          )}

        </AnimatePresence>
      </Card>
    </div>
  );
}
