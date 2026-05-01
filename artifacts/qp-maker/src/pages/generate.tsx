import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { motion, AnimatePresence } from "framer-motion";
import { Check, ChevronRight, Settings2, FileText, Download, RotateCw, Edit } from "lucide-react";
import { toast } from "sonner";
import { format } from "date-fns";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { departments, subjects, questionBank } from "@/lib/mock-data";
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
  rbtLevels: z.array(z.string()).min(1, "Select at least one RBT level")
});

const steps = [
  { id: 1, name: "Configuration", icon: Settings2 },
  { id: 2, name: "Strategy", icon: FileText },
  { id: 3, name: "Preview", icon: Check }
];

export default function GeneratePaper() {
  const [currentStep, setCurrentStep] = useState(1);
  const [isAutoGenerate, setIsAutoGenerate] = useState(true);
  const [generatedQuestions, setGeneratedQuestions] = useState<any[]>([]);

  const form = useForm<z.infer<typeof step1Schema>>({
    resolver: zodResolver(step1Schema),
    defaultValues: {
      maxMarks: 50,
      rbtLevels: ["L1", "L2", "L3"],
      duration: "1.5 hrs"
    }
  });

  const generateQuestions = () => {
    // Mock generation
    const selected = questionBank.slice(0, 10);
    setGeneratedQuestions(selected);
    setCurrentStep(3);
    toast.success("Question paper generated successfully!");
  };

  const handleDownload = () => {
    toast.success("Paper downloaded successfully as .docx");
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-3xl font-bold text-foreground font-serif tracking-tight">Generate Question Paper</h1>
        <p className="text-muted-foreground">Follow the steps to configure and generate an AI-powered question paper.</p>
      </div>

      {/* Progress Wizard */}
      <div className="relative">
        <div className="absolute top-1/2 left-0 w-full h-1 bg-muted -translate-y-1/2 rounded-full overflow-hidden">
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
                <div 
                  className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-colors duration-300 ${
                    isCompleted ? "bg-primary border-primary text-primary-foreground" : 
                    isCurrent ? "bg-background border-primary text-primary" : 
                    "bg-background border-muted text-muted-foreground"
                  }`}
                >
                  <step.icon className="h-5 w-5" />
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
          {currentStep === 1 && (
            <motion.div
              key="step1"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <Form {...form}>
                <form onSubmit={form.handleSubmit(() => setCurrentStep(2))}>
                  <CardHeader>
                    <CardTitle>Paper Configuration</CardTitle>
                    <CardDescription>Set up the basic details for the examination.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <FormField
                        control={form.control}
                        name="department"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Department</FormLabel>
                            <Select onValueChange={field.onChange} defaultValue={field.value}>
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue placeholder="Select department" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                {departments.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                              </SelectContent>
                            </Select>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                      
                      <FormField
                        control={form.control}
                        name="examType"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Exam Type</FormLabel>
                            <Select onValueChange={field.onChange} defaultValue={field.value}>
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue placeholder="Select exam type" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="IAT-1">First Internal Assessment (IAT-1)</SelectItem>
                                <SelectItem value="IAT-2">Second Internal Assessment (IAT-2)</SelectItem>
                                <SelectItem value="IAT-3">Third Internal Assessment (IAT-3)</SelectItem>
                                <SelectItem value="End-Sem">End Semester</SelectItem>
                              </SelectContent>
                            </Select>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name="subjectName"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Subject Name</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g. Machine Learning" {...field} />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name="subjectCode"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Subject Code</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g. 21AI51" {...field} />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <div className="grid grid-cols-2 gap-4">
                        <FormField
                          control={form.control}
                          name="semester"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>Semester</FormLabel>
                              <Select onValueChange={field.onChange} defaultValue={field.value}>
                                <FormControl>
                                  <SelectTrigger>
                                    <SelectValue placeholder="Select" />
                                  </SelectTrigger>
                                </FormControl>
                                <SelectContent>
                                  {[1,2,3,4,5,6,7,8].map(s => <SelectItem key={s} value={`${s}`}>{s}th</SelectItem>)}
                                </SelectContent>
                              </Select>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                        <FormField
                          control={form.control}
                          name="batch"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>Batch</FormLabel>
                              <FormControl>
                                <Input placeholder="e.g. 2022-26" {...field} />
                              </FormControl>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <FormField
                          control={form.control}
                          name="maxMarks"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>Max Marks</FormLabel>
                              <FormControl>
                                <Input type="number" {...field} />
                              </FormControl>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                        <FormField
                          control={form.control}
                          name="duration"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>Duration</FormLabel>
                              <Select onValueChange={field.onChange} defaultValue={field.value}>
                                <FormControl>
                                  <SelectTrigger>
                                    <SelectValue placeholder="Select" />
                                  </SelectTrigger>
                                </FormControl>
                                <SelectContent>
                                  <SelectItem value="1 hr">1 hr</SelectItem>
                                  <SelectItem value="1.5 hrs">1.5 hrs</SelectItem>
                                  <SelectItem value="2 hrs">2 hrs</SelectItem>
                                  <SelectItem value="3 hrs">3 hrs</SelectItem>
                                </SelectContent>
                              </Select>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                      </div>

                      <FormField
                        control={form.control}
                        name="dateOfIat"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Date of IAT</FormLabel>
                            <FormControl>
                              <Input type="date" {...field} />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name="teachingDept"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Teaching Department</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g. AIML Dept" {...field} />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    </div>

                    <Separator />

                    <FormField
                      control={form.control}
                      name="rbtLevels"
                      render={() => (
                        <FormItem>
                          <div className="mb-4">
                            <FormLabel className="text-base">RBT Levels to Include</FormLabel>
                            <FormDescription>Select the Bloom's taxonomy levels appropriate for this test.</FormDescription>
                          </div>
                          <div className="flex flex-wrap gap-4">
                            {["L1", "L2", "L3", "L4", "L5", "L6"].map((level) => (
                              <FormField
                                key={level}
                                control={form.control}
                                name="rbtLevels"
                                render={({ field }) => {
                                  return (
                                    <FormItem
                                      key={level}
                                      className="flex flex-row items-start space-x-3 space-y-0"
                                    >
                                      <FormControl>
                                        <Checkbox
                                          checked={field.value?.includes(level)}
                                          onCheckedChange={(checked) => {
                                            return checked
                                              ? field.onChange([...field.value, level])
                                              : field.onChange(
                                                  field.value?.filter(
                                                    (value) => value !== level
                                                  )
                                                )
                                          }}
                                        />
                                      </FormControl>
                                      <FormLabel className="font-normal">
                                        {level}
                                      </FormLabel>
                                    </FormItem>
                                  )
                                }}
                              />
                            ))}
                          </div>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </CardContent>
                  <CardFooter className="bg-muted/30 flex justify-end p-4 border-t">
                    <Button type="submit" className="px-8">
                      Next Step <ChevronRight className="ml-2 h-4 w-4" />
                    </Button>
                  </CardFooter>
                </form>
              </Form>
            </motion.div>
          )}

          {currentStep === 2 && (
            <motion.div
              key="step2"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Question Selection Strategy</CardTitle>
                    <CardDescription>Configure how questions should be selected for the paper.</CardDescription>
                  </div>
                  <div className="flex items-center space-x-2 bg-muted p-2 rounded-lg">
                    <Label htmlFor="auto-mode" className={`text-sm cursor-pointer ${!isAutoGenerate ? 'font-bold' : ''}`}>Manual</Label>
                    <Switch 
                      id="auto-mode" 
                      checked={isAutoGenerate} 
                      onCheckedChange={setIsAutoGenerate}
                    />
                    <Label htmlFor="auto-mode" className={`text-sm cursor-pointer ${isAutoGenerate ? 'font-bold text-primary' : ''}`}>AI Auto-Generate</Label>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-8">
                {isAutoGenerate ? (
                  <div className="space-y-8">
                    <div className="space-y-4">
                      <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Difficulty Distribution</h3>
                      <div className="space-y-6 bg-muted/20 p-6 rounded-lg border border-muted/50">
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <Label>Easy (L1, L2)</Label>
                            <span className="text-sm font-medium">30%</span>
                          </div>
                          <Slider defaultValue={[30]} max={100} step={5} />
                        </div>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <Label>Medium (L3, L4)</Label>
                            <span className="text-sm font-medium">50%</span>
                          </div>
                          <Slider defaultValue={[50]} max={100} step={5} />
                        </div>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <Label>Hard (L5, L6)</Label>
                            <span className="text-sm font-medium">20%</span>
                          </div>
                          <Slider defaultValue={[20]} max={100} step={5} />
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">CO Coverage Targets</h3>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        {["CO1", "CO2", "CO3", "CO4", "CO5"].map((co, i) => (
                          <div key={co} className="space-y-2 p-4 border rounded-lg bg-card">
                            <Label className="text-center block w-full">{co}</Label>
                            <Input type="number" defaultValue={20} className="text-center" />
                            <span className="text-xs text-center block text-muted-foreground">%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12 bg-muted/20 rounded-lg border border-dashed">
                    <FileText className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-4" />
                    <h3 className="text-lg font-medium">Manual Selection Mode</h3>
                    <p className="text-sm text-muted-foreground max-w-md mx-auto mt-2">
                      In a full implementation, this would show a searchable data table of all questions where you can manually select the ones you want.
                    </p>
                  </div>
                )}
              </CardContent>
              <CardFooter className="bg-muted/30 flex justify-between p-4 border-t">
                <Button variant="outline" onClick={() => setCurrentStep(1)}>Back</Button>
                <Button onClick={generateQuestions} className="px-8">
                  Generate Paper <Zap className="ml-2 h-4 w-4" />
                </Button>
              </CardFooter>
            </motion.div>
          )}

          {currentStep === 3 && (
            <motion.div
              key="step3"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <CardHeader className="flex flex-row items-center justify-between border-b pb-4">
                <div>
                  <CardTitle>Preview & Download</CardTitle>
                  <CardDescription>Review the generated paper before downloading.</CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={() => setCurrentStep(2)}>
                    <Edit className="mr-2 h-4 w-4" /> Edit
                  </Button>
                  <Button variant="outline" size="sm" onClick={generateQuestions}>
                    <RotateCw className="mr-2 h-4 w-4" /> Regenerate
                  </Button>
                  <Button size="sm" onClick={handleDownload}>
                    <Download className="mr-2 h-4 w-4" /> Download .docx
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="p-0 bg-muted/30">
                <div className="p-6 overflow-x-auto">
                  <div className="min-w-[800px] bg-white p-8 shadow-sm border mx-auto text-black">
                    <PaperPreview formData={form.getValues()} questions={generatedQuestions} />
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

function Zap(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}
