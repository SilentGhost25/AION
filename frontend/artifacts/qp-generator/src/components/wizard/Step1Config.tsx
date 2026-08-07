import { PaperConfig } from "@workspace/api-client-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { DatePicker } from "@/components/ui/date-picker"
import { ArrowRight, BookOpen, Clock, Hash, Calendar, GraduationCap } from "lucide-react"
import { format } from "date-fns"

export function Step1Config({
  config,
  setConfig,
  onNext
}: {
  config: PaperConfig
  setConfig: (c: PaperConfig) => void
  onNext: () => void
}) {
  const isValid = config.institutionDepartment && config.subjectName && config.subjectCode && config.batch && config.maxMarks > 0;

  return (
    <div className="max-w-3xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
      <Card className="border-t-4 border-t-primary shadow-lg">
        <CardHeader className="bg-slate-50/50 border-b">
          <div className="flex items-center gap-2 text-primary mb-2">
            <BookOpen className="h-5 w-5" />
            <span className="font-semibold uppercase tracking-wider text-xs">Step 1 of 3</span>
          </div>
          <CardTitle className="text-2xl text-slate-800">Examination Details</CardTitle>
          <CardDescription>
            Configure the structural metadata for the question paper. This information will appear in the header of the exported PDF.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-8 space-y-8">
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label htmlFor="dept">Institution Department <span className="text-red-500">*</span></Label>
              <Input 
                id="dept"
                placeholder="e.g. Dept. of Computer Science" 
                value={config.institutionDepartment}
                onChange={e => setConfig({ ...config, institutionDepartment: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch">Batch <span className="text-red-500">*</span></Label>
              <div className="relative">
                <GraduationCap className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input 
                  id="batch"
                  className="pl-9"
                  placeholder="e.g. 2022-26" 
                  value={config.batch}
                  onChange={e => setConfig({ ...config, batch: e.target.value })}
                />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label htmlFor="subjectName">Subject Name <span className="text-red-500">*</span></Label>
              <Input 
                id="subjectName"
                placeholder="e.g. Machine Learning" 
                value={config.subjectName}
                onChange={e => setConfig({ ...config, subjectName: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="subjectCode">Subject Code <span className="text-red-500">*</span></Label>
              <div className="relative">
                <Hash className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input 
                  id="subjectCode"
                  className="pl-9"
                  placeholder="e.g. 21AI51" 
                  value={config.subjectCode}
                  onChange={e => setConfig({ ...config, subjectCode: e.target.value })}
                />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-2">
              <Label>Exam Type</Label>
              <Select 
                value={config.examType} 
                onValueChange={(val: any) => setConfig({ ...config, examType: val })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select Exam Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="IAT1">IAT-1</SelectItem>
                  <SelectItem value="IAT2">IAT-2</SelectItem>
                  <SelectItem value="SEE">SEE</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Semester</Label>
              <Select 
                value={config.semester.toString()} 
                onValueChange={(val) => setConfig({ ...config, semester: parseInt(val, 10) })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select Semester" />
                </SelectTrigger>
                <SelectContent>
                  {[1,2,3,4,5,6,7,8].map(sem => (
                    <SelectItem key={sem} value={sem.toString()}>Semester {sem}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Date of Exam</Label>
              <DatePicker 
                date={config.dateOfExam ? new Date(config.dateOfExam) : undefined}
                setDate={(date) => setConfig({ ...config, dateOfExam: date ? format(date, "yyyy-MM-dd") : undefined })}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-50 p-4 rounded-lg border">
            <div className="space-y-2">
              <Label htmlFor="maxMarks">Max Marks</Label>
              <Input 
                id="maxMarks"
                type="number" 
                value={config.maxMarks}
                onChange={e => setConfig({ ...config, maxMarks: parseInt(e.target.value, 10) || 0 })}
              />
            </div>
            <div className="space-y-2">
              <Label>Duration</Label>
              <Select 
                value={config.duration} 
                onValueChange={(val) => setConfig({ ...config, duration: val })}
              >
                <SelectTrigger className="w-full">
                  <Clock className="mr-2 h-4 w-4 text-muted-foreground" />
                  <SelectValue placeholder="Select Duration" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1 hr">1 hr</SelectItem>
                  <SelectItem value="1.5 hrs">1.5 hrs</SelectItem>
                  <SelectItem value="2 hrs">2 hrs</SelectItem>
                  <SelectItem value="3 hrs">3 hrs</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

        </CardContent>
        <CardFooter className="bg-slate-50/50 border-t py-4 flex justify-end">
          <Button 
            size="lg" 
            onClick={onNext} 
            disabled={!isValid}
            className="font-semibold shadow-md transition-all active:scale-95"
          >
            Configure Structure
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
