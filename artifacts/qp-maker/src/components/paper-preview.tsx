import React from "react";

interface PaperPreviewProps {
  formData?: any;
  questions: any[];
}

export function PaperPreview({ formData, questions }: PaperPreviewProps) {
  const defaults = {
    examType: "IAT-1",
    department: "Computer Science & Engineering",
    subjectName: "Machine Learning",
    subjectCode: "21AI51",
    semester: "5",
    maxMarks: 50,
    batch: "2022-26",
    duration: "1.5 hrs",
    dateOfIat: "2023-10-12",
    teachingDept: "AIML Dept"
  };

  const data = formData ? { ...defaults, ...formData } : defaults;

  return (
    <div className="font-serif text-[11pt] leading-tight space-y-4 text-black">
      {/* Header */}
      <div className="text-center border-b-2 border-black pb-4 space-y-1">
        <h1 className="font-bold text-lg">Dayananda Sagar Academy of Technology & Management</h1>
        <p className="text-sm">(Autonomous Institute under VTU) Affiliated to VTU | Approved by AICTE</p>
        <p className="text-sm">Accredited by NAAC with A+ Grade | 6 Programs Accredited by NBA</p>
        <p className="font-bold pt-2">Department of {data.department}</p>
        <p className="font-bold underline">{data.examType}</p>
      </div>

      {/* Meta Info Table */}
      <table className="w-full text-sm border-collapse border border-black mb-4">
        <tbody>
          <tr>
            <td className="border border-black p-1 font-bold w-1/4">Subject / Course:</td>
            <td className="border border-black p-1 w-1/4">{data.subjectName}</td>
            <td className="border border-black p-1 font-bold w-1/4">Subject Code:</td>
            <td className="border border-black p-1 w-1/4">{data.subjectCode}</td>
          </tr>
          <tr>
            <td className="border border-black p-1 font-bold">Semester:</td>
            <td className="border border-black p-1">{data.semester}</td>
            <td className="border border-black p-1 font-bold">Max. Marks:</td>
            <td className="border border-black p-1">{data.maxMarks}</td>
          </tr>
          <tr>
            <td className="border border-black p-1 font-bold">Batch:</td>
            <td className="border border-black p-1">{data.batch}</td>
            <td className="border border-black p-1 font-bold">Duration:</td>
            <td className="border border-black p-1">{data.duration}</td>
          </tr>
          <tr>
            <td className="border border-black p-1 font-bold">Date of IAT:</td>
            <td className="border border-black p-1">{data.dateOfIat}</td>
            <td className="border border-black p-1 font-bold">Teaching Dept:</td>
            <td className="border border-black p-1">{data.teachingDept}</td>
          </tr>
        </tbody>
      </table>

      {/* RBT Legend */}
      <div className="text-xs space-y-1 my-4 bg-gray-50 p-2 border border-gray-300">
        <p><span className="font-bold">Course Outcomes (COs):</span> CO1, CO2, CO3, CO4, CO5</p>
        <p><span className="font-bold">Revised Bloom's Taxonomy (RBT) Levels:</span> L1-Remember, L2-Understand, L3-Apply, L4-Analyze, L5-Evaluate, L6-Create</p>
      </div>

      {/* Instruction */}
      <div className="font-bold text-center my-4">
        Instruction: Answer the following questions
      </div>

      {/* Questions Table */}
      <table className="w-full text-sm border-collapse border border-black mb-8">
        <thead>
          <tr className="bg-gray-100">
            <th className="border border-black p-2 w-12 text-center">Q.No</th>
            <th className="border border-black p-2 text-left">Questions</th>
            <th className="border border-black p-2 w-16 text-center">Marks</th>
            <th className="border border-black p-2 w-16 text-center">COs</th>
            <th className="border border-black p-2 w-16 text-center">RBTL</th>
          </tr>
        </thead>
        <tbody>
          {questions.slice(0, 10).map((q, idx) => {
            const isOr = idx % 2 === 1;
            return (
              <React.Fragment key={idx}>
                {isOr && (
                  <tr>
                    <td colSpan={5} className="border-x border-black p-1 text-center font-bold bg-gray-50">OR</td>
                  </tr>
                )}
                <tr>
                  <td className="border border-black p-2 text-center align-top">{idx + 1}</td>
                  <td className="border border-black p-2 align-top">{q.text}</td>
                  <td className="border border-black p-2 text-center align-top">{q.marks}</td>
                  <td className="border border-black p-2 text-center align-top">{q.co}</td>
                  <td className="border border-black p-2 text-center align-top">{q.bloomLevel}</td>
                </tr>
              </React.Fragment>
            );
          })}
        </tbody>
      </table>

      {/* CO Table */}
      <div className="mt-8 page-break-inside-avoid">
        <h3 className="font-bold mb-2">Course Outcomes:</h3>
        <table className="w-full text-sm border-collapse border border-black">
          <thead>
            <tr className="bg-gray-100">
              <th className="border border-black p-2 w-16 text-center">COs</th>
              <th className="border border-black p-2 text-left">Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-black p-2 text-center font-bold">CO1</td>
              <td className="border border-black p-2">Understand the fundamental concepts of machine learning.</td>
            </tr>
            <tr>
              <td className="border border-black p-2 text-center font-bold">CO2</td>
              <td className="border border-black p-2">Apply learning algorithms to real-world problems.</td>
            </tr>
            <tr>
              <td className="border border-black p-2 text-center font-bold">CO3</td>
              <td className="border border-black p-2">Analyze and evaluate the performance of models.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
