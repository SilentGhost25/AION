// ─────────────────────────────────────────────────────────────────────────────
//  EXAM & DEPARTMENT CONSTANTS
//  Single source for all dropdown options, exam types, departments, semesters.
// ─────────────────────────────────────────────────────────────────────────────

import type { ExamType } from "@/types";

export const EXAM_TYPES: ExamType[] = ["IAT-1", "IAT-2", "SEE"];

export const EXAM_LABELS: Record<ExamType, string> = {
  "IAT-1": "Internal Assessment Test 1",
  "IAT-2": "Internal Assessment Test 2",
  "SEE": "Semester End Examination",
};

export const EXAM_MAX_MARKS: Record<ExamType, number> = {
  "IAT-1": 50,
  "IAT-2": 50,
  "SEE": 100,
};

export const EXAM_DURATIONS: Record<ExamType, string> = {
  "IAT-1": "1.5 hrs",
  "IAT-2": "1.5 hrs",
  "SEE": "3 hrs",
};

export const MODULES = [1, 2, 3, 4, 5] as const;
export type ModuleNumber = (typeof MODULES)[number];

export const SEMESTERS = ["1", "2", "3", "4", "5", "6", "7", "8"] as const;
export const SEM_SUFFIX = ["st", "nd", "rd", "th", "th", "th", "th", "th"] as const;

export const DEPARTMENTS = [
  "Artificial Intelligence & Machine Learning",
  "Computer Science & Engineering",
  "Electronics & Communication Engineering",
  "Electrical & Electronics Engineering",
  "Mechanical Engineering",
  "Civil Engineering",
  "Information Science & Engineering",
  "Chemical Engineering",
] as const;

export type Department = (typeof DEPARTMENTS)[number];

/** Subject catalogue per department — extend as needed. */
export const DEPARTMENT_SUBJECTS: Record<string, Array<{ code: string; name: string }>> = {
  "Artificial Intelligence & Machine Learning": [
    { code: "21AI51", name: "Machine Learning" },
    { code: "21AI52", name: "Deep Learning" },
    { code: "21AI53", name: "Natural Language Processing" },
    { code: "21AI54", name: "Computer Vision" },
    { code: "21AI55", name: "Data Structures and Algorithms" },
    { code: "21AI41", name: "Design and Analysis of Algorithms" },
    { code: "21AI42", name: "Database Management Systems" },
  ],
  "Computer Science & Engineering": [
    { code: "21CS51", name: "Operating Systems" },
    { code: "21CS52", name: "Computer Networks" },
    { code: "21CS53", name: "Software Engineering" },
    { code: "21CS41", name: "Design and Analysis of Algorithms" },
  ],
  "Electronics & Communication Engineering": [
    { code: "21EC51", name: "Digital Signal Processing" },
    { code: "21EC52", name: "VLSI Design" },
    { code: "21EC53", name: "Wireless Communication" },
  ],
};
