// ─────────────────────────────────────────────────────────────────────────────
//  TRAINING DATA — extracted from real uploaded question papers
//
//  These records represent patterns the AI has learned from past exam papers.
//  Each `TrainingPaper` stores a full parsed paper with all questions, marks,
//  CO, and Bloom level metadata. The AI generation engine uses these to:
//    1. Learn question framing styles per Bloom level
//    2. Learn mark distribution patterns
//    3. Learn which questions typically require diagrams
//    4. Use as few-shot examples in generation prompts
//
//  To add more papers: follow the same structure below.
// ─────────────────────────────────────────────────────────────────────────────

export interface TrainingQuestion {
  slNo: number;
  text: string;
  marks: number;
  co: string;
  bloom: string;
  hasDiagram: boolean;
  diagramDescription?: string;
  subParts?: Array<{ label: string; text: string; marks: number; bloom: string; co: string }>;
}

export interface TrainingPaper {
  id: string;
  title: string;
  subjectName: string;
  subjectCode: string;
  examType: "IAT-1" | "IAT-2" | "SEE";
  department: string;
  maxMarks: number;
  duration: string;
  marksPerQuestion: number | "variable";
  questionCount: number;
  orPairs: number;
  courseOutcomes: Array<{ code: string; text: string }>;
  questions: TrainingQuestion[];
  learnType: "full-paper";
  source: string;
}

// ─────────────────────────────────────────────────────────────────────────────
//  PAPER 1: Artificial Intelligence — IAT (10 questions × 10 marks)
//  5 OR pairs, Modules 1-2, CO1-CO2, L1-L2
// ─────────────────────────────────────────────────────────────────────────────
const AI_IAT_PAPER_1: TrainingPaper = {
  id: "train-ai-iat-001",
  title: "AI IAT Paper — Foundations & Search Strategies",
  subjectName: "Artificial Intelligence",
  subjectCode: "21AI41",
  examType: "IAT-1",
  department: "Artificial Intelligence & Machine Learning",
  maxMarks: 50,
  duration: "1.5 hrs",
  marksPerQuestion: 10,
  questionCount: 10,
  orPairs: 5,
  courseOutcomes: [
    { code: "CO1", text: "Understand basic idea of AI and its foundations" },
    { code: "CO2", text: "Apply uninformed and informed search strategies for problem solving" },
    { code: "CO3", text: "Analyze various interfacing techniques and quantifying uncertainties" },
    { code: "CO4", text: "Ability to solve various problems using AI search strategies" },
    { code: "CO5", text: "Ability to build chatbot and games using AI algorithms" },
  ],
  questions: [
    { slNo: 1,  text: "Interpret the categories of Artificial Intelligence with an example? What is Artificial Intelligence?", marks: 10, co: "CO1", bloom: "L1", hasDiagram: false },
    { slNo: 2,  text: "Brief on foundation of Artificial Intelligence?", marks: 10, co: "CO1", bloom: "L1", hasDiagram: false },
    { slNo: 3,  text: "Explain in detail the history of artificial intelligence?", marks: 10, co: "CO1", bloom: "L1", hasDiagram: false },
    { slNo: 4,  text: "Elucidate the properties of Task Environments.", marks: 10, co: "CO1", bloom: "L1", hasDiagram: false },
    { slNo: 5,  text: "Write a python program to implement A* algorithm.", marks: 10, co: "CO2", bloom: "L2", hasDiagram: false },
    { slNo: 6,  text: "Write a python program to solve water jug problem using DFS search strategy.", marks: 10, co: "CO2", bloom: "L2", hasDiagram: false },
    { slNo: 7,  text: "Illustrate the working of intelligent agent with block diagram. Write simple reflex agent program for two state vacuum environment.", marks: 10, co: "CO1", bloom: "L1", hasDiagram: true, diagramDescription: "Block diagram of intelligent agent architecture" },
    { slNo: 8,  text: "Explain 5 components of problem-solving using an 8 puzzle problem.", marks: 10, co: "CO1", bloom: "L1", hasDiagram: false },
    { slNo: 9,  text: "What is PEAS? Explain different agent types with their PEAS descriptions.", marks: 10, co: "CO2", bloom: "L1", hasDiagram: false },
    { slNo: 10, text: "Explain the following structure of agents: (i) model-based agent (ii) goal-based agent (iii) utility-based agent (iv) learning agent.", marks: 10, co: "CO2", bloom: "L1", hasDiagram: false },
  ],
  learnType: "full-paper",
  source: "image_1784260407755.png",
};

// ─────────────────────────────────────────────────────────────────────────────
//  PAPER 2: Artificial Intelligence — IAT (10 questions × 10 marks)
//  5 OR pairs, Modules 1-3, CO1-CO3, L2-L4
//  Notable: several questions include graph/tree diagrams
// ─────────────────────────────────────────────────────────────────────────────
const AI_IAT_PAPER_2: TrainingPaper = {
  id: "train-ai-iat-002",
  title: "AI IAT Paper — Graph Search & Wumpus World",
  subjectName: "Artificial Intelligence",
  subjectCode: "21AI41",
  examType: "IAT-2",
  department: "Artificial Intelligence & Machine Learning",
  maxMarks: 50,
  duration: "1.5 hrs",
  marksPerQuestion: 10,
  questionCount: 10,
  orPairs: 5,
  courseOutcomes: [
    { code: "CO1", text: "Understand basic idea of AI and its foundations" },
    { code: "CO2", text: "Apply uninformed and informed search strategies for problem solving" },
    { code: "CO3", text: "Analyze various interfacing techniques and quantifying uncertainties" },
    { code: "CO4", text: "Ability to solve various problems using AI search strategies" },
    { code: "CO5", text: "Ability to build chatbot and games using AI algorithms" },
  ],
  questions: [
    { slNo: 1,  text: "Illustrate the separation property of GRAPH-SEARCH on a rectangular-grid problem.", marks: 10, co: "CO1", bloom: "L2", hasDiagram: false },
    { slNo: 2,  text: "List out the differences between Informed and Uninformed Search.", marks: 10, co: "CO1", bloom: "L2", hasDiagram: false },
    { slNo: 3,  text: "Apply DFS for below graph with source node=8, goal=13 and write the traversal.", marks: 10, co: "CO2", bloom: "L3", hasDiagram: true, diagramDescription: "Weighted directed graph with nodes 1-14, root=8" },
    { slNo: 4,  text: "Apply PEAS specification for Wumpus world.", marks: 10, co: "CO2", bloom: "L3", hasDiagram: false },
    { slNo: 5,  text: "Analyze depth limited search and iterative depth first search and justify which algorithm is best suited for below graph to reach the goal node 3.", marks: 10, co: "CO3", bloom: "L4", hasDiagram: true, diagramDescription: "Tree diagram showing nodes 1-5, root=1, children=[2,3], grandchildren=[4,5]" },
    { slNo: 6,  text: "Analyze the Wumpus World and Justify the following: (i) There is no pit in P[1,1] (ii) There is no breeze in b[1,1] (iii) There is no wumpus in W[2,2] (iv) There is no pit in P[2,2] (v) No pit in P[1,2].", marks: 10, co: "CO3", bloom: "L4", hasDiagram: false },
    { slNo: 7,  text: "Apply the knowledge of greedy best first search and solve 8 puzzle problem below. Start State: [[1,2,3],[8,_,6],[7,5,4]] Goal State: [[1,2,3],[8,_,4],[7,6,5]].", marks: 10, co: "CO2", bloom: "L3", hasDiagram: true, diagramDescription: "Two 3×3 grid states — Start State and Goal State for 8-puzzle" },
    { slNo: 8,  text: "Apply A* algorithm for below graph to find the shortest path from starting node = a and goal node = z.", marks: 10, co: "CO2", bloom: "L3", hasDiagram: true, diagramDescription: "Weighted graph with nodes a,b,c,d,e,f,z and heuristic values annotated" },
    { slNo: 9,  text: "Analyze the Wumpus World and construct Knowledge base (KB) using model checking approach.", marks: 10, co: "CO3", bloom: "L4", hasDiagram: false },
    { slNo: 10, text: "Analyze the concepts of heuristic function and implement a program in python for Travelling sales person problem.", marks: 10, co: "CO3", bloom: "L4", hasDiagram: false },
  ],
  learnType: "full-paper",
  source: "image_1784260415245.png + image_1784260418415.png",
};

// ─────────────────────────────────────────────────────────────────────────────
//  PAPER 3: Sensors and Transducers — SEE (Module-based, 100 marks)
//  Format: 5 modules × (Q + OR), each question has sub-parts a/b/c
//  Marks per question: 20 (split as a+b+c or a+b variants)
// ─────────────────────────────────────────────────────────────────────────────
const SENSORS_SEE_PAPER_1: TrainingPaper = {
  id: "train-bee-see-001",
  title: "Sensors and Transducers — SEE",
  subjectName: "Sensors and Transducers",
  subjectCode: "BEE404",
  examType: "SEE",
  department: "Electrical & Electronics Engineering",
  maxMarks: 100,
  duration: "3 hrs",
  marksPerQuestion: "variable",
  questionCount: 10,
  orPairs: 5,
  courseOutcomes: [
    { code: "CO1", text: "Explain the working principles of sensors and transducers" },
    { code: "CO2", text: "Apply knowledge of sensing devices to practical measurement problems" },
    { code: "CO3", text: "Analyze performance characteristics and select appropriate sensors" },
    { code: "CO4", text: "Design signal conditioning circuits for sensor outputs" },
    { code: "CO5", text: "Evaluate data acquisition systems and telemetry" },
  ],
  questions: [
    // Module 1 — Q1 & Q2 (OR)
    {
      slNo: 1, text: "Module 1 — Q1 (a+b+c)", marks: 20, co: "CO1", bloom: "L3", hasDiagram: false,
      subParts: [
        { label: "a", text: "Write a short note on importance and role of sensors in technology.", marks: 10, bloom: "L3", co: "CO2" },
        { label: "b", text: "Explain working principle and operating mechanism of Temperature sensor.", marks: 6, bloom: "L2", co: "CO1" },
        { label: "c", text: "Define the following: Accuracy and Precision.", marks: 4, bloom: "L1", co: "CO1" },
      ],
    },
    {
      slNo: 2, text: "Module 1 — Q2 (OR) (a+b+c)", marks: 20, co: "CO1", bloom: "L3", hasDiagram: false,
      subParts: [
        { label: "a", text: "Write a short note on sensor selection criteria and trade-offs.", marks: 10, bloom: "L3", co: "CO2" },
        { label: "b", text: "Explain working principle and operating mechanism of pressure sensor.", marks: 6, bloom: "L2", co: "CO1" },
        { label: "c", text: "Define the following: Sensitivity and Resolution.", marks: 4, bloom: "L1", co: "CO1" },
      ],
    },
    // Module 2 — Q3 & Q4 (OR)
    {
      slNo: 3, text: "Module 2 — Q3 (a+b+c)", marks: 20, co: "CO1", bloom: "L2", hasDiagram: false,
      subParts: [
        { label: "a", text: "Explain working principle and operating mechanism of Fibre optic sensor.", marks: 10, bloom: "L2", co: "CO1" },
        { label: "b", text: "Explain the applications of Optical sensor.", marks: 5, bloom: "L3", co: "CO2" },
        { label: "c", text: "Elaborate on operation of Switch sensor.", marks: 5, bloom: "L3", co: "CO2" },
      ],
    },
    {
      slNo: 4, text: "Module 2 — Q4 (OR) (a+b+c)", marks: 20, co: "CO1", bloom: "L2", hasDiagram: false,
      subParts: [
        { label: "a", text: "Explain working principle and operating mechanism of Strain gauges.", marks: 10, bloom: "L2", co: "CO1" },
        { label: "b", text: "Explain the applications of acoustic touch sensor.", marks: 5, bloom: "L3", co: "CO2" },
        { label: "c", text: "Elaborate on operation of Piezo resistive sensor.", marks: 5, bloom: "L3", co: "CO2" },
      ],
    },
    // Module 3 — Q5 & Q6 (OR)
    {
      slNo: 5, text: "Module 3 — Q5 (a+b)", marks: 20, co: "CO1", bloom: "L2", hasDiagram: false,
      subParts: [
        { label: "a", text: "Describe the working principle and applications of MEMS and NANO sensor.", marks: 10, bloom: "L2", co: "CO1" },
        { label: "b", text: "Explain the applications of sensors in navigation.", marks: 10, bloom: "L3", co: "CO2" },
      ],
    },
    {
      slNo: 6, text: "Module 3 — Q6 (OR) (a+b)", marks: 20, co: "CO1", bloom: "L2", hasDiagram: false,
      subParts: [
        { label: "a", text: "Describe the working principle and applications of Film sensor and touch screen sensor.", marks: 10, bloom: "L2", co: "CO1" },
        { label: "b", text: "Explain the applications of sensors in drone.", marks: 10, bloom: "L3", co: "CO2" },
      ],
    },
    // Module 4 — Q7 & Q8 (OR)
    {
      slNo: 7, text: "Module 4 — Q7 (a+b+c)", marks: 20, co: "CO1", bloom: "L1", hasDiagram: false,
      subParts: [
        { label: "a", text: "What is transducer? How are they classified?", marks: 8, bloom: "L1", co: "CO1" },
        { label: "b", text: "What are advantages of electrical transducers?", marks: 6, bloom: "L3", co: "CO2" },
        { label: "c", text: "Explain variable reluctance transducer.", marks: 6, bloom: "L3", co: "CO2" },
      ],
    },
    {
      slNo: 8, text: "Module 4 — Q8 (OR) (a+b+c)", marks: 20, co: "CO1", bloom: "L1", hasDiagram: false,
      subParts: [
        { label: "a", text: "Explain the working of Piezoelectric accelerometer. List the advantages, disadvantages and applications of Piezoelectric Transducers.", marks: 8, bloom: "L1", co: "CO1" },
        { label: "b", text: "Explain the working of LVDT with advantages, disadvantages and applications.", marks: 6, bloom: "L3", co: "CO2" },
        { label: "c", text: "Explain displacement measurement using Hall effect transducers.", marks: 6, bloom: "L3", co: "CO2" },
      ],
    },
    // Module 5 — Q9 & Q10 (OR)
    {
      slNo: 9, text: "Module 5 — Q9 (a+b+c)", marks: 20, co: "CO1", bloom: "L2", hasDiagram: false,
      subParts: [
        { label: "a", text: "With the help of block diagram, explain the working of telemetering system.", marks: 8, bloom: "L2", co: "CO1" },
        { label: "b", text: "Explain briefly the amplitude and frequency modulation.", marks: 6, bloom: "L3", co: "CO2" },
        { label: "c", text: "What do you mean by filter and filtering? How are filters classified?", marks: 6, bloom: "L1", co: "CO1" },
      ],
    },
    {
      slNo: 10, text: "Module 5 — Q10 (OR) (a+b+c)", marks: 20, co: "CO1", bloom: "L2", hasDiagram: false,
      subParts: [
        { label: "a", text: "Draw the block diagram of generalized Data Acquisition system and explain briefly.", marks: 8, bloom: "L2", co: "CO1" },
        { label: "b", text: "Explain the working of multichannel analog multiplexed data acquisition system.", marks: 6, bloom: "L3", co: "CO2" },
        { label: "c", text: "Explain R-2R ladder D/A converter and PWM.", marks: 6, bloom: "L1", co: "CO1" },
      ],
    },
  ],
  learnType: "full-paper",
  source: "image_1784260515329.png + image_1784260524140.png + image_1784260548018.png + image_1784260550722.png",
};

// ─────────────────────────────────────────────────────────────────────────────
//  EXPORTS
// ─────────────────────────────────────────────────────────────────────────────

export const TRAINING_PAPERS: TrainingPaper[] = [
  AI_IAT_PAPER_1,
  AI_IAT_PAPER_2,
  SENSORS_SEE_PAPER_1,
];

// ── Derived lookup helpers ────────────────────────────────────────────────────

/** All unique question texts across all papers — used as few-shot framing examples */
export const FRAMING_EXAMPLES = TRAINING_PAPERS.flatMap(p =>
  p.questions.flatMap(q => {
    const base = [{ text: q.text, marks: q.marks, co: q.co, bloom: q.bloom, examType: p.examType, hasDiagram: q.hasDiagram }];
    const subs = (q.subParts ?? []).map(s => ({
      text: s.text, marks: s.marks, co: s.co, bloom: s.bloom, examType: p.examType, hasDiagram: false,
    }));
    return [...base, ...subs];
  })
);

/** Mark distribution samples — used by the randomization engine */
export const MARKS_SAMPLES = TRAINING_PAPERS.flatMap(p =>
  p.questions.map(q => ({
    examType: p.examType,
    marks: q.marks,
    co: q.co,
    bloom: q.bloom,
  }))
);

/** Questions that have diagrams — used to train the diagram classifier */
export const DIAGRAM_SIGNALS = TRAINING_PAPERS.flatMap(p =>
  p.questions
    .filter(q => q.hasDiagram)
    .map(q => ({
      text: q.text,
      co: q.co,
      bloom: q.bloom,
      diagramDescription: q.diagramDescription,
    }))
);

/** Stats for the Settings → AI Training Data panel */
export const TRAINING_STATS = {
  totalPapers: TRAINING_PAPERS.length,
  totalQuestions: FRAMING_EXAMPLES.length,
  diagramQuestions: DIAGRAM_SIGNALS.length,
  byExamType: {
    "IAT-1": TRAINING_PAPERS.filter(p => p.examType === "IAT-1").length,
    "IAT-2": TRAINING_PAPERS.filter(p => p.examType === "IAT-2").length,
    "SEE": TRAINING_PAPERS.filter(p => p.examType === "SEE").length,
  },
  byBloom: Object.fromEntries(
    ["L1","L2","L3","L4","L5","L6"].map(l => [
      l,
      FRAMING_EXAMPLES.filter(q => q.bloom === l).length,
    ])
  ),
};
