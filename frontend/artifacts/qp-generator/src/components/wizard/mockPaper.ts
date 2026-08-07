// ---------------------------------------------------------------------------
// TEMPORARY TEST DATA — remove once the backend is connected.
//
// This produces a sample generated paper (with a deliberate mix of short and
// very large questions) so the preview / row-expansion can be verified without
// a running backend. The real flow uses the backend response from
// POST /api/paper/generate instead.
// ---------------------------------------------------------------------------
import { PaperConfig, GeneratedPaper } from "@workspace/api-client-react"
import { QuestionWithSubs } from "./Step2Preview"

// A very long block of text to prove the row grows to fit huge questions.
const HUGE = `Consider a large-scale distributed system that must maintain strong consistency across geographically separated data centres. Design and fully describe a data structure and accompanying algorithm that supports the following operations efficiently: (1) insertion of a key-value pair with an associated vector-clock timestamp, (2) conflict detection and resolution when two replicas concurrently update the same key, (3) range queries that return all keys within a lexicographic interval in sorted order, and (4) garbage collection of tombstoned entries after a configurable retention window. For each operation, derive the worst-case and amortised time complexity, justify your bounds with a rigorous argument, discuss the space overhead introduced by the metadata, and explain in detail how your design behaves under network partitions with respect to the CAP theorem. Finally, compare your approach against a naive centralised solution and a CRDT-based solution, clearly stating the trade-offs in terms of latency, throughput, memory footprint, and implementation complexity, and recommend which approach is most appropriate for a system serving 10 million requests per second.`

const MEDIUM = `Explain the divide-and-conquer strategy used by merge sort. Trace the complete execution of merge sort on the array [38, 27, 43, 3, 9, 82, 10], clearly showing every split and merge step, and analyze the time and space complexity of the algorithm.`

const SHORT = `Define a binary search tree and state its ordering property.`

export function buildTestPaper(config: PaperConfig): GeneratedPaper {
  const isSEE = config.examType === "SEE"
  const blockMarks = Math.floor(config.maxMarks / 5)

  // Helper to split a block's marks across sub-questions.
  const split = (n: number): number[] => {
    const base = Math.floor(blockMarks / n)
    const rem = blockMarks - base * n
    return Array.from({ length: n }, (_, i) => base + (i < rem ? 1 : 0))
  }

  const questions: QuestionWithSubs[] = Array.from({ length: 10 }).map((_, i) => {
    const qNo = i + 1
    const isOr = i % 2 === 1
    const co = `CO${Math.min(5, Math.ceil(qNo / 2))}`

    // Mix it up: some questions get one huge sub-question, some get a couple
    // of normal ones, one gets a short single line — so row heights vary widely.
    let subQuestions
    if (qNo === 1 || qNo === 6) {
      // Single, enormous question — should force the row to grow tall.
      subQuestions = [
        { label: "a", text: HUGE, marks: blockMarks, co, rbt: "L5" },
      ]
    } else if (qNo === 4) {
      // Short one-liner — should render a compact row.
      subQuestions = [
        { label: "a", text: SHORT, marks: split(1)[0], co, rbt: "L1" },
      ]
    } else if (qNo === 9) {
      // Mixed: one huge + one medium in the same block.
      const [m1, m2] = split(2)
      subQuestions = [
        { label: "a", text: MEDIUM, marks: m1, co, rbt: "L3" },
        { label: "b", text: HUGE, marks: m2, co, rbt: "L5" },
      ]
    } else {
      // Normal two/three sub-question block.
      const n = isSEE ? 3 : 2
      const marks = split(n)
      const texts = [MEDIUM, SHORT, `Illustrate the concept for question ${qNo} with a suitable example and a neat labelled diagram.`]
      subQuestions = Array.from({ length: n }, (_, s) => ({
        label: String.fromCharCode(97 + s),
        text: texts[s % texts.length],
        marks: marks[s],
        co,
        rbt: `L${(s % 4) + 2}`,
      }))
    }

    return {
      qNo,
      text: subQuestions.map(s => `${s.label}) ${s.text}`).join("\n"),
      marks: blockMarks,
      co,
      rbt: subQuestions[0].rbt,
      sectionNumber: qNo,
      isOrQuestion: isOr,
      subQuestions,
    }
  })

  return {
    config,
    questions,
    courseOutcomes: [
      "Understand fundamental data structures including arrays and linked lists",
      "Apply stack and queue operations to solve computational problems",
      "Implement tree data structures and traversal algorithms",
      "Analyze graph algorithms and hashing techniques",
      "Evaluate and compare sorting and searching algorithms",
    ],
    coCoverage: { co1: 20, co2: 20, co3: 20, co4: 20, co5: 20 },
    syllabusCoverage: { s1: 20, s2: 20, s3: 20, s4: 20, s5: 20 },
  }
}
