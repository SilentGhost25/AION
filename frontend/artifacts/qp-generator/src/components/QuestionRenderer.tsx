import React from "react"
import katex from "katex"
import "katex/dist/katex.min.css"

export interface QuestionSegment {
  type       : "text" | "math" | "figure" | "table"
  value?     : string
  latex?     : string
  display?   : boolean
  figure_id? : string
  alt_text?  : string
  render_url?: string
  caption?   : string
  markdown?  : string
}

export interface QuestionIntegrity {
  encoding_clean      : boolean
  math_validated      : boolean
  injection_clean     : boolean
  grounding_score     : number
  round_trip_verified : boolean
}

export interface QuestionData {
  question_id : string
  question_no : number
  sub_label?  : string
  marks       : number
  bloom       : string
  co          : string
  segments    : QuestionSegment[]
  integrity?  : QuestionIntegrity
}

export function QuestionRenderer({ question }: { question: QuestionData }) {
  // Safety: never render a question that failed integrity
  if (question.integrity && (!question.integrity.encoding_clean || !question.integrity.math_validated)) {
    return (
      <div className="p-3 my-2 text-xs font-semibold text-rose-700 bg-rose-50 border border-rose-200 rounded-md flex items-center gap-2">
        <span>⚠</span> Question integrity check failed — not rendered
      </div>
    )
  }

  return (
    <div className="question-content leading-relaxed text-slate-800">
      {question.segments.map((seg, i) => {
        switch (seg.type) {
          case "text":
            return <span key={i}>{seg.value} </span>

          case "math":
            return (
              <KaTeXRenderer
                key={i}
                latex={seg.latex || ""}
                displayMode={seg.display ?? false}
                onError={(err) => {
                  console.error(`KaTeX error for ${question.question_id}:`, err)
                  return <code className="px-1.5 py-0.5 rounded bg-slate-100 font-mono text-xs text-rose-600">{seg.latex}</code>
                }}
              />
            )

          case "figure":
            return (
              <figure key={i} className="my-3 text-center">
                <img src={seg.render_url} alt={seg.alt_text || "Figure"} className="max-w-full h-auto mx-auto rounded border shadow-sm" />
                {seg.caption && <figcaption className="text-xs text-slate-500 mt-1 font-medium">{seg.caption}</figcaption>}
              </figure>
            )

          case "table":
            return (
              <pre key={i} className="my-2 p-2 bg-slate-50 border rounded text-xs font-mono overflow-x-auto">
                {seg.markdown || seg.value}
              </pre>
            )

          default:
            return null
        }
      })}
    </div>
  )
}

export function KaTeXRenderer({
  latex,
  displayMode,
  onError
}: {
  latex: string
  displayMode: boolean
  onError: (err: Error) => React.ReactElement
}) {
  if (!latex || latex.trim() === "" || latex.includes("\uFFFD")) {
    return <span className="text-rose-500 font-mono text-xs">[invalid equation]</span>
  }

  try {
    const html = katex.renderToString(latex, {
      displayMode,
      throwOnError: true,
      strict: false,
    })
    return (
      <span
        className={displayMode ? "block my-2 text-center" : "inline-block px-1"}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    )
  } catch (err) {
    return onError(err as Error)
  }
}
