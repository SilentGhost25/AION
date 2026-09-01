import React from "react"
import katex from "katex"
import "katex/dist/katex.min.css"

// Matches: \command, \command_{...}, \command^{...}, \command_{...}^{...}, \text{...}
const LATEX_RE = /\\\((?:[^\\]|\\(?!\)))*?\\\)|\$[^$]+\$|\\(?:[a-zA-Z]+)(?:\{[^}]*\}|\^[^{]*|_{[^}]*})*/g

export function MathText({ text, className }: { text: string; className?: string }) {
  if (!text) return null

  // Quick check: if no LaTeX commands, just render plain text
  if (!text.includes("\\")) {
    return <span className={className}>{text}</span>
  }

  const parts: React.ReactNode[] = []
  let lastIdx = 0
  let match: RegExpExecArray | null

  const re = new RegExp(LATEX_RE.source, "g")
  while ((match = re.exec(text)) !== null) {
    // Text before the math
    if (match.index > lastIdx) {
      parts.push(<span key={`t-${lastIdx}`}>{text.slice(lastIdx, match.index)}</span>)
    }

    const latex = match[0]
    try {
      const html = katex.renderToString(latex, {
        throwOnError: false,
        strict: false,
        displayMode: false,
      })
      parts.push(
        <span
          key={`m-${match.index}`}
          className="inline-block px-0.5"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      )
    } catch {
      // Fallback: show raw LaTeX in a subtle style
      parts.push(
        <code key={`m-${match.index}`} className="px-1 py-0.5 rounded bg-slate-100 font-mono text-[0.9em]">
          {latex}
        </code>
      )
    }

    lastIdx = match.index + latex.length
  }

  // Remaining text
  if (lastIdx < text.length) {
    parts.push(<span key={`t-end`}>{text.slice(lastIdx)}</span>)
  }

  return <span className={className}>{parts}</span>
}
