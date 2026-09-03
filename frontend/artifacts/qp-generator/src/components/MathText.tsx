import React from "react"
import katex from "katex"
import "katex/dist/katex.min.css"

// Heals tab-corrupted LaTeX tokens resulting from unescaped JSON/transport encoding
function healLatexTokens(raw: string): string {
  if (!raw) return ""
  return raw
    .replace(/[\t ]+imes\b/g, " \\times ")
    .replace(/[\t ]+ext\{/g, " \\text{")
    .replace(/[\t ]+heta\b/g, " \\theta ")
    .replace(/[\t ]+au\b/g, " \\tau ")
    .replace(/[\t ]+frac\{/g, " \\frac{")
    .replace(/[\t ]+sqrt\{/g, " \\sqrt{")
    .replace(/[\t ]+cdot\b/g, " \\cdot ")
    .replace(/[\t ]+approx\b/g, " \\approx ")
    .replace(/[\t ]+pm\b/g, " \\pm ")
    .replace(/[\t ]+pi\b/g, " \\pi ")
    .replace(/[\t ]+mu\b/g, " \\mu ")
}

// Matches: \( ... \), $ ... $, \[ ... \], or standalone \command expressions
const LATEX_RE = /\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]|\$([^\$\n]+)\$|\\(?:[a-zA-Z]+)(?:\{[^}]*\}|\^[^{ \t\n]*|_{[^} \t\n]*})*/g

export function MathText({ text, className }: { text: string; className?: string }) {
  if (!text) return null

  const cleaned = healLatexTokens(text)

  // Quick check: if no math markers or backslashes, return plain text
  if (!cleaned.includes("\\") && !cleaned.includes("$")) {
    return <span className={className}>{cleaned}</span>
  }

  const parts: React.ReactNode[] = []
  let lastIdx = 0
  let match: RegExpExecArray | null

  const re = new RegExp(LATEX_RE.source, "g")
  while ((match = re.exec(cleaned)) !== null) {
    if (match.index > lastIdx) {
      parts.push(<span key={`t-${lastIdx}`}>{cleaned.slice(lastIdx, match.index)}</span>)
    }

    const rawMatch = match[0]
    // Extract inner content without \( \), \[ \], or $ $
    let expr = match[1] ?? match[2] ?? match[3] ?? rawMatch
    const isDisplay = rawMatch.startsWith("\\[")

    try {
      const html = katex.renderToString(expr.trim(), {
        throwOnError: false,
        strict: false,
        displayMode: isDisplay,
      })
      parts.push(
        <span
          key={`m-${match.index}`}
          className="inline-block px-0.5 align-baseline"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      )
    } catch {
      parts.push(
        <code key={`m-${match.index}`} className="px-1 py-0.5 rounded bg-slate-100 font-mono text-[0.9em]">
          {rawMatch}
        </code>
      )
    }

    lastIdx = match.index + rawMatch.length
  }

  if (lastIdx < cleaned.length) {
    parts.push(<span key={`t-end`}>{cleaned.slice(lastIdx)}</span>)
  }

  return <span className={className}>{parts}</span>
}
