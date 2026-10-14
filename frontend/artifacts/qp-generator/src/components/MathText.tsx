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

interface LatexToken {
  type: "text" | "math"
  expr?: string
  raw?: string
  display?: boolean
  value?: string
}

function scanLatex(text: string): LatexToken[] {
  const tokens: LatexToken[] = []
  let i = 0
  let textStart = 0

  const flushText = (endIdx: number) => {
    if (endIdx > textStart) {
      tokens.push({ type: "text", value: text.slice(textStart, endIdx) })
    }
  }

  while (i < text.length) {
    // Delimited: \( ... \)
    if (text.startsWith("\\(", i)) {
      const end = text.indexOf("\\)", i + 2)
      if (end !== -1) {
        flushText(i)
        tokens.push({ type: "math", expr: text.slice(i + 2, end), raw: text.slice(i, end + 2), display: false })
        i = end + 2
        textStart = i
        continue
      }
    }
    // Delimited: \[ ... \]
    if (text.startsWith("\\[", i)) {
      const end = text.indexOf("\\]", i + 2)
      if (end !== -1) {
        flushText(i)
        tokens.push({ type: "math", expr: text.slice(i + 2, end), raw: text.slice(i, end + 2), display: true })
        i = end + 2
        textStart = i
        continue
      }
    }
    // Delimited: $ ... $ (inline)
    if (text[i] === "$" && !text.startsWith("$$", i)) {
      const end = text.indexOf("$", i + 1)
      if (end !== -1 && !text.slice(i + 1, end).includes("\n")) {
        flushText(i)
        tokens.push({ type: "math", expr: text.slice(i + 1, end), raw: text.slice(i, end + 1), display: false })
        i = end + 1
        textStart = i
        continue
      }
    }
    // Standalone LaTeX command: \frac{...}{...}, \sqrt{...}, etc. with balanced braces
    if (text[i] === "\\" && i + 1 < text.length && /[a-zA-Z]/.test(text[i + 1])) {
      let j = i + 1
      while (j < text.length && /[a-zA-Z]/.test(text[j])) j++
      // Consume any arguments with balanced braces or scripts
      while (j < text.length) {
        if (text[j] === "{" || (text[j] === "^" && text[j + 1] === "{") || (text[j] === "_" && text[j + 1] === "{")) {
          if (text[j] === "^" || text[j] === "_") j++
          let depth = 1
          j++
          while (j < text.length && depth > 0) {
            if (text[j] === "{") depth++
            else if (text[j] === "}") depth--
            j++
          }
        } else if (text[j] === "^" || text[j] === "_") {
          j += 2
        } else {
          break
        }
      }
      const raw = text.slice(i, j)
      flushText(i)
      tokens.push({ type: "math", expr: raw, raw, display: false })
      i = j
      textStart = i
      continue
    }
    i++
  }
  flushText(text.length)
  return tokens
}

export function MathText({ text, className }: { text: string; className?: string }) {
  if (!text) return null

  const cleaned = healLatexTokens(text)

  // Quick check: if no math markers or backslashes, return plain text
  if (!cleaned.includes("\\") && !cleaned.includes("$")) {
    return <span className={className}>{cleaned}</span>
  }

  const tokens = scanLatex(cleaned)
  return (
    <span className={className}>
      {tokens.map((tok, idx) => {
        if (tok.type === "text") {
          return <span key={idx}>{tok.value}</span>
        }
        try {
          const html = katex.renderToString((tok.expr || "").trim(), {
            throwOnError: false,
            strict: false,
            displayMode: tok.display ?? false,
          })
          return (
            <span
              key={idx}
              className="inline-block px-0.5 align-baseline"
              dangerouslySetInnerHTML={{ __html: html }}
            />
          )
        } catch {
          return (
            <code key={idx} className="px-1 py-0.5 rounded bg-slate-100 font-mono text-[0.9em]">
              {tok.raw}
            </code>
          )
        }
      })}
    </span>
  )
}
