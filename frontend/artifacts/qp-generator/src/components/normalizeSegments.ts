import type { QuestionSegment } from "./QuestionRenderer"

/**
 * Single canonicalization point for question segments.
 * Math latex is OPAQUE. No regex. No unescape. No markdown.
 */
export function normalizeSegments(segments: QuestionSegment[]): QuestionSegment[] {
  return segments.map((seg) => {
    if (seg.type === "math") {
      return { ...seg, latex: String(seg.latex ?? ""), display: seg.display ?? false }
    }
    if (seg.type === "text") {
      return { ...seg, value: String(seg.value ?? "") }
    }
    return seg
  })
}

/**
 * Split question_text at [MATH:block_id] markers into segments.
 * The latex from math_blocks is passed OPAQUE to KaTeX.
 */
export function buildSegments(
  questionText: string,
  mathBlocks: Array<{ block_id: string; latex: string; display_mode?: boolean }>,
): QuestionSegment[] {
  if (!questionText) return []

  const blockMap = new Map<string, { latex: string; display: boolean }>()
  for (const b of mathBlocks ?? []) {
    if (b.block_id && b.latex) {
      blockMap.set(b.block_id, { latex: b.latex, display: b.display_mode ?? false })
    }
  }

  const segments: QuestionSegment[] = []
  const re = /\[MATH:([^\]]+)\]/g
  let lastIdx = 0
  let match: RegExpExecArray | null

  while ((match = re.exec(questionText)) !== null) {
    if (match.index > lastIdx) {
      const chunk = questionText.slice(lastIdx, match.index)
      if (chunk) segments.push({ type: "text", value: chunk })
    }
    const block = blockMap.get(match[1])
    if (block) {
      segments.push({ type: "math", latex: block.latex, display: block.display })
    } else {
      segments.push({ type: "text", value: `[${match[1]}]` })
    }
    lastIdx = match.index + match[0].length
  }

  if (lastIdx < questionText.length) {
    const chunk = questionText.slice(lastIdx)
    if (chunk) segments.push({ type: "text", value: chunk })
  }

  if (segments.length === 0 && questionText) {
    segments.push({ type: "text", value: questionText })
  }

  return normalizeSegments(segments)
}
