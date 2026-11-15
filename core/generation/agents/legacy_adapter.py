"""
Convert v3 OrchestratorResult into the legacy run_pipeline return shape.

The legacy pipeline returns `(paper_parts, full_paper)`:
    paper_parts : List[dict]  — one entry per module, with questions
    full_paper  : List[dict]  — flat list of questions across all modules

Each question is a dict with (at minimum) these keys:
    slot_id, question_text, solution, marking_scheme,
    marks, bloom, co, module_id,
    image_path, figure_caption, table_data, diagram_request

This adapter is the ONLY place that knows about the legacy shape.
If the shape changes, this file changes. Agents do not.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Dict, List, Tuple

from .generated_question import GeneratedQuestion
from .orchestrator_contracts import OrchestratorResult


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def to_legacy_shape(
    result: OrchestratorResult,
    context: Any,
) -> Tuple[List[dict], List[dict]]:
    """
    Return `(paper_parts, full_paper)` matching the legacy pipeline.

    paper_parts : one entry per module, with module metadata + questions
    full_paper  : flat list of all question dicts (used by DOCX export)
    """
    if not result.success:
        # Legacy shape on failure: empty outputs.
        return [], []

    questions = list(result.questions or [])
    paper_spec = getattr(context, "paper_spec", None)
    artifact = getattr(context, "artifact", None)

    # Build flat list of question dicts
    full_paper: List[dict] = [_to_question_dict(q) for q in questions]

    # Group by module
    by_module: Dict[str, List[dict]] = defaultdict(list)
    for q_dict in full_paper:
        by_module[q_dict["module_id"]].append(q_dict)

    # Build paper_parts
    paper_parts: List[dict] = []
    if paper_spec is not None and hasattr(paper_spec, "module_count"):
        for module_idx in range(1, paper_spec.module_count + 1):
            module_id = f"module_{module_idx}"
            module_questions = by_module.get(module_id, [])
            paper_parts.append({
                "module_id": module_id,
                "module_index": module_idx,
                "module_title": _module_title(artifact, module_idx),
                "questions": module_questions,
                "question_count": len(module_questions),
            })
    else:
        # No spec: emit one part per module_id we saw.
        for module_id, module_questions in sorted(by_module.items()):
            paper_parts.append({
                "module_id": module_id,
                "module_index": _module_idx_from_id(module_id),
                "module_title": "",
                "questions": module_questions,
                "question_count": len(module_questions),
            })

    # Attach QA report to both for downstream consumers
    qa = _qa_dict(result.qa_report)
    for part in paper_parts:
        part["qa_report"] = qa

    return paper_parts, full_paper


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _extract_subpart_texts(text: str, n_parts: int) -> List[str]:
    if n_parts <= 1 or not text:
        return [text]
    import re
    # Match outer subquestion markers (strictly letters a-e):
    # 1. Parenthesized: (a), (b), (c)
    # 2. Letter with closing paren: a), b), c)
    # 3. Letter with dot preceded by whitespace: " a. ", " b. " (avoiding numbers/decimals)
    # 4. Explicit part: "Part a:", "Part A:"
    pattern = (
        r"(?:^|\n|(?<=[.:;?!]\s)|(?<=\s))"
        r"(?:"
        r"\(([a-eA-E])\)"
        r"|\(([ivxIVX]{1,4})\)"
        r"|\(([1-9])\)"
        r"|([a-eA-E])\)"
        r"|(?<=\s)([a-eA-E])\.)"
        r"\s+"
    )
    matches = list(re.finditer(pattern, text))
    if not matches:
        print(f"[LEGACY-ADAPTER] warning: no subpart markers [a-e] found in text: {text[:80]!r}", flush=True)
        if n_parts > 1:
            sentences = [s.strip() for s in re.split(r'(?<=[.?!;])\s+', text) if s.strip()]
            if len(sentences) >= n_parts:
                k, m = divmod(len(sentences), n_parts)
                sub_texts = []
                idx = 0
                for i in range(n_parts):
                    take = k + (1 if i < m else 0)
                    chunk = " ".join(sentences[idx:idx + take]).strip()
                    sub_texts.append(chunk)
                    idx += take
                return sub_texts
        return [text for _ in range(n_parts)]

    stem = text[:matches[0].start()].strip()
    sub_texts = []
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sub_text = text[start:end].strip()
        sub_texts.append(f"{stem} {sub_text}".strip() if stem else sub_text)

    if len(sub_texts) == n_parts:
        return sub_texts
    elif len(sub_texts) > n_parts:
        return sub_texts[:n_parts - 1] + [" ".join(sub_texts[n_parts - 1:])]
    else:
        padded = [f"{stem} {text}".strip() if stem else text for _ in range(n_parts - len(sub_texts))]
        return sub_texts + padded


def _strip_shared_prefix(sub_texts: list, min_words: int = 6) -> list:
    """Remove duplicated opening phrases from sub-parts 2+ when they share
    a long prefix with sub-part 1."""
    if len(sub_texts) < 2:
        return sub_texts
    words_a = sub_texts[0].split()
    result = [sub_texts[0]]
    task_verbs = {
        "what", "why", "how", "when", "where", "which",
        "explain", "describe", "list", "state", "define",
        "calculate", "compute", "determine", "derive", "find",
        "propose", "identify", "analyze", "analyse", "evaluate",
        "compare", "differentiate", "design", "discuss", "write",
        "mention", "show", "prove", "illustrate", "demonstrate",
        "assess", "justify", "critique", "construct", "develop",
        "suggest", "recommend", "outline", "summarize",
        "consider", "examine", "investigate", "explore",
    }
    for sub in sub_texts[1:]:
        words_sub = sub.split()
        shared = 0
        for w1, w2 in zip(words_a, words_sub):
            if w1.lower().rstrip(".,;:!?") == w2.lower().rstrip(".,;:!?"):
                shared += 1
            else:
                break
        if shared < min_words:
            result.append(sub)
            continue
        tail = " ".join(words_sub[shared:]).strip()
        if not tail or len(tail.split()) < 4:
            result.append(sub)
            continue
        first_word = tail.split()[0].lower().rstrip(".,;:!?")
        if first_word in task_verbs:
            result.append(tail)
        else:
            clean_tail = tail[0].lower() + tail[1:] if tail else tail
            result.append(f"Explain the principles and mechanisms of {clean_tail}")
    return result


def _clean_topic(concept: str) -> str:
    """Sanitize a topic string: strip bullets, relative clauses, leading participles, etc."""
    if not concept:
        return "the core module concept"
    t = str(concept).strip()
    t = re.sub(r"^[▪•→◆■\*\-\d\.\)\s]+", "", t).strip()
    t = re.sub(r"[▪•→◆■]+", " ", t).strip()
    t = re.split(r",\s*(?:where|which|that|who|when)\b", t, flags=re.IGNORECASE)[0].strip()
    t = t.rstrip(",;:.- ")
    t = re.sub(r"^(?:executed\s+by|where|which|that|by|when|and|or|for|during|using)\s+", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s+in\s+detail$", "", t, flags=re.IGNORECASE).strip()
    if t.lower().startswith("the "):
        t = t[4:].strip()
    if not t or len(t) < 3:
        return "the core principles"
    return t


def _specific_overlap(s1: str, s2: str) -> tuple:
    generic = {
        'the', 'of', 'in', 'by', 'which', 'a', 'an', 'and', 'for', 'to', 'is', 'are', 'with',
        'describe', 'explain', 'outline', 'role', 'method', 'methods', 'way', 'ways', 'how', 
        'ensure', 'ensures', 'use', 'using', 'detail', 'system', 'systems', 'computer', 'computers', 
        'following', 'state', 'list', 'define', 'respect', 'provide', 'briefly'
    }
    def get_terms(s):
        w = [t for t in re.findall(r'[a-zA-Z]+', s.lower()) if t not in generic]
        res = set()
        for x in w:
            for suf in ['ingly', 'ously', 'fully', 'ment', 'tion', 'sion', 'ness', 'able', 'ible', 'ing', 'ied', 'ies', 'est', 'ers', 'ely', 'ity', 'ive', 'ous', 'ful', 'ed', 'er', 'ly', 'es', 's']:
                if len(x) > len(suf) + 2 and x.endswith(suf):
                    x = x[:-len(suf)]
                    break
            if x not in generic:
                res.add(x)
        return res
    t1 = get_terms(s1)
    t2 = get_terms(s2)
    common = t1 & t2
    min_len = min(len(t1), len(t2))
    containment = len(common) / min_len if min_len > 0 else 0.0
    return common, containment


def _to_question_dict(q: GeneratedQuestion) -> dict:
    """Convert one GeneratedQuestion to the legacy question dict with synthesized subquestions."""
    partition = [int(m) for m in (q.partition or ([q.marks] if q.marks else [10]))]
    letters = ["a", "b", "c", "d", "e"]
    sub_texts = _extract_subpart_texts(q.question_text, len(partition))
    sub_texts = _strip_shared_prefix(sub_texts)

    # Detect subquestion overlap/duplicates and record structured validation telemetry.
    # Reliability first: Never fabricate or overwrite pedagogical subquestion text with synthetic templates.
    overlap_detected = False
    overlap_details = []
    if len(partition) > 1:
        for i in range(len(sub_texts)):
            for j in range(i):
                w_i = set(re.findall(r"\w+", sub_texts[i].lower()))
                w_j = set(re.findall(r"\w+", sub_texts[j].lower()))
                jaccard = len(w_i & w_j) / max(1, len(w_i | w_j))
                common, containment = _specific_overlap(sub_texts[i], sub_texts[j])
                if sub_texts[i].strip().lower() == sub_texts[j].strip().lower() or jaccard >= 0.40 or (containment >= 0.6 and len(common) >= 2):
                    overlap_detected = True
                    overlap_details.append({
                        "sub_i": letters[i] if i < len(letters) else f"sub_{i+1}",
                        "sub_j": letters[j] if j < len(letters) else f"sub_{j+1}",
                        "jaccard": round(jaccard, 3),
                        "containment": round(containment, 3),
                    })

    sub_questions = [
        {
            "letter": letters[i] if i < len(letters) else f"sub_{i+1}",
            "text": sub_texts[i] if i < len(sub_texts) else q.question_text,
            "marks": partition[i],
            "co": q.co,
            "bloom": q.bloom,
        }
        for i in range(len(partition))
    ]

    telemetry = dict(q.telemetry or {})
    if overlap_detected:
        telemetry["subquestion_overlap"] = True
        telemetry["overlap_details"] = overlap_details

    return {
        "slot_id": q.slot_id,
        "module_id": q.module_id,
        "global_q_idx": q.global_q_idx,
        "text": q.question_text,
        "question_text": q.question_text,
        "sub_questions": sub_questions,
        "subQuestions": sub_questions,
        "solution": q.solution,
        "marking_scheme": list(q.marking_scheme or []),
        "marks": sum(partition),
        "total_marks": sum(partition),
        "partition": partition,
        "bloom": q.bloom,
        "co": q.co,
        "topic": q.topic,
        "image_path": q.image_path,
        "figure_caption": q.figure_caption,
        "references_image": q.references_image,
        "diagram_request": q.diagram_request,
        "generation_source": q.generation_source,
        "attempts": q.attempts,
        "telemetry": telemetry,
        "subquestion_overlap": overlap_detected,
        "overlap_details": overlap_details,
    }


def _module_title(artifact: Any, module_idx: int) -> str:
    """Extract a module title from the artifact, if available."""
    if artifact is None:
        return f"Module {module_idx}"
    titles = getattr(artifact, "module_titles", None)
    if isinstance(titles, list) and 1 <= module_idx <= len(titles):
        return titles[module_idx - 1]
    return f"Module {module_idx}"


def _module_idx_from_id(module_id: str) -> int:
    try:
        return int(module_id.split("_")[1])
    except (IndexError, ValueError):
        return 0


def _qa_dict(qa_report: Any) -> dict:
    if qa_report is None:
        return {}
    if hasattr(qa_report, "to_dict"):
        return qa_report.to_dict()
    return {}
