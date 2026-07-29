"""
AION Module: Question Generator
Maturity:    v0.1 — RAG² (REVERSE ASSESSMENT GENERATION) ENGINE
Upgrades to: Fine-Tuned Fine-Grained Academic Examiner LLM / MoE
Contract:    concept: Concept -> GeneratedQuestion (see schemas.py)
             MUST enforce ideal_answer generation BEFORE question_text generation.
"""

import os
import re
import random
import time
from .schemas import Concept, GeneratedQuestion
from .content_validator import validate_chunk, clean_chunk
from .llm import get_llm


# ─────────────────────────────────────────────────────────────
# Shared LLM runner
# ─────────────────────────────────────────────────────────────

def _run_prompt(prompt: str, max_tokens: int = 250) -> str:
    try:
        res = get_llm().generate(prompt)
        if res:
            return res
    except Exception:
        pass
    return ""


# ─────────────────────────────────────────────────────────────
# Multiple prompt templates — randomly selected each run
# This forces different phrasing and question styles
# ─────────────────────────────────────────────────────────────

_TURBO_PROMPTS = [

    # Template A — Explain + Derive
    """\
You are a senior VTU examiner setting questions for B.E. semester exams.

SOURCE MATERIAL:
\"\"\"{chunk}\"\"\"

Generate ONE complete VTU-style exam question worth {marks} marks at Bloom level {bloom}.

Requirements:
- Begin with ONE single VTU verb: Explain, Derive, Analyse, or Discuss
- Pick ONE verb only. Do not list multiple verbs.
- Must reference specific concepts from the source material
- Must be academically rich — not a single-line definition
- For {marks} marks, the question must have proportional depth
- Do NOT write phrases like "as described in the source", "as per the material",
  "from the source material", "as outlined in the text", "from the document".
- Do NOT mention any author names, researcher names, or person names.
- Do NOT reference any Figure, Table, Diagram, Program, or Listing numbers.
- Do NOT mention any book titles, chapter numbers, or publication names.
- Write a clean, self-contained exam question with zero external references.
- Output ONLY the question text. No answer, no note, no prefix.

Question:""",

    # Template B — Compare/Contrast
    """\
You are a VTU exam paper setter for engineering semester exams.

SOURCE MATERIAL:
\"\"\"{chunk}\"\"\"

Write ONE original VTU-style descriptive question worth {marks} marks at Bloom level {bloom}.

Style guide:
- Begin with ONE verb: Compare, Contrast, Differentiate, Evaluate, or Justify
- Pick ONE verb only. Do not list multiple verbs.
- Draw TWO specific concepts from the source material and ask the student to compare them
- The question must be complete and end with a period or question mark
- Do NOT reference the source document. Write as a clean exam question.
- Do NOT mention any author names, researcher names, or person names.
- Do NOT reference any Figure, Table, Diagram, Program, or Listing numbers.
- Do NOT mention any book titles, chapter numbers, or publication names.
- Write a clean, self-contained exam question with zero external references.
- Output ONLY the raw question text. No labels, no notes, no answer.

Question:""",

    # Template C — Multi-part structured
    """\
You are an experienced VTU university examiner.

SOURCE MATERIAL:
\"\"\"{chunk}\"\"\"

Create ONE structured VTU exam question worth {marks} marks at Bloom level {bloom}.

Use this EXACT multi-part format and fill in the blanks with real concepts from the source:
(1) Explain [first concept] and its role in [domain].
(2) Compare [first concept] with [second concept].
(3) Derive or Justify [a key result or application].

Rules:
- Replace the bracketed placeholders with actual concepts from the source material
- Each sub-part must be a complete grammatical sentence
- Do NOT write "as per the source", "from the material", or any reference to the document
- Do NOT mention any author names, researcher names, or person names.
- Do NOT reference any Figure, Table, Diagram, Program, or Listing numbers.
- Do NOT mention any book titles, chapter numbers, or publication names.
- Write a clean, self-contained exam question with zero external references.
- Output ONLY the question. No answer, no note, no preamble.

Question:""",

    # Template D — Application
    """\
You are a VTU question paper setter focused on applied engineering concepts.

SOURCE MATERIAL:
\"\"\"{chunk}\"\"\"

Formulate ONE VTU-style exam question worth {marks} marks at Bloom level {bloom}.

Focus on APPLICATION — begin with ONE of these verbs:
Illustrate, Design, Construct, Apply, or Examine

Rules:
- Pick ONE verb only. Do not list multiple verbs.
- The question must ask the student to apply a concept to a real scenario
  OR illustrate with a worked example
  OR design a solution using a concept from the source
- Do NOT reference "the source material", "the document", "the text", "the notes"
- Do NOT mention any author names, researcher names, or person names.
- Do NOT reference any Figure, Table, Diagram, Program, or Listing numbers.
- Do NOT mention any book titles, chapter numbers, or publication names.
- Write a clean, self-contained exam question with zero external references.
- The question must be complete. End with a period or question mark.
- Output ONLY the question. No answer, no prefix, no note.

Question:""",

    # Template E — Critical analysis
    """\
You are a senior VTU examiner writing higher-order thinking questions.

SOURCE MATERIAL:
\"\"\"{chunk}\"\"\"

Write ONE critical-analysis VTU exam question worth {marks} marks at Bloom level {bloom}.

Rules:
- Begin with ONE verb only: Analyse, Evaluate, Examine, Justify, or Assess
- Pick ONE verb. Do not write a list of verbs like "Analyse, Evaluate, Critically examine..."
- Challenge the student to think beyond memorization
- Be specific to the concepts in the source material
- Do NOT reference "the source", "the material", "the document", "the text"
- Do NOT mention any author names, researcher names, or person names.
- Do NOT reference any Figure, Table, Diagram, Program, or Listing numbers.
- Do NOT mention any book titles, chapter numbers, or publication names.
- Write a clean, self-contained exam question with zero external references.
- Write as a clean standalone exam question
- End the question with a period or question mark
- Output ONLY the question text. Nothing else.

Question:""",
]

_TURBO_STOP = [
    "Ideal Answer",
    "Ideal answer",
    "ideal answer",
    "Marking Scheme",
    "marking scheme",
    "Note:",
    "note:",
    "Answer:",
    "Explanation:",
    "Here is",
    "Here's",
    "here is",
    "here's",
    "Q2)",
    "Q2.",
    "Q3)",
    "---",
    "===",
    "```",
    # Source reference stops
    "as described in the source",
    "as outlined in the source",
    "as mentioned in the source",
    "as per the source",
    "from the source material",
    "in the source material",
    "as provided in the source",
    "as given in the source",
    "refer to the source",
    "as per the material",
    "from the material",
    # Leaked verb list stops
    "Analyse, Evaluate",
    "Explain, Compare, Derive",
    "Critically examine",
    ", Evaluate,",
    ", Critically",
    # Author/figure stops
    "as described by",
    "according to",
    "Figure ",
    "Fig. ",
    "Table ",
    "Program ",
    "Listing ",
    "as shown in Figure",
    "as described in Figure",
]


# ─────────────────────────────────────────────────────────────
# Turbo post-cleaner
# ─────────────────────────────────────────────────────────────

def _post_clean(text: str) -> str:
    """
    Cleans turbo output.
    Never drops sentences — only removes artifacts and fixes punctuation.
    """
    t = text.strip()

    # ── Cut at answer/note markers ────────────────────────────
    cut_patterns = [
        r"\n.*?Ideal Answer.*",
        r"\n.*?Marking Scheme.*",
        r"\n.*?Note\s*:.*",
        r"\n.*?Answer\s*:.*",
        r"\n.*?Explanation\s*:.*",
        r"\n.*?Here (is|are).*",
        r"\n.*?---.*",
        r"\n.*?===.*",
    ]
    for pat in cut_patterns:
        t = re.sub(pat, "", t, flags=re.S | re.I).strip()

    # ── Remove unwanted prefixes ──────────────────────────────
    prefix_patterns = [
        r"^(Here is a|Here\'s a|Question:|Q\d*[.)]\s*)",
        r"^Descriptive Exam Question\s*:?\s*",
        r"^VTU.*?Question\s*:?\s*",
        r"^\*\*.*?\*\*\s*",
    ]
    for pat in prefix_patterns:
        t = re.sub(pat, "", t, flags=re.I).strip()

    # ── FIX 1: Remove leaked prompt verb lists ────────────────
    # These appear when the model copies from the prompt examples
    leaked_verb_lists = [
        r"^Analyse,\s*Evaluate,\s*Critically examine,\s*Justify,\s*(or\s*)?Assess[,.]?\s*",
        r"^Explain,\s*Compare,\s*Derive,\s*Analyse[,.]?\s*",
        r"^Explain\s*/\s*Compare\s*/\s*Derive[,.]?\s*",
        r"^(Explain|Compare|Derive|Analyse|Illustrate|Discuss|Evaluate|Justify|Design|Examine|Interpret),\s*(Evaluate|Compare|Derive|Analyse|Illustrate|Discuss|Evaluate|Justify|Design|Examine)[,.]?\s*",
    ]
    for pat in leaked_verb_lists:
        t = re.sub(pat, "", t, flags=re.I).strip()

    # ── FIX 2: Remove source material references ──────────────
    source_ref_patterns = [
        # Inline references like "decisionss material" artifact
        r"\w+s\s+material\b",                          # broken word + material
        r",?\s*s material\b",                          # stray "s material"
        # Standard reference phrases
        r",?\s*(as described|as outlined|as mentioned|as stated|as discussed|"
        r"as given|as provided|as presented|as defined|as shown|as indicated|"
        r"as noted|as specified|as listed|as found|as contained|as included|"
        r"as covered|as elaborated|as detailed|as highlighted|as summarized)"
        r"\s+in\s+(the\s+)?(source|given|provided|above|following|attached|"
        r"reference|study|course|academic|this|that)?\s*"
        r"(material|text|passage|excerpt|content|document|textbook|notes|"
        r"reading|literature|context|information|data|details|description|"
        r"explanation|definition|concept|theory|topic|subject|chapter|"
        r"section|paragraph|book|article|paper|source|resource)",
        r",?\s*as per\s+(the\s+)?(source|material|text|notes|textbook|"
        r"document|reading|context|above|given)",
        r",?\s*from\s+(the\s+)?(source|material|text|notes|textbook|"
        r"document|reading|context)",
        r",?\s*per\s+(the\s+)?(source|material|text|notes|textbook|"
        r"document|reading|context)",
        r",?\s*\(from\s+(the\s+)?(source|material|text|notes|textbook|"
        r"document)\)",
        r",?\s*(provided|outlined|described|mentioned|given|contained|"
        r"included|covered|detailed|highlighted|summarized)\s+in\s+(the\s+)?"
        r"(source|material|text|notes|textbook|document|reading|context)",
        r",?\s*(refer\s+to|see)\s+(the\s+)?(source|material|text|notes|"
        r"textbook|document)",
        r",?\s*\(refer\s+to\s+(the\s+)?(source|material|text|notes|"
        r"textbook|document)\)",
        r",?\s*as (described|outlined|mentioned|stated|discussed|given|"
        r"provided|presented|defined|shown|indicated|noted|specified|listed|"
        r"found|contained|included|covered|elaborated|detailed|highlighted|"
        r"summarized)\s+above",
        r",?\s*in\s+(the\s+)?(source|material|text|notes|textbook|"
        r"document|reading|context|above)\s+(material|text|provided|given|"
        r"document|textbook|notes)?",
        r",?\s*based on\s+(the\s+)?(source|material|text|notes|textbook|"
        r"document|reading|context|above|given|provided)",
        r",?\s*according to\s+(the\s+)?(source|material|text|notes|"
        r"textbook|document|reading|context|above|given|provided)",
    ]
    for pat in source_ref_patterns:
        t = re.sub(pat, "", t, flags=re.I).strip()

    # ── FIX 4: Remove author, book, figure, table references ─
    academic_ref_patterns = [
        # Explicit author references: "by Donald Knuth", "according to Russell and Norvig"
        r",?\s*(by|according to|as (described|proposed|suggested|introduced|"
        r"formulated|defined|stated|shown|demonstrated|proved|discussed|"
        r"presented|outlined|explained|noted|mentioned|illustrated|developed|"
        r"designed|invented|coined|established|discovered|theorized|"
        r"hypothesized|argued|advocated|recommended|concluded) by)\s+"
        r"(Russell|Norvig|Knuth|Brooks|Turing|Dijkstra|Cormen|"
        r"Sedgewick|Tanenbaum|Stallings|Pressman|Sommerville|"
        r"Hopcroft|Ullman|Sipser|Goodfellow|Bengio|Bishop|"
        r"Mitchell|Haykin|Duda|Hart|Stork|Pearl|Nilsson|"
        r"Winston|Rich|Knight|Luger|Poole|Mackworth|"
        r"Horowitz|Sahni|Anderson|Rajaraman|Shalev-Shwartz|"
        r"Yegnanarayana|Kanetkar|Balagurusamy|Reema|Lipschutz|"
        r"[A-Z][a-z]+\s+[A-Z][a-z]+)\b",

        # Possessive known author: "Russell & Norvig's", "Knuth's", "Turing's"
        r"\b(Russell|Norvig|Knuth|Brooks|Turing|Dijkstra|Cormen|"
        r"Sedgewick|Tanenbaum|Stallings|Pressman|Sommerville|"
        r"Hopcroft|Ullman|Sipser|Goodfellow|Bengio|Bishop|"
        r"Mitchell|Haykin|Duda|Hart|Stork|Pearl|Nilsson|"
        r"Winston|Rich|Knight|Luger|Poole|Mackworth|"
        r"Horowitz|Sahni|Anderson|Rajaraman|Shalev-Shwartz|"
        r"Yegnanarayana|Kanetkar|Balagurusamy|Reema|Lipschutz"
        r")'s?\b",

        # Figure/Table references: "Figure 3.4", "Table 2.1", "Fig. 5"
        r",?\s*(as )?(described|shown|illustrated|depicted|presented|"
        r"given|displayed|drawn|plotted|listed|tabulated|represented|"
        r"demonstrated|explained|outlined|summarized|highlighted|"
        r"indicated|mentioned|noted|specified|defined|seen|found|"
        r"appearing|visible|observable)?\s*"
        r"(in|from|of|per|see|refer|as per|as in|as shown in|"
        r"as described in|as illustrated in|as depicted in|"
        r"as presented in|as given in|as displayed in)?\s*"
        r"(Figure|Fig\.|Fig|Table|Tab\.|Diagram|Chart|Graph|"
        r"Exhibit|Illustration|Plate|Map|Drawing|Sketch|"
        r"Plot|Panel|Box|Frame|Screen|Display|View|Window|"
        r"Appendix|App\.|Section|Sec\.|Chapter|Ch\.|"
        r"Page|Pg\.|Equation|Eq\.|Algorithm|Alg\.|"
        r"Listing|List\.|Code|Snippet|Example|Ex\.|"
        r"Problem|Prob\.|Exercise|Exer\.|Case|"
        r"Scenario|Situation|Instance)"
        r"\s*\.?\s*\d+(\.\d+)*\s*[a-d]?",

        # Standalone figure refs: "Figure 3.4", "as shown in Figure 2"
        r"\s*\(?\s*(Figure|Fig\.|Fig|Table|Tab\.|Diagram|Chart)\s*\.?\s*"
        r"\d+(\.\d+)*\s*[a-d]?\s*\)?",

        # Book/chapter references
        r",?\s*(as (described|discussed|mentioned|stated|outlined|"
        r"explained|presented|shown|noted|specified|listed|found|"
        r"contained|included|covered|detailed|highlighted|"
        r"summarized|elaborated) in)\s+(the\s+)?(book|chapter|"
        r"textbook|text|module|unit|section|paragraph|page|"
        r"lecture|slide|handout|notes|reading|reference|"
        r"publication|journal|paper|article|document|manual|"
        r"guide|handbook|encyclopedia|dictionary|glossary|"
        r"appendix|supplement|addendum|errata|corrigendum|"
        r"preface|foreword|introduction|abstract|summary|"
        r"conclusion|bibliography|index|table of contents)",

        # Program/listing references: "Program 1.22", "Listing 3.1"
        r"\s*\(?\s*(Program|Listing|Code|Snippet|Algorithm|"
        r"Procedure|Function|Routine|Subroutine|Method|Class|"
        r"Module|Package|Library|Framework|API|Interface|"
        r"Implementation|Version|Release|Build|Revision|"
        r"Edition|Volume|Issue|Number|Part|Section|Chapter|"
        r"Appendix|Exhibit)\s+\d+(\.\d+)*\s*\)?",

        # "The Mythical Man-Month", book titles in quotes
        r",?\s*(from|in|per|as per|as described in|as outlined in|"
        r"as mentioned in|as stated in|as discussed in)\s+"
        r"['\"][^'\"]+['\"]",
    ]
    for pat in academic_ref_patterns:
        t = re.sub(pat, "", t, flags=re.I).strip()

    # ── FIX 3: Clean up broken punctuation after removals ─────
    # e.g. "decisions. In your..." becomes clean after "s material" removed
    t = re.sub(r"\s{2,}", " ", t)                      # collapse double spaces
    t = re.sub(r"\s+([.,;:?!])", r"\1", t)             # remove space before punct
    t = re.sub(r"([.,;:])\s*([.,;:])", r"\1", t)       # remove double punctuation
    t = re.sub(r"\n{3,}", "\n\n", t)                   # collapse excess newlines

    # ── FIX 5: Clean orphaned fragments after removal ─────────
    # Fix "as described  and discuss" → "and discuss"
    t = re.sub(r"\s{2,}", " ", t)
    # Fix leading conjunctions after removal: "and discuss..." → "Discuss..."
    t = re.sub(r"^\s*(and|or|but|also|moreover|furthermore|additionally|"
               r"however|therefore|hence|thus|consequently|meanwhile|"
               r"subsequently|finally|lastly|firstly|secondly)\s+",
               "", t, flags=re.I).strip()
    # Fix "described. How" → "How"
    t = re.sub(r"^(described|outlined|mentioned|stated|discussed|"
               r"given|provided|presented|defined|shown|indicated|"
               r"noted|specified|listed|found|contained|included|"
               r"covered|elaborated|detailed|highlighted|summarized)"
               r"\s*[.,]?\s*", "", t, flags=re.I).strip()
    # Capitalize first letter after cleanup
    if t and t[0].islower():
        t = t[0].upper() + t[1:]

    # ── Remove markdown ───────────────────────────────────────
    t = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", t)
    t = re.sub(r"\*+", "", t)

    # ── Remove trailing mark annotations ─────────────────────
    t = re.sub(r"\s*\[\s*\d+\s*[Mm]arks?\s*\]\s*$", "", t)
    t = re.sub(r"\s*\(\s*\d+\s*[Mm]arks?\s*\)\s*$", "", t)

    # ── Fix terminal punctuation ──────────────────────────────
    t = t.strip().rstrip(",:;")
    if t and t[-1] not in ".?!":
        if re.match(r"^(what|how|why|when|where|which|who)", t, re.I):
            t += "?"
        else:
            t += "."

    # ── Minimum length guard ──────────────────────────────────
    if len(t.split()) < 8:
        t += " Explain its significance with a suitable example."

    return t


# ─────────────────────────────────────────────────────────────
# Domain guard — rejects hallucinated off-topic questions
# ─────────────────────────────────────────────────────────────

# Terms that should NEVER appear in a CS/software engineering paper
_OFF_DOMAIN_TERMS = re.compile(
    r"\b("
    r"transformer|resistor|capacitor|inductor|voltage|current|watt|ampere|"
    r"kva|kw|kwh|ohm|circuit|diode|transistor|rectifier|inverter|"
    r"mechanical|thermodynamic|fluid|hydraulic|pneumatic|"
    r"chemistry|biology|anatomy|physiology|"
    r"balance sheet|ledger|depreciation|amortization|"
    r"lathe|mill|drill|weld|forge|cast"
    r")\b",
    re.I,
)

# Terms that MUST appear in a CS/SE question to be accepted
_CS_DOMAIN_TERMS = re.compile(
    r"\b("
    r"algorithm|data|program|software|system|function|recursion|"
    r"complexity|analysis|design|code|memory|variable|type|struct|"
    r"array|tree|graph|search|sort|loop|class|object|method|"
    r"compiler|runtime|stack|queue|pointer|file|process|thread|"
    r"network|database|query|schema|index|key|hash|bit|byte|"
    r"iteration|specification|requirement|module|interface|"
    r"abstraction|inheritance|polymorphism|encapsulation"
    r")\b",
    re.I,
)

def _is_valid_cs_question(question: str) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).
    Rejects questions from wrong academic domain.
    """
    if _OFF_DOMAIN_TERMS.search(question):
        match = _OFF_DOMAIN_TERMS.search(question).group()
        return False, f"off_domain_term:'{match}'"

    if not _CS_DOMAIN_TERMS.search(question):
        return False, "no_cs_domain_terms_found"

    return True, "ok"


# ─────────────────────────────────────────────────────────────
# VTU Mark & Bloom Taxonomy Mapper
# ─────────────────────────────────────────────────────────────

def _assign_marks_and_bloom(question_text: str) -> tuple:
    """
    Dynamically maps question complexity to VTU standard marks and Bloom level.
    
    VTU Standard Mark Patterns:
        2  marks — Definition / Single concept (Bloom 1: Remember)
        5  marks — Explanation with example (Bloom 2: Understand)
        8  marks — Compare / Contrast / Application (Bloom 3: Apply)
        10 marks — Multi-part / Design / Derive (Bloom 4: Analyze)
        15 marks — Critical analysis / Case study (Bloom 5: Evaluate)
        20 marks — Comprehensive / Research-level (Bloom 6: Create)
    
    Bloom Taxonomy Levels:
        1 — Remember (define, list, state, identify)
        2 — Understand (explain, describe, discuss, summarize)
        3 — Apply (illustrate, demonstrate, apply, solve, calculate)
        4 — Analyze (analyse, compare, contrast, differentiate, derive)
        5 — Evaluate (evaluate, assess, justify, critique, recommend)
        6 — Create (design, construct, develop, formulate, propose)
    
    Returns:
        (marks: int, bloom_level: int)
    """
    q = question_text.lower().strip()
    words = q.split()
    word_count = len(words)

    # ── Complexity scoring ────────────────────────────────────
    complexity_score = 0

    # ── BLOOM 6: Create (20 marks) ────────────────────────────
    create_indicators = [
        "design and implement", "develop a complete", "construct a system",
        "formulate a comprehensive", "propose a novel", "create a framework",
        "build a prototype", "architect a solution", "engineer a",
        "develop an algorithm that", "design a complete system",
    ]
    if any(x in q for x in create_indicators):
        complexity_score += 25

    # ── BLOOM 5: Evaluate (15 marks) ──────────────────────────
    evaluate_indicators = [
        "critically evaluate", "critically examine", "critically analyse",
        "assess the effectiveness", "justify your answer with",
        "evaluate the advantages and disadvantages",
        "assess the implications", "critique the approach",
        "recommend with justification", "evaluate and recommend",
        "assess the performance", "justify with suitable reasoning",
    ]
    if any(x in q for x in evaluate_indicators):
        complexity_score += 18

    # ── BLOOM 4: Analyze (10 marks) ───────────────────────────
    analyze_indicators = [
        "(1)", "(2)", "(3)",                    # Multi-part questions
        "compare and contrast", "differentiate between",
        "derive the expression", "derive the formula",
        "derive or justify", "analyse and compare",
        "analyse the role", "analyse the impact",
        "analyse how", "design an algorithm",
        "step-by-step example", "with a worked example",
        "design a scenario", "analyse the significance",
        "derive the time complexity", "derive the relationship",
    ]
    if any(x in q for x in analyze_indicators):
        complexity_score += 12

    # ── BLOOM 3: Apply (8 marks) ──────────────────────────────
    apply_indicators = [
        "illustrate with", "with suitable example",
        "with an example", "apply the concept",
        "solve the following", "calculate the",
        "demonstrate how", "show how",
        "implement the", "use the algorithm",
        "apply this to", "illustrate your solution",
        "illustrate this by", "work out",
    ]
    if any(x in q for x in apply_indicators):
        complexity_score += 8

    # ── BLOOM 2: Understand (5 marks) ─────────────────────────
    understand_indicators = [
        "explain the concept", "explain the significance",
        "explain the role", "explain how",
        "describe the", "discuss the",
        "summarize the", "outline the",
        "what is meant by", "what are the",
        "explain in detail", "discuss in brief",
    ]
    if any(x in q for x in understand_indicators):
        complexity_score += 5

    # ── BLOOM 1: Remember (2 marks) ───────────────────────────
    remember_indicators = [
        "define ", "list the", "state the",
        "identify the", "name the", "mention the",
        "what is ", "write the definition",
    ]
    # Only match if it is a SHORT question (< 20 words)
    if word_count < 20 and any(x in q for x in remember_indicators):
        complexity_score += 2

    # ── Word count bonus ──────────────────────────────────────
    # Longer questions tend to be more complex
    if word_count > 80:
        complexity_score += 5
    elif word_count > 50:
        complexity_score += 3
    elif word_count > 30:
        complexity_score += 1

    # ── Verb-based scoring (first word analysis) ──────────────
    first_word = words[0] if words else ""

    bloom_6_verbs = ["design", "construct", "develop", "formulate", "propose", "create", "build", "architect"]
    bloom_5_verbs = ["evaluate", "assess", "justify", "critique", "recommend", "judge", "defend"]
    bloom_4_verbs = ["analyse", "analyze", "compare", "contrast", "differentiate", "derive", "examine", "investigate"]
    bloom_3_verbs = ["apply", "illustrate", "demonstrate", "solve", "calculate", "implement", "use", "show"]
    bloom_2_verbs = ["explain", "describe", "discuss", "summarize", "outline", "classify", "interpret"]
    bloom_1_verbs = ["define", "list", "state", "identify", "name", "recall", "recognize", "label"]

    if first_word in bloom_6_verbs:
        complexity_score += 8
    elif first_word in bloom_5_verbs:
        complexity_score += 6
    elif first_word in bloom_4_verbs:
        complexity_score += 4
    elif first_word in bloom_3_verbs:
        complexity_score += 3
    elif first_word in bloom_2_verbs:
        complexity_score += 2
    elif first_word in bloom_1_verbs:
        complexity_score += 1

    # ── Final mapping to VTU standard marks ───────────────────
    if complexity_score >= 25:
        return 20, 6    # 20 Marks | Bloom 6: Create
    elif complexity_score >= 18:
        return 15, 5    # 15 Marks | Bloom 5: Evaluate
    elif complexity_score >= 12:
        return 10, 4    # 10 Marks | Bloom 4: Analyze
    elif complexity_score >= 8:
        return 8, 3     # 8 Marks  | Bloom 3: Apply
    elif complexity_score >= 4:
        return 5, 2     # 5 Marks  | Bloom 2: Understand
    else:
        return 2, 1     # 2 Marks  | Bloom 1: Remember


# ─────────────────────────────────────────────────────────────
# Turbo generator
# ─────────────────────────────────────────────────────────────

def generate_turbo(
    concept,
    marks:       int   = 5,
    temperature: float = None,   # None = randomize per call
) -> GeneratedQuestion:
    """
    Turbo mode — question only, no answer, no critic.

    Diversity mechanisms:
      1. Random prompt template selected per call
      2. Temperature randomized per call (0.60 – 0.85)
      3. Random concept offset injected into chunk selection
      4. num_predict scales dynamically with marks
    """

    # ── Extract concept fields ────────────────────────────────
    if isinstance(concept, dict):
        chunk      = concept.get("content") or concept.get("text") or ""
        bloom      = concept.get("bloom_level") or concept.get("bloom_dna") or 2
        concept_id = concept.get("concept_id", "unknown")
    else:
        chunk      = getattr(concept, "content", "") or getattr(concept, "text", "")
        bloom      = getattr(concept, "bloom_dna", None) or getattr(concept, "bloom_level", 2) or 2
        concept_id = getattr(concept, "concept_id", "unknown")

    # ── Dynamic context window ────────────────────────────────
    # Give the model more material for higher-mark questions
    context_limit = {
        2:  800,
        5:  1500,
        10: 2200,
        20: 3000,
    }.get(marks, 1500)

    chunk = chunk[:context_limit].strip()

    # ── Dynamic num_predict (output length) ───────────────────
    # More marks = longer, richer question allowed
    num_predict = {
        2:  120,    # short single question
        5:  280,    # rich single question
        10: 480,    # multi-part (1)(2)(3) question
        20: 700,    # full structured question
    }.get(marks, 280)

    # ── Randomize temperature for diversity ───────────────────
    # Different every run so model explores different phrasings
    if temperature is None:
        temperature = round(random.uniform(0.60, 0.85), 2)

    # ── Pick a random prompt template ─────────────────────────
    prompt_template = random.choice(_TURBO_PROMPTS)
    prompt = prompt_template.format(chunk=chunk, marks=marks, bloom=bloom)

    model = os.environ.get("AION_MODEL", "qwen2.5:3b")

    options = {
        "temperature":    temperature,
        "num_predict":    num_predict,
        "num_ctx":        4096,
        "top_p":          round(random.uniform(0.88, 0.95), 2),  # slight variation
        "repeat_penalty": 1.12,
        "stop":           _TURBO_STOP,
        "seed":           int(time.time() * 1000) % 2**31,       # true randomness
        "keep_alive":     "30m",
    }

    raw = ""

    # ── Try get_llm() first ───────────────────────────────────
    try:
        raw = get_llm().generate(prompt, temperature=temperature, options=options)
    except Exception:
        pass

    # ── Ollama package fallback ───────────────────────────────
    if not raw:
        try:
            import ollama
            response = ollama.generate(
                model   = model,
                prompt  = prompt,
                options = options,
            )
            raw = response.get("response", "").strip()
        except Exception:
            pass

    # ── HTTP fallback ─────────────────────────────────────────
    if not raw:
        try:
            import requests
            host    = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            payload = {
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": options,
            }
            r   = requests.post(f"{host}/api/generate", json=payload, timeout=60)
            r.raise_for_status()
            raw = r.json().get("response", "").strip()
        except Exception:
            pass

    # ── Last resort ───────────────────────────────────────────
    if not raw:
        sentences = re.split(r"(?<=[.?!])\s+", chunk)
        stem      = sentences[0][:80] if sentences else chunk[:80]
        raw       = f"Explain the concept of {stem} and discuss its significance in the context of the subject."

    question_text = _post_clean(raw)

    # ── Domain validation ─────────────────────────────────────
    domain_ok, domain_reason = _is_valid_cs_question(question_text)

    if not domain_ok:
        # Retry once with a different template and lower temperature
        prompt_template = random.choice(_TURBO_PROMPTS)
        prompt          = prompt_template.format(chunk=chunk, marks=marks, bloom=bloom)
        options["temperature"] = 0.55
        options["seed"]        = int(time.time() * 1000) % 2**31

        raw = ""
        try:
            raw = get_llm().generate(prompt, temperature=0.55, options=options)
        except Exception:
            pass

        if not raw:
            try:
                import ollama
                response = ollama.generate(model=model, prompt=prompt, options=options)
                raw      = response.get("response", "").strip()
            except Exception:
                pass

        if raw:
            question_text  = _post_clean(raw)

    marks, bloom = _assign_marks_and_bloom(question_text)

    return GeneratedQuestion(
        concept_id    = concept_id,
        ideal_answer  = None,
        question_text = question_text,
        marks         = marks,
        bloom_level   = bloom,
    )


# ─────────────────────────────────────────────────────────────
# BALANCED / DEEP MODE  — RAG² Answer-First Pipeline
# ─────────────────────────────────────────────────────────────

def generate(concept: Concept, mode: str = "balanced") -> GeneratedQuestion:
    """
    RAG² enforced for balanced & deep modes.
    Turbo bypasses answer-first generation and LLM critic entirely.

    Modes:
      turbo    — question only, ~3-5 s/question
      balanced — 10 marks, 250-300 words
      deep     — 20 marks, 400-500 words
    """
    mode = mode.lower().strip()

    # ── Turbo: delegate immediately ───────────────────────────
    if mode == "turbo":
        return generate_turbo(concept, marks=5)

    # ── Mode config ───────────────────────────────────────────
    if mode == "deep":
        marks       = 20
        max_tokens  = 500
        word_limit  = "400–500 words"
        extra       = (
            "Include:\n"
            "- Clear definition\n"
            "- Explanation\n"
            "- Advantages and disadvantages\n"
            "- Comparison (if applicable)\n"
            "- Conclusion"
        )
    else:
        mode        = "balanced"
        marks       = 10
        max_tokens  = 300
        word_limit  = "250–300 words"
        extra       = (
            "Include definition, key points, and brief conclusion.\n"
            "No extra commentary. Exam-ready structured format."
        )

    # ── Step 0: Validate & clean concept prose ────────────────
    cleaned_content = clean_chunk(concept.content)
    quality         = validate_chunk(cleaned_content)

    if not quality.is_valid:
        return GeneratedQuestion(
            concept_id    = concept.concept_id,
            ideal_answer  = "[SKIPPED: Non-academic code or noise fragment]",
            question_text = f"[INVALID CONCEPT SKIPPED: {quality.reason}]",
            marks         = 0,
            bloom_level   = 0,
        )

    clean_snippet = re.sub(r"https?://\S+", "", cleaned_content)
    clean_snippet = re.sub(
        r"\b[\w_\-]+\.(py|html|js|css|json|yaml)\b", "", clean_snippet
    )
    clean_snippet = clean_snippet.strip().rstrip(".")

    # ── Step 1: Ideal Answer (Target Generation) ──────────────
    answer_prompt = f"""You are an academic exam expert for VTU engineering.
Generate a VTU-style descriptive answer.
Marks: {marks}
Word limit: {word_limit}
{extra}

Based ONLY on this academic concept:
{clean_snippet}

Rules:
- Focus ONLY on academic concepts, definitions, and theory.
- Do NOT generate answers about code syntax, file paths, or variable names.

Ideal Answer:"""

    ideal_answer = _run_prompt(answer_prompt, max_tokens=max_tokens)

    if not ideal_answer:
        sentences = re.split(r"(?<=[.?!])\s+", clean_snippet)
        key_points = [s.strip() for s in sentences[:5] if len(s.strip()) > 20]
        if key_points:
            ideal_answer = "Key points:\n" + "\n".join(f"- {kp}" for kp in key_points)
        else:
            ideal_answer = f"The concept relates to: {clean_snippet[:300]}"

    # ── Step 2: Reverse-generate the Question ─────────────────
    question_prompt = f"""You are a VTU university examiner.
Write ONE descriptive exam question using VTU command verbs.
(Explain / Define / Derive / Compare / Analyse / Illustrate / Discuss)
Target Marks: {marks}

Ideal Answer:
{ideal_answer}

Rules:
- NO questions about code syntax, file names, or variable names.
- Focus on academic concept understanding.
- Output ONLY the question. No prefix, no note, no answer.

Exam Question:"""

    question_text = _run_prompt(question_prompt, max_tokens=150)

    if not question_text:
        snippet = clean_snippet[:80]
        if "defined as" in clean_snippet.lower() or "is a" in clean_snippet.lower():
            question_text = (
                f"Define and explain the concept of '{snippet}'. "
                f"What are its key characteristics and applications?"
            )
        elif "algorithm" in clean_snippet.lower() or "method" in clean_snippet.lower():
            question_text = (
                f"Explain the working of '{snippet}' with suitable examples. "
                f"What are its advantages and limitations?"
            )
        elif "theorem" in clean_snippet.lower() or "law" in clean_snippet.lower():
            question_text = (
                f"State and explain '{snippet}'. "
                f"Derive the key result and discuss its significance."
            )
        else:
            question_text = (
                f"Explain in detail: '{snippet}'. "
                f"Discuss its significance in the context of the subject."
            )

    if not question_text.strip().endswith("?"):
        question_text = question_text.strip() + "?"

    marks, bloom_level = _assign_marks_and_bloom(question_text)

    return GeneratedQuestion(
        concept_id    = concept.concept_id,
        ideal_answer  = ideal_answer,
        question_text = question_text,
        marks         = marks,
        bloom_level   = bloom_level,
    )
