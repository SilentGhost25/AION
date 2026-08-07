# Exam Paper Generator — Backend Architecture Blueprint

> **Status:** Planning / blueprint only. No code written yet.
> **Target hardware:** NVIDIA L40 (48 GB VRAM), 60 GB RAM, LAN-only (college project, not global scale).
> **Hard budget:** upload → parse → OCR → embed → store → generate a paper, all within **5–6 minutes**.
> **Model constraint:** ≤ 16B parameters (chosen: Qwen2.5-14B-Instruct).

---

## 0. Guiding Principles (the "logic" carried over from the demo)

The demo questions felt authentic because they followed hard rules. The backend encodes these same rules as **prompt constraints + schema enforcement + validators**, not as free-form generation:

1. **Marks distribution** uses only university-realistic splits (5+5, 6+4, 8+2 for 10-mark; 6+8+6, 7+7+6 for 20-mark). Never arbitrary (3+7, 1+9).
2. **CO mapping** is driven by module/syllabus position (Q1-2 → CO1, Q3-4 → CO2, …).
3. **Bloom's (RBT) → verb mapping** is a lookup table. Marks scale with cognitive level.
4. **Sub-question progression** builds up within a block: (a) define/explain → (b) apply/implement → (c) analyze/evaluate.
5. **Contextual specificity**: concrete examples pulled from the uploaded material (real algorithms, real data, real diagrams), never vague "explain X."
6. **Schema enforcement** (constrained decoding) makes malformed output *impossible* — marks that don't sum, missing CO/RBT, invalid Bloom's levels can't be emitted.

These six rules are the backbone. Everything below is infrastructure to serve them reliably within the time budget.

---

## 1. Technology Stack

| Layer | Choice | Why |
|---|---|---|
| **Web framework** | FastAPI + Uvicorn | Pydantic mirrors the existing Zod/OpenAPI contract 1:1; async; auto OpenAPI docs |
| **LLM serving** | Ollama + custom Modelfile | Easiest to stand up; Modelfile is tunable (SYSTEM/PARAMETER); user wants to edit Modelfiles |
| **Model** | Qwen2.5-14B-Instruct (Q5_K_M GGUF, ~11 GB) | ≤16B, strong instruction-following + structured output; leaves VRAM for OCR/embed/rerank |
| **Structured output** | Ollama `format` = full JSON Schema | Token-level schema enforcement via XGrammar (NOT plain `format:json` which is syntax-only) |
| **Document parsing** | Docling | Auto-skips OCR on native-text PDFs; unified DoclingDocument JSON with per-element bbox + page provenance |
| **OCR (scanned)** | EasyOCR on GPU (Docling default) | GPU-native, ~4–11 pages/sec on L40 → fits budget; one-line switch to Nemotron/RapidOCR later |
| **DOCX** | Docling native + `zipfile` fallback | Native for text/raster; zipfile reads `word/media/` for SmartArt/DrawingML |
| **Figure extraction** | PyMuPDF `get_images()` + `get_image_rects()` | Pull embedded diagrams with bounding boxes to REUSE originals |
| **Embeddings** | BGE-M3 | Multilingual, strong retrieval, dense+sparse; for RAG grounding + dedup |
| **Reranking** | BGE-reranker-v2-m3 | High-ROI second-stage precision boost |
| **Vector store** | SQLite + sqlite-vec | Zero-ops, single-file, perfect for LAN/college scale; stores chunks + embeddings + question cache |
| **Faithfulness gate** | LettuceDetect (ModernBERT) | Catch hallucinations — question must be grounded in uploaded content |
| **Diagram generation** | LLM JSON-spec → networkx/matplotlib/graphviz | For diagram *variations* (vs. reusing originals) |

---

## 2. Where It Lives (Monorepo Integration)

New peer artifact, replacing the Express `api-server`:

```
E:\Exam-Paper-Generator\
├── artifacts/
│   ├── api-server/            # EXISTING Express+OpenAI backend → RETIRE after cutover
│   ├── api-server-py/         # ← NEW Python/FastAPI local-LLM backend
│   │   ├── .replit-artifact/artifact.toml   # kind="api", localPort=8080, paths=["/api"], uvicorn run cmd
│   │   ├── pyproject.toml                    # fastapi, uvicorn, ollama, docling, sentence-transformers, sqlite-vec, pymupdf, pydantic
│   │   ├── Modelfile                         # Qwen2.5-14B custom persona + params
│   │   ├── app/
│   │   │   ├── main.py                       # FastAPI app, CORS (allow LAN origin), mount /api router
│   │   │   ├── config.py                     # settings: OLLAMA_URL, model name, thresholds, paths
│   │   │   ├── routers/
│   │   │   │   ├── health.py                 # GET /api/healthz
│   │   │   │   ├── ingest.py                 # POST /api/ingest  (parse+OCR+embed+store)
│   │   │   │   ├── generate.py               # POST /api/generate/block, POST /api/paper/generate
│   │   │   │   └── regenerate.py             # POST /api/regenerate/sub
│   │   │   ├── schemas.py                    # Pydantic mirrors of openapi.yaml + new sub-question fields
│   │   │   ├── services/
│   │   │   │   ├── parser.py                 # Docling orchestration, OCR toggle, figure extraction
│   │   │   │   ├── embedder.py               # BGE-M3 embed + BGE-reranker
│   │   │   │   ├── retriever.py              # RAG: query → top-K chunks (per module)
│   │   │   │   ├── generator.py              # prompt build + Ollama call + schema enforce
│   │   │   │   ├── validator.py              # marks-sum, CO, RBT, Bloom's checks
│   │   │   │   ├── dedup.py                  # cosine-sim vs question cache
│   │   │   │   ├── faithfulness.py           # LettuceDetect gate
│   │   │   │   └── diagrams.py               # reuse originals + JSON-spec render variations
│   │   │   ├── prompts/
│   │   │   │   ├── system.txt                # DSATM/VTU persona
│   │   │   │   ├── block_generate.jinja      # per-block generation template
│   │   │   │   └── verbs.py                  # Bloom's L1-L6 → verb lists + marks ranges
│   │   │   └── db/
│   │   │       ├── store.py                  # sqlite-vec init, migrations
│   │   │       └── models.py                 # documents, chunks, figures, question_cache tables
│   │   └── data/                             # sqlite db file, uploaded originals, extracted figures
```

**Deployment decision:** The Python backend takes over `/api` on port 8080. The Express `api-server` is retired (or kept on a different port for reference during cutover). Only one artifact owns `paths=["/api"]` in a deployment.

---

## 3. API Contract (extends the existing spec)

The current `openapi.yaml` only has `GET /healthz` + `POST /paper/generate` with a **flat** `GeneratedQuestion`. We extend the spec (keeping it the single source of truth so Zod + react-query regenerate) with:

### 3.1 New/extended schemas
```yaml
SubQuestionOut:
  label: string          # a, b, c
  text: string
  marks: integer
  co: string             # CO1..CO5
  rbt: string            # L1..L6
  parts: [string]        # nested (i)(ii)(iii), optional, no marks
  figures: [FigureRef]   # optional attached diagrams

FigureRef:
  kind: "original" | "generated"
  url: string            # /api/figures/{id}.png  OR data URL
  caption: string

GeneratedQuestion:       # EXTENDED — add:
  subQuestions: [SubQuestionOut]   # the real per-row data
  # (existing flat fields kept as fallback / base-schema compatibility)
```

### 3.2 Endpoints

| Method | Path | Purpose | Timing |
|---|---|---|---|
| `GET` | `/api/healthz` | liveness + model-loaded status | instant |
| `POST` | `/api/ingest` | one call per uploaded file: parse → OCR (if scanned) → extract figures → chunk → embed → store. Returns `{docId, pageCount, ocrUsed, figureCount, elapsedMs}` | bulk of budget |
| `POST` | `/api/generate/block` | **per-block on-demand** (matches current UI): `{docId, sectionNumber, blockMarks, numSubs, examType, module}` → one block with schema-enforced `subQuestions[]` | ~5–15s |
| `POST` | `/api/regenerate/sub` | single sub-question re-roll (the ↻ button): `{docId, blockContext, subLabel, marks, targetRbt, avoid:[cachedTexts]}` → one `SubQuestionOut` | ~3–8s |
| `POST` | `/api/paper/generate` | orchestrates all 10 blocks (kept for contract compat / one-shot mode) | within budget |
| `GET` | `/api/figures/{id}` | serve an extracted/generated figure image | instant |

**Backwards-compat:** `/api/paper/generate` keeps its `PaperInput → GeneratedPaper` shape; the frontend can migrate incrementally from client-side fake generation to real per-block calls.

---

## 4. Data Flow

### 4.1 Ingest (once per uploaded document — the slow path)
```
Upload (scanned PDF / DOCX / typed PDF)
   │
   ▼
Docling parse ──► native text? ──► skip OCR (fast, ~thousands pg/min)
   │                   │
   │                   └─► scanned? ──► EasyOCR-GPU (~4-11 pg/sec)
   ▼
Figure extraction (PyMuPDF get_images + get_image_rects) ──► store originals + bbox + page
   ▼
Chunking (layout-aware, ~512 tokens, keep module/section metadata)
   ▼
BGE-M3 embed each chunk ──► store {chunk, embedding, module, page, docId} in sqlite-vec
   ▼
Return {docId, pageCount, ocrUsed, figureCount, elapsedMs}
```

### 4.2 Per-block generation (the fast, on-demand path)
```
Frontend: "Generate Questions" for block N
   │  {docId, sectionNumber, blockMarks=10, numSubs=2, examType, module}
   ▼
Retriever: RAG query for module N ──► top-K chunks ──► BGE-rerank ──► grounded context
   ▼
Planner (rule-based, no LLM):
   • marks split (pick valid pattern: 5+5 / 6+4 / 8+2)
   • RBT progression ([L2,L3] IAT / [L1,L3,L4] SEE)
   • CO = CO{module}
   • per-sub verb list from Bloom's table
   ▼
Prompt build (Jinja): persona + grounded context + per-sub instructions + JSON schema
   ▼
Ollama call (Qwen2.5-14B, format=JSON Schema) ──► schema-enforced subQuestions[]
   ▼
Validators:
   • sum(marks) == blockMarks   (regenerate if fail)
   • co matches, rbt in L1-L6
   ▼
Dedup: embed each question, cosine-sim vs question_cache
   • > 0.85 similar? ──► regenerate with avoid-list + temp bump
   ▼
Faithfulness gate (LettuceDetect): question grounded in context?
   • hallucinated? ──► regenerate or flag
   ▼
Diagram step (if question needs a figure):
   • reuse: attach matching original from figures table
   • variation: LLM JSON-spec ──► networkx/matplotlib render
   ▼
Store accepted questions in question_cache (embedding + text + docId)
   ▼
Return block to frontend
```

---

## 5. The Generation Prompt (encodes the demo logic)

`prompts/block_generate.jinja`:
```
System: You are an expert VTU question paper setter for {{subject}} at DSATM.

Block config:
- Total marks: {{block_marks}}
- Sub-questions: {{num_subs}}, labeled (a),(b),(c)...
- Marks per sub: {{marks_split}}          # e.g. [6,4]
- Module {{module}} → assign CO{{module}} to all subs
- Exam: {{exam_type}}  ({{ 'deeper, application-heavy' if exam_type=='SEE' else 'concept-check' }})

Reference material (use CONCRETE examples from this — real algorithms, data, diagrams):
{{grounded_context}}

Per sub-question instructions:
{% for s in subs %}
({{s.label}}) {{s.marks}} marks, Bloom's {{s.rbt}} — use verbs like: {{s.verbs}}.
   Example intent: {{s.hint}}
{% endfor %}

Rules:
- Each sub-question is complete and standalone.
- Add specificity: "with a neat diagram", "trace step-by-step", "show intermediate results".
- Do NOT invent facts not in the reference material.
- Avoid these (already used): {{avoid_list}}

Output ONLY valid JSON matching the provided schema.
```

`prompts/verbs.py` (Bloom's table):
```python
BLOOMS = {
  "L1": {"verbs": ["Define","List","State","Identify","Recall"],       "marks": (2,6)},
  "L2": {"verbs": ["Explain","Describe","Discuss","Illustrate","Summarize"], "marks": (4,8)},
  "L3": {"verbs": ["Implement","Apply","Trace","Construct","Solve","Demonstrate"], "marks": (6,10)},
  "L4": {"verbs": ["Analyze","Compare","Differentiate","Examine"],     "marks": (8,14)},
  "L5": {"verbs": ["Evaluate","Justify","Assess","Critique","Recommend"], "marks": (10,20)},
  "L6": {"verbs": ["Design","Create","Formulate","Develop"],           "marks": (12,20)},
}
```

### JSON Schema passed to Ollama (`format`)
```json
{
  "type": "object",
  "properties": {
    "subQuestions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "label": {"type": "string", "pattern": "^[a-z]$"},
          "text":  {"type": "string", "minLength": 20},
          "marks": {"type": "integer", "minimum": 1},
          "co":    {"type": "string", "pattern": "^CO[1-5]$"},
          "rbt":   {"type": "string", "enum": ["L1","L2","L3","L4","L5","L6"]},
          "parts": {"type": "array", "items": {"type": "string"}},
          "needsFigure": {"type": "boolean"}
        },
        "required": ["label","text","marks","co","rbt"]
      }
    }
  },
  "required": ["subQuestions"]
}
```

---

## 6. Database Schema (SQLite + sqlite-vec)

```sql
documents(       id, filename, exam_subject, uploaded_at, page_count, ocr_used )
chunks(          id, doc_id, module, page, text, embedding BLOB )        -- vec index
figures(         id, doc_id, page, bbox, image_path, caption, phash )    -- reuse index
question_cache(  id, doc_id, sub_text, embedding BLOB, marks, co, rbt, created_at )  -- dedup
papers(          id, doc_id, config_json, created_at )                   -- history
```

- **Dedup query:** `sqlite-vec` cosine top-1 against `question_cache.embedding`; if score > 0.85 → regenerate.
- **Figure reuse:** match a question's topic embedding against `figures.caption` embeddings + perceptual hash to avoid duplicate diagrams.

---

## 7. The 5–6 Minute Budget (worst case: scanned PDFs)

| Phase | Worst-case estimate | Notes |
|---|---|---|
| Upload + Docling parse | 5–15 s | native pages nearly free |
| OCR (scanned, ~40 pg) | 10–60 s | EasyOCR-GPU @ 4 pg/s = ~10s; degraded scans slower |
| Figure extraction | 2–5 s | PyMuPDF is fast |
| Chunk + BGE-M3 embed | 10–30 s | batch on GPU |
| **Ingest subtotal** | **~30–120 s** | leaves ≥3 min for generation |
| Per-block generate ×10 | 10 × ~8–15 s = 80–150 s | can parallelize 2–3 blocks (VRAM permitting) |
| Dedup + faithfulness | +1–3 s per block | overlaps generation |
| **Total** | **~2.5–4.5 min** | within 5–6 min with margin |

**Safety levers if over budget:** batch OCR pages, cap top-K, parallelize block generation, pre-warm the model (keep-alive), skip faithfulness on regenerate.

---

## 8. VRAM Budget (48 GB L40)

| Component | VRAM |
|---|---|
| Qwen2.5-14B Q5_K_M | ~11 GB |
| KV cache (context ~8k) | ~2–4 GB |
| BGE-M3 embedder | ~2 GB |
| BGE-reranker-v2-m3 | ~2 GB |
| EasyOCR (GPU) | ~1–2 GB |
| LettuceDetect (ModernBERT) | ~1 GB |
| **Total** | **~20–22 GB** | comfortable headroom for batching/parallel blocks |

---

## 8.5. Serving the Frontend from the Linux Server (single-origin deployment)

**Requirement:** the React frontend must run on the Linux server and behave exactly as it does in dev — same layout, same interactions, same DSATM letterhead/print output — with zero code divergence between dev and server.

### Why this is already easy (verified against the current code)
- `vite.config.ts` reads `PORT` and `BASE_PATH` from env, `host: '0.0.0.0'`, and `allowedHosts: true` — so it binds on the LAN with no edits.
- `build` emits static assets to `artifacts/qp-generator/dist/public`.
- The API client (`lib/api-client-react/custom-fetch.ts`) issues **relative `/api/...` requests** by default and only prepends an absolute base if `setBaseUrl()` is called. Leaving it unset means the browser hits the *same origin* that served the page.

### The deployment model: FastAPI serves the built frontend (recommended)
Serve the compiled React bundle as static files **from the same FastAPI process** that owns `/api`. One origin, one port, no CORS, no `setBaseUrl` needed.

```
Browser ──► http://<server-lan-ip>:8080/
                  │
    ┌─────────────┴─────────────────────────────┐
    │ FastAPI (Uvicorn) on Linux server          │
    │  • GET /api/*        → backend routers      │
    │  • GET /figures/*    → served images        │
    │  • GET /*            → StaticFiles(dist/public) + SPA fallback to index.html │
    └────────────────────────────────────────────┘
```

Implementation notes for `app/main.py`:
- Mount API routers under `/api` **first** (order matters).
- Then `app.mount("/", StaticFiles(directory="dist/public", html=True))` for the SPA.
- Add an SPA catch-all: any non-`/api`, non-file route returns `index.html` (so client-side routing via `wouter` works on refresh/deep-links).
- Because page and API share the origin, `custom-fetch` relative `/api` calls resolve correctly with **no CORS config** and **no frontend changes**.

### Build & run on Linux (the exact steps the plan will script)
```bash
# 1. Build the frontend (on the server, or build elsewhere and copy dist/public)
pnpm --filter @workspace/qp-generator build      # → artifacts/qp-generator/dist/public

# 2. Point FastAPI at that dist dir (config.py: FRONTEND_DIST=../qp-generator/dist/public)

# 3. Run the backend (which now also serves the SPA)
uvicorn app.main:app --host 0.0.0.0 --port 8080
#   → open http://<server-lan-ip>:8080/ from any machine on the LAN
```

### Alternative (only if they must be separate origins)
If the frontend is ever served by a separate process (e.g. `vite preview` or nginx on a different port), then:
- Set `VITE`-time `BASE_PATH` if hosted under a sub-path.
- Call `setBaseUrl("http://<server-lan-ip>:8080")` once at app startup to point the client at the backend.
- Add the frontend origin to FastAPI CORS `allow_origins`.
This is documented as a fallback; **single-origin (FastAPI-served) is the default** because it makes the frontend "just work" identically to dev with no divergence.

### Print/PDF fidelity on the server
The DSATM letterhead, logos, and per-sub-question table rely only on CSS `print:` classes and bundled assets (already in `dist/public` after build) — nothing dev-server-specific. Served statically, `Ctrl-P` / Export PDF renders identically to local dev. Logos must be imported as bundled assets (not dev-only paths) so they resolve under the built `base` path — this is covered by the still-pending "fix wrong logos" task.

### Cross-platform note (Windows dev → Linux server)
- No OS-specific paths in the frontend; Vite handles asset URLs. The `.npmrc` `store-dir`/`package-import-method=copy` tweak was a Windows-USB workaround and is irrelevant on the Linux server (native pnpm hardlinks work fine there).
- Ollama, Docling, EasyOCR, PyTorch, sqlite-vec all run natively on Linux (in fact better — CUDA on the L40 is first-class on Linux).

---

## 9. Frontend Wiring Changes (later, after backend works)

Currently `Step2Rules.tsx` fakes generation (`setTimeout`). Migration:
1. `handleGenerateAll` (per-block button) → `POST /api/generate/block`.
2. `regenerateSub` (↻ button) → `POST /api/regenerate/sub`.
3. File upload in `QuestionCard` → `POST /api/ingest` (returns `docId`, stored per block).
4. Extend `@workspace/api-client-react` by regenerating from the updated `openapi.yaml`.
5. Keep the client-side demo path intact (Load Demo Data) so demos work without a live server.

---

## 10. Build Phases (vertical slice first)

- **Phase 1 — Vertical slice (prove the budget):** Ollama + Modelfile up; FastAPI skeleton; `/api/ingest` (Docling + EasyOCR + sqlite-vec) + `/api/generate/block` (RAG + schema-enforced generation + marks validator). Wire ONE block end-to-end from the frontend. **Measure real timing on the L40.**
- **Phase 2 — Quality gates:** dedup cache + faithfulness gate + full Bloom's/marks validators.
- **Phase 3 — Diagrams:** reuse originals (PyMuPDF) + generate variations (JSON-spec → networkx).
- **Phase 4 — Full paper + polish:** `/api/paper/generate` orchestration, SEE format, regenerate endpoint, serve the built frontend from the server (LAN).

---

## 11. Open Decisions (confirm before Phase 1)

1. **Backend location** — `artifacts/api-server-py/` in the monorepo? (recommended)
2. **Retire Express `api-server`** on cutover, or keep it parked on another port?
3. **Ollama already installed on the L40 server?** If not, Phase 1 includes install steps + a localhost stub so scaffolding can proceed before the server is wired.
4. **Extend `openapi.yaml`** with sub-question/figure fields now (so the contract round-trips rich data), or keep flat and carry richness only in the frontend for now?
```
