# AION Debug Log

Running record of changes, errors, fixes, and open items.
Update this file after every session on the server.

---

## Format

Each session entry has:
- **Date** — YYYY-MM-DD
- **Commits** — git hashes that landed
- **Changed** — what files/functions were modified
- **Errors** — bugs or failures discovered
- **Fixes** — what resolved them
- **Open** — items still needing work
- **Next session** — where to start

---

## Session: 2026-10-13 (Phase 1 Cleanup + Pipeline Fixes)

### Commits landed
- `6329ddcd5` — chore: remove redundant launchers, backup files, Windows-only scripts
- `52f321f6a` — fix(api): update Groq model IDs to current lineup
- `8fb9738f9` — fix(evaluators): correct Bloom verb and marking scheme false positives
- `bfe4fbedf` — fix(pipeline): deterministic post-processing for preamble stripping and marking rescale
- `cfd6d4bc2` — fix(pipeline): plumb evidence_summary to GeneratedQuestion and expand regex preamble strip

### Changed
- **Removed:** 11 redundant launchers, 5 .bak files, obsolete Front-End-Builder/ and mockup-sandbox/ dirs
- **Kept:** `aiq` (authoritative launcher), `start_vllm.sh`, `start_vllm.ps1`
- **`core/api/providers.py`** — Groq model IDs updated:
  - text_models: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`
  - vision_models: `qwen/qwen3.8-27b`
- **`core/generation/agents/local_evaluators.py`** — Bloom verb evaluator now:
  - Skips leading stopwords ("a", "the", "this", etc.)
  - Accepts adjacent Bloom levels (L3 accepts L2/L4 verbs)
- **`core/generation/agents/prompts.py`** — marking scheme prompt now enforces exact sum
- **`core/generation/agents/deterministic_repairs.py`** — added:
  - `repair_forbidden_preamble` (regex, case-insensitive, punctuation-flexible)
  - `repair_marking_scheme_rescale` (proportional rescale with final-entry clamp)
  - diagnostic logging in `try_deterministic_repair`
- **`core/generation/agents/writing_agent.py`** — preamble strip at generation time; evidence_summary population
- **`core/generation/agents/generated_question.py`** — added `evidence_summary: str = ""` field
- **`core/generation/agents/evaluation_prompts.py`** — `MAX_EVIDENCE_SUMMARY_CHARS` 300 → 600
- **`core/generation/agents/evaluation_agent.py`** — `[EVAL]` diagnostic logging

### Errors encountered and fixed
| Error | Root cause | Fix |
|-------|-----------|-----|
| Multi-file crash: `expected str, bytes or os.PathLike object, not list` | `Path(file_path)` on a list in SSE yield | Guard with `isinstance(file_path, list)` |
| Groq returning 404 CONFIG_404 | Model IDs (`llama-3.3-70b-versatile`) no longer served | Updated to `gpt-oss-120b`, `gpt-oss-20b` |
| 7/10 questions flagged `MARKING_SCHEME_MISMATCH` | LLM produced sums ≠ 10 | Deterministic rescale + prompt enforcement |
| Bloom verb `'a'` rejected | First word was article | Skip stopwords |
| Bloom verb `'design'` rejected at L3 | Verb not in L3 list | Adjacent-level acceptance |
| Every API-flagged question was `UNGROUNDED` | `evidence_summary` field did not exist; audit received empty string | Added field; populated from `plan.evidence_blocks` |
| Forbidden preamble still appeared after prompt fix | LLM ignoring negative instruction | Deterministic regex strip at generation + repair time |

### Pipeline state after last run (`cfd6d4bc2`)
- `[AGENT:planning] plans=10`
- `[AGENT:writing] questions=10 failures=0`
- `[HTTP-API] provider=groq status=OK`
- `[EVAL]` — 4 pass, 2 `UNGROUNDED`
- `[AGENT:refinement] flagged=6 resolved=9 unresolved=1`
- `paper_status: BLOCKED` (1 unresolved slot)
- `degraded=False` — semantic audit is running properly

### Open items
1. **`unresolved=1` slot** — LLM repair budget exhausted (5/5 used, 6 flagged).
   - Fix ready: raise `max_llm_repairs_per_paper` from 5 to 8 in `refinement_agent.py`
2. **Module 1 UNGROUNDED** — evidence sample too small for module 1's scope.
   - Fix ready: expand from 3 blocks × 400 chars to 5 blocks × 500 chars
3. **`MULTI_PART_STRUCTURE_MISMATCH`** on modules 2, 3 (`[6, 4]` slots) — no deterministic handler.
   - Phase 3 item
4. **Sibling duplication** — Q1/Q2 occasionally near-identical.
   - Phase 3: pass sibling stems as context to WritingAgent
5. **`slot_count: 11` in QA report** vs. actual 10 — cosmetic bug in checking_agent
6. **VLM endpoint not configured** — `AION_VLM_URL` unset. Visual path inert.
7. **15 legacy test failures** in `tests/test_*.py` — categorized earlier as test debt

### Environment notes
- **Server Python:** `aion` conda env, Python 3.11
- **Laptop Python:** 3.14 (do not run tests here)
- **vLLM:** Qwen2.5-14B-Instruct-AWQ on `http://localhost:8000`
- **Groq key:** exported in `~/.bashrc` and `.env.server` (must be rotated — was pasted in chat)
- **Test PDFs:** `workspace/uploads/{b56ada8d-c9a,e59d96e4-d0b,081fcfb8-bf2}/original.pdf`

### Next session: start here
1. Apply the two pending one-line fixes (repair budget, evidence sample size)
2. Re-run smoke test
3. If `paper_status: PASS`, move to Phase 3 (VLM wiring, sibling context, test cleanup)
4. If still BLOCKED, inspect the new `[EVAL]` verdicts

---

## Session: 2026-10-13 (Part 2) - First PASS

### Commits landed
- 52f321f6a - Groq model IDs updated to gpt-oss-120b
- 8fb9738f9 - Bloom verb and marking scheme false positive fixes
- bfe4fbedf - deterministic preamble strip and marking rescale
- cfd6d4bc2 - evidence_summary plumbed to GeneratedQuestion
- 0434fc0df - LLM repair budget 5 to 8, evidence sample 3x400 to 5x500

### Files changed
- core/api/providers.py - Groq model IDs
- core/generation/agents/local_evaluators.py - stopword skip, adjacent Bloom
- core/generation/agents/prompts.py - marking sum enforced
- core/generation/agents/deterministic_repairs.py - preamble strip, marking rescale, DET-REPAIR logs
- core/generation/agents/writing_agent.py - preamble strip, evidence summary
- core/generation/agents/generated_question.py - evidence_summary field
- core/generation/agents/evaluation_prompts.py - MAX_EVIDENCE_SUMMARY_CHARS 300 to 600
- core/generation/agents/evaluation_agent.py - EVAL diagnostic logging
- core/generation/agents/refinement_agent.py - max_llm_repairs 5 to 8

### Pipeline state - FIRST PASS
- paper_status: PASS
- questions: 10
- unresolved: 0
- EVAL: all 6 API-audited slots passed Groq
- DET-REPAIR: 4 MULTI_PART_STRUCTURE_MISMATCH, all resolved by LLM tier
- degraded_reasons: empty
- slot_count: 10

### Open items for next session
1. Web UI test not yet run - smoke passes, UI path untested
2. AION_MODEL in backend process is qwen2.5:14b (WRONG)
   vLLM serves Qwen/Qwen2.5-14B-Instruct-AWQ
   Fix in ~/.bashrc and .env.server, restart backend
   Will 404 on write calls from UI
3. Frontend not running (port 3000 returns 000)
   Start with: ./aiq frontend
4. Sibling duplication in module 1 - Q1/Q2 near-identical
   Phase 3: pass sibling stems as context to WritingAgent
5. MULTI_PART_STRUCTURE_MISMATCH costs 4 LLM repairs per paper
   Deterministic handler would save budget
6. VLM endpoint not configured - AION_VLM_URL unset
7. 15 legacy test failures in tests/test_*.py - test debt
8. Groq key rotation pending - key was pasted in chat

### Next session: start here
1. Fix AION_MODEL:
   grep -rn 'qwen2.5:14b' ~/.bashrc ~/AIQ/AION/.env.server
   Change to: Qwen/Qwen2.5-14B-Instruct-AWQ
   source ~/.bashrc
   ./aiq stop && ./aiq backend
2. Start frontend: ./aiq frontend (separate terminal)
3. Verify services:
   curl localhost:8100/health
   curl localhost:3000/
   curl localhost:8000/v1/models
   cat /proc/$(pgrep -f aion_api.py | head -1)/environ | tr '\0' '\n' | grep AION_MODEL
4. Open browser, upload 3 IoT PDFs, generate
5. Capture: [AGENT:*], [CONTRACT GATE], [QUALITY GATE] from backend terminal
6. If PASS through UI, pipeline is production-ready

