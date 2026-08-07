# Frontend Isolation Review

## Goal
Keep **only** the frontend (`artifacts/qp-generator`) and **its direct dependencies** (`lib/api-spec`, `lib/api-client-react`, `lib/api-zod`), designed to integrate cleanly with **any backend** that implements the OpenAPI contract.

---

## ✅ KEEP — Pure Frontend & Contract

### 1. Core Frontend Application
**Path:** `artifacts/qp-generator/`
- React + Vite app with upload-only workflow (2 steps: Config+Upload → Preview)
- Wired to generated hooks from `@workspace/api-client-react`
- Uses relative `/api` paths (works with same-origin serving or CORS proxy)
- **No backend logic** — calls `POST /api/paper/generate`, displays response

### 2. API Contract (Spec)
**Path:** `lib/api-spec/`
- OpenAPI 3.1 spec in `openapi.yaml` — the **single source of truth**
- Defines request/response schemas for paper generation
- Used by orval to generate client libraries
- **Backend-agnostic** — any server implementing this spec works

### 3. Generated React Client
**Path:** `lib/api-client-react/`
- Auto-generated from `lib/api-spec` via orval
- `useGeneratePaper` hook, `customFetch` with `setBaseUrl()`
- **No server logic** — pure client-side wrapper

### 4. Generated Zod Validators
**Path:** `lib/api-zod/`
- Auto-generated Zod schemas from the spec
- Used for runtime validation (optional, can be dropped if not needed)

### 5. Root Build Orchestration
**Files:** `package.json`, `pnpm-workspace.yaml`, `pnpm-lock.yaml`, `tsconfig.base.json`, `tsconfig.json`, `.npmrc`, `.gitignore`
- Workspace setup for the 4 packages above
- After removal, these will reference only the frontend packages

### 6. Assets & Documentation
**Paths:** `attached_assets/`, `README.md`
- Logos, reference images
- README can be updated to reflect frontend-only scope

---

## ❌ REMOVE — Backend, Database, Integrations

### 7. Express Backend Server
**Path:** `artifacts/api-server/`
- **Old Node.js backend** — you said you've already built a Python backend
- Contains routes, database logic, server setup
- **DELETE:** This is completely replaced by your Python backend

### 8. Mockup Sandbox
**Path:** `artifacts/mockup-sandbox/`
- Prototyping/preview tool (unrelated to production frontend)
- **DELETE:** Not part of the production question-paper app

### 9. Database Layer
**Path:** `lib/db/`
- Drizzle ORM schemas and migrations
- **DELETE:** Your Python backend has its own DB layer (SQLite+sqlite-vec per BACKEND_ARCHITECTURE.md)

### 10. OpenAI Integrations (Server-Side)
**Paths:** `lib/integrations-openai-ai-server/`, `lib/integrations/openai_ai_integrations/`
- Server-side AI integration wrappers
- **DELETE:** Your Python backend uses Ollama (Qwen2.5), not OpenAI

### 11. OpenAI Integrations (React Hooks)
**Path:** `lib/integrations-openai-ai-react/`
- React hooks for OpenAI streaming (unused in your frontend)
- **DELETE:** Not needed for the upload → backend → display flow

### 12. Build Scripts
**Path:** `scripts/`
- Post-merge hooks, workspace utilities
- **DELETE:** If these are just for backend orchestration; **KEEP** if they support frontend build/deploy

### 13. Backend Architecture Doc
**Path:** `BACKEND_ARCHITECTURE.md`
- Your Python backend design (Ollama, Docling, sqlite-vec)
- **DECISION:** Keep as reference documentation, or move to the Python backend repo

---

## ✅ Clean Integration Points

After removal, the frontend will:

1. **Call a single endpoint:** `POST /api/paper/generate` with `PaperInput` (config + 10 sections with uploaded content)
2. **Receive structured questions:** `GeneratedPaper` with backend-generated sub-questions, marks, CO, BL
3. **Work with any deployment:**
   - Same-origin: Python FastAPI serves `/` → frontend build, `/api` → backend routes
   - Separate origins: frontend calls `setBaseUrl("https://backend.example.com")` before mutations
4. **No hardcoded assumptions:** Backend decides everything (sub-question count, marks, CO, BL)

---

## 🔧 Post-Removal Checklist

After I delete the backend/infra packages:

1. **Update `pnpm-workspace.yaml`** — remove deleted package paths
2. **Update root `package.json`** — remove irrelevant scripts (if any)
3. **Verify `lib/api-spec/openapi.yaml`** is complete (currently it's flat; see integration notes below)
4. **Clean `pnpm-lock.yaml`** — run `pnpm install` to regenerate without deleted packages
5. **Update README.md** — document that this is a frontend-only repo

---

## 📋 Integration Notes (Current Schema Gap)

**Current state:** The wire schema (`GeneratedQuestion`) is **flat** — no `subQuestions` field. The frontend runtime-casts to `QuestionWithSubs` (adds `subQuestions?: SubQuestion[]`) for display.

**Options for your Python backend:**

### Option A: Keep the flat schema (simplest)
- Backend returns `GeneratedQuestion[]` with top-level `text/marks/co/rbt`
- Frontend continues to cast and treats each question as a single row
- **Pro:** No spec changes needed
- **Con:** Loses per-sub-question detail (marks, CO, BL)

### Option B: Extend the spec (recommended)
- Add `subQuestions` array to `GeneratedQuestion` in `openapi.yaml`:
  ```yaml
  GeneratedQuestion:
    properties:
      # ... existing fields
      subQuestions:
        type: array
        items:
          type: object
          required: [label, text, marks, co, rbt]
          properties:
            label: { type: string }
            text: { type: string }
            marks: { type: number }
            co: { type: string }
            rbt: { type: string }
            parts: { type: array, items: { type: string } }
            images: { type: array, items: { type: string } }
  ```
- Run `pnpm -F @workspace/api-spec run codegen` to regenerate types
- Frontend gets native TypeScript types matching backend output
- **Pro:** Clean contract, proper validation, no runtime cast
- **Con:** Requires one schema update (5 minutes)

---

## 🚀 Ready to Proceed?

**Confirm before I delete:**

1. ✅ Remove `artifacts/api-server/` (old Express backend)
2. ✅ Remove `artifacts/mockup-sandbox/` (prototype tool)
3. ✅ Remove `lib/db/` (Drizzle ORM)
4. ✅ Remove `lib/integrations*/` (OpenAI wrappers)
5. ✅ Remove `scripts/` (or keep if needed for frontend build)
6. ⚠️ **Decision:** Keep or move `BACKEND_ARCHITECTURE.md`?

After removal, the repo will contain:
- `artifacts/qp-generator/` — the React frontend
- `lib/api-spec/` — OpenAPI contract
- `lib/api-client-react/` — generated hooks
- `lib/api-zod/` — generated validators
- Root workspace files (package.json, pnpm-workspace.yaml, etc.)

Type **"proceed"** to delete, or tell me which items to keep/adjust.
