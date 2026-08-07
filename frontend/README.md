# DSATM AI Question Paper Generator

A faculty assessment portal for generating structured internal-assessment (IAT) question papers. Faculty configure the exam details and section rules through a guided wizard, and the app produces a print-ready question paper.

## Run & Operate

- `pnpm --filter @workspace/qp-generator run dev` — run the frontend wizard (defaults to port 5173)
- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string
- Optional frontend env: `PORT` (default `5173`), `BASE_PATH` (default `/`)

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Frontend: React 19 + Vite 7, Tailwind CSS 4, Radix UI, wouter, TanStack Query
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/qp-generator` — the frontend wizard (Step 1 config → Step 2 rules → Step 3 preview)
- `artifacts/api-server` — Express API
- `lib/api-spec/openapi.yaml` — source of truth for the API contract
- `lib/api-zod`, `lib/api-client-react` — generated Zod schemas and React query hooks
- `lib/db` — Drizzle schema and migrations

## Product

Faculty enter institution/subject details and per-section marking rules, then generate a formatted question paper that can be reviewed, edited, and printed directly from the browser.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
