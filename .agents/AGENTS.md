# Project Rules & Release Baseline

# Project Rules & Release Baseline

## Current Production Baseline
- **Tag**: `v1.1.0`
- **Branch**: `mod_eval`
- **Core Enhancements in v1.1.0**:
  - Module-level concurrency & dynamic port allocation (`start_server.ps1`, `start_server.sh`)
  - Cross-module deduplication registry (`shared_generated_texts`) & Jaccard audit pass in `v0_1/main.py`
  - Bloom verb and operation harmonization (`BLOOM_VERB_LEVEL_MAP`) across linter, critic, and exporters
  - Fail-closed subject archetype registry
  - RapidOCR word-count gating (`OCR_WORD_COUNT_THRESHOLD=100`) & persistent extraction caching
  - Sparse module indexing in `v0_1/segmenter.py`
  - Topical neighborhood chunk distribution & `EXTERNAL` depth exclusion in `v0_1/chunk_image_mapper.py`

## Release Baseline (Golden State / Rollback Target)
- **Tag**: `v1.0.0` and `v1`
- **Golden Commit**: `6e3631a`
- **Reversion Directive**: If the user requests to revert to the previous golden version, check out or restore tag `v1.0.0` / `v1`.

## Core Features in v1.0.0
- **Typed Contract System**: `v0_1/contracts.py`
- **Execution Auditor**: `v0_1/execution_auditor.py`
- **Unified Pipeline Orchestrator**: `v0_1/unified_pipeline.py`
- **VTU Marks Enforcement**: Strict 6+4=10 marks per IA question, 8+6+6=20 marks per SEE question.
- **Server Deployment**: `.env.server`, `v0_1/llm_server.py`, `start_server.sh`, `start_server.ps1`.
