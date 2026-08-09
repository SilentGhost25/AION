# Project Rules & Release Baseline

## Release Baseline (Golden State)
- **Tag**: `v1.0.0` and `v1`
- **Golden Commit**: `6e3631a`
- **Reversion Directive**: If the user requests to revert to the previous golden version, check out or restore tag `v1.0.0` / `v1`.

## Core Features in v1.0.0
- **Typed Contract System**: `v0_1/contracts.py`
- **Execution Auditor**: `v0_1/execution_auditor.py`
- **Unified Pipeline Orchestrator**: `v0_1/unified_pipeline.py`
- **VTU Marks Enforcement**: Strict 6+4=10 marks per IA question, 8+6+6=20 marks per SEE question.
- **Server Deployment**: `.env.server`, `v0_1/llm_server.py`, `start_server.sh`, `start_server.ps1`.
