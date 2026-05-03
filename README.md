# AgentKit Multi-Agent Orchestration Lab

Production-style Python lab for the AgentKit design document. It demonstrates loop detection, coordinator/sub-agent isolation, structured context passing, memory compaction with preserved case facts, and a planner-executor workflow.

## What Is Included

- Phase 1: Circuit breaker decorator that opens after 3 consecutive tool failures.
- Phase 2: Coordinator with only delegation tools, plus isolated Finance and HR agents.
- Phase 3: Minimal `ContextPayload` JSON handoffs without full chat history.
- Phase 4: Regex case-fact extraction before memory compaction, with missing-field flags.
- Phase 5: Planner schema validation and executor dependency handling.
- Observability: structured JSON log events for circuit, delegation, context, memory, plans, and steps.
- Offline-safe execution: demos and tests run deterministically without making LLM calls.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

OpenAI/OpenRouter keys are optional for this lab implementation. The included phase demos default to deterministic local execution so tests do not spend tokens.

## Run Phase Demos

```powershell
python -m phases.phase1_run
python -m phases.phase2_run
python -m phases.phase3_run
python -m phases.phase4_run
python -m phases.phase5_run
```

## Test

```powershell
python -m pytest
```

## Project Map

- `circuit_breaker.py`: loop detection and circuit breaker state.
- `agents_config.py`: agent definitions and tool ownership boundaries.
- `context.py`: minimal structured handoff payloads.
- `tools/`: database, finance, HR, and delegation tools.
- `memory/`: `CaseFacts`, deterministic extractor, and compaction.
- `planner/`: plan schema, validation, and local planner.
- `executor/`: dependency-aware step execution.
- `observability/`: structured JSON logger.
- `phases/`: runnable demos matching the design phases.
- `tests/`: unit/integration coverage for the major behaviors.

