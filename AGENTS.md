# SUMOFlowAI — instructions for Codex

## Mission
Make this existing SUMOFlowAI repository experimentally reliable and reproducible for a potential peer-reviewed IEEE conference submission. Preserve the graduation-project implementation as the research baseline. **Never optimize the code merely to manufacture better numbers.** Read `research/IEEE_VALIDATION_HANDOFF.md` before making or proposing changes.

## Research integrity rules
- Treat all historical figures as **reported, not independently validated**, until reproduced from the exact run artifacts. Never invent metrics, experiment runs, significance, citations, model usage, or completed tests.
- Distinguish code behavior, artifact evidence, and interpretation. Keep the provenance (Git commit, model weight hash, config, seed, profile, SUMO version, log/output paths) for every new run.
- Never silently use random/uninitialized DQN agents, simulated BiLSTM predictions, stale checkpoints, default zero metrics, or fallback features and describe them as trained model results.
- All published baseline-versus-adaptive metrics must use consistent measurement definitions, SUMO scenarios, and sampling horizons. Include trip completion, incomplete vehicles, and teleports.
- Keep the historical May 2026 files immutable. A June 9/10 Google Drive folder creation/upload time **does not prove** a new training run. Confirm file modification times and generation provenance.
- Do not rewrite the thesis or paper's claims until the relevant measurements have been checked. Never claim real-world deployment or YOLO-driven signal control without explicit working code and experiment evidence.

## Working agreements
- Work only on the user-selected research branch; never commit directly to the default branch.
- Before changing code, inspect the relevant files and summarize intended changes, existing tests, and acceptance criteria. Execute **one phase at a time** and request approval before beginning the next phase or modifying evaluation semantics.
- Prefer incremental, reviewable diffs, fixtures, typed interfaces, explicit failures, and tests. Do not make unrelated architecture or UI refactors.
- Do not modify historical checkpoints, model files, raw datasets, or Drive artifacts. Never commit large generated XML/logs, credentials, API keys, or local environments.
- Keep a running, dated change/evidence ledger under `research/` that records commands, output paths, completed validations, warnings, remaining blockers, and exact commit SHAs. Mark unrun tests as unrun; report failures transparently.
- Separate evaluation and training into distinct seeds. Use paired simulation scenarios and preserve demand/route inputs and SUMO settings.
- Do not commit, push, create PRs, run expensive large-scale experiments, install unexpected dependencies, or access external credentials without explicit human approval. Provide the proposed commands first.

## First task
Perform a **read-only phase-0 audit**, using `research/IEEE_VALIDATION_HANDOFF.md` and actual repository files. Report a per-issue evidence table, recommended minimal fix, test design, experiment costs, and order of work. Do not edit source files until the user approves phase 1.
