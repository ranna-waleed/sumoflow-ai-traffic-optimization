# SUMOFlowAI Phase 1A evidence ledger

**Date:** 2026-10-25 (Africa/Cairo)  
**Branch:** `research/ieee-reproducibility`  
**Starting commit:** `5b93cd21502e09deef3c4a8100a1c7d4b0e49c2b`  
**Scope:** Evaluation correctness, provenance, and automated tests only.

## Changes

- Added strict, episode-specific, atomic checkpoint validation with per-file SHA-256 records.
- Retained `load_latest()` exclusively as the existing best-effort training-resume path.
- Replaced default-zero XML parsing with streaming, validity-aware parsing.
- Added completed-trip CO2, unfinished-vehicle, teleport, collision, and completion fields.
- Replaced the live-sampled evaluator with per-run SUMO XML outputs and manifests.
- Retired the historical-JSON `evaluate_fair` comparison path.
- Added strict provenance comparability validation and synthetic unit tests.

## Commands and results

```text
python -m compileall -q DeepQN/agent/dqn_agent.py DeepQN/evaluation DeepQN/tests/test_metrics_phase1a.py DeepQN/tests/test_checkpoints_phase1a.py
```

Result: **passed** (exit 0).

```text
python -m unittest DeepQN.tests.test_metrics_phase1a DeepQN.tests.test_checkpoints_phase1a -v
```

First result: **15 passed, 1 cleanup error**. The failing test exposed an open XML
file handle after an attribute-validation exception on Windows. `_stream_elements`
was changed to own and close its binary stream explicitly.

Second result: **16 passed in 5.344 seconds** (exit 0).

After adding a regression test that verifies temporary SUMO configs cannot
reuse historical FCD or tripinfo output paths, the final result was:
**19 passed in 5.155 seconds** (exit 0), including strict rejection of a
missing checkpoint directory and step-length-aware CO2 integration.

```text
python -m DeepQN.evaluation.evaluate --help
```

Initial result: **failed** because importing the CLI imported TraCI eagerly.
The SUMO environment import was moved into the simulation execution path.
Final result: **passed** without TraCI or a running SUMO instance.

## Not run

- `pytest`: unavailable in the active Python 3.12 environment; no package was installed.
- SUMO smoke or full simulations: explicitly outside Phase 1A.
- Historical report regeneration: requires authenticated checkpoints and paired run artifacts.
- Training, BiLSTM validation, and reward tests: outside Phase 1A.

## Warnings and remaining blockers

- `DeepQN/checkpoints/` is absent locally, so no trained-policy evaluation was run.
- Historical output files do not have Phase 1A manifests and are not automatically comparable.
- A baseline-generation command remains a Phase 1B concern; Phase 1A only provides the
  common XML parser and manifest/comparison contract.
- Model mode is recorded as supplied by the caller; BiLSTM mode enforcement belongs to Phase 2.
- Existing unrelated worktree changes were not modified, staged, or reverted.

## Generated outputs

No simulation XML, checkpoint, model, historical report, or raw result was generated or modified.

## Phase 1A correction round

- Separated fixed-horizon execution success from network clearance and retained running/waiting counts.
- Classified tripinfo records as successful arrivals, removed/vaporized, or unfinished; undefined means remain null when no vehicle arrives.
- Distinguished total-network CO2 per successful arrival from optional genuine per-arrival trip emissions, and made emission step scaling explicit.
- Added additional-file hashes, effective-config hashes, output hashes/integrity checks, and collision refusal for all new artifacts.
- Rejected non-finite/negative measurements and loaded checkpoint dictionaries with restricted `weights_only=True` where supported.

Attempted `python -m pytest ... -q`: **not executed** because pytest is not installed (`No module named pytest`); no dependency was installed.

Executed `python -m unittest DeepQN.tests.test_metrics_phase1a DeepQN.tests.test_checkpoints_phase1a -v`:
first correction run **26 passed, 1 failed** (message-text regression only); after correction, **27 passed in 5.261 seconds**.

Executed `python -m compileall -q DeepQN/agent/dqn_agent.py DeepQN/evaluation` and `git ... diff --check`: both exited 0; Git reported only line-ending conversion warnings. No SUMO process was started.

Final post-review execution after summary reconciliation checks: **27 tests passed in 5.739 seconds** (exit 0).

## Phase 1A.2 scientific reconciliation

- Defined throughput as SUMO-reported arrivals (non-negative `arrival`) and now tracks teleport/vaporization-tagged arrivals independently from tagged non-arrival removals and unfinished records.
- Corrected trip-level emissions to use SUMO's cumulative nested `CO2_abs`; a mean is available only when every reported arrival has a valid measurement.
- Zero baselines now produce a null percentage plus an explanation while retaining absolute differences and other valid comparisons.
- Bumped the metric contract to `sumo-xml-reported-arrivals-and-horizon-v3` so earlier semantics cannot compare silently.
- Read-only artifact inventory found no local 3,482-record Tahrir tripinfo or 1,218-record Taksim tripinfo. The locally present files are different/incomplete runs; they were not substituted for the reviewed matched pairs.

Final command: `python -m unittest DeepQN.tests.test_metrics_phase1a DeepQN.tests.test_checkpoints_phase1a -v` — **31 tests passed in 5.515 seconds**. Compilation and `git diff --check` exited 0; only existing line-ending warnings were emitted. No SUMO process was started.
