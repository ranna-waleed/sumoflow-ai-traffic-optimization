# Phase 3.0D final readiness ledger

Date: 2026-10-27 (Africa/Cairo)  
Branch: `research/ieee-reproducibility`  
HEAD: `895ebb8a664b1979d137b54ccb07b6e4af6375e4`  
Upstream: `origin/research/ieee-reproducibility`, ahead 0 / behind 0  
Final decision: **Phase 3.1 NO-GO**

## Scope and repository integrity

The complete Phase 2C and Phase 3.0A–3.0C evidence was reviewed against executable source.
Phase 3.0C is committed at `892812a` and its calibration protocol is separately tracked at
`895ebb8`. The protocol SHA-256 is
`0bf45ff5bcb2f265227178b9f3aa425a7948fc1366173079c4f9e3ae1ae51740`, exactly matching
the Phase 3.0C manifest reference.

No model was trained, no full Tahrir/Taksim demand was executed, no dependency was installed,
and no commit, push, approval field or trust root was changed. Pre-existing unrelated worktree
changes were preserved. Test-created detector XML was deleted and tracked bytecode was restored.

## Principal findings

| Severity | Finding | Classification | Required correction |
|---|---|---|---|
| Critical | No approved five-pool physical demand registry exists. | Blocks forecast/DQN scientific production. | Generate, independently review, hash and freeze explicit realizations and partitions. |
| Critical | Tahrir E1 geometry and signal sequencing lack independent traffic-engineering approval. | Blocks a field-calibrated Tahrir claim; synthetic scope may be transparently limited. | Obtain external evidence or explicitly restrict claims to uncalibrated simulation. |
| Critical | No compatible trained seven-input forecast checkpoint exists. | Future output of forecast training, but forecast training inputs/seeds must first be frozen. | Allocate forecast seeds and execute approved leakage-free training/selection. |
| Critical | No four-profile reward calibration artifact exists. | Blocks DQN training under the publication reward contract. | Produce training/calibration-only traces under the committed 0.95 protocol. |
| Critical | Policy indices `0..4` and RNG seeds `11001..11005` are not explicitly bound at runtime; pairing rule is null. | Blocks publication evaluation/inference. | Freeze the seed contract and pairing/crossing rule before outcomes. |
| Critical | Phase 3.0C candidate validation is not invoked by `evaluate.py`. | Candidate is documentation, not execution authority. | Implement a separately approved full-manifest preflight before future publication execution. |
| High | Hierarchical analysis is specified but not implemented; five policy seeds give weak tail resolution. | Nonblocking for preparatory stages; blocks publication inference. | Freeze and verify analysis code; report seed-level effects and sensitivity. |
| High | Existing bounded E1 tests do not validate arbitrary Tahrir lane-change/teleport geometry. | Nonblocking for bounded integrity; blocks overgeneralized physical claims. | Retain fail-closed exclusions or obtain bounded/external evidence for the claimed scope. |

Missing checkpoints and held-out results are expected products of future authorized stages, not
prerequisites to starting those stages. Missing demand/calibration/mapping/seed/budget contracts
are prerequisites.

## Verification executed

All scientific tests and required validation commands exited 0. One auxiliary trailing-space
diagnostic initially exited 1 because it treated intentional two-space Markdown hard breaks as
errors; the corrected untracked-file check permits exactly those hard breaks and passed. No file
was changed to conceal this diagnostic result.

1. Focused bounded integration:

   `python -m unittest DeepQN.tests.test_phase2c2a_detector_characterization DeepQN.tests.test_phase2b2a_detector_events DeepQN.tests.test_phase2b2b_methodology.StrictForecastTests DeepQN.tests.test_phase2b2b_methodology.StrictForecastStartupTests DeepQN.tests.test_phase3b_forecast_contract DeepQN.tests.test_phase3c_research_protocol -v`

   Result: **53 tests passed in 21.972 seconds; 0 failed; 0 skipped.** This includes bounded
   real-SUMO E1 evidence, exact/adjacent time boundaries, teleport rejection, detector
   installation, causal history, strict startup and Phase 3.0C contracts.

2. Full regression:

   `python -m unittest discover -s DeepQN/tests -p "test_*.py" -v`

   Result: **235 tests passed in 148.557 seconds; 0 failed; 1 skipped.** The skip is the
   quarantined legacy interactive diagnostic requiring an explicit checkpoint episode.
   Existing bounded SUMO fixtures ran; no Tahrir/Taksim experiment ran.

3. Final validation (recorded after document creation): non-writing Python compilation of changed
   sources, JSON parsing, hash-reference checks, and staged/unstaged CRLF-aware diff checks.

## Deliverables and hashes

| File | SHA-256 |
|---|---|
| `research/2026-10-27-phase3d-end-to-end-contract-audit.md` | `7048b38f9c7a634b82e8a540980c6b197c2d7a7c16b0fa557fe64345b94ce9e6` |
| `research/2026-10-27-phase3d-integration-readiness-matrix.json` | `8ad1fdbe2c51ff3847e8c3b83dcbcce45362593d64d5eb0bd12c8e0e1c35dfba` |
| `research/2026-10-27-phase3d-pre-experiment-dependency-plan.md` | `4a568bae957ae757456cd26b1e4ff7b6abafadf22ce18955489ca11bdbc81975` |
| `research/2026-10-27-phase3d-seed-allocation-candidate.json` | `dbaf6fa3e16548773489b26f36cf53f90295e930f71e380689222896eb9da192` |

The ledger is omitted from its own hash table to avoid a circular self-hash.

## Readiness boundary

- Bounded software-integrity testing: ready.
- Transparently uncalibrated synthetic simulation study: conditional, after the listed contracts
  and artifacts are frozen.
- Field-calibrated Tahrir case study: not ready.
- Real-world deployment: not ready.

Passing tests establish code-contract behavior, not traffic improvement, empirical calibration,
signal safety, external validity, publication approval or Phase 3.1 authorization.
