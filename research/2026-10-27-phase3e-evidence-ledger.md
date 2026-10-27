# Phase 3.0E evidence ledger

Date: 2026-10-27 (Africa/Cairo)  
Branch: `research/ieee-reproducibility`  
Starting HEAD: `280244b` (upstream aligned; ahead 0 / behind 0)  
Decision: **Phase 3 remains NO-GO**

## Work completed

- Added an offline, fail-closed Gate-0 contract module. It validates separately named seed roles,
  writes a verified five-pool usage registry atomically, and verifies a complete immutable preflight
  before any future runtime construction.
- Added a version-2 seed-allocation candidate with forecast seeds `12001..12005`. The proposed
  matched rule remains deliberately unfrozen; it cannot pass the executable preflight.
- Exercised the complete demand generation -> realization manifest -> five-pool split registry ->
  usage-subpool registry path on bounded engineering fixtures.
- Assessed all four historical profile templates for offline explicit-demand generation. No
  production route, registry, calibration trace or model was generated.
- Defined the uncalibrated synthetic claim boundary and recorded detector/TLS restrictions.
- Created a machine-readable Gate-0 checklist with false authority fields and empty trust roots.

## Files created

| File | Purpose |
|---|---|
| `DeepQN/evaluation/gate0.py` | Offline seed, usage-registry and immutable-preflight validation. |
| `DeepQN/tests/test_phase3e_gate0.py` | Adversarial bounded Gate-0 and five-pool tests. |
| `research/2026-10-27-phase3e-seed-allocation-candidate.json` | Explicit but unfrozen identity mapping. |
| `research/2026-10-27-phase3e-synthetic-scope-candidate.md` | Claim boundary and topology decisions. |
| `research/2026-10-27-phase3e-demand-and-seed-readiness.md` | Demand feasibility, pool access and staged readiness. |
| `research/2026-10-27-phase3e-gate0-checklist.json` | Non-executable prerequisite register. |
| `research/2026-10-27-phase3e-evidence-ledger.md` | This evidence record. |

## Verification record

Focused test executed before final documentation:

`python -m unittest DeepQN.tests.test_phase3e_gate0 -v`

Result: **4 passed in 1.074 seconds; 0 failed; 0 skipped; exit 0.**

Broader focused command:

`python -m unittest DeepQN.tests.test_phase3a_demand_expansion DeepQN.tests.test_phase2c2b_demand_provenance DeepQN.tests.test_phase3c_research_protocol DeepQN.tests.test_phase3e_gate0 -v`

Result: **40 passed in 18.397 seconds; 0 failed; 0 skipped; exit 0.** The existing
bounded SUMO expansion checks ran; no Tahrir/Taksim profile ran.

Full command:

`python -m unittest discover -s DeepQN/tests -p "test_*.py" -v`

Result: **239 passed in 152.349 seconds; 0 failed; 1 skipped; exit 0.** The skip is the
quarantined legacy interactive diagnostic requiring an explicitly selected checkpoint episode.

Non-writing compilation parsed both new Python files successfully. JSON parsing validated both
Phase 3.0E JSON files. Staged and unstaged `git diff --check` exited 0; Git emitted only the known
LF-to-CRLF warning for the pre-existing modified `DeepQN/tests/test_metrics_phase1a.py`. A separate
check of every new Phase 3.0E text file found no trailing whitespace other than intentional
two-space Markdown line breaks. Test-created detector XML was removed and the three test-modified
tracked bytecode files were restored; unrelated worktree changes were untouched.

Final repository checks reported HEAD `280244b7ff3d341ede5878bcd53f29889955880d` and upstream
divergence `0 0`.

## Deliverable hashes

| File | SHA-256 |
|---|---|
| `DeepQN/evaluation/gate0.py` | `a51345cd8bb73a96f9775eabe6d1f877b08bbd24a5247eed0631c6c686d42399` |
| `DeepQN/tests/test_phase3e_gate0.py` | `5a6efd3a710355abd8949ec277314e2c6c9d8acba14311ba35acfc3dd1684e80` |
| `research/2026-10-27-phase3e-seed-allocation-candidate.json` | `165886b1d20ea1d9847c0c0117273f2afe2821972200a9e66c1ffd11efd9272e` |
| `research/2026-10-27-phase3e-synthetic-scope-candidate.md` | `a58670ecd7a82c11cdb416bf7021c166241d56dfe7f8c1375a8c543921242824` |
| `research/2026-10-27-phase3e-demand-and-seed-readiness.md` | `64755662e1faf7876824def132dc0c3287ccdf847cb6cc40fef2ef82d68772cf` |
| `research/2026-10-27-phase3e-gate0-checklist.json` | `c83e7c7802c030de24104cd74b97c293570311881b772b09c285473352220093` |

The ledger is omitted from its own hash table to avoid a circular self-hash.

## Scientific limitations

- Passing fixture tests prove software behavior only, not Tahrir calibration or model performance.
- No approved five-pool Tahrir realization registry, expansion specification or usage registry
  exists.
- Forecast boundary approval and signal-safety decisions remain external and empty.
- No publication-compatible forecast checkpoint or DQN ensemble exists.
- Four-profile reward reference values and approved training budget remain absent.
- The candidate experiment manifest is documentation, not runtime authority; the new preflight is
  intentionally not connected to a runner.
- Field-calibrated, real-world and deployment claims remain prohibited.

No dependency was installed; no SUMO scenario or training ran; no historical artifact, model,
checkpoint, unrelated worktree file, trust root or approval field was modified; no commit or push
was made.
