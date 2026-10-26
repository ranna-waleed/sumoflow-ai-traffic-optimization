# Phase 3.0B forecast-contract readiness ledger

**Date:** 2026-10-27 (Africa/Cairo)
**Branch:** `research/ieee-reproducibility`
**Starting HEAD:** `5ac56d79d8b935242aea36140ed2c576365216a8`
**Upstream at start:** same SHA; no fetch, pull, commit or push performed.
**Gate:** publication forecast training and Phase 3 performance experiments remain NO-GO.

## Repository and source findings

- The checked-out branch and local `origin/research/ieee-reproducibility` both resolved to
  the remotely recorded Phase 3.0A SHA.
- Existing unrelated `.vscode`, `test_metrics_phase1a.py`, backend-image, untracked
  forecasting initializer and stray status-name changes were not modified.
- SUMO documents `simulationStep` as advancing to the requested/current next step,
  `getTime` as current simulator time, and E1 TraCI values as belonging to the last step.
- v3 offline event binning used `floor(entry_time)+1`, while live adapter collection assigns
  last-step E1 rows to the current post-step time. Exact integer entry timestamps therefore
  differed between offline and live paths. This is corrected and versioned rather than
  relabelled silently.
- The historical trainer has a 30-by-4 sequence output and MSE objective; the strict research
  path is a single four-count expected-value output. No historical weight was modified.

## Implementation

- `DeepQN/forecasting/research_contract.py`: right-closed reporting-time conversion,
  nonnegative Poisson loss, causal zero/persistence/moving-average/historical-mean baselines,
  count scoring by realization, and five-pool physical leakage validation.
- `lstm/research_data.py`: v4 right-closed dataset contract, exact event reporting-bin
  conversion, explicit history/target definitions and target shape in manifests.
- `DeepQN/forecasting/inference.py`: v2 nonnegative-count inference contract, softplus output,
  frozen 7/60/128x2/bidirectional/dropout-0.2/four-output architecture, strict dataset/time/
  activation/loss metadata and restricted checkpoint loading.
- `DeepQN/tests/test_phase3b_forecast_contract.py`: bounded CPU model, loss, serialization,
  baseline, scoring and five-pool leakage adversarial tests.
- Existing detector and strict-inference tests were updated only where their accepted
  fixtures must declare or assert the new versioned semantics.
- `research/2026-10-27-phase3b-forecast-contract-candidate.md`: mathematical and scientific
  contract with explicit unsupported claims and blockers.

## Verification state

The pre-ledger focused command completed with **49 tests: 49 passed, 0 failed, 0 skipped**.
One earlier focused run had 48 passes and one expected-message mismatch after the new
fail-closed bin helper rejected an outside event earlier in the call path; the legacy test
expectation was aligned to the same `no recorded sample bin` meaning. A NumPy warning from
eager `where` evaluation at a valid zero target was also removed using masked computation.
The subsequent focused forecasting subset completed with **30/30 passed**, and six changed
Python files compiled in memory. Final full-suite, hash and diff outcomes are appended below
after execution.

## Scientific limitations

- The mapping and detectors remain unapproved; synthetic fixtures are not publication data.
- No forecast model was trained and no accuracy, calibration or traffic benefit is reported.
- Poisson NLL is a preregistered candidate, not empirically superior; overdispersion has not
  been measured. Negative-binomial use would require a separate versioned decision.
- The five-pool validator depends on the canonical physical fingerprint supplied by the
  separately frozen demand-provenance workflow; hashes establish identity, not independence
  or scientific approval by themselves.
- Real relocation, teleport and approved Tahrir collection controls remain unresolved.
- Publication trust roots and external decision fields remain empty.

## Final verification record

```text
python -m unittest DeepQN.tests.test_phase3b_forecast_contract \
  DeepQN.tests.test_phase2b1_contracts \
  DeepQN.tests.test_phase2b2a_detector_events \
  DeepQN.tests.test_phase2b2b_methodology.StrictForecastTests -v
50 run, 50 passed, 0 failed, 0 skipped; 11.472 s; exit 0.

python -m unittest discover -s DeepQN/tests -p "test_*.py" -q
226 run, 226 passed, 0 failed, 1 skipped; 164.337 s; exit 0.

python -m unittest DeepQN.tests.test_phase3b_forecast_contract -q
8 run, 8 passed, 0 failed, 0 skipped; 0.422 s; exit 0 after import cleanup.

python -c "...compile every DeepQN/lstm Python source in memory..."
66 Python files compiled; exit 0.

python -c "...parse model config, mapping and external decision-register JSON..."
3 contract JSON files parsed; exit 0.
```

The full-suite skip is the previously quarantined legacy interactive diagnostic and is not
reported as a pass. Full discovery invoked only existing bounded SUMO fixtures; no Tahrir or
Taksim run occurred. Test-created detector XML was removed and tracked bytecode modified by
imports was restored. No dependency was installed.

Final pre-ledger SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `DeepQN/forecasting/research_contract.py` | `91da17222512505fd111d513ece472c58f961f421b13211e368d4850eb1fa513` |
| `DeepQN/forecasting/inference.py` | `6dd34f30a133cf9b86d9562d6e75d1cb73e4e35093185e1743f6ef59801eb6dc` |
| `lstm/research_data.py` | `eee0fa8ed6ed5c8b4a2615bbe07006ccd1c0f35743c972390bdf2b120e25c99f` |
| `DeepQN/tests/test_phase3b_forecast_contract.py` | `5af5098592fad392b8e2689d8d78243b892a123354c09327a3f440b7b3c79740` |
| `DeepQN/tests/test_phase2b2a_detector_events.py` | `bd019a02345bfdf2f2c3cc4fa88ecdd82d655d45863cf671de0424a3b93d3123` |
| `DeepQN/tests/test_phase2b2b_methodology.py` | `bd4fb3aa3e081f12f66099eb35c2fb7992b851aaaa56e130b18f29192e44923e` |
| `research/2026-10-27-phase3b-forecast-contract-candidate.md` | `7b0977238f78e6abed029cbb76cec1a0cef306ae208870072c479738ad275d53` |

No publication approval field, trust root or external decision was changed. No model/data
training, real-data collection, commit or push occurred. Phase 3.0C and Phase 3.1 were not
started.
