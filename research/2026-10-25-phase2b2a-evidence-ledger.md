# SUMOFlowAI Phase 2B.2A evidence ledger

**Date:** 2026-10-25 (Africa/Cairo)  
**Branch:** `research/ieee-reproducibility`  
**Starting commit:** `101384280eef99f90412868babce9a19d00c4e0c`  
**Scope:** exact approach-entry events and causal live data collection only.

No model was trained, no RL reward/replay/policy code was changed, and no full
Tahrir or Taksim simulation was run.

## Implemented files

- `DeepQN/forecasting/detector_events.py`
  - Defines detector/event schema `sumo-e1-approach-entry-v1`.
  - Generates one lane-specific E1 at a common 0.10 m plane from the start of
    every mapped approach edge and validates lane IDs and lengths.
  - Collects TraCI `getVehicleData()` records by vehicle ID and exact SUMO entry
    timestamp. Repeated API responses are deduplicated; a later crossing is
    retained; simultaneous reports for the same vehicle/TLS/time on different
    lane detectors fail as ambiguous.
  - Combines E1 counts with causal approach speed and 100-second TraCI
    accumulated-waiting observations without changing the seven-feature order.
- `DeepQN/forecasting/validate_detector_fixture.py`
  - Runs the eight-second fixture through TraCI, redirects detector outputs to
    an isolated directory, records E1 events and live feature rows, and closes
    the connection deterministically.
- `DeepQN/tests/fixtures/detector_sumo/`
  - Deterministic four-approach, one-TLS demand; four E1 detectors and four
    independent instantaneous-loop references. A vehicle scheduled exactly at
    the eight-second end tests the end boundary.
- `lstm/research_data.py`
  - Dataset contract is now
    `causal-junction-arrivals-60s-to-30s-v3-detector-capable`.
  - Accepts explicitly mapped E1 records, bins a SUMO event in the reporting
    step `[t-1,t)`, preserves 60-second features and four 30-second targets,
    records the detector mapping hash/event source, and validates actual
    SHA-256 syntax for snapshot, demand, and SUMO-configuration provenance.
  - Inferred snapshot labels remain supported only as the explicitly
    provisional historical research path.
  - Publication readiness requires every source to be detector-backed *and* an
    independently traffic-engineering-approved mapping. Current mappings do
    not meet the second condition, so validation remains fail-closed.
- `DeepQN/tests/test_phase2b2a_detector_events.py`
  - Adds detector placement, deduplication, re-entry, ambiguous multi-lane,
    bounded real-SUMO, causal target, reproducible artifact, and readiness tests.
- `DeepQN/tests/test_phase2b1_contracts.py`
  - Updates synthetic configuration provenance to a real SHA-256 value.

## Formal event contract

One approach entry is one SUMO E1 `getVehicleData()` record with:

- the E1 placed on a declared lane at the versioned mapped approach-edge-start
  plane;
- ownership by exactly one `(TLS, direction, edge, lane)` mapping;
- SUMO vehicle ID and E1 sub-step `entryTime` as identity/time evidence;
- a unique `(detector ID, vehicle ID, exact entryTime)` key;
- reporting in the causal row at step end `t` for entry time in `[t-1,t)`.

The leave time may be `-1` when TraCI first reports an entry for a vehicle still
on the loop. It is not required to count the entry. Last-step occupancy,
last-step vehicle IDs, and aggregate `nVehEntered` are not substituted for the
entry record. Same-TLS/same-vehicle/same-time events on different lane loops are
rejected because a lane change can make their physical interpretation
ambiguous. A later distinct entry timestamp is a legitimate re-entry.

Relevant official SUMO documentation:

- E1 definition/output and lane-change caveat:
  https://sumo.dlr.de/userdoc/Simulation/Output/Induction_Loops_Detectors_%28E1%29.html
- TraCI E1 values and `getVehicleData()` fields:
  https://sumo.dlr.de/docs/TraCI/Induction_Loop_Value_Retrieval.html
- Python API tuple structure:
  https://sumo.dlr.de/daily/pydoc/traci/_inductionloop.html

## Tahrir structural detector evidence

The existing seven-TLS approach mapping covers every non-internal incoming edge
of each declared TLS. Detector generation produced **46 lane-specific E1s** on
the ten mapped edges. Generated detector mapping SHA-256:
`afd17243b1d154ecf073cca645c3a58dfc4b7e4875e105206be6d44b8fed4a06`.

The shortest mapped lanes are the five lanes of `10873186#8`, each only 0.20 m.
This invalidated the initial proposed 1 m placement in a real regression. The
uniform plane was corrected to 0.10 m and all 46 lanes then validated
structurally. This does **not** prove that these edge starts are the desired
traffic-engineering measurement boundaries. No Tahrir traffic was simulated,
and the mapping remains `traffic_engineering_validated=false`.

## Bounded SUMO 1.26.0 evidence

Environment: Python 3.12.0, NumPy 1.26.2, PyTorch 2.7.1 CPU, installed SUMO
1.26.0 Windows release, TraCI package reporting `0.0.0`.

Command (exit 0):

```powershell
python -m DeepQN.forecasting.validate_detector_fixture DeepQN/tests/fixtures/detector_sumo/detector.sumocfg DeepQN/tests/fixtures/detector_sumo/mapping.json DeepQN/tests/fixtures/tiny_sumo/tiny.net.xml D:/temp/sumoflow-phase2b2a-final/outputs D:/temp/sumoflow-phase2b2a-final/result.json
```

Expected and observed supported cases:

| Case | Expected | Observed |
|---|---:|---:|
| Ordinary first-step entries | 4 unique vehicles | 4 TraCI E1 records |
| Independent instant-loop `enter` reference | 4 | 4, same vehicle IDs |
| Multiple entries during one simulation step | 4 | 4 |
| Remaining on detector | counted once | south vehicle left at 1.00; one entry |
| Enter/leave between one-second observations | retained | three instant-loop leave records at 0.00 precision; E1 entries retained |
| Simulation-start entry | retained | four entry times 0.0 |
| Vehicle scheduled exactly at end | excluded | no E1 event |

TraCI initially reported `leaveTime=-1` for all four entries; independent instant
output later recorded leave events. This confirms that entry counting must not
wait for a completed leave record. Final `result.json` SHA-256:
`5c2a84c20460b365d31056bdfafbfff379de9ff6958d72220237043502cedaf1`.

Committed fixture hashes are: route
`bf63570f7c75f8923484fc132e4a3693d294beced81d6b4f610f92b7295f7d08`,
SUMO config `124c3dbbd99f11c03093211710d28b355db87fc2fa48d7c6299d004873996263`,
additional detectors `af2cd09916a6b77d47bd17ba2ebe6fa724b80c6b6829da79ad18bd235c7077a8`,
and approach mapping `316516c88306706823e62152e730242c224804b963262e29090284b6097c4163`.

## Dataset/provenance evidence

The deterministic synthetic detector-backed fixture has six windows in each of
train, validation, and held-out test, three distinct demand IDs/hashes, a
60-second input, and a 30-second target. Moving/counting the event affects the
future target and not the preceding input. Repeated generation produced equal
artifact and manifest-payload hashes. Example artifact hashes:

- `train.npz`: `eaef2caadf36663db7fd906ccc46cbdad698a7e63c46e297e175766aa5f54aee`
- `validation.npz`: `fe0549279a24a4efc490f0bc2c8d1167b2d412797e95d4260e3a4ccb962f34f2`
- `test.npz`: `1ed61e16218100dd58380c55e6d4a475440aa371db8b91ea3b46d07a94ab538e`
- `scaler.json`: `2a5e565139d22badb0719d5943f17ebe0fa65353e8b596bb4bc083e45cf62f7f`
- manifest payload: `13a7905fe87bfa557b1de8df379628e50403b7888239aa2491adee717217e8d3`

The example is a contract fixture, not training data. Its readiness is false.
Artifacts were written only under `D:/temp` and were not committed.

## Verification commands and results

```powershell
python -m unittest DeepQN.tests.test_phase2b1_contracts DeepQN.tests.test_phase2b2a_detector_events -v
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
python -m compileall -q DeepQN/evaluation DeepQN/forecasting lstm/research_data.py DeepQN/tests
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" diff --check
```

- Focused contracts: **26 passed**, 0 failed, 0 skipped, 6.195 seconds.
- Full discovery: **71 discovered; 70 passed; 1 explicitly skipped legacy
  interactive diagnostic; 0 failed**, 15.410 seconds.
- Compilation: exit 0.
- Diff check: exit 0; only Git LF/CRLF warnings.

## Unresolved semantics and training prohibition

- No bounded real fixture exercised an actual lane change across parallel E1s.
  Ambiguous duplicate reports fail closed in deterministic tests.
- No forced teleport crossing was produced. A teleport across/onto a boundary
  could have detector-specific behavior and remains unverified.
- Later re-entry was verified at collector-contract level, not with a cyclic
  real-SUMO route.
- The tiny fixture demonstrates E1 detection of sub-step detector traversal,
  but not a vehicle traversing an entire short mapped edge between snapshots.
- Tahrir placement has structural coverage only. The 0.20 m edge requires
  independent review before treating its 0.10 m plane as scientifically useful.
- There is no Taksim detector mapping.
- The live collector is not connected to DQN or historical BiLSTM inference;
  doing so or training belongs to a later approved phase.

Therefore publication-grade BiLSTM training remains prohibited. Authorization
requires traffic-engineering approval of every boundary, real multi-lane and
teleport characterization (or a preregistered exclusion/failure rule), a
production collection manifest with genuine frozen config/demand hashes, and
an approved mapping whose hash is carried through dataset and model manifests.

## Final detector/data-contract correction

### Reproduced integrity gaps and corrections

- **Sampling continuity:** detector-backed sources bypassed the inferred
  extractor's time-gap check. `build_causal_windows()` now validates every
  source before selecting an event mode. Timestamps must be finite, monotonic,
  unique, and differ by exactly one second. Missing, duplicate, reversed,
  two-second, and fractional gaps are rejected for both detector-backed and
  snapshot-inferred sources.
- **Event accounting:** every event is validated against the exact detector ID,
  TLS, direction, and lane in the hashed detector mapping. Vehicle identity,
  entry/leave timestamps, deterministic order, and exact event identity must be
  valid and unique. Entry times are assigned to the SUMO reporting interval
  `[t-1,t)` using `floor(entry)+1`; events exactly at 1.0 belong to bin 2,
  events just below 1.0 belong to bin 1, and events just above 1.0 belong to bin
  2. Every resulting bin must exist in the snapshot timeline.
- **Crop policy:** the only supported policy is
  `reject_outside_recorded_interval`. Legitimate events outside an intentionally
  selected interval must be placed in a separate immutable raw source; they
  cannot be silently dropped from a cropped source.
- **Event serialization:** `sumo-e1-event-file-v1` is deterministic canonical
  JSON containing the event-contract version, source identity, detector-mapping
  hash, network hash, and ordered event records. Loading revalidates the entire
  contract and returns the actual file SHA-256.
- **File-backed provenance:** detector-backed sources now require actual
  snapshot, event, SUMO-config, and frozen-demand files. Their bytes are hashed
  and checked against the source specification. The in-memory events must equal
  the immutable event file. Dataset generation also requires exact approach,
  network, and generated detector-mapping hashes; altered files are rejected.
- **Generator support:** `generate_dataset_from_manifest()` now supports
  `event_source_mode="sumo_e1"`, loads the event file, verifies every declared
  input hash, propagates event paths/hashes and mapping identity, and rejects
  mixed event modes across train/validation/test.
- **Waiting-time contract:** every detector-backed source config must explicitly
  contain `waiting-time-memory=100`; absent or different values fail. The
  bounded runner verifies the source config and also passes
  `--waiting-time-memory 100` explicitly to SUMO. Its result records the value.
- **Publication approval:** the mutable
  `DetectorMapping.traffic_engineering_validated` boolean is no longer consulted
  for readiness. A future approval must supply an evidence ID, an evidence file
  whose bytes match its SHA-256, and the exact approved detector-mapping hash.
  Synthetic fixtures remain not publication-ready.

### Files changed in this correction

- `DeepQN/forecasting/detector_events.py`
- `DeepQN/forecasting/validate_detector_fixture.py`
- `lstm/research_data.py`
- `DeepQN/tests/test_phase2b1_contracts.py`
- `DeepQN/tests/test_phase2b2a_detector_events.py`
- `research/2026-10-25-phase2b2a-evidence-ledger.md`

No historical result, model, checkpoint, reward, replay buffer, or policy code
was changed.

### Hash evidence

Canonical synthetic event-file hashes generated from the corrected serializer:

- train: `bc2f949e0e6e9e9cba5dc622ecf17d2c6ecdae66d67bbf88e3049cbc5ab5b7be`
- validation: `283721a638d40589ad996a44b77a6ca729ccea256837cc662a483c1601345d90`
- test: `45f19b13a58836234bc50ee49dd019a6a911868d4f71295cc3730c071b714ef4`

The detector-backed tensor contents remain deterministic. The corrected tests
regenerate the dataset twice from unchanged file-backed inputs and require
equal artifact and manifest-payload hashes. Tensor/scaler hashes remain:

- `train.npz`: `eaef2caadf36663db7fd906ccc46cbdad698a7e63c46e297e175766aa5f54aee`
- `validation.npz`: `fe0549279a24a4efc490f0bc2c8d1167b2d412797e95d4260e3a4ccb962f34f2`
- `test.npz`: `1ed61e16218100dd58380c55e6d4a475440aa371db8b91ea3b46d07a94ab538e`
- `scaler.json`: `2a5e565139d22badb0719d5943f17ebe0fa65353e8b596bb4bc083e45cf62f7f`

These are synthetic contract artifacts, not scientific training data.

### Final verification

```powershell
python -m unittest DeepQN.tests.test_phase2b1_contracts DeepQN.tests.test_phase2b2a_detector_events -v
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
python -m compileall -q DeepQN/evaluation DeepQN/forecasting lstm/research_data.py DeepQN/tests
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" diff --check
```

- Focused: **32 passed**, 0 failed, 0 skipped, in 7.921 seconds.
- Full discovery: **77 discovered; 76 passed; 1 explicit legacy diagnostic
  skip; 0 failed**, in 18.481 seconds.
- Compilation: exit 0.
- Diff check: exit 0; Git emitted only LF/CRLF warnings.

The accepted bounded E1 evidence was preserved. It was exercised by the
existing bounded regression because the collection runner's explicit waiting
configuration and event-file serialization changed; no full scenario ran.

### Remaining blockers

Real multi-lane lane changes, teleport entry behavior, cyclic re-entry, full
short-edge traversal, and independent Tahrir boundary approval remain
unresolved. They are not represented as validated. The publication-readiness
gate remains false, and Phase 2B.2B has not begun.

## Final fail-closed integrity correction

### Reproductions and corrections

- **Readiness authority:** a caller-supplied
  `{"publication_training_ready": true}` previously passed without inspecting
  any evidence. Readiness now requires the actual on-disk manifest and its
  payload hash; all dataset and raw source hashes; the current dataset,
  feature, E1-event, sampling, waiting-memory, partition, mapping, and network
  contracts; and a separate immutable approval record bound to the exact
  approach, detector, and network hashes. The approval evidence hash must also
  be registered by a separately human-reviewed repository change in
  `TRUSTED_MAPPING_APPROVALS`. That trust root is intentionally empty. No
  reviewer, approval, date, or evidence was invented, so current Tahrir and
  synthetic data remain not publication-ready.
- **Offline multi-lane ambiguity:** serialized events with the same TLS,
  vehicle, and exact entry timestamp on different detector IDs are now rejected
  by the common event validator. Serialization, loading, and dataset building
  all use that validator. The live collector now stages identity changes and
  commits them only after the complete batch validates; a failed batch cannot
  poison a retry. A later event at a distinct timestamp remains legitimate.
- **Snapshot substitution:** detector-backed construction now reloads the
  authoritative snapshot JSONL and requires exact dataclass equality with the
  observations used to build windows. Regressions independently alter speed,
  accumulated waiting, edge, vehicle ID, and timestamp while leaving the
  verified source file unchanged; every substitution is rejected.

### Files changed by this final correction

- `DeepQN/forecasting/detector_events.py`
- `lstm/research_data.py`
- `DeepQN/tests/test_phase2b2a_detector_events.py`
- `research/2026-10-25-phase2b2a-evidence-ledger.md`

The earlier event and generated dataset hashes recorded above remain the actual
deterministic synthetic-fixture hashes. They are integrity evidence only and
do not confer scientific approval.

### Commands and exact outcomes

```powershell
python -m unittest DeepQN.tests.test_phase2b2a_detector_events -v
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
python -m compileall -q DeepQN lstm
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" diff --check
```

- Targeted final run: **16 passed, 0 failed, 0 skipped**, 3.954 seconds,
  exit 0.
- Full final discovery: **81 discovered; 80 passed; 1 explicit legacy
  diagnostic skipped; 0 failed**, 20.934 seconds, exit 0.
- Compilation: exit 0.
- Diff check: exit 0; Git emitted only existing LF/CRLF conversion warnings.
- During implementation, the first targeted run passed 16 tests. After adding
  deeper contract verification, a second targeted run exposed two errors caused
  by passing an obsolete second argument to `_validate_waiting_memory`; that
  call was corrected and the final targeted run passed. The first full
  discovery then had one assertion-message mismatch in a pre-existing Phase
  2B.1 test (**81 run, 1 failure, 1 skip**); the error text was made backward
  compatible without weakening validation, and the full final run above
  passed.

No additional long SUMO scenario was launched. Full discovery exercised only
the existing bounded fixtures.

### Remaining scientific limitations

No approval exists in the trust root. Independent Tahrir traffic-engineering
review remains absent. Real multi-lane lane changes, teleport entry behavior,
cyclic re-entry, and full short-edge traversal remain unresolved beyond the
accepted bounded cases. These limitations are fail-closed; code correctness
and passing regressions do not constitute scientific validation. Phase 2B.2B
has not begun.
