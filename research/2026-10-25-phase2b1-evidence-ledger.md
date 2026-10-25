# SUMOFlowAI Phase 2B.1 evidence ledger

**Date:** 2026-10-25 (Africa/Cairo)  
**Branch:** `research/ieee-reproducibility`  
**Starting commit:** `53e96656bdcf221e549e3ec8699cd6b69ffce107`  
**Scope:** research measurement and forecasting data contracts only.

No model was trained, no reward/policy/replay semantics changed, and no Tahrir or Taksim simulation was executed.

## Changed and created files

### Evaluation measurement contract

- `DeepQN/evaluation/metrics.py`
  - Bumped the metric definition to `sumo-xml-demand-normalized-horizon-wait-v4` (`:10`).
  - Added scheduled, inserted, arrived, running, removed, and undeparted population fields plus insertion/completion rates and horizon-normalized waiting (`:34-45`).
  - Added strict streaming explicit-demand parsing (`:141-180`) and ID-based reconciliation (`:239-309`).
  - Distinguished SUMO 1.26 horizon marker `vaporized="end"` from genuine non-arrival removals (`:206-225`).
  - Verified reported `departDelay` against the frozen demand schedule when present.
  - Preserved arrival-conditional waiting, duration, time-loss, and trip-emission means as secondary outcomes.
  - Added the primary absolute/percentage comparison without fabricating percentages for zero baselines (`:107-117`).
- `DeepQN/evaluation/evaluate.py` and `baseline.py`
  - Both now request `tripinfo-output.write-unfinished=true`, `tripinfo-output.write-undeparted=true`, and identical unscaled emission output (`evaluate.py:65-67`).
  - Both pass the exact route files to the strict demand reconciler (`evaluate.py:200-201`, `baseline.py:83-84`).
- `DeepQN/evaluation/provenance.py`
  - The effective configuration hash retains both tripinfo completeness switches as behavior-relevant settings.

### Bounded congestion fixture

- `DeepQN/tests/fixtures/congested_sumo/backlog.rou.xml`
- `DeepQN/tests/fixtures/congested_sumo/backlog.sumocfg`
- `DeepQN/tests/fixtures/congested_sumo/backlog_eval.yaml`

The fixture reuses the tiny one-TLS network, schedules ten due vehicles on one constrained approach at time zero, includes one vehicle scheduled after the horizon, uses an eight-second horizon and seed 23, and is guarded by a 20-second wall-clock timeout.

### Forecasting contracts

- `DeepQN/forecasting/contracts.py`
  - Defines schema `junction-approach-arrivals-v1`, ordered features `north, south, east, west, total, avg_speed, avg_waiting`, and units `veh/s, veh/s, veh/s, veh/s, veh/s, m/s, s` (`:12-15`).
  - Validates TLS IDs, edge existence, edge destination, lane geometry, travel-heading direction, and globally unique boundary ownership (`:45-117`).
  - Counts a vehicle only when it enters a boundary edge; remaining on the edge cannot create repeated arrival events (`:121-179`).
- `DeepQN/configs/forecast_mappings/tahrir_v1.json`
  - Defines junction-specific boundaries for the seven configured Tahrir TLS agents.
  - SHA-256: `275d8494f9a408ddee515b63402bdac27bba041730726231d35f428ac04c0f98`.
  - No Taksim mapping was created.
- `lstm/research_data.py`
  - Defines 60-second histories and four 30-second future directional arrival-count targets (`:25-28,110-151`).
  - Partitions by whole source runs before windowing. Overlap is allowed only within a partition; identical source content cannot appear in multiple sources/partitions (`:84-108`).
  - Fits the feature scaler exclusively from training inputs (`:159-173`).
  - Writes non-overwriting train/validation/test tensors, scaler, source/config/seed/mapping/network provenance, and artifact hashes (`:188-241`).
  - Provides a JSON-manifest-driven generation entry point (`:244-271`).

The new dataset pipeline is not connected to the historical `train_lstm.py` or live `BiLSTMAdapter`; doing so, changing the model, and training belong to a later approved phase.

### Tests and legacy diagnostic

- `DeepQN/tests/test_phase2b1_contracts.py`
  - Adds demand reconciliation, censored waiting, teleport/removal/running/undeparted cases, shared measurement settings, real SUMO backlog, mapping validation, feature ordering, single-crossing behavior, causality, partition leakage, scaler schema, artifact hashing, and collision refusal tests.
- `DeepQN/tests/test_ai_behavior.py`
  - Is explicitly skipped during discovery because it is an interactive legacy diagnostic rather than isolated unit tests (`:83-87`).
  - Explicit execution now requires a checkpoint episode and uses strict `load_for_evaluation`, so an absent checkpoint directory fails instead of printing a trained-agent success claim (`:50-58,89-102`).

## Actual SUMO 1.26 backlog semantics

The first corrected bounded run exposed the following actual SUMO records:

- Inserted but unfinished vehicles: `depart >= 0`, `arrival=-1`, and `vaporized="end"`; their accumulated `waitingTime` and `departDelay` are populated.
- Undeparted due vehicles: `depart=-1`, `arrival=-1`, `vaporized="end"`, `duration=0`, and `departDelay=H-intended_depart`.
- Summary at the final reported step distinguishes cumulative `inserted`, current `running`, current `waiting`, `arrived`, and `ended`.
- The `end` vaporization value is a fixed-horizon output marker. It must not be counted as a removal and does not increment summary `ended`.
- SUMO's embedded effective configuration showed `write-undeparted`; with that setting, SUMO emitted both inserted-unfinished and undeparted records. The patched source configuration nevertheless explicitly requests both switches for clarity and comparability.

The parser reconciles:

\[
N_{inserted}=N_{arrived}+N_{running}+N_{removed}
\]

and

\[
N_{undeparted}=N_{summary\ waiting}.
\]

Every due demand ID must have exactly one tripinfo record, and no unknown record may enter the denominator.

## Validated primary outcome

For every vehicle due by horizon (H):

\[
d_i^{pre}=\begin{cases}
t_i^{depart}-t_i^{intended}, & \text{inserted},\\
H-t_i^{intended}, & \text{undeparted},
\end{cases}
\]

\[
W_H=\frac{\sum_{i\in due}(d_i^{pre}+w_i(H))}{N_{due}}.
\]

This includes censored pre-insertion backlog and on-network accumulated waiting, preventing a controller from improving the primary result by withholding vehicles from the network.

Final explicit fixture output directory:

`D:\temp\sumoflow-phase2b1-validation-20261025-final`

| Measure | Result |
|---|---:|
| Horizon / seed | 8 s / 23 |
| Due scheduled demand | 10 |
| Inserted / running | 2 / 2 |
| Undeparted | 8 |
| Reported arrivals / removals | 0 / 0 |
| Demand-normalized horizon waiting | 6.9 s/due vehicle |
| Censored pre-insertion delay | 6.8 s/due vehicle |
| Mean departure delay among inserted vehicles | 2.0 s |
| Insertion / completion rate | 0.2 / 0.0 |
| Completed-trip waiting/travel/time-loss means | undefined (`null`) |
| Total network CO2 | 32293.13 mg |
| Teleports / collisions | 0 / 0 |
| Valid / horizon reached | yes / yes |

Output hashes:

- Tripinfo: `f22dbed5351982c7355b4caaf48abc6266f8dc2dfa2e2617ea7003003f3eb850`
- Emission: `1f56fc7c485d0ae95fd1999285e3d7ebcbc73eda9a1ae71564751817274b17eb`
- Summary: `1e17459b1a7e53ec52b608fabfedec8bdb7b43846a682893a54e7e78e47c2204`

The manifest records SUMO 1.26.0, Python 3.12.0, NumPy 1.26.2, Torch 2.7.1 CPU, Git SHA, dirty state, route/network/additional/config hashes, seed, output paths, and completion status.

## Tahrir junction-local mapping evidence

Direction means SUMO travel heading, not the geographic side from which a vehicle originated.

| TLS | Entry edge | Heading | Direction slot |
|---|---|---:|---|
| 315744796 | `690516091#0` | 11.71 deg | north |
| 315744796 | `28718647#3` | 281.49 deg | west |
| 96621100 | `690516091#1` | 350.72 deg | north |
| 96621100 | `10873186#8` | 39.09 deg | north |
| 2031414903 | `50211834#1` | 94.89 deg | east |
| 2031414903 | `10873186#5` | 167.01 deg | south |
| 2031414899 | `10873191#3` | 326.19 deg | north |
| 6288771431 | `10873191#4` | 331.57 deg | north |
| 271064234 | `10873193#1` | 31.28 deg | north |
| 315743335 | `28718874#1` | 81.28 deg | east |

All edges exist, contain lanes, terminate at the declared controlled TLS, and are unique across the mapping. Empty direction lists are deliberate where a TLS has no incoming edge in that travel-heading sector; they are not missing-data fallbacks.

The four forecast values supplied to a future DQN agent are therefore defined as that junction's four local directional future entry counts. They are not global network demand. The current historical adapter remains unchanged and does not yet satisfy this contract.

## Dataset causality and leakage policy

- At decision second (t), input contains exactly seconds (t-59,\ldots,t).
- Target contains crossings in (t+1,\ldots,t+30), summed by direction in vehicles/30 s.
- Feature extraction processes snapshots monotonically and rejects non-one-second gaps, duplicate vehicle observations, negative/nonfinite speed or waiting, unknown mappings, and duplicate edge ownership.
- A vehicle remaining on a boundary edge is not recounted. A later crossing at a different downstream junction is a separate local event because every boundary edge has one owner.
- Train, validation, and held-out test partitions consist of distinct source runs. Sliding windows may overlap within one source/partition but never cross a source or partition boundary.
- Reused source hashes are rejected even when source IDs differ.
- The scaler records ordered feature names/units and is fitted only from training inputs. Targets are physical counts and are not fitted using validation/test data.

The causality regression changes a crossing after the input endpoint and demonstrates that the input tensor is identical while the future target changes.

## Commands and results

### Focused development run

```text
python -m unittest DeepQN.tests.test_metrics_phase1a DeepQN.tests.test_phase2b1_contracts -v
```

First run: 32 passed, one failed. The test placed both altered future events inside the same 30-second target window. The fixture was corrected to move one event outside that target; no production logic changed.

Second run: **33 passed in 3.143 s**, exit 0.

### Final broad discovery

```text
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
```

Result: **56 discovered, 55 executed and passed, 1 explicitly skipped**, in 14.754 s, exit 0.

The skip is only `test_ai_behavior`, labeled `legacy interactive diagnostic; run explicitly with a checkpoint episode`.

### Compilation

```text
python -m compileall -q DeepQN/evaluation DeepQN/forecasting lstm/research_data.py DeepQN/tests/test_phase2b1_contracts.py
```

Exit 0.

### Legacy diagnostic failure safety

```text
python -m DeepQN.tests.test_ai_behavior 200
```

Captured exit: **1**. Exact cause: `checkpoint directory not found: DeepQN\checkpoints`. It did not report trained agents as loaded.

### Bounded final SUMO fixture

```text
python -c "...run_fixed_time_episode(... wall_clock_timeout_s=20)..."
```

Exit 0 in approximately 11.45 s. Only the tiny congestion fixture was run.

`git diff --check` produced only Git-for-Windows LF/CRLF conversion warnings and no whitespace error.

## Compatibility implications and limitations

- Metric v4 is intentionally not comparable with historical v1-v3 reports or manifests. All publication comparisons must be regenerated with explicit frozen vehicle demand and complete tripinfo output.
- Route files containing `<flow>` or `<trip>` elements, procedural/non-numeric departures, duplicate IDs, or missing explicit demand are rejected. They must first be expanded and frozen to explicit vehicles.
- The primary metric has been validated for explicit vehicle demand and fixed-horizon SUMO 1.26 output. Person/container demand is not supported.
- The actual congestion fixture contained zero teleports; teleport-tagged arrivals and genuine non-arrival removals are covered by deterministic XML regression, not by this real fixture.
- The Tahrir boundary mapping is geometrically and structurally validated but has not been exercised in a full Tahrir run or independently reviewed against traffic-engineering intent.
- There is no verified Taksim mapping.
- The data pipeline consumes normalized one-second snapshot JSONL. A production TraCI/FCD extractor that emits accumulated vehicle waiting and source manifests remains to be implemented before dataset generation at scale.
- The historical BiLSTM model, adapter, and DQN policies remain incompatible with the new local-arrival schema. They were neither modified nor trained.
- Reward equations, replay transitions, policy networks, and model architectures remain unchanged.
- Existing `.vscode/settings.json` and backend image changes were not modified, staged, or reverted.

## Stop gate

Phase 2B.1 ends here. Connecting the new contract to inference/training, altering reward/replay semantics, generating full datasets, or training models requires separate approval.

## Phase 2B.1 scientific-contract correction

### Corrected contracts

- `DeepQN/evaluation/metrics.py` now uses metric contract
  `sumo-xml-demand-normalized-horizon-wait-v5`. A valid run must have the
  primary demand-normalized waiting value to be compared. Undefined
  arrival-conditional secondary means no longer abort the primary comparison:
  each unavailable secondary returns a null absolute difference, null
  percentage, and an explanation. A genuine zero baseline retains its absolute
  difference but has an explicitly undefined percentage.
- Eligible scheduled demand is the half-open SUMO execution interval
  `[begin, end)`. Both begin and end are passed by the existing fixed-time and
  controlled XML loaders. An empty eligible population yields null primary and
  rate values, never an observed zero.
- A bounded SUMO 1.26.0 probe (`begin=5`, `end=10`, one-second step) verified:
  departure 4 was absent; departure 5 was inserted; departure 9 was inserted;
  departure 10 was emitted only as an undeparted `vaporized="end"` tripinfo
  boundary marker; departure 11 was absent. The final summary step was time 9
  with two inserted vehicles. The parser validates and excludes the exact-end
  marker from the eligible denominator; any unexpected execution/removal state
  remains an error.
- `lstm/research_data.py` contract v2 requires a frozen demand-realization ID
  and hash plus the snapshot and effective SUMO-config hashes. Identical demand
  IDs or hashes cannot cross partitions even when snapshots differ. Every NPZ
  now carries source ID, TLS ID, input/target start and end timestamps, and
  partition alongside `x` and `y`; all NPZ files remain hashed by the manifest.
- The forecast event label is explicitly
  `inferred-consecutive-1s-edge-entry-provisional-v1`. The first snapshot only
  initializes occupancy; staying on an edge is not recounted; disappearance
  followed by re-entry is a new inferred event. A complete enter-and-leave
  event between samples, teleport semantics, and physical crossing accuracy
  are not observable from snapshots. The dataset manifest therefore records
  `publication_training_ready=false`, and the fail-closed readiness validator
  rejects it. No E1 reference has yet established exact labels.
- The causal waiting feature is frozen as the mean of TraCI
  `vehicle.getAccumulatedWaitingTime()` for vehicles present on the mapped
  approach edges at sample time, with `--waiting-time-memory=100` seconds.
  Every source must record the same 100-second setting. This is a rolling-memory
  state feature, not journey-total tripinfo waiting.
- Paired baseline discovery now uses the actual non-overwriting filename
  `manifest_fixed_time_{profile}_seed{seed}_run{run_index}.json`.

### New regression fixture and tests

- Added `DeepQN/tests/fixtures/demand_window_sumo/window.{rou.xml,sumocfg}`.
- Added comparison tests for two valid zero-arrival runs, zero baselines,
  missing primary values, one-sided missing completed-trip means, and mixed
  arrival counts.
- Added nonzero-begin, empty-denominator, exact-end boundary, paired-manifest,
  demand-realization leakage, exported metadata, provisional-readiness,
  first-snapshot, and legitimate re-entry tests.

### Exact verification record

Direct bounded probe (exit 0):

```powershell
sumo -c DeepQN/tests/fixtures/demand_window_sumo/window.sumocfg --tripinfo-output D:/temp/window-trip.xml --summary-output D:/temp/window-summary.xml --tripinfo-output.write-unfinished true --tripinfo-output.write-undeparted true --no-step-log true
```

Focused suite (exit 0):

```powershell
python -m unittest DeepQN.tests.test_metrics_phase1a DeepQN.tests.test_phase2b1_contracts -v
```

Result: 40 passed in 2.889 seconds.

The first broad run discovered 64 tests and produced 62 passes, one explicit
skip, and one failure. The failure was only an overly narrow expected error
string: production correctly rejected the cross-partition demand reuse as an
ID/hash/partition conflict. The assertion was broadened and the complete suite
was rerun.

Final commands:

```powershell
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
python -m compileall -q DeepQN/evaluation DeepQN/forecasting lstm/research_data.py DeepQN/tests
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" diff --check
```

Final results: **64 discovered; 63 executed and passed; 1 explicit skip; 0
failures** in 13.418 seconds. Unit-test exit 0, compile exit 0, and diff-check
exit 0 (Git emitted only its existing LF/CRLF conversion warnings). The broad
suite executed only bounded tiny SUMO fixtures; no Tahrir or Taksim run occurred.

### Remaining scientific limitations and stop gate

- Snapshot-difference approach-entry targets remain provisional. They can miss
  short-edge crossings between samples and do not independently distinguish
  teleports from physical crossings. Publication-grade forecast training is
  intentionally blocked until an exact TraCI event or E1 detector reference is
  validated.
- The 100-second accumulated-waiting feature is a proposed frozen contract;
  neither the historical training data nor adapter is proven to have used it.
- Metric v5 is incompatible with historical metric contracts and requires new
  matched outputs. Historical artifacts were not modified.
- No model was trained, no reward/replay/policy semantics changed, no dependency
  was installed, and no commit or push was made. Phase 2B.2 and Phase 3 remain
  outside this correction.

## Final boundary-classification regression correction

`reconcile_demand()` previously used `math.isclose(depart, end_s,
abs_tol=1e-9)` for exact-end records. Its implicit relative tolerance could
classify `71999.99995` as both eligible demand and an end boundary when
`end_s=72000`. Exact-end classification now requires `depart == end_s` and
explicitly excludes every ID already in the eligible `[begin, end)` set.

The new regression uses `begin=71900`, `end=72000`, and explicit departures at
`71999.99995`, `72000`, and `72000.00005`. It verifies that only the fractional
near-end vehicle enters the denominator, the exact-end undeparted SUMO marker
remains accepted as a boundary record, and the after-end vehicle is excluded.
The near-end vehicle reconciles as inserted and cannot be silently dropped.

Verification:

```powershell
python -m unittest DeepQN.tests.test_metrics_phase1a DeepQN.tests.test_phase2b1_contracts -v
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
python -m compileall -q DeepQN/evaluation DeepQN/forecasting lstm/research_data.py DeepQN/tests
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" diff --check
```

- Focused: **44 passed**, 0 failed, in 4.216 seconds; exit 0.
- Full discovery: **65 discovered, 64 executed and passed, 1 explicit legacy
  diagnostic skip, 0 failed**, in 13.797 seconds; exit 0.
- Compilation: exit 0.
- Diff check: exit 0; only existing LF/CRLF warnings were emitted.

No full scenario, training, dependency installation, commit, or push occurred.
