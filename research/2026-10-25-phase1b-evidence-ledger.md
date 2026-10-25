# SUMOFlowAI Phase 1B evidence ledger

**Date:** 2026-10-25 (Africa/Cairo)  
**Branch:** `research/ieee-reproducibility`  
**Starting commit:** `5b93cd21502e09deef3c4a8100a1c7d4b0e49c2b` (dirty Phase 1 worktree)  
**Scope:** tiny CPU-only SUMO 1.26.0 fixed-time/random end-to-end validation.

## Preflight

- `python -c "import traci; ..."`: exit 0, TraCI import succeeded.
- `sumo --version`: exit 0, Eclipse SUMO 1.26.0 Windows release build.
- No dependency was installed and no historical scenario/output was modified.

## Commands and exit status

```text
netconvert --node-files DeepQN/tests/fixtures/tiny_sumo/tiny.nod.xml --edge-files DeepQN/tests/fixtures/tiny_sumo/tiny.edg.xml --tls.guess true --output-file DeepQN/tests/fixtures/tiny_sumo/tiny.net.xml
```

Exit 0: `Success.`

The initial full test run exited 1 after 31 unit tests passed: the real integration exposed Tahrir-edge queries from the hard-coded BiLSTM adapter and then an opaque missing-provenance error. After isolating the fixture with a no-forecast adapter, a second run exposed SUMO's `meanSpeed=-1.00` unavailable sentinel. These failures were retained as evidence and fixed with regression coverage.

```text
python -m unittest DeepQN.tests.test_metrics_phase1a DeepQN.tests.test_checkpoints_phase1a DeepQN.tests.test_tiny_sumo_phase1b -q
```

Final exit 0: `Ran 34 tests in 9.617s` / `OK`.

```text
python -m DeepQN.evaluation.tiny_sumo_validation --output-root D:\temp\sumoflow-phase1b-validation-20261025-final2 --seed 17 --repeat 2 --base-port 18963
```

Exit 0:

```json
{"identical_raw_output_hashes": false, "reproducible_controller_diagnostics": true, "reproducible_metrics": true, "runs": 2}
```

## Run results (identical parsed metrics in both repetitions)

| Metric | Fixed time | Explicit random policy |
|---|---:|---:|
| Valid / reached 40 s horizon | yes (last step 39 s) | yes (last step 39 s) |
| Tripinfo records / summary arrived | 4 / 4 | 4 / 4 |
| Running + waiting at end | 0 | 0 |
| Teleports / collisions | 0 / 0 | 0 / 0 |
| Mean waiting (s) | 3.0 | 4.0 |
| Mean travel (s) | 21.75 | 23.0 |
| Mean time loss (s) | 7.0075 | 7.9125 |
| Network CO2 (mg) | 363031.77000000014 | 373227.5800000001 |
| Network CO2 / reported arrival (mg) | 90757.94250000003 | 93306.89500000002 |
| Mean trip `CO2_abs` (mg) | 90451.54000000001 | 93000.4925 |

Random policy diagnostics: controller label `random`, checkpoint manifest null, one controlled TLS (`center`), 8 decision requests, 4 requested switches, and an action-trace SHA-256 recorded. Both repetitions had identical diagnostics/action hash.

The paired manifests passed `validate_comparability`: same seed, SUMO version, metric definition, horizon, step length, effective SUMO configuration, and hashes for the network, route file, and additional TLS file. Both output sets have recorded SHA-256 hashes and passed integrity validation.

## Reproducibility interpretation

Parsed metrics and the random action-trace hash were identical across two fresh output directories. This is controlled reproducibility for this fixture: demand is fixed, vehicle `sigma=0`, seed is explicit, inputs are hashed, and the random controller uses the same seeded generator.

Raw XML hashes differed between repetitions because SUMO embeds generation timestamps and the temporary output/configuration paths in XML comments. The ledger therefore does not claim byte-for-byte reproducibility; every raw file is nevertheless individually hashed and integrity checked.

## Bugs found and corrected

1. The old launcher discarded the process handle/output, slept for three seconds, and used GUI-oriented flags. It now uses `traci.start` retrying connection, an explicitly checked port, owned shutdown with `traci.close(wait=True)`, and a diagnostic log.
2. Fixed-horizon evaluation could terminate when demand emptied. `terminate_on_empty` now defaults to the historical training behavior (`true`) and is explicitly false for the fixed-horizon fixture.
3. Git provenance failed on Windows because `safe.directory` used backslashes. It now uses the Git-compatible forward-slash path.
4. Missing provenance errors now identify exact fields.
5. Real SUMO summary output uses `meanSpeed=-1.00` when unavailable. The parser skips exactly that sentinel and still rejects other negative/nonfinite speeds.
6. The BiLSTM adapter ignores its configured `use_real_model: false` setting and queried Tahrir edges. Phase 1B does not alter BiLSTM semantics; the smoke runner injects a deterministic no-forecast adapter and records `model_mode=not_applicable_random_policy`.

## Limits

- Four vehicles and one TLS validate plumbing, not controller effectiveness or statistical inference.
- The fixture cleared by the horizon, so unfinished-vehicle handling remains unit-tested rather than exercised here.
- No teleport occurred in this tiny execution; teleport reconciliation remains covered by synthetic regression tests and historical evidence.
- Raw SUMO XML is not byte-identical due to metadata comments.
- Production BiLSTM mode/provenance remains a Phase 2 blocker.
- No trained checkpoint, Tahrir/Taksim simulation, reward change, training, or large experiment was performed.
