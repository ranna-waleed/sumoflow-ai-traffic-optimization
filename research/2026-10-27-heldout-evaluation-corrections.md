# Held-out evaluation numerical and collision correction — 2026-10-27

Scope: nonpublication synthetic V6 held-out evaluation only. The failed
`full-seed42004-r1` directory was preserved and no SUMO run was started.

## Corrections

- `DeepQN/env/sumo_env.py`: lane occupancy values in the closed interval
  `[-1e-12, 0)` are treated as floating-point zero and recorded through
  `occupancy_zero_correction_*` diagnostics. Values below `-1e-12`, NaN, and
  infinity remain fail-closed. Positive v4 bounded saturation is unchanged.
- `DeepQN/evaluation/evaluate.py`: the observed SUMO wording
  `collision with vehicle` is an adverse traffic-behavior event. Deduplicated
  collision details must equal the structured SUMO collision counter;
  disagreement fails telemetry validation. Route/stop/signal/output failures
  remain execution-integrity failures.
- Regression coverage was added to
  `DeepQN/tests/test_phase3_synthetic_inputs.py` and
  `DeepQN/tests/test_heldout_dqn_evaluation.py`.

## Executed verification

```text
python -m unittest DeepQN.tests.test_phase2b2b_methodology DeepQN.tests.test_phase3_synthetic_inputs DeepQN.tests.test_heldout_dqn_evaluation DeepQN.tests.test_phase2a_methodology DeepQN.tests.test_phase2b3_baselines -v
Ran 107 tests in 36.595s
OK
```

The exact reported occupancy `-6.687083418913758e-18` was accepted as zero.
`-1.0001e-12`, `-0.01`, NaN, and infinity were rejected. The exact observed
`flow_584.1`/`flow_604.1` collision warning was retained once after log-source
deduplication and reconciled with structured `collisions=1`.

No performance or causality claim is made. The collision was emitted during
the fresh untrained-DQN member, not the trained-DQN member.

## R2 postprocessing correction

The preserved R2 untrained run proved that the earlier parser counted two
different semantics as detailed emergency-braking events:

- structured `statistic.xml`: `emergencyBraking=1`, `collisions=1`;
- error log: one per-vehicle braking warning and one collision warning;
- message log: aggregate `Emergency Braking: 1` and
  `Teleports: 0 (Collisions: 1)` lines;
- SUMO diagnostic log: zero relevant lines.

The aggregate message line is now retained as source evidence and reconciled
as a summary counter, but is not counted as a second per-vehicle event. Raw,
deduplicated, summary, and structured counts are reported separately. A
structured count without a detailed warning remains explicitly labelled
`structured_count_exceeds_detailed_warnings`; details exceeding the structured
counter, malformed telemetry, or conflicting summaries fail closed.

Completed action traces are now atomically written immediately after SUMO
shutdown and before telemetry/metric postprocessing. The paired orchestrator
atomically records each controller as it completes and records postprocessing
failures without generating comparisons. A new `--reuse-fixed-report` option
allows a hash- and identity-validated fixed-time member to be reused in a new
attempt.

R2 XML was parsed read-only after correction: 2,106 scheduled/inserted, 2,105
arrived, one running, zero teleports, one collision, valid horizon, and valid
demand reconciliation. R2 remains useful partial metric and traffic-event
evidence, but not a complete DQN evaluation record because its action trace was
never written by the old ordering and cannot be reconstructed.

R2's embedded SUMO statistic configuration additionally records seed `42`,
despite the evaluation manifest label `42004`. The inherited training seed was
passed as a TraCI launch CLI override after the `.sumocfg` was patched. The
controlled evaluator now explicitly sets the launch seed to the paired seed and
records `effective_sumo_seed` in controller diagnostics. Pair validation checks
that binding. Consequently, R2's untrained member is standalone partial
evidence only and is not seed-comparable with its seed-42004 fixed-time member.
