# Phase 3 exploratory training diagnostic ledger

Date: 2026-10-27 (Africa/Cairo)  
Branch: `research/ieee-reproducibility`  
Classification: **nonpublication, uncalibrated synthetic-simulation diagnostic**

## Preflight

- SUMO: Eclipse SUMO 1.26.0, Windows release build.
- Network: all seven configured DQN TLS IDs exist among nine frozen programs.
- Lane inventory: all 46 configured incoming lanes exist in the network.
- Configured green-phase indices exist in the corresponding programs.
- Forecast mode: `no_forecast`; no checkpoint or forecast model was loaded.
- Observation: 37-dimensional v3 contract with fail-closed lane queries.
- Physical trace: `research_fail_closed`.
- Seeds: Python/NumPy/Torch `41001`; SUMO `41002`.
- Horizon: night profile `[79200, 86400]`, one-second steps, ten-second decisions,
  `terminate_on_empty=false`.
- Resource bound: 900 wall-clock seconds; isolated TraCI port 18813 and output root.
- Focused strict observation/trace/config tests: 37 passed, 0 failed, 0 skipped.

## Commands and results

`python -m DeepQN.training.diagnostic_train --config DeepQN/configs/diagnostics/night_one_episode_no_forecast.yaml`

Three non-overwriting attempts were made while repairing diagnostic configuration portability:

1. `run-20261026T225147Z`: exit 1 before TraCI. SUMO rejected duplicate output options because
   the source file uses the `<o>` alias. No decisions, transitions or checkpoints.
2. `run-20261026T225221Z`: exit 1 before TraCI. The copied configuration retained a relative
   `gui-settings-file`. No decisions, transitions or checkpoints.
3. `run-20261026T225255Z`: SUMO 1.26 loaded the exact network, additional files and night route,
   and started at 79200. SUMO then emitted source-program safety warnings. A telemetry-only
   Q-array shape error occurred before the first action. TraCI closed at 79200.

The two patcher defects and Q-array flattening defect were corrected for a future retry. The run was
not repeated after the safety warnings because the requested preflight required safe, consistent
signal behavior and the `96621100` sequencing anomaly lacks independent traffic-engineering review.

## Actual third-attempt telemetry

| Field | Actual value |
|---|---:|
| Complete episode | No |
| Simulated duration | 0 s |
| Wall-clock duration | 6.312 s |
| Python process CPU time | 6.28125 s |
| CPU/wall ratio | 0.9951 |
| Device | CPU |
| CUDA available / allocated | false / 0 bytes |
| RAM peak | unavailable; Windows process sampler returned no value |
| Decisions / replay transitions | 0 / 0 |
| Optimizer updates | 0 |
| Loss | unavailable |
| Q-value sample | unavailable; telemetry failed before recording |
| Epsilon | 1.0 for every fresh agent |
| Requested/applied actions | none |
| Reward/components | none |
| Vehicles | loaded 1, inserted 0, running 0, waiting 0 |
| Termination | TraCI requested termination at 79200 after diagnostic failure |
| Checkpoint | none |

SUMO console warnings:

- unsafe green phase 0 at native TLS `96621068` because lane `10873186#4_0` is targeted by two
  `G` links;
- missing yellow phase in TLS `96621100` when switching to phase 0;
- unsafe green phase 2 at TLS `96621100` because lane `10873186#9_1` is targeted by two `G` links.

No teleport or collision occurred before shutdown. The small XML files in the third attempt are
startup/termination artifacts, not valid episode outcomes.

## Evidence paths

- Diagnostic configuration: `DeepQN/configs/diagnostics/night_one_episode_no_forecast.yaml`
- Diagnostic runner: `DeepQN/training/diagnostic_train.py`
- Failed attempt reports and effective configurations:
  `DeepQN/diagnostics/phase3_exploratory_night/run-*/`
- Third SUMO log: `D:/temp/sumoflow-sumo-xvbtxb5h.log`

Any future checkpoint created by this pathway would contain legacy diagnostic weight dictionaries
only. It would lack the immutable whole-ensemble research manifest and must never be labeled
publication-compatible.

## Decision and next action

No exploratory budget is recommended yet because a single complete episode has not passed. Obtain
an explicit traffic-engineering disposition for the three native-program warnings—especially the
controlled `96621100` transition—then rerun the same one-episode command. If that completes, a
reasonable next diagnostic is 3–5 night episodes with unchanged inputs and seeds allocated in
advance; this would examine runtime stability and whether the 1,000-transition replay threshold is
crossed, not establish performance or convergence.

No approval flag or trust root changed. No publication training, historical checkpoint load,
commit, push or full multi-profile experiment occurred.
