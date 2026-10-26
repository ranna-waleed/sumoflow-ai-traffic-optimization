# Phase 2B.3 evidence ledger — controller baselines and experiment preparation

Date: 2026-10-26  
Branch: `research/ieee-reproducibility`  
Starting commit: `cbc28fbc0632b2d3325e9f7bb38cc2afd5a70a41`  
Scope: bounded engineering implementation and validation only. No model training,
Tahrir/Taksim simulation, production calibration, commit, or push was performed.

## Implemented files

- `DeepQN/evaluation/controller_baselines.py`: versioned, fail-closed static and
  actuated signal-program validation; network controlled-link-width validation;
  complete movement-map validation; max-pressure calculation and an environment
  action provider. The controller equation is
  `P(p) = sum_{m served by p} s_m (q_up(m) - q_down(m))`; deterministic ties use
  the smallest phase index. Queue inputs are last-step halted-vehicle counts.
- `DeepQN/evaluation/baseline.py`: native SUMO static/actuated runner with explicit
  program identity/hash, no model/checkpoint, XML metric parsing, non-overwriting
  outputs, bounded wall-clock timeout, and a `secondary_full_operation`
  classification. It refuses to label a native run as the primary matched warm-up
  comparison because no phase-synchronized native handoff has been approved.
- `DeepQN/evaluation/evaluate.py`: accepts only an explicitly supplied validated
  max-pressure action provider, applies the existing 60-second comparative warm-up,
  and records movement/network hashes and pressure definition in controller
  diagnostics. DQN/random behavior is retained.
- `DeepQN/evaluation/reward_calibration.py`: immutable
  `sumoflow-reward-calibration-v1` manifest using only `training` or `calibration`
  partitions. It records file/demand hashes, units, positive-sample counts, NumPy
  version and the explicitly linear positive quantile. Output remains
  `publication_approved: false`.
- `DeepQN/evaluation/compute_pilot.py`: bounded callable and real-TraCI fixture
  timing. Memory means peak Python allocations reported by `tracemalloc`; the
  TraCI sample is explicitly a combined SUMO-step/IPC round trip and is not a
  full-scale estimate.
- `DeepQN/agent/replay_buffer.py`: complete state serialization/restoration for
  uniform and prioritized replay, including transitions, capacity, isolated RNG,
  PER priorities/cursor/size/beta and immutable-copy behavior.
- `DeepQN/tests/fixtures/controller_baselines/actuated.add.xml`: genuine SUMO
  actuated fixture program with variable green bounds and explicit `max-gap`,
  `detector-gap`, and `passing-time` parameters. SHA-256:
  `c29c5f6602a65cafd843378e682dcad941607fb612f33f205a3db9f4abc06b61`.
- `DeepQN/tests/fixtures/controller_baselines/actuated.sumocfg`: deterministic
  seed-17, 40-second fixture configuration. SHA-256:
  `5d675e2b8641214ae63c6902ef2ce25831e304a553c9de5e7ed32f1bd3759d7a`.
- `DeepQN/tests/fixtures/controller_baselines/max_pressure.json`: all sixteen
  controlled movements of the tiny network, explicit phase eligibility and unit
  saturation weights. SHA-256:
  `f35146e6a278b3832e230d2fb345efc94539498cf9c16d52ce43cac19a6d93fd`.
- `DeepQN/tests/test_phase2b3_baselines.py`: deterministic and bounded regression
  tests for all additions.

No unrelated `.vscode`, backend image, `test_metrics_phase1a.py`,
`DeepQN/forecasting/__init__.py`, or stray `tatus --short` change was edited.

## Controller evidence and compatibility

The tiny fixture fixed-time program was parsed as static with the exact phase
durations `[10, 3, 10, 3]` seconds and complete green/yellow states. SUMO 1.26.0
loaded `actuated_fixture` as `TRAFFICLIGHT_TYPE_ACTUATED`; a bounded native run
produced valid tripinfo, emissions, summary, hashes, and a complete manifest with
`model_mode=not_applicable` and no checkpoint. This validates the fixture and
runner, not Tahrir traffic performance.

The max-pressure fixture definition is accepted only when every physical TLS link
is present exactly once, each movement exists in the network, the declared phase
serves the link, and every saturation weight and queue is finite and valid. The
action provider reads every needed queue before returning a complete request, so a
query failure cannot partially apply a signal request. The existing environment
continues to implement yellow, red-hold, minimum-green, maximum-green, requested
versus executed action traces, and physical fail-closed checks; the full regression
suite exercises those contracts.

Tahrir is **fail-closed** for actuated and max-pressure research runs. Its inspected
network programs are static; no approved actuated phase/detector design, complete
movement/saturation map, or independent traffic-engineering approval was supplied.
The tiny definitions must not be transplanted. Taksim was not assessed in this
phase.

## Initialization/fairness contract

The primary DQN/max-pressure policy comparison uses the existing one-second,
60-second `phase_controller_keep_v1` initialization and first decision at
`begin + 60 s`. Max-pressure is routed through the same controlled evaluator.
Native SUMO fixed-time and actuated programs operate from the simulation begin;
stopping them at 60 seconds and handing control back is not known to preserve their
native controller state. They are therefore labelled
`secondary_full_operation`, and attempts to label the current native runner as a
primary matched-policy comparison fail. A future approved protocol must define
phase/program state synchronization at handoff before combining these estimates.
The frozen demand/outcome denominator remains `[begin, end)`.

## Reward-reference calibration

The implemented pipeline can select scales for delay (vehicle-seconds), CO2 (mg),
arrivals (vehicles), and local pressure (vehicle-seconds). It rejects held-out
validation/test/evaluation partitions, invalid/missing demand hashes, negative or
nonfinite samples, missing positive support, and an existing destination. Values
are the prespecified linear 0.95 quantile of positive eligible interval samples by
default. Zeros remain valid observations but cannot define a positive divisor.

No genuine Tahrir calibration input or approval was supplied. Consequently no
production constants were generated and synthetic/tiny inputs remain engineering
fixtures only. Final test outcomes must never be added to a calibration manifest.

## Bounded computational-cost evidence

Installed SUMO: `Eclipse SUMO sumo 1.26.0`; Python: 3.12.0; platform:
Windows 11 AMD64; processor string: AMD64 Family 25 Model 80 Stepping 0,
AuthenticAMD. One separately executed three-step actuated-fixture pilot reported:

- SUMO/TraCI initialization wall time: `1.5713086999894585 s`.
- Three simulation-step plus TraCI round trips: `517600`, `635900`, and `298200 ns`.
- Simulation-loop wall time: `0.0022276000236161053 s`.
- Peak traced Python allocation: `27674 bytes`.

These are single-run engineering timings. They do not isolate SUMO compute from IPC,
do not include a neural network, do not estimate Tahrir/Taksim cost, and have no
uncertainty estimate. The harness enforces a step bound and wall-clock failure;
native SUMO evaluation independently retains the subprocess timeout.

## Training-resume audit

Source inspection found that research checkpoints already contain online/target
weights, optimizer, epsilon, update/step counters, Python/NumPy/Torch CPU RNG states,
episode and provenance. The current training entry point still calls the legacy
best-effort `load_latest`, starts its loop at episode 1, and derives sequential or
random profile choice afresh. Research checkpoints do **not** currently include the
replay buffer, PER state, CUDA RNG state, environment/SUMO RNG state, demand schedule
cursor, or sufficient orchestration state to resume at the identical next episode.
Nor does the training path restore all stored RNG fields.

This phase adds and tests deterministic replay/PER state restoration as a primitive,
but deliberately does not alter the training algorithm or checkpoint schema. The
training pipeline is therefore **not interruption-equivalent and not ready to be
described as completely resumable**.

## Commands and results

Executed commands (all exit status 0 unless noted):

1. `python -m unittest DeepQN.tests.test_phase2b3_baselines -v`
   - Final quiet repeat: 19 tests run, 19 passed, 0 failed, 0 skipped,
     8.412 seconds.
2. `python -m unittest discover -s DeepQN/tests -p "test_*.py" -v`
   - Final quiet repeat: 134 tests run in 29.033 seconds: 133 passed, 0 failed,
     one deliberate skip (`test_ai_behavior`, legacy interactive diagnostic).
     An earlier run exposed a subprocess ResourceWarning because the new pilot used
     non-waiting TraCI close; the implementation was corrected to wait for process
     shutdown, and the final focused/full runs emitted no such warning.
3. A seven-file in-memory `compile(...)` command, avoiding tracked bytecode writes:
   - `compiled 7 changed Python files`; exit 0.
4. `git ... diff --check`, `git ... diff --stat`, and `git ... status --short` are
   - CRLF-aware `git diff --check --ignore-space-at-eol`: exit 0. Git printed only
     existing line-ending conversion notices for tracked files.
5. `sumo --version` printed `Eclipse SUMO sumo 1.26.0`. On this Windows build the
   executable itself returns status 1 for the version request; the version text is
   nevertheless explicit. No simulation failure was inferred from that status.

## Go/no-go for the following training phase

**NO-GO for publication training or Phase 3 experiments.** Required first:

- independent approval of the exact Tahrir movement/saturation map and actuated
  phase/detector program, with hashes;
- an approved common initialization analysis for native full-operation versus
  60-second policy-handoff comparisons;
- approved training/calibration demand artifacts and frozen reward references;
- integration of replay/PER, CUDA and all process RNG state, episode/profile/demand
  cursor, and orchestration state into an atomically published training-resume
  checkpoint, followed by an interruption-equivalence test;
- publication approval of the forecasting detector mapping/data contract remains
  separately required by the Phase 2B.2A gate.

Code correctness on tiny fixtures is not scientific validation of controller
quality, forecast benefit, training convergence, or external validity. No
performance-improvement claim is made.

## Final baseline execution and calibration-provenance correction

Correction date: 2026-10-26. This section supersedes the earlier v1 controller
and calibration descriptions where they conflict.

### Reproduced defects

1. `MaxPressureActionProvider` accessed an invented public `env.controllers`
   member. Production `SumoEnv` stores private `_phase_controllers`, so the original
   provider could not execute against the real environment. The original unit test
   concealed this by creating a test double with the nonexistent member.
2. A pressure-selected phase was converted to binary `switch=1` without proving
   that the selected target was the only next green. This could misstate target
   selection as physical execution on three-green or red-hold topologies.
3. Each movement declared only one service phase even if the physical signal state
   served that link in multiple phases.
4. The native runner parsed an arbitrary `tls_program_path` but SUMO executed the
   `.sumocfg` inputs. An unrelated file could therefore supply the recorded identity.
5. Calibration trusted a sample's self-declared partition and a syntactically weak
   hash instead of verifying an immutable demand artifact and frozen split registry.

### Corrections

- Controller contract is now `sumoflow-controller-baselines-v2`.
  `SumoEnv.phase_controller_states()` exposes frozen decision-boundary snapshots;
  it does not expose mutable controller instances. The provider uses this real API.
- The binary adapter accepts exactly two green phases, no red-hold, and an exact
  match between pressure-service phases and controller greens. Three-green and
  green/red-hold configurations fail before an action is applied. This is an
  explicit topology restriction, not a new DQN action space.
- `last_selected_targets` records pressure targets separately from binary requests.
  Physical execution remains determined by `action_decisions` and the TraCI trace.
  A bounded real `SumoEnv` regression demonstrated target phase 2, an initially
  deferred switch under minimum-green with physical phase still 0, followed by an
  accepted request and observed physical phase 2.
- Max-pressure schema is now `sumoflow-max-pressure-v2`; every movement declares
  the complete `phase_indices` set. The loader derives all `G/g` service phases
  from the network and requires exact equality. A movement served in two phases
  contributes its weighted upstream-minus-downstream pressure to both.
- Native program sources must be the effective network or an actual
  `additional-files` input. The patched configuration is then launched through
  TraCI, and the selected program ID, SUMO type, phase states, durations,
  actuated min/max limits, and parameters are canonicalized and matched to the
  declared program hash. Mismatch produces a failed manifest and no valid result.
- Calibration contract is now `sumoflow-reward-calibration-v2`. Publication-
  candidate mode requires a caller-supplied expected hash for a frozen
  `sumoflow-calibration-split-registry-v1` file. Registry membership binds each
  realization to the actual source file/hash and demand artifact file/hash.
  Duplicate IDs, malformed SHA-256 values, changed source/demand content, unknown
  membership, conflicting partitions, and demand reuse across held-out and
  eligible partitions fail. Engineering-fixture mode remains explicitly separate.
  Every generated manifest still sets `publication_approved=false`.

### Fixture identities after correction

- Actuated source SHA-256:
  `c29c5f6602a65cafd843378e682dcad941607fb612f33f205a3db9f4abc06b61`.
- Actuated configuration SHA-256:
  `5d675e2b8641214ae63c6902ef2ce25831e304a553c9de5e7ed32f1bd3759d7a`.
- Max-pressure v2 mapping SHA-256:
  `3ba1f398b23b43372fb501b744b1a32f83dd4191ee08bcf85976f2fddd0049cf`.
- Canonical actuated program SHA-256 observed both in source and live SUMO:
  `9bb7294ffd0272272a5b79db77831fbe085fbb664ef6f5bf3033fd33d67ae8b1`.
- Canonical static fixture program SHA-256:
  `18ecfaed1d23f274d9f70beafb4172584edd4e65ff723de67941fb47dc484265`.

### Correction verification

- Focused command:
  `python -m unittest DeepQN.tests.test_phase2b3_baselines -v`
  final quiet repeat executed 25 tests: 25 passed, 0 failed, 0 skipped in
  14.564 seconds.
- Full command:
  `python -m unittest discover -s DeepQN/tests -p "test_*.py" -v`
  executed 140 tests: 139 passed, 0 failed, one deliberate legacy diagnostic
  skip, in 33.647 seconds.
- Real SUMO was used only for bounded tiny fixtures: live program identity,
  mismatch rejection, native XML generation, static schedule fidelity, actuated
  activation, and the real-environment max-pressure request/physical transition.
- Compilation and final CRLF-aware diff results are recorded in the final handoff.
  Eight changed Python files compiled successfully using in-memory `compile(...)`.
  CRLF-aware `git -c core.whitespace=cr-at-eol diff --check` exited 0 after
  normalizing the newly inserted block to the file's CRLF convention; Git emitted
  only line-ending conversion notices for other pre-existing files.

### Remaining restrictions

The topology restriction means max-pressure cannot run on any three-green or
red-hold TLS until a separately reviewed safe target-phase adapter is implemented.
No Tahrir movement/service/saturation map or actuated program has independent
traffic-engineering approval. No production split registry, approved calibration
demand, or approved reward references exist. Native full-operation baselines remain
`secondary_full_operation`; they are not represented as equivalent to the common
60-second policy warm-up. Phase 3 and training remain no-go.
