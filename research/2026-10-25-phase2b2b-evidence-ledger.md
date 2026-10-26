# SUMOFlowAI Phase 2B.2B evidence ledger

**Completed:** 2026-10-26 (Africa/Cairo)  
**Branch:** `research/ieee-reproducibility`  
**Starting commit:** `b27d8081a79d0a37145d235a511f864cda6e850c`  
**Scope:** strict inference, interval reward, action execution traces, replay
terminal semantics, and future-run reproducibility infrastructure. No model was
trained and no full Tahrir/Taksim scenario was run.

## Implemented contracts

### Strict forecasting

`DeepQN/forecasting/inference.py` defines
`junction-bilstm-60s-to-30s-v1`. A verified model requires seven features in
the exact Phase 2B.1 order and units, 60 consecutive one-second samples, four
nonnegative next-30-second junction-local arrival counts, a declared four-value
DQN state scale, the detector event schema, and exact model/scaler/dataset/
mapping/network hashes. The dataset manifest and inference manifest must agree
on the approach, detector, and network identities. Loading uses
`torch.load(..., weights_only=True)` and strict tensor loading.

Live strict collection uses the validated E1 detector mapping, exact
`getVehicleData()` events, and the same approach speed and 100-second
accumulated-waiting definitions as dataset preparation. Missing history,
artifacts, mappings, detector data, feature order, or model compatibility raises
an explicit error. There is no proxy fallback.

`no_forecast` is a separate named experimental condition that deliberately
places four zeros in the existing 37-dimensional DQN state. These zeros are an
ablation definition, not a replacement for a failed model. The repository's
research configuration selects `no_forecast`; no valid new BiLSTM exists.

The historical checkpoint remains incompatible: its recurrent input tensor has
width six, while the new architecture requires seven inputs. This was inspected
with restricted loading and exercised through the strict adapter. Mock model
outputs were used only in tests.

### Interval reward

For internal SUMO steps (j\in I_k), the implemented shared global components
are:

\[
D_k=\sum_j \Delta t_j\sum_{l\in L}h_l(j)\quad[\mathrm{vehicle\ seconds}],
\]

\[
E_k=\sum_j \Delta t_j\sum_{e\in E}c_e(j)\quad[\mathrm{mg}],\qquad
A_k=\sum_j a(j)\quad[\mathrm{vehicles}].
\]

Each controlled lane has exactly one TLS owner and each edge is deduplicated.
For each TLS, local pressure is the population standard deviation of incoming
lane halting counts. (P_k) is the timestep integral of the mean local pressure
over TLSs, in vehicle-seconds. This is a local queue-imbalance measure, not a
claim of classical upstream-minus-downstream max pressure.

\[
r_k=\operatorname{clip}\left(-\alpha D_k/D_{ref}
-\beta E_k/E_{ref}+\gamma A_k/A_{ref}-\delta P_k/P_{ref},
r_{min},r_{max}\right).
\]

All references must be finite and positive. Every raw component, the unclipped
weighted value, clipped value, and actual interval duration are exposed for
logging. Actual differences in consecutive SUMO times determine
(\Delta t_j), so shortened final intervals and non-unit timesteps are handled.
The old endpoint-difference `RewardFunction` remains available only under the
explicit `legacy_endpoint_v1` contract. The research YAML selects
`interval_v1`.

### Actions and physical traces

The learned action remains a **request**: keep (`0`) or request switch (`1`).
`ActionDecision` records acceptance, deferral reason, pre-action green phase,
and whether a transition started. Reasons distinguish accepted keep/switch,
minimum-green deferral, yellow deferral, and forced maximum-green switching.
Yellow/red/minimum-green behavior is not bypassed.

For every executed SUMO step, the environment records start/end times, actual
TraCI phase index, and red/yellow/green state per TLS. Evaluation writes a
non-overwriting JSON trace and separately hashes requested and physical traces.
The environment also returns an in-memory canonical SHA-256.

### Replay and terminal semantics

Replay records requested actions, the reward for the same executed interval,
`terminated`, `truncated`, and actual interval duration. Failed/invalid
transitions are rejected. Environmental empty-demand termination is distinct
from configured-horizon truncation; a nondivisible final interval is retained
with its shorter duration. Double-DQN targets bootstrap through truncation and
stop only for true termination:

\[
y=r+\gamma(1-\mathbb{1}_{terminated})Q_{target}
(s',\arg\max_a Q_{online}(s',a)).
\]

Uniform and prioritized replay now use isolated RNG objects rather than
mutating global Python/NumPy RNG state. Historical `done` callers retain an
alias whose bootstrap meaning is termination.

### Reproducibility and checkpoints

`seed_all()` records/seeds Python, NumPy, Torch, and CUDA when available and can
enable deterministic Torch algorithms. SUMO accepts an explicit separately
recorded seed. `ResearchRunContract` requires frozen demand identity/hash,
configuration hash, independent training/SUMO seeds, forecast condition,
dataset provenance, Git SHA, and dirty state, and refuses overwrite.

Future `sumoflow-dqn-checkpoint-v2` ensemble files add optimizer state,
architecture/config hash, run-manifest hash, Python/NumPy/Torch RNG state,
TLS identity, episode, and counters. The whole ensemble is preflighted for
collisions and strict evaluation checks shared run provenance and architecture.
Legacy best-effort resume remains separately available. No historical
checkpoint was modified.

The preregistered design remains five independent training seeds per forecast
condition and ten paired evaluation demand realizations per trained policy,
controller, and profile. None were executed.

## Files created or modified

- `DeepQN/forecasting/inference.py` — strict model/no-forecast contracts and
  detector-backed live feature collection.
- `DeepQN/env/reward.py` — interval reward plus explicitly retained legacy
  reward.
- `DeepQN/env/sumo_env.py` — explicit forecast mode, action decisions,
  physical traces, actual-step reward sampling, termination/truncation.
- `DeepQN/agent/replay_buffer.py` — terminal/truncation/duration fields and
  isolated RNGs.
- `DeepQN/agent/dqn_agent.py` — termination-aware targets and immutable v2
  research ensemble checkpoints.
- `DeepQN/training/reproducibility.py` — seed and run-contract infrastructure.
- `DeepQN/training/train.py` — explicit RNG seeding and transition semantics.
- `DeepQN/evaluation/evaluate.py` — non-overwriting requested/physical trace
  artifact and hashes.
- `DeepQN/configs/dqn_config.yaml` — explicit `no_forecast`, `interval_v1`,
  reward references, and SUMO seed.
- `DeepQN/tests/test_phase2b2b_methodology.py` — deterministic contract tests.
- This ledger.

Pre-existing `.vscode`, backend image, `test_metrics_phase1a.py`, untracked
`DeepQN/forecasting/__init__.py`, and untracked `tatus --short` changes were not
modified as part of this phase.

## Verification

Executed commands:

```powershell
python -m unittest DeepQN.tests.test_phase2b2b_methodology DeepQN.tests.test_phase2a_methodology DeepQN.tests.test_checkpoints_phase1a -v
python -m unittest DeepQN.tests.test_phase2b2b_methodology -v
python -m unittest DeepQN.tests.test_phase2b2b_methodology DeepQN.tests.test_tiny_sumo_phase1b -v
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
python -m compileall -q DeepQN lstm
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" -c core.whitespace=cr-at-eol diff --check
```

- Initial focused compatibility run: **26 passed**, 0 failed, 0 skipped,
  11.689 seconds.
- Expanded Phase 2B.2B focused run before the checkpoint test: **10 passed**,
  0 failed, 0 skipped, 5.280 seconds.
- Final focused inference/reward/replay plus tiny integration run: **13 passed**,
  0 failed, 0 skipped, 11.435 seconds.
- Final full discovery: **92 discovered; 91 passed; 1 explicit legacy
  diagnostic skipped; 0 failed**, 23.285 seconds.
- Compilation: exit 0.
- Diff check with CRLF-aware `cr-at-eol`: exit 0. Default Git whitespace
  reporting treats this repository's carriage returns as trailing whitespace;
  no source line contains trailing spaces beyond the CRLF terminator.

Full discovery exercised only existing bounded tiny fixtures. It did not run
Tahrir or Taksim. Test predictions and tiny rewards are contract fixtures and
are not scientific performance evidence.

## Remaining limitations and compatibility consequences

- There is no compatible trained seven-input/four-count BiLSTM; strict model
  mode therefore cannot produce a scientifically valid prediction today.
- Publication training remains prohibited: the Phase 2B.2A mapping trust root
  is empty, and Tahrir boundary review plus unresolved detector behaviors remain
  blockers.
- Historical forecast-conditioned DQN weights are not compatible evidence for
  the new forecast/reward/state semantics. Both forecast and no-forecast DQN
  conditions require new training after separate approval.
- Reward normalization constants in the YAML are implementation defaults, not
  scientifically calibrated references. They must be frozen using training or
  calibration data without final-test leakage before training.
- The local queue-dispersion pressure term is implemented and testable but must
  not be described as classical max-pressure control. A separate max-pressure
  baseline is still required.
- Deterministic mocks, seeds, trace equality, and passing tests demonstrate
  software behavior only. They do not validate forecast accuracy, policy
  quality, traffic benefit, or publication claims.

Phase 2B.3 and Phase 3 were not started.

## Final physical-trace and research-checkpoint integrity correction (2026-10-26)

### Corrected physical trace contract

`SumoEnv.step()` now records three distinct boundary observations for every
executed simulator step: `start_boundary`, `end_boundary_before_controller`,
and `after_controller_update`. The controller update occurs only after the
end-boundary physical query and therefore describes state requested for the
next simulator interval. `executed_state` is populated only when the two
TraCI-observed interval boundaries are both available and identical. When they
differ, the trace records `execution_evidence = "boundary_changed_or_unavailable"`
and does not claim that either endpoint was the state for the complete interval.
This corrects the reproduced case in which yellow phase 1 was executed but a
post-step controller transition caused green phase 2 to be attributed to that
completed interval.

Research evaluation uses `physical_trace_mode = "research_fail_closed"`. A
failed TraCI physical-phase query raises an explicit runtime failure before an
action-trace result or hash can be returned. The previous controller-internal
estimate is available only as the explicitly labeled
`legacy_controller_estimate` in `legacy_estimate` mode. It is never accepted as
physical execution evidence. The tests also exposed and corrected a separate
red-hold bookkeeping defect: an explicit `yellow_to_next` transition into the
configured red-hold phase now sets `in_red_hold` correctly.

The bounded real-SUMO fixture was rerun. It verified that every recorded
boundary sample came from `traci_physical_query`, that samples were present,
and that stable intervals produced physical execution evidence. This validates
the corrected transport and trace schema for the fixture; it does not prove
that endpoint samples reconstruct an unobserved internal phase change.

### Research checkpoint publication and loading

`save_research_ensemble()` now stages every agent checkpoint and the complete
`sumoflow-dqn-ensemble-v2` manifest in an isolated temporary directory. Agent
files are published first and the ensemble manifest is renamed last. On any
publication exception, every file published by that attempt is removed, so an
incomplete ensemble is not presented by a completion manifest. Existing target
paths remain non-overwritable.

`load_for_research_evaluation()` requires the whole-ensemble manifest and
verifies its schema, episode, exact configured TLS membership, run-manifest
hash, checkpoint filenames, constituent SHA-256 hashes, per-agent v2 schemas,
TLS identities, architecture hashes, required provenance/RNG fields, and state
dictionary compatibility. Every checkpoint is restricted-loaded with
`torch.load(..., weights_only=True)` and probed before any live ensemble is
mutated. A failure therefore leaves all live agents unchanged. Schema-less
historical checkpoints remain available only through the explicitly named
`load_legacy_for_evaluation()` path; the historical loader rejects any
schema-declared research checkpoint.

### Discount timing decision

The research configuration explicitly selects `discount_mode =
"decision_step_v1"`. The implemented target remains

`y_k = r_k + gamma * (1 - terminated_k) * Q_target(s_{k+1})`.

Thus one factor of gamma is applied per policy decision, including a shortened
final decision interval; `interval_duration_s` is retained for truncation and
provenance but does not exponentiate gamma. Unsupported discount modes fail
configuration validation. No elapsed-time discounting was introduced and no
model was trained.

### Correction files

- `DeepQN/env/sumo_env.py`: boundary-timed fail-closed physical tracing and
  corrected red-hold state.
- `DeepQN/agent/dqn_agent.py`: atomic ensemble publication, strict manifest
  validation, research-only loading, explicit legacy loading, and declared
  decision-step discounting.
- `DeepQN/evaluation/evaluate.py`: strict whole-ensemble research loader.
- `DeepQN/configs/dqn_config.yaml`: fail-closed trace and discount modes.
- `DeepQN/tests/test_phase2b2b_methodology.py`: adversarial trace, checkpoint,
  publication-failure, mutation-safety, and discount tests.
- `DeepQN/tests/test_tiny_sumo_phase1b.py`: real-SUMO physical-source assertions.
- This evidence ledger.

### Correction verification

Executed commands and outcomes:

```powershell
python -m unittest DeepQN.tests.test_phase2b2b_methodology DeepQN.tests.test_checkpoints_phase1a -v
python -m unittest DeepQN.tests.test_tiny_sumo_phase1b -v
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
```

- The first focused run reproduced one red-hold assertion failure: **26 run,
  25 passed, 1 failed**. The implementation was corrected rather than weakening
  the assertion.
- Corrected focused run: **26 passed, 0 failed, 0 skipped**, 4.545 seconds.
- Bounded real-SUMO run: **2 passed, 0 failed, 0 skipped**, 9.640 seconds.
- Final full discovery: **99 discovered; 98 passed; 1 explicit legacy
  diagnostic skipped; 0 failed**, 19.431 seconds.

Final compilation and Git checks are recorded below after execution. No
Tahrir/Taksim run, training, dependency installation, commit, or push occurred.

```powershell
python -m compileall -q DeepQN lstm
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" -c core.whitespace=cr-at-eol diff --check
```

- Compilation: exit 0, no diagnostics. The three tracked cache files changed by
  compilation were restored exactly; source changes were retained.
- CRLF-aware Git diff check: exit 0. Git emitted only line-ending conversion
  notices for `evaluate.py`, the pre-existing `test_metrics_phase1a.py` change,
  and `test_tiny_sumo_phase1b.py`; it reported no whitespace error.

### Remaining scientific limitations

- Boundary sampling proves observed endpoints, not every physical state within
  an interval. A differing endpoint is deliberately classified as unresolved.
- The bounded real-SUMO fixture did not exercise every possible yellow,
  all-red, teleport, or externally modified signal sequence.
- Atomic cleanup is tested against an injected rename failure on the local
  filesystem. It is not a distributed transaction or protection against
  process/power loss at every machine-instruction boundary.
- Decision-step discounting is now explicit and reproducible, but its scientific
  suitability relative to elapsed-time discounting remains an experimental
  design choice for newly trained models.
- These checks validate code contracts and integrity behavior, not traffic
  benefit, forecast accuracy, learned-policy quality, or publication claims.

## Strict forecast cold-start and live-sensor preflight correction (2026-10-26)

### Reproduced blocker and selected initialization contract

The strict adapter correctly rejected `predict_for()` without exactly 60
consecutive observations, while `SumoEnv.reset()` previously requested an
initial state immediately after clearing adapter history. Consequently a real
`verified_model` environment could not complete reset. Existing tiny SUMO
integration used `no_forecast` and did not exercise this path.

The corrected contract uses a **60-second observed controller warm-up**. At
each one-second SUMO step the environment issues a keep request through the
same phase-safety controller, advances SUMO, updates the controller, and then
collects the real E1/snapshot feature row. The first forecast-dependent policy
decision occurs at `simulation begin + 60 seconds`. Research DQN evaluation
requires `initial_control_protocol = "phase_controller_keep_v1"`, one-second
steps, and exactly 60 warm-up seconds for both `verified_model` and
`no_forecast` comparison conditions.

The frozen demand-normalization interval remains `[configured begin,
configured end)`. Vehicles, waiting, departure backlog, arrivals, emissions,
and signal evolution during warm-up remain in SUMO's XML outputs and therefore
in the study outcomes. Warm-up does not create a learned-policy action, replay
transition, or RL reward. After history completion, the reward accumulator is
reset before the first DQN transition. The manifest records the protocol,
duration, sample count, arrivals, first decision timestamp, and unchanged
demand window. This is a prospective experimental contract; changing it later
requires design approval and new matched runs.

Reset is transactional with respect to evaluation validity: detector/history
preflight or prediction failure closes TraCI and propagates the error. The
evaluation runner records a failed manifest and cannot return a valid
controller result.

### Detector and waiting-memory preflight

Before strict warm-up, `StrictForecastAdapter.validate_runtime_installation()`
now verifies:

- the configured network file exists and matches the inference manifest's
  SHA-256;
- every detector in the exact hashed detector mapping is present in the live
  TraCI induction-loop ID list;
- every required detector reports the mapped lane and position; and
- the verified detector-mapping and network identities are returned for run
  diagnostics.

Missing detectors, wrong lanes/positions, altered networks, or missing mapping
state fail explicitly. Snapshot inference is not used as a fallback. Actual E1
events remain subject to the existing deterministic ownership and duplicate
validation.

Research launch now requires explicit
`simulation.waiting_time_memory_s = 100` and passes
`--waiting-time-memory 100` to SUMO. Missing or contradictory values fail
before startup. The launch value and detector preflight are retained in
controller diagnostics rather than inferred from SUMO defaults.

### Research/legacy separation

The controlled evaluation wrapper explicitly marks its copied configuration as
research evaluation. Research construction requires both `bilstm.mode` and
`physical_trace_mode`; it rejects `legacy_adapter`, missing keys, and any
waiting-memory value other than 100. Historical defaults remain available only
outside this explicitly marked research path. Random-policy tiny integration
remains an integration diagnostic and is not treated as a forecast comparison.

### Files changed in this focused correction

- `DeepQN/forecasting/inference.py`: strict history readiness and live E1/
  network installation preflight.
- `DeepQN/env/sumo_env.py`: fail-closed research-mode configuration, live
  warm-up, first-decision timing, waiting-memory launch enforcement, and
  startup cleanup.
- `DeepQN/evaluation/evaluate.py`: marks controlled runs as research runs,
  requires the matched forecast protocol for DQN, and records warm-up/sensor/
  waiting-memory diagnostics.
- `DeepQN/configs/dqn_config.yaml`: explicit 100-second waiting memory and the
  shared 60-second `phase_controller_keep_v1` warm-up.
- `DeepQN/tests/fixtures/tiny_sumo/tiny_eval.yaml`: explicit no-forecast,
  fail-closed physical trace, and waiting-memory settings; zero warm-up remains
  appropriate for this random-policy integration diagnostic.
- `DeepQN/tests/test_phase2b2b_methodology.py`: cold-start, priming, first
  decision, equal-treatment, detector identity, research-mode, and launch
  preflight regressions.
- This evidence ledger.

### Exact verification

```powershell
python -m unittest DeepQN.tests.test_phase2b2b_methodology -v
python -m unittest DeepQN.tests.test_phase2b2b_methodology DeepQN.tests.test_phase2a_methodology DeepQN.tests.test_tiny_sumo_phase1b -v
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
python -m compileall -q DeepQN lstm
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" -c core.whitespace=cr-at-eol diff --check
```

- Initial focused correction suite: **24 passed, 0 failed, 0 skipped**, 6.981
  seconds, exit 0.
- Final focused methodology/preflight/tiny-integration suite: **37 passed, 0
  failed, 0 skipped**, 10.859 seconds, exit 0.
- Final full discovery: **106 discovered; 105 passed; 1 explicit legacy
  diagnostic skipped; 0 failed**, 21.001 seconds, exit 0.
- After adding the warm-up physical signal/action trace, the final focused
  rerun was **25 passed, 0 failed, 0 skipped**, 5.596 seconds, exit 0; the final
  full rerun was **106 discovered; 105 passed; 1 explicit legacy diagnostic
  skipped; 0 failed**, 21.400 seconds, exit 0.
- Compilation: exit 0, no diagnostics. The three tracked cache files changed
  by compilation were restored exactly.
- CRLF-aware diff check: exit 0; only Git-for-Windows line-ending conversion
  notices were emitted.

Full discovery reran only the existing bounded SUMO fixtures. No new long SUMO
run was needed: cold-start control, detector installation, hash identity, and
launch-command behavior were deterministically exercised. The real bounded
tiny fixture remained `no_forecast`; no compatible verified BiLSTM artifact
exists, so no real strict-model performance or startup success is claimed.

### Remaining restrictions and scientific limitations

- No publication-approved Tahrir mapping or compatible seven-input model
  exists. A full strict Tahrir startup remains prohibited and unvalidated.
- Runtime preflight proves detector ID/lane/position agreement and file hash
  identity. It does not independently validate traffic-engineering placement,
  unresolved lane changes, teleports, cyclic re-entry, or short-edge behavior.
- The selected warm-up affects traffic by design and is included in every
  outcome. Its suitability must remain frozen before training/evaluation; it
  must not be tuned using final test demand.
- No forecast accuracy, controller benefit, or publication performance claim
  follows from synthetic predictors, mocks, or passing contract tests.

Phase 2B.3 and Phase 3 were not started.

## Final episode-reset and research-sensor integrity correction (2026-10-26)

### Reproduced failures and corrections

1. **Episode-local phase state survived reset.** The previous reset path
   manually cleared `phase_elapsed_steps`, `in_yellow`, and `yellow_elapsed`,
   but omitted `in_red_hold` and `forced_switch`. `_PhaseController.reset()` is
   now the single complete reset contract and restores the initial green phase,
   elapsed counters, yellow state, red-hold state, and forced-switch flag.
   Regression tests dirty the controller as if an episode ended in red hold and
   as if it ended during a forced yellow transition, then reuse the same
   environment and confirm that no state survives.

2. **Initial TLS initialization was warning-only.** Reset now commands the
   configured initial green phase and immediately queries the physical TraCI
   phase and red/yellow/green state. Research reset requires an available,
   nonempty physical state and the exact commanded phase. Failure, mismatch, or
   an unavailable query closes TraCI and aborts reset. Confirmed initial states
   are retained in `_initial_tls_states`. The bounded tiny SUMO integration
   exercised the successful physical initialization path; deterministic tests
   exercised mismatch failure.

3. **Lane sensor exceptions became invented zeros.** Research observations now
   use `lane_observation_mode = "research_fail_closed"`. Halting number,
   waiting time, and occupancy are read separately; an exception or nonfinite
   result raises with the exact TLS ID, lane ID, and measurement name. A
   post-warm-up failure prevents `step()` from returning a transition, so the
   evaluation runner cannot mark the run complete. Zero substitution remains
   only in the explicitly labeled `legacy_zero_fill` mode outside research
   evaluation.

4. **Research accepted explicit legacy physical tracing.** A configuration
   with `research_evaluation=true` now requires
   `physical_trace_mode="research_fail_closed"`; an explicit
   `legacy_estimate` value is rejected. The separately labeled legacy path is
   preserved outside research evaluation.

### Files changed in this correction

- `DeepQN/env/sumo_env.py`: complete controller reset contract, physical
  initialization confirmation, strict research trace-mode requirement, and
  measurement-specific lane sensing.
- `DeepQN/configs/dqn_config.yaml`: explicit fail-closed research lane mode.
- `DeepQN/tests/fixtures/tiny_sumo/tiny_eval.yaml`: explicit fail-closed lane
  mode for the bounded research integration fixture.
- `DeepQN/tests/test_phase2b2b_methodology.py`: repeated-episode, initial TLS,
  strict trace configuration, three lane-query, legacy-label, and post-warm-up
  failure regressions.
- This evidence ledger.

No reward, forecast architecture, checkpoint, replay, or experiment-design
semantics were changed.

### Exact verification record

```powershell
python -m unittest DeepQN.tests.test_phase2b2b_methodology -v
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
python -m compileall -q DeepQN lstm
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" -c core.whitespace=cr-at-eol diff --check
```

- Focused regressions: **30 passed, 0 failed, 0 skipped**, 5.866 seconds,
  exit 0.
- Full discovery: **111 discovered; 110 passed; 1 explicit legacy diagnostic
  skipped; 0 failed**, 20.634 seconds, exit 0.
- Compilation: exit 0, no diagnostics. The three tracked cache files modified
  by compilation were restored exactly.
- CRLF-aware diff check: exit 0. Git emitted only line-ending conversion
  notices and no whitespace error.

The full suite reran only existing bounded SUMO fixtures. No Tahrir/Taksim run,
training, dependency installation, commit, or push occurred.

### Remaining restrictions

- Successful initialization confirms the phase index and current signal-state
  string at the reset boundary; it does not reconstruct unobserved internal
  phase changes later in an interval. The existing boundary-trace limitation
  remains.
- Fail-closed lane reads establish data availability and identity, not sensor
  calibration or traffic-engineering validity.
- No compatible publication-ready BiLSTM or approved Tahrir detector mapping
  exists, so strict full-scenario forecast evaluation remains prohibited.
- Passing tests establish software contract behavior only, not forecast
  accuracy, learned-controller quality, traffic benefit, or publication-ready
  evidence.

Phase 2B.3 and Phase 3 were not started.

## DQN observation normalization and research-action integrity correction (2026-10-26)

### Occupancy defect and corrected observation contract

SUMO TraCI `lane.getLastStepOccupancy()` supplies percentage points. The prior
state builder clipped that raw value directly into `[0, 1]`, producing 0.25 for
0.25% and saturating every value of 1% or greater at 1. The corrected lane
feature validates a finite raw value in `[0, 100]`, divides it by 100, and only
then applies observation clipping.

| Raw TraCI occupancy | DQN occupancy feature |
|---:|---:|
| 0% | 0.0 |
| 0.25% | 0.0025 |
| 1% | 0.01 |
| 5% | 0.05 |
| 30% | 0.30 |
| 80% | 0.80 |
| 100% | 1.0 |

Negative values, values above 100%, NaN, and infinity fail closed in research
mode. The state remains 37-dimensional. Research configuration now binds the
semantic version `sumoflow-dqn-observation-v2-occupancy-percent`, which is also
recorded in controller diagnostics. This correction does not change the
separate seven-feature BiLSTM contract. Historical DQN checkpoint compatibility
is **not** implied: unchanged tensor dimensions do not make weights trained on
the old saturated occupancy representation valid for this observation contract.

### Comparative warm-up enforcement

Every research evaluation must explicitly declare the boolean
`simulation.comparative_research`. A comparative run fails when
`require_matched_forecast_warmup` is absent or false, or when the declared
protocol is not the shared 60-second, one-second-sampled
`phase_controller_keep_v1` initialization. Both `verified_model` and
`no_forecast` DQN conditions therefore enter their first learned decision after
the same observed initialization.

The tiny random-policy fixture explicitly declares
`comparative_research=false`, uses no warm-up, and remains an integration
diagnostic rather than a matched research evaluation. The common DQN warm-up
remains inside the frozen demand-normalization and XML measurement window.
Fixed-time and future controller baselines include traffic from that same full
window, while the DQN policies share a documented initial-control segment; the
protocol must be frozen and reported when interpreting controller comparisons.

### Atomic research action preflight

Before any `_PhaseController.apply_action()` call, `SumoEnv.step()` validates
the complete action dictionary. Comparative research requires exactly the
configured TLS ID set. Unknown/extra IDs, missing IDs, booleans, floats,
negative values, and integers outside `{0, 1}` are rejected. Validation occurs
before controller mutation, and adversarial tests confirm every controller's
state is unchanged after each invalid dictionary. A valid complete multi-TLS
request retains the established keep/switch request semantics and records the
requested actions separately from execution decisions.

### Files changed in this correction

- `DeepQN/env/sumo_env.py`: versioned observation contract, occupancy-percent
  conversion, comparative-run declaration, mandatory matched warm-up, and
  atomic action-dictionary validation.
- `DeepQN/evaluation/evaluate.py`: explicit comparative/non-comparative run
  classification and observation-contract provenance.
- `DeepQN/configs/dqn_config.yaml`: research observation-contract declaration.
- `DeepQN/tests/fixtures/tiny_sumo/tiny_eval.yaml`: explicit versioned
  observation contract and non-comparative diagnostic classification.
- `DeepQN/tests/test_phase2b2b_methodology.py`: numerical occupancy,
  invalid-value, matched-warm-up, and action-integrity regressions.
- This evidence ledger.

No reward, BiLSTM representation, checkpoint publication, replay, or phase
safety semantics were changed.

### Exact verification

```powershell
python -m unittest DeepQN.tests.test_phase2b2b_methodology -v
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
python -m compileall -q DeepQN lstm
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" -c core.whitespace=cr-at-eol diff --check
```

- Focused regressions: **34 passed, 0 failed, 0 skipped**, 5.909 seconds,
  exit 0.
- Full discovery: **115 discovered; 114 passed; 1 explicit legacy diagnostic
  skipped; 0 failed**, 21.135 seconds, exit 0.
- Compilation: exit 0, no diagnostics. The three tracked cache files modified
  by compilation were restored exactly.
- CRLF-aware diff check: exit 0; only line-ending conversion notices were
  emitted.

No training, full Tahrir/Taksim run, dependency installation, commit, or push
occurred. Correct software normalization and input validation are not evidence
of policy quality or traffic benefit. Phase 2B.3 and Phase 3 were not started.
