# Phase 2C.3 scientific protocol audit and correction ledger

**Date:** 2026-10-26  
**Branch:** `research/ieee-reproducibility`  
**Audited starting commit:** `75557e668fb9fb968810040e1e6b8882e59f9f2d`  
**Decision:** **Phase 3 NO-GO.** This phase establishes neither model performance nor real-world validity.

## Evidence classes and scope

This dossier uses `source-verified`, `artifact-verified`, `bounded-runtime-reproduced`,
`authoritative-documentation`, `related-work report`, and `unresolved`. The supplied
Deep Research review was used as independent-review input; claims below were checked
against repository files or the cited primary source. No training, Tahrir/Taksim run,
dependency installation, commit, or push was performed.

## Targeted correction: DQN lane occupancy

### Reproduction and authoritative semantics

- **Source-verified defect:** `DeepQN/env/sumo_env.py` previously accepted the result
  of `traci.lane.getLastStepOccupancy()` in `[0,100]` and divided it by 100. The old
  contract in `DeepQN/configs/dqn_config.yaml` called it an occupancy percentage.
- **Authoritative documentation:** SUMO's lane TraCI protocol defines the value as
  total vehicle length on the lane divided by lane length, i.e. a dimensionless
  fraction: <https://sumo.dlr.de/docs/TraCI/Lane_Value_Retrieval.html>.
- **Version-pinned source:** SUMO 1.26.0's lane implementation maintains the net vehicle
  length sum separately from the length-with-gap sum; the audited source is pinned at
  <https://github.com/eclipse-sumo/sumo/blob/v1_26_0/src/microsim/MSLane.cpp>.
- **Bounded-runtime-reproduced with installed SUMO 1.26.0:** in the tiny fixture, one
  5 m vehicle on a 92.8 m lane produced
  `0.05387931034482759 == 5 / 92.8`. The prior implementation produced
  `0.0005387931034482759` for the DQN.
- **Distinct E1 contract:** the same SUMO installation returned `100.0` for a fully
  occupied E1 detector step. SUMO documents E1 occupancy as occupied time in percent,
  potentially above 100 with simultaneous vehicles:
  <https://sumo.dlr.de/docs/Simulation/Output/Induction_Loops_Detectors_(E1).html>.
  E1 semantics were not changed.
- **Documentation discrepancy:** the installed 1.26.0 Python lane wrapper docstring says
  “occupancy in %”; the wire-protocol documentation, source path, and actual runtime
  value establish the lane-length fraction used here. The bounded regression prevents
  relying on that ambiguous docstring.

### Corrected contract

The 37-dimensional layout is unchanged. Its lane occupancy element is now

\[
o_l=\operatorname{clip}(\mathrm{TraCI\ lane\ occupancy}_l,0,1).
\]

Research mode rejects negative, greater-than-one, NaN, and infinite inputs. The
contract is `sumoflow-dqn-observation-v3-lane-occupancy-fraction` at
`DeepQN/env/sumo_env.py` and `DeepQN/configs/dqn_config.yaml`. New research checkpoint
and ensemble schemas are v3 and carry the exact observation contract. Strict research
loading rejects v2 manifests/checkpoints; the explicit historical loader remains
separate. `DeepQN/evaluation/evaluate.py` now supplies the configured observation
contract to that strict loader.

This supersedes the Phase 2B.2B ledger's conclusion that lane occupancy should be
divided by 100. It does **not** supersede its fail-closed lane-query behavior, its
37-dimensional layout, or the seven-feature forecasting contract. Existing v2 weights
are semantically incompatible with v3 even though their tensor shapes may match.

## Related work: Fawzy et al. (2026)

Fawzy, N., Mohamed, M. A., Amer, H. M., and Ata, M. M., “Scenario based traffic
optimization in Egypt performance gains through simulation modeling,” *Scientific
Reports*, 16, 10812 (2026), DOI
<https://doi.org/10.1038/s41598-026-41535-x> (open access).

### Comparison with SUMOFlowAI

| Dimension | Fawzy et al. | SUMOFlowAI protocol implication |
|---|---|---|
| Site and network | Umm Kulthum Square, Mansoura; OSM/QGIS/Google Earth geometry and manual checking | Tahrir-derived SUMO network. Neither study validates the other site or establishes transfer. |
| Intervention | Scenario 2 simultaneously changes lane connections, turning movements, signal locations, durations, and states | Proposed contribution is forecast augmentation of otherwise matched Dueling Double DQN arms. Geometry and demand must remain frozen. |
| Causal attribution | Reported gains are joint effects of road and signal modifications; the components are confounded | Forecast contribution requires the same controller, observation layout, reward, training budget, demand, seeds, and horizon except for forecast inputs. |
| Baseline | “Existing” Scenario 1 versus a manually optimized combined scenario | Fixed-time is a secondary reference; no-forecast DQN is the primary ablation. Actuated and topology-supported max-pressure are additional baselines. |
| Horizon and censoring | 999 s and 3599 s; trip metrics coexist with substantial running/insertion backlog | Use the frozen fixed-horizon, demand-normalized waiting burden plus insertion, completion, running, undeparted, throughput, teleports, and failures. Conditional trip means cannot stand alone. |
| Emissions | Paper reports trip/emission outcomes for unequal completed populations | Report network totals and demand-normalized quantities under identical demand/horizon; label completed-trip means as conditional. |
| Repetition and uncertainty | The reviewed article text does not identify SUMO version, random seeds, independent replications, or uncertainty intervals | Record SUMO version and every seed; use five independent training seeds and ten paired held-out demand seeds per profile/controller, with hierarchical paired uncertainty. |
| Geographic evidence | Satellite/OSM construction and geometric measurements | Geographic resemblance is not field-demand calibration. SUMOFlowAI still lacks measured Tahrir counts, queues, travel times, saturation flow, and calibration/validation targets. |
| Contribution | Manual scenario/geometric and signal redesign in an Egyptian case study | Narrow simulation-method contribution: causal forecast augmentation for independent multi-junction Dueling Double DQN. No present real-world deployment or superiority claim. |

### Tables 6 and 7: methodological reading

The paper reports the same 6,252 loaded vehicles for both 3,599 s scenarios, while
more than 5,000 remain awaiting insertion; its narrative gives 5,078 waiting in
Scenario 1. It also reports 500 running vehicles, but unequal arrivals/completions:
672 versus 715. At 999 s, reported completed counts are 262 versus 317. These are
meaningful capacity outcomes, not nuisance records. Comparisons of waiting, duration,
or emissions only among vehicles that finished are conditional on different selected
populations. “Loaded,” “inserted,” “waiting to insert,” “running,” “ended,” and
“arrived” must therefore be reported and reconciled separately.

The article's own summary definitions distinguish loaded, inserted, awaiting insertion,
ended, and arrived, and its abstract reports both different completed counts and
conditional trip metrics. Its reported gains belong to its Mansoura scenarios only.
They are neither SUMOFlowAI results nor calibration evidence for the Tahrir-derived
network.

### Methodological lessons adopted

1. Freeze identical demand bytes and geometry across controller arms; do not attribute
   a combined geometry-and-signal intervention to a controller alone.
2. Predeclare a fixed horizon and report all population states. A controller must not
   appear favorable by preventing insertion or by leaving difficult trips incomplete.
3. Make demand-normalized horizon waiting the primary outcome; retain departure delay,
   insertion/completion rates, throughput, running/undeparted counts, conditional trip
   metrics, teleports, and network emissions as explicit complementary outcomes.
4. Hash the SUMO version, network, routes, additional files, controller, seed, and raw
   outputs. Use paired replications and interval estimates.
5. Describe satellite/OSM construction as geographic construction—not empirical traffic
   calibration—unless independent field observations support calibration and validation.

## Revised scientific-readiness matrix

### Blockers to a simulation-only experiment

| Priority | Verified finding | Evidence | Required closure |
|---|---|---|---|
| P0 | v2 lane occupancy compressed the DQN feature by 100 | Source and bounded SUMO 1.26 test | Corrected to v3; all DQNs must be trained from scratch under v3. |
| P0 | No compatible, publication-approved seven-input forecast checkpoint exists | `lstm/models/config.json` declares seven inputs; strict inference requires 7-in/4-out at `DeepQN/forecasting/inference.py`; Phase 2A artifact-shape evidence found the historical checkpoint incompatible | Approve mapping/dataset evidence, version architecture, train only on frozen training partitions, and evaluate held-out forecasts. |
| P0 | Research BiLSTM count head is unconstrained linear | `lstm/train_lstm.py` final `nn.Linear` at line 52; strict inference rejects negative features/predictions | Freeze a nonnegative output/count-likelihood contract before training. |
| P0 | Historical flow demand is not an explicit vehicle-identity denominator | `DeepQN/evaluation/metrics.py` rejects `<flow>`/`<trip>` for the primary contract | Deterministically expand each publication demand realization to immutable explicit vehicle IDs. |
| P0 | Publication reward references are missing | YAML values at `DeepQN/configs/dqn_config.yaml:256-257` are engineering values; calibration infrastructure does not approve production constants | Derive/freeze references from training/calibration partitions only. |
| P0 | Evaluation design is not frozen in config | `DeepQN/configs/dqn_config.yaml:263` still specifies 3 episodes, not the planned 10 paired demand seeds | Create and hash a publication protocol manifest; do not use an ad hoc override. |
| P1 | Learned-policy uncertainty is not yet estimable | No v3 training runs exist | Five independent training seeds; ten paired held-out demand seeds/profile/controller; training seed is the independent learned-policy unit. |
| P1 | “Max-pressure” applies conventionally to only two controlled TLS | Phase 2B.3/2C.2D source-derived movement inventory | Report seven-TLS study results separately from the two-TLS conventional competing-phase subset; label gate control precisely. |
| P1 | `96621100` native red→yellow→green cycle differs from commanded controller transitions | Frozen program and Phase 2C.2D sequence audit | Keep source anomaly disclosed; obtain traffic-engineering review before any physical-safety claim. Do not silently reorder. |
| P1 | Detector boundary meaning is not independently traffic-engineering approved | Phase 2C.2A/2C.2D ledgers | Keep publication forecast training prohibited until approval evidence is bound to the exact mapping hash. |

### Additional blockers to real-world claims

| Finding | Consequence |
|---|---|
| No measured Tahrir turning counts, queues, travel times, saturation flows, or independent calibration/validation split | The model may be called Tahrir-derived, not empirically calibrated Tahrir traffic. |
| Signal safety and detector placement lack independent traffic-engineering approval | No safety-certified, deployment-ready, or operational recommendation claim. |
| Camera detections do not drive the evaluated DQN path | No YOLO-driven live-control claim. |
| Taksim eight-agent compatibility remains conditional | No transfer/generalization claim until topology, state, mapping, policy identity, and evaluation are validated. |

## Provisional experiment contract (not execution authorization)

- Primary hypothesis: under frozen synthetic/unvalidated Tahrir-derived demand, does a
  strictly validated causal 30 s forecast improve demand-normalized fixed-horizon waiting
  relative to the same v3 Dueling Double DQN with four forecast slots set to zero?
- Arms: fixed-time; feasible native actuated; topology-qualified max-pressure; v3 DQN
  without forecast; v3 DQN with verified forecast. Random policy remains diagnostic.
- Five independent training seeds per learned arm; ten paired held-out evaluation demand
  seeds per profile/controller. Never tune on evaluation demand.
- Report per-run raw outcomes, training-seed-aware paired uncertainty, effect sizes and
  intervals. Correct multiplicity for secondary outcomes. Report failed/teleported runs
  by arm and preregister handling rather than deleting outcome-dependent failures.
- No model is assumed to outperform a baseline.

## Changed files

- `DeepQN/env/sumo_env.py`: v3 contract; preserve validated lane fraction; strict range.
- `DeepQN/configs/dqn_config.yaml`: v3 research observation declaration.
- `DeepQN/configs/controller_contracts/tahrir_controller_review_candidate_v1.json`:
  refreshed only the bound `dqn_config.yaml` SHA-256 after that declared contract changed;
  scientific approval remains false and all approval fields remain empty.
- `DeepQN/tests/fixtures/tiny_sumo/tiny_eval.yaml`: fixture contract follows v3.
- `DeepQN/agent/dqn_agent.py`: v3 research checkpoint/ensemble schema and observation
  binding; v2 is incompatible with strict v3 evaluation.
- `DeepQN/evaluation/evaluate.py`: supplies observation contract to strict loader.
- `DeepQN/tests/test_phase2b2b_methodology.py`: corrected deterministic occupancy values,
  strict invalid-value tests, v3 checkpoint tests, explicit v2 rejection.
- `DeepQN/tests/test_phase2c3_occupancy_contract.py`: bounded SUMO 1.26 lane-vs-E1 proof.
- This dossier: literature integration, readiness matrix, evidence and stopping decision.

## Verification record

Focused command executed:

```powershell
python -m unittest DeepQN.tests.test_phase2b2b_methodology.ResearchLaneObservationTests DeepQN.tests.test_phase2b2b_methodology.ActionReplayReproducibilityTests DeepQN.tests.test_phase2c3_occupancy_contract -v
```

Result: **19 tests run, 19 passed, 0 failed, 0 skipped** in 9.880 s. The bounded test
launched only the existing tiny and detector fixtures.

Full regression, compilation, hash, and CRLF-aware diff results are appended below after
execution.

### Final verification

Commands and outcomes:

```powershell
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
# First run: 183 run, 1 failure, 1 skipped (128.488 s).
# The sole failure was the expected source-binding guard: changing dqn_config.yaml
# invalidated its SHA-256 in the unapproved Tahrir controller-review candidate.

python -m unittest DeepQN.tests.test_phase2c2d_tahrir_controller_contract.TahrirControllerContractTests.test_sources_are_actual_hash_bound_files -v
# 1 run, 1 passed, 0 failed, 0 skipped (0.013 s), after refreshing that binding.

python -m unittest discover -s DeepQN/tests -p "test_*.py" -q
# Final run: 183 run, 183 passed, 0 failed, 1 skipped (127.105 s).

python -c "from pathlib import Path; files=sorted(list(Path('DeepQN').rglob('*.py'))+list(Path('lstm').rglob('*.py'))); [compile(p.read_text(encoding='utf-8-sig'), str(p), 'exec') for p in files]; print(f'compiled {len(files)} Python source files')"
# compiled 61 Python source files; exit 0; no bytecode intentionally retained.
```

The sole skip is `test_ai_behavior`, the explicitly quarantined legacy interactive
diagnostic. It was not represented as an executed pass.

Post-change hashes:

| Artifact | SHA-256 |
|---|---|
| `DeepQN/configs/dqn_config.yaml` | `eb60241f8433c2f72667c3f15c2f7df4c91d78b2ce6d1a64a989ca46efd3a8f4` |
| `DeepQN/configs/controller_contracts/tahrir_controller_review_candidate_v1.json` | `2ee35e47b01a9fa5c5866f3e829ccfaa12bdd2a018d39a77031fff73032d47dd` |
| `DeepQN/tests/test_phase2c3_occupancy_contract.py` | `1ac5d302510c7db4446485f81f9d26bc53c2de5c8eefec0e31212daeeb030db9` |

The candidate remains `publication_approved=false`; refreshing a content binding is not
scientific approval. Test-generated detector XML was removed and test-modified tracked
bytecode was restored. No historical result or checkpoint changed.

## Remaining limitations and gate

Passing software tests establishes the corrected observation and provenance contracts;
it does not validate traffic demand, signal safety, forecast accuracy, reward quality, or
controller performance. The publication BiLSTM, v3 DQNs, production reward references,
explicit-ID demands, and frozen evaluation protocol do not yet exist. Phase 3 remains
**NO-GO**.

## Final protocol-package and v3 training-contract consistency addendum

### Reproduced gap and correction

**Source-verified before correction:** strict v3 evaluation ensembles were bound to the
v3 observation identifier, but `ResumeContracts.validate()` accepted any nonempty
observation label and `train_research()` constructed `MultiAgentDQN` without supplying
an observation contract. A self-consistent but semantically stale resume contract could
therefore reach the future publication training pathway.

**Corrected:** `DeepQN/training/research_resume.py:68-146` retains generic semantic
identifiers for engineering interruption tests but adds a publication-only observation
configuration SHA-256 and validator. The canonical payload binds simulator timestep and
waiting/lane modes, profile clocks, TLS/lane/phase inventories, all state normalization,
forecast configuration, and DQN state dimension. It requires the exact
`sumoflow-dqn-observation-v3-lane-occupancy-fraction` identifier, matching 37-dimensional
state/DQN declarations, and rejects any post-snapshot change. `train_research()` invokes
the gate before agent creation and passes the v3 identifier into the ensemble at
`DeepQN/training/train.py:88-92`. Empty external trust roots still prohibit actual
research training.

Generic snapshot round trips using labels such as `obs-v1` remain valid tests of atomic
serialization and interruption equivalence. They are explicitly not publication-mode
validation. New regressions cover matching v3, both v2/v3 mismatch directions, missing,
non-string, blank and unknown identifiers, missing/malformed configuration hashes,
post-publication configuration mutation, state-dimension mismatch, deterministic YAML
round trips, and the production entry point. Strict evaluation's v2-under-v3 rejection
remains covered by `test_phase2b2b_methodology.py`.

### Extreme-short-lane occupancy finding

SUMO 1.26.0 source distinguishes capped brutto occupancy from TraCI lane netto
occupancy. In `MSLane.cpp` 3287-3297, brutto occupancy uses `MIN2(1., ...)`; in lines
3300-3311, `getNettoOccupancy()` returns vehicle-length sum divided by lane length with
no cap. Thus a value above one is possible in pathological overlapping/collision or
geometry states; source alone does not establish which Tahrir short lanes can produce
it under valid motion. The v3 implementation deliberately rejects nonfinite or
out-of-range lane values in research mode instead of clipping them, preserving the
defined feature and surfacing a run-integrity failure. No new Tahrir run was performed.

### External-review package

Created without approvals or fabricated identities:

- `research/2026-10-26-phase2c3-external-review-dossier.md`
- `research/2026-10-26-phase2c3-experimental-protocol-candidate.md`
- `research/2026-10-26-phase2c3-external-decision-register.json`
- `research/2026-10-26-phase2c3-reviewer-questions.md`

The package incorporates Fawzy et al. (2026), DOI
<https://doi.org/10.1038/s41598-026-41535-x>, including the 6,252 loaded vehicles,
substantial insertion backlog, 500 running vehicles, and unequal completions reported
at 3,599 seconds. Those are related-work observations only. The candidate protocol
separates the simulation-only estimand from unsupported real-world claims and records
current-code discrepancies rather than changing defaults.

### Executed verification

```text
python -m unittest DeepQN.tests.test_phase2c3_training_observation_contract
    DeepQN.tests.test_phase2c2c_training_resume
    DeepQN.tests.test_phase2b2b_methodology.ActionReplayReproducibilityTests -v
Result: 39 run, 39 passed, 0 failed, 0 skipped; exit 0.

python -m unittest discover -s DeepQN/tests -p "test_*.py" -q
Result: 194 run, 194 passed, 0 failed, 1 skipped; 130.113 s; exit 0.

python -c "... compile changed Python files ..."
Result: compiled 3 changed Python files; exit 0.

python -m json.tool research/2026-10-26-phase2c3-external-decision-register.json
Result: valid JSON; exit 0.

git -c safe.directory="..." -c core.whitespace=cr-at-eol diff --check
Result: no errors; exit 0. Line-ending conversion warnings were informational.
```

The one skip is the quarantined legacy interactive `test_ai_behavior` diagnostic and is
not reported as a pass. Full discovery launched only already-bounded test fixtures; no
Tahrir/Taksim simulation or training occurred. Test-generated detector XML was deleted,
and tracked bytecode modified by discovery was restored.

### Final artifact hashes

| Artifact | SHA-256 |
|---|---|
| `DeepQN/training/research_resume.py` | `1453380efa8df85a23a2cb02b52999ff6c26e8dc93a5797dc595f994bdd2ceba` |
| `DeepQN/training/train.py` | `23e0753b28f53b29bae1c92f3666a642cfd138d0ec28101af7e76cb12dbfeb83` |
| `DeepQN/tests/test_phase2c3_training_observation_contract.py` | `4b54fc05cbf76722587decc27a0b842661a1dbe0d9d32971502596899193b717` |
| External-review dossier | `0398a6fe59f723fa0c163cddfa35b11ba50aba15a45fa042dd13fa86ecbf9fc4` |
| Experimental-protocol candidate | `cdaf159c6b8f4d7e6e03cf94bbbc5985bd4d8de8fef542a30c4390ec97a85413` |
| External-decision register | `074d1baf1cf99cd8bdebb7d037b1fba22ad36743d93c3beab2902ca916fbbf80` |
| Reviewer questions | `70c8e40d4257267bded934337688f118ab0830c9cb0d856fd56857c2ae39afb5` |

The checked-out commit remained `75557e668fb9fb968810040e1e6b8882e59f9f2d`.
Unrelated `.vscode`, backend image, `test_metrics_phase1a.py`, forecasting initializer,
and stray worktree files were not edited or removed. No dependency was installed and no
commit or push was made. Phase 3 remains **NO-GO** pending every external decision and
missing artifact enumerated in the package.

## Final semantic-validation and evidence-register integrity correction

### Independently reproduced early-gate weakness

Before correction, a direct publication-validator probe recomputed the canonical hash
after each mutation. All four self-consistent but semantically invalid configurations
were accepted (command exit 0):

```text
legacy_zero_fill: ACCEPTED
features_per_lane_4: ACCEPTED
step_length_5: ACCEPTED
warmup_0: ACCEPTED
```

This was specifically an early-validator weakness. Later `SumoEnv` construction already
rejects some strict-research violations, but that is not a substitute for validating a
publication resume contract before agent creation or snapshot restoration. In
particular, the old gate checked only the declared 37 dimensions, not the feature-layout
arithmetic.

### Corrected publication-only semantics

`DeepQN/training/research_resume.py:148-203` now requires:

- exactly 10 lanes × 3 lane features + 2 junction features + 5 global features = 37;
- matching DQN state dimension;
- `research_fail_closed` lane observation;
- finite positive queue, wait, occupancy, and phase-elapsed normalization references;
- one-second simulation steps and explicit 100-second waiting memory;
- exactly 60 seconds of common observed history under
  `phase_controller_keep_v1`; and
- one of the two intended configuration modes, `no_forecast` or `verified_model`.

The last item validates an arm's configuration identity only. It does not approve,
authenticate, or load a forecast checkpoint. Generic engineering snapshot contracts and
historical loaders remain outside this publication-only semantic gate. Publication
trust roots remain the empty tuple in `DeepQN/training/train.py`.

The adversarial tests at
`DeepQN/tests/test_phase2c3_training_observation_contract.py:164-286` cover all four
reproductions, missing layout fields, bad arithmetic, every normalization reference with
zero/negative/NaN/infinity/missing values, invalid modes/protocols/timing, valid v3 arms,
generic snapshot compatibility, closed authority, and absence of agent construction or
restoration calls after semantic failure. The mocked entry-point test establishes call
ordering only; it is not represented as real SUMO training.

### Decision-register integrity

Every unresolved entry in
`research/2026-10-26-phase2c3-external-decision-register.json` now contains one or more
repository-relative source paths and verified SHA-256 values. Bindings cover the exact
Tahrir network, forecast mapping, controller-review candidate, DQN configuration,
reward-calibration implementation, and relevant Phase 2C evidence/protocol files. A
regression reads every referenced file and recomputes its digest.

The register deliberately does not hash itself. `publication_approved` and
`phase3_authorized` remain false; reviewers remain empty; every status is unresolved;
external decisions, reviewer identities, and approval evidence hashes remain null. These
local identity checks establish unchanged bytes only, never external scientific approval.

### Verification

```text
python -m unittest DeepQN.tests.test_phase2c3_training_observation_contract -v
Result: 19 run, 19 passed, 0 failed, 0 skipped; 4.543 s; exit 0.

python -m unittest discover -s DeepQN/tests -p "test_*.py" -q
Result: 202 run, 202 passed, 0 failed, 1 skipped; 132.941 s; exit 0.
```

The single skip remains the quarantined legacy interactive diagnostic. Full discovery
used only bounded fixtures. No full scenario or training was run. Test-generated detector
XML was removed and tracked bytecode modified by discovery was restored.

Pre-ledger final hashes for corrected artifacts:

| Artifact | SHA-256 |
|---|---|
| `DeepQN/training/research_resume.py` | `1fa852a5bde2b7f511d8e4b0b6caba5a1173c8f7a190f03b2a60a40969a0f105` |
| `DeepQN/tests/test_phase2c3_training_observation_contract.py` | `1ca39d23c8e329d1b5a53c08784386bbe690bcba6098d9731200f2b24b9bd3ed` |
| `research/2026-10-26-phase2c3-external-decision-register.json` | `8726fc5569ac6381e89f4132caeadf42ceb623eaa8d415d5a85935a2fc63659a` |

Remaining blockers are unchanged: independent mapping/signal review, explicit-ID frozen
demand and splits, an approved nonnegative forecast-output contract and trained model,
publication reward references, frozen protocol authorization, and field evidence for
any real-world claim. Phase 3 remains **NO-GO**.
