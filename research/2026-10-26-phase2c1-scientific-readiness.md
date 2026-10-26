# SUMOFlowAI Phase 2C.1 — Tahrir scientific-readiness audit

**Date:** 2026-10-26 (Africa/Cairo)  
**Branch:** `research/ieee-reproducibility`  
**Starting commit:** `c52b3f166cba3dabee2d3e7b261582a918eb581e`  
**Scope:** evidence review, bounded safeguard corrections, and protocol design.
No model was trained, no Tahrir/Taksim traffic experiment was run, and no
performance result was generated.

## Evidence language

- **Source-verified** means derived from the checked-out source or XML.
- **Artifact-verified** means derived from bytes whose SHA-256 is recorded.
- **Test-reproduced** means exercised by an executed automated test.
- **Previously bounded evidence** identifies an accepted earlier fixture; it is
  not represented as a new test in this phase.
- **Proposed/frozen-on-approval** is a protocol decision, not observed evidence.
- **Unresolved** means that the required scientific evidence is absent.

## 1. Frozen Tahrir inputs inspected

| Artifact | SHA-256 |
|---|---|
| `DeepQN/configs/dqn_config.yaml` | `c7f31e298d83f8402fd02df2dcb75d4efe9237e96952f10aaa90635f5833f83d` |
| `DeepQN/configs/forecast_mappings/tahrir_v1.json` | `275d8494f9a408ddee515b63402bdac27bba041730726231d35f428ac04c0f98` |
| `simulation/maps/tahrirupdated.net.xml` | `a7f25235f208db932735983653e00b3af659d09d04169ef5807265433c2db708` |
| `simulation/maps/tahrir.add.xml` | `87125278f5f806a769e3223d70278adbc02c88010c4e80fc32813a6ae885eac5` |
| `simulation/maps/vehtype.add.xml` | `e97ddd34ad616596f80a536531cf35e4d82df162057e24a52c726432742bd835` |
| morning configuration / route | `252bf33da83a58506254c2b96b90631505f814a86eaa82e3b6e74c0ae6e1aea0` / `e6563d8c0ef6b1ffc23df3591b0f746a2f3b5b57b480517819abf4e68115ce03` |
| midday configuration / route | `de5bc9ff80389d26ea27d4546eda3de7b79db1f08f838da7dcfd090ccef0bdee` / `35d56764f98806c10aa6194558800b51be92a92f176b1e8fa886aae93cc94a9f` |
| evening configuration / route | `f15eeea8240c5a10aefc0773018e60a6202a4c16d16293a9ab7384f16a167f33` / `c5190b7ec68cc8daff6a9a10d278b0df7932c19732bf7e82d567e96844df7abd` |
| night configuration / route | `9deb687ccd44ceb069181be4031694119f77a68f9f0c0a2c44110cc2b98b19cc` / `1e84789b2fac27aaff9d336a22cb0fc60b83ad08fc032046bf1671003223396a` |

The four supplied route files are historical realized demands, not a genuine
multi-seed training/evaluation population. The profile configurations also set
different teleport thresholds (180/240/180/300 seconds). Those settings may be
profile-specific, but must remain identical across controllers within a profile.

## 2. Seven-junction topology and controller-readiness matrix

The network contains nine `tlLogic` elements, at
`simulation/maps/tahrirupdated.net.xml:775-816`; the DQN configuration selects
seven (`DeepQN/configs/dqn_config.yaml:46-171`). The unselected TLS
`6288771435` and `96621068` continue under their native programs and must also be
hash-bound in every experiment even though they are not policy agents.

| Controlled TLS | Physical static phases and controlled movements | DQN binary request | Native fixed | Actuated | Current max-pressure | Forecast boundary |
|---|---|---|---|---|---|---|
| `315744796` | green 0 serves 4 links; yellow 1; green 2 serves 6 links; yellow 3 | Structurally compatible with configured next-green cycle | Eligible after whole-program live preflight | No approved program/detectors | **Topology-compatible only** (two green, no red hold); movement/saturation definition absent | 10 E1s on `28718647#3`, `690516091#0`; structurally complete, unapproved |
| `96621100` | all-red 0; yellow 1; green 2 serves 10 links | Compatible with explicit green/red-hold semantics | Eligible | No approved program/detectors | **Rejected** by current two-green/no-red-hold adapter | 10 E1s; `10873186#8` has five 0.20 m lanes and a 0.10 m detector plane; unapproved |
| `2031414903` | green 0 serves 2 links; yellow 1; green 2 serves 2 links; yellow 3 | Structurally compatible | Eligible | No approved program/detectors | **Topology-compatible only**; definition absent | 9 E1s on `50211834#1`, `10873186#5`; unapproved |
| `2031414899` | green 0 serves 4; yellow 1; all-red 2 | Compatible with explicit green/red-hold semantics | Eligible | No approved program/detectors | **Rejected** by current adapter | 4 E1s on `10873191#3`; unapproved |
| `6288771431` | green 0 serves 4; yellow 1; all-red 2 | Compatible with explicit green/red-hold semantics | Eligible | No approved program/detectors | **Rejected** by current adapter | 4 E1s on `10873191#4`; unapproved |
| `271064234` | green 0 serves 3; yellow 1; all-red 2 | Compatible with explicit green/red-hold semantics | Eligible | No approved program/detectors | **Rejected** by current adapter | 4 E1s on `10873193#1`; unapproved |
| `315743335` | green 0 serves 3; yellow 1; all-red 2 | Compatible with explicit green/red-hold semantics | Eligible | No approved program/detectors | **Rejected** by current adapter | 5 E1s on `28718874#1`; unapproved |

### Movement and observation qualifications

- **Source-verified:** every physical controlled connection was enumerated from
  the network's `connection@tl/linkIndex`; every link is served by the green
  phase listed above. The configured phase cycles are at
  `DeepQN/configs/dqn_config.yaml:50-166`, and the physical programs are at
  `simulation/maps/tahrirupdated.net.xml:775-816`.
- **Source-verified:** configured `incoming_lanes` equals the complete junction
  `incLanes` list, but does not always equal the controlled-from-lane set.
  Uncontrolled incoming lanes are: `28718647#3_4` at `315744796`; lanes 2–3 of
  `50211834#1` and lanes 2–4 of `10873186#5` at `2031414903`;
  `10873193#1_0` at `271064234`; and lanes 0–1 of `28718874#1` at
  `315743335`. Retaining them in the observation may be defensible as local
  context, but they must not be described as controlled movements without an
  engineering explanation.
- **Source-verified:** the current max-pressure action adapter deliberately
  rejects red-hold and non-two-green topologies
  (`DeepQN/evaluation/controller_baselines.py:329-370`). It therefore cannot run
  a seven-agent Tahrir condition. Supporting the five red-hold junctions requires
  a separately reviewed safe target/control formulation; mapping a selected
  pressure phase to a generic binary switch is prohibited.
- **Source-verified:** no Tahrir actuated program, detector/gap design, or
  max-pressure v2 movement/saturation file exists. No eligible performance run
  can be produced by transplanting the tiny fixture.

## 3. Forecast-boundary audit and E1 limitations

The approach mapping lists all seven TLS at
`DeepQN/configs/forecast_mappings/tahrir_v1.json:6-12`. Loading it against the
network establishes ten distinct incoming edges with no cross-TLS reuse. The
generated mapping contains 46 lane-specific E1 detectors, hash
`afd17243b1d154ecf073cca645c3a58dfc4b7e4875e105206be6d44b8fed4a06`.
Detector generation uses a 0.10 m edge-start plane
(`DeepQN/forecasting/detector_events.py:18,142-177`).

This is structural consistency, not traffic-engineering approval. The trust
root is intentionally empty (`lstm/research_data.py:41`), and the generated
mapping defaults to `traffic_engineering_validated=False`
(`DeepQN/forecasting/detector_events.py:38`). Publication forecast training is
therefore fail-closed.

### Accepted evidence and exclusions

The Phase 2B.2A bounded four-approach fixture already established ordinary
entry, multiple same-step entry, sub-step loop traversal, simulation-start
entry, exact-end exclusion, and agreement with an independent instantaneous-loop
reference. It was not repeated.

No available bounded fixture provides an independent, physically meaningful
reference for the following Tahrir-specific cases, so this phase does **not**
claim them as validated:

| Gap | Required bounded design | Current decision |
|---|---|---|
| Lane change across parallel E1s | Two-lane approach, commanded lane change spanning both loops, compare E1 records with FCD lane/position trace | Same-TLS/same-vehicle/same-time duplicate remains fail-closed; exclude affected realization if triggered |
| Teleport across/onto boundary | Induce a documented SUMO teleport on a short congested approach; compare teleport log, FCD discontinuity and E1 | Unresolved; any teleport touching a forecast boundary invalidates that source until characterized |
| Cyclic re-entry | Closed route crosses the same plane twice; independent FCD crossing times | Contract accepts distinct timestamps, but real behavior is unverified; publication collection requires this test or exclusion of cyclic routes |
| Entire short-edge traversal in one second | Short approach with pre/post FCD positions and E1/instant-loop references | E1 is expected to retain sub-step records, but full-edge traversal is unverified |
| Five 0.20 m lanes on `10873186#8` | Exact-network startup plus controlled traffic crossing and an independent reference | 0.10 m placement is structurally valid only; require specific engineering approval or remap the boundary |

Merely loading a detector or synthesizing an event would not resolve these
questions. No new Tahrir SUMO run was launched.

## 4. Comparison protocol frozen on approval

### Conditions and scopes

1. **Primary matched-policy scope:** forecast-free Dueling Double DQN and
   verified-BiLSTM Dueling Double DQN. Both retain the 37-dimensional observation,
   the same four forecast slots (zeros for no-forecast), identical
   `phase_controller_keep_v1` control for the first 60 observed seconds, first
   learned decision at `begin+60`, identical reward/action/phase safety, training
   budget, demands and seeds. Only the forecast content differs.
2. **Primary scope once eligible:** max-pressure may join the matched 60-second
   handoff only after all seven TLS have approved movement/service/saturation and
   topology-safe control definitions.
3. **Secondary full-operation scope:** native fixed-time and SUMO-actuated run
   from the simulation begin. They use the same frozen demand and outcome window,
   but are not phase-synchronized at the DQN handoff. Their estimates must be in
   a separately labelled table/stratum and must never be pooled with the primary
   matched-policy estimand (`DeepQN/evaluation/baseline.py:50-61`).
4. Random policy remains an engineering diagnostic, not a research baseline.

All controllers use the same `[begin,end)` realized demand, SUMO version,
network/additional/route bytes, step length, horizon, teleport/collision settings,
waiting-memory setting, output options and metric contract. Runs fail on sensor,
physical trace, output, provenance, population-reconciliation, checkpoint, or
forecast-contract errors. A failed run is not imputed as zero and is not silently
rerun under a new seed; the failure and preregistered replacement policy are
reported.

### Outcomes

- **Single primary outcome:** demand-normalized horizon waiting burden, seconds
  per due vehicle, including censored pre-insertion delay and on-network waiting
  (`DeepQN/evaluation/metrics.py:268-309`). Lower is better.
- Secondary: insertion/completion rates, arrivals/hour, running and undeparted
  counts, departure delay, backlog, conditional completed-trip waiting/duration/
  time loss, total network CO2, network CO2 per reported arrival, teleports,
  removals/vaporization, collisions, physical action interventions and failures.
- Forecast endpoints, evaluated separately on the held-out forecast test set:
  per-direction MAE/RMSE/count bias and aggregate scores versus zero,
  persistence, historical mean and moving average. Controller outcomes cannot
  substitute for forecast accuracy.

### Seeds, estimands and inference

- Five independent training seeds per DQN condition. Proposed immutable IDs:
  `11001-11005`; the same IDs pair common randomness across the two conditions.
- Ten paired held-out evaluation demand realizations per controller and profile.
  Proposed immutable IDs: `21001-21010`. Every trained DQN checkpoint is
  evaluated on all ten; baselines use the identical ten demands.
- Four preregistered profiles remain separate strata. A profile-weighted overall
  estimand requires weights frozen before outcomes are inspected; an unweighted
  average is not silently substituted.
- Primary contrast: verified-forecast DQN minus no-forecast DQN in primary waiting
  burden. The directional scientific hypothesis is improvement (negative
  contrast), but report a two-sided 95% interval and the full distribution.
  Training seed—not 40 demand/profile runs—is the independent learned-policy
  unit. Use a hierarchical paired analysis/cluster bootstrap over training seed
  and demand realization, retaining profile strata.
- Five paired training seeds give weak tail resolution (a two-sided exact
  sign-flip test cannot attain 0.05). Results are estimation-first; no strong
  confirmatory significance claim is permitted solely from pseudo-replicated
  evaluation episodes. Increasing training seeds requires a separately approved
  compute amendment, not post-result discretion.
- Secondary controller contrasts/outcomes use Holm correction within a frozen
  family; effect sizes and intervals are reported regardless of significance.
  No controller is assumed to outperform another.

## 5. Dataset partitions and reward calibration

Required immutable registry partitions are: training demand; calibration demand
for reward reference scales only; validation demand for checkpoint/budget/model
selection; forecast held-out test; and controller evaluation demand. Identity,
route bytes, generator/version/parameters, base seed, network/config hashes and
partition must be frozen before training. Byte-identical demand cannot represent
multiple realizations or cross partitions.

No such genuine Tahrir registry exists. The four historical routes do not supply
five independent training seeds plus ten held-out evaluation realizations. No
production reward calibration values were generated. Calibration remains
`publication_approved=false`; its units and quantile method must be frozen from
approved training/calibration-only interval traces, never validation/test/evaluation.

## 6. Interruption-equivalent training readiness

The training entry point seeds processes, then uses best-effort legacy
`load_latest` (`DeepQN/training/train.py:90-111`), restarts episode numbering at
one, and regenerates profile selection. Research ensemble checkpoints store
weights, target weights, optimizer, epsilon/counters and Python/NumPy/Torch CPU
RNG (`DeepQN/agent/dqn_agent.py:364-429`), while replay buffers can now serialize
their own content/RNG (`DeepQN/agent/replay_buffer.py:113-130,253-279`). These
pieces are not integrated into a complete training-state checkpoint. CUDA RNG,
replay/PER state, next episode/profile/demand cursor and orchestration manifest
are not restored by the training path.

**Proposed bounded restoration acceptance test (not implemented here):** run a
small CPU synthetic training schedule for `K+M` decision updates; separately run
`K`, atomically save the complete ensemble/replay/RNG/schedule state, restore in
a fresh process, and run `M`. Compare next profile/demand identity, sampled replay
indices, epsilon/update counters, actions, losses, online/target/optimizer tensors
and final checkpoint hashes. Exercise uniform and prioritized replay and a save
failure. CUDA gets a separate RNG-state/tolerance test. Checkpointing occurs only
at an episode boundary; mid-SUMO interruption restarts the episode and cannot be
called bitwise continuation.

Current status: **not interruption-equivalent; training no-go**.

## 7. Phase 2C.1 safeguard corrections

Two code defects authorized by the phase request were reproduced and corrected:

1. Live native-program verification could block indefinitely before the already
   bounded evaluation subprocess. It now runs in an isolated spawn process with
   an explicit finite positive wall-clock limit, forced termination, and propagated
   diagnostics (`DeepQN/evaluation/controller_baselines.py:181-222`). Native
   publication baselines require and record that limit
   (`DeepQN/evaluation/baseline.py:62-66,88-121`).
2. A split registry could assign byte-identical demand to two realization IDs
   in the same partition. The registry now permits exactly one realization owner
   per verified demand hash, regardless of partition
   (`DeepQN/evaluation/reward_calibration.py:60-82`).

Regression coverage is at `DeepQN/tests/test_phase2b3_baselines.py:65-72,
216-244,285-292`. These safeguards do not make a Tahrir controller or calibration
scientifically approved.

## 8. Required approvals and go/no-go matrix

| Gate | Evidence still required | Status |
|---|---|---|
| Tahrir forecast mapping | Independent traffic-engineering approval bound to approach mapping `275d…f98`, detector mapping `afd1…a06`, and network `a7f2…708`; resolve 0.20 m edge and exceptional event policy | **NO-GO** |
| Genuine forecast dataset | Frozen multi-seed E1/snapshot/demand/config sources; approved mapping; leakage-safe partitions; held-out baseline evaluation | **NO-GO** |
| Seven-TLS max-pressure | Complete v2 movements, all service phases, saturation assumptions, downstream lanes; safe red-hold targeting for five TLS; bounded physical tests and approval | **NO-GO** |
| Tahrir actuated | Explicit loaded programs, detectors/gaps/min/max timing for declared TLS scope; independent review and bounded activation tests | **NO-GO** |
| Native fixed | All nine executed programs and full effective input hashes; explicit timeout | **Engineering-ready, scientific demand set absent** |
| DQN training | Genuine split/calibration registry, approved forecast condition, frozen budget rule, complete interruption-equivalent resume, five-seed compute approval | **NO-GO** |
| Evaluation | Ten held-out paired demands/profile, eligible checkpoints/controllers, storage budget and failure policy | **NO-GO** |

### Final Phase 2C.1 decision

**No-go for AI training and large experiments.** The code can support bounded
engineering checks, but the forecast mapping is unapproved, strong adaptive
baselines are incomplete for the seven-TLS topology, genuine partitions and
reward calibration are absent, and training restoration is not yet complete.
Passing unit/tiny-fixture tests establishes code behavior only; it does not
validate forecast benefit, controller superiority, or Tahrir traffic realism.

## 9. Executed verification

Commands executed from the repository root:

```powershell
python -m unittest DeepQN.tests.test_phase2b3_baselines -v
python -m unittest discover -s DeepQN/tests -p "test_*.py"
python -c "...compile each of the four changed Python files in memory..."
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" -c core.whitespace=cr-at-eol diff --check
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" status --short
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" rev-parse HEAD
```

- Focused Phase 2B.3/safeguard suite: **27 passed, 0 failed, 0 skipped** in
  18.573 seconds, exit 0. This includes bounded real-SUMO tiny tests and the new
  enforceable-preflight-timeout regression.
- Final full discovery after the shutdown check: **142 executed; 141 passed,
  0 failed, 1 explicitly skipped legacy diagnostic** in 43.834 seconds, exit 0.
- In-memory compilation: four changed Python files compiled, exit 0.
- CRLF-aware diff check: exit 0; Git emitted only line-ending conversion notices.
- HEAD remained `c52b3f166cba3dabee2d3e7b261582a918eb581e`.
- Automated tests updated three tracked Python cache files; they were restored
  exactly. Pre-existing `.vscode`, backend-image, `test_metrics_phase1a.py`,
  `DeepQN/forecasting/__init__.py`, and stray `tatus*` worktree changes were not
  edited. No commit or push occurred.

Several full-discovery invocations were started while capturing complete output;
the final logged invocation above is the authoritative count. Its log was written
outside the repository at `D:/temp/phase2c1-full-tests-final.log`.

