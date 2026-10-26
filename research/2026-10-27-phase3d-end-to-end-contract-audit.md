# Phase 3.0D end-to-end scientific-readiness audit

Date: 2026-10-27 (Africa/Cairo)  
Branch: `research/ieee-reproducibility`  
Audited HEAD: `895ebb8a664b1979d137b54ccb07b6e4af6375e4`  
Upstream: `origin/research/ieee-reproducibility`, ahead 0 / behind 0  
Decision: **Phase 3.1 NO-GO**

## Evidence standard and repository state

This audit distinguishes executable source, hash-verified artifacts, bounded test evidence,
proposed contracts and external scientific evidence. Passing fixtures demonstrate software
behavior only. No DQN/BiLSTM training, Tahrir/Taksim experiment or performance comparison was
run. The Phase 3.0C protocol is tracked at HEAD and its actual SHA-256
`0bf45ff5bcb2f265227178b9f3aa425a7948fc1366173079c4f9e3ae1ae51740` exactly matches
the experiment manifest. Publication approval, Phase 3 authorization and trust roots remain
false/empty.

Pre-existing unrelated changes to `.vscode/settings.json`,
`DeepQN/tests/test_metrics_phase1a.py`, and the two backend images, plus the pre-existing
untracked paths, were left untouched.

## End-to-end trace

1. Fixed-number historical flows can be expanded deterministically and independently
   regenerated (`demand_expansion.py:151,336`). This proves byte reproducibility, not that the
   historical template is empirically calibrated or that copies are independent demand.
2. Explicit realizations, ID-insensitive physical fingerprints and split registries exist in
   code (`demand_provenance.py:201,260,493,564`). No approved five-pool Tahrir registry exists.
3. Effective configurations can bind route/network/additional bytes and SUMO seed
   (`demand_provenance.py:626-750`). The current YAML SHA is correctly marked provisional.
4. E1 mappings validate lanes/positions and live installations; events reject ambiguous
   same-time multi-lane identity and teleport intersections (`detector_events.py:56-94`,
   `inference.py:157-202`). The Tahrir geometry remains unapproved.
5. One-second seven-feature rows form 60-second causal histories and targets in `(t,t+30]`
   (`research_contract.py:14-16`; `research_data.py:221-286`). Training-only scaling is
   contract-tested. Overlapping windows are subsamples, never independent experimental units.
6. `ResearchBiLSTM` produces nonnegative four-count outputs (`inference.py:49-75`) and strict
   loading rejects the incompatible historical six-input checkpoint. No compatible trained
   checkpoint or forecast-training seed schedule exists.
7. `SumoEnv` enforces observation v3, 37 values, strict lane sensors, exact action dictionaries,
   verified/no-forecast modes, one-second sampling and common 60-second initialization
   (`sumo_env.py:327-451,461-584,786-856`). First learned actions occur after warm-up.
8. `IntervalRewardFunction` integrates delay and pressure in vehicle-seconds, edge CO2 in mg,
   and arrivals in vehicles (`reward.py:77-158`). Warm-up is included in horizon outcomes but
   reward is reset before learned decisions (`sumo_env.py:504-509`). Calibration uses only
   training/calibration traces, independently recomputes demand fingerprints and requires all
   four profiles (`research_protocol.py:54-181`). No calibrated references exist.
9. Strict checkpoint loading and run-manifest output exist, but the five-policy by ten-demand
   publication schedule is not executable: the seed-pairing rule, checkpoint identities,
   approved demands and training budget remain unresolved. The complete-coverage validator is
   at `research_protocol.py:327-383`.
10. XML metrics retain due, inserted, arrived, running and undeparted populations; horizon
    waiting includes censored pre-insertion delay (`metrics.py:268-309`). Teleports/collisions
    and output integrity are recorded. The equal-profile descriptive estimator is correct, but
    hierarchical inference is not implemented and no outcomes exist.

The detailed machine-readable incompatibility matrix is
`research/2026-10-27-phase3d-integration-readiness-matrix.json`.

## Seed-identity reconciliation

The values `0..4` in `validate_publication_pairing()` are replicate indices, whereas
`11001..11005` in `proposed_seed_schedule()` are proposed actual DQN RNG seeds
(`demand_provenance.py:762-788`). Treating both as `training_seed` is ambiguous and therefore a
critical blocker.

The versioned candidate `sumoflow-publication-seed-allocation-candidate-v1` separates policy
replicate index, DQN RNG seed, forecast-training seed, demand-generation seed, SUMO execution
seed, profile, evaluation demand and checkpoint identity. Its SHA-256 is
`dbaf6fa3e16548773489b26f36cf53f90295e930f71e380689222896eb9da192`.
It preserves the previously proposed DQN and evaluation seed numbers but remains unapproved.
Forecast seeds are null, checkpoint identities are future products, and
`training_seed_pairing_rule` is null. Consequently the publication pairing gate remains closed.
Every future eligible checkpoint must cross every frozen held-out demand in its profile, with
both arms using the same physical fingerprint and SUMO seed.

## Bounded integration and boundary evidence

Executed bounded tests use SUMO 1.26 fixtures and no production demand. Real E1 evidence covers
ordinary entries, repeated responses, multiple events, short-loop/sub-step behavior, cyclic
re-entry and lane-change characterization. Exact and adjacent timestamp-to-bin semantics are
also tested. These establish the software time convention: TraCI last-step E1 events are labeled
at the completed post-step boundary; histories end at decision time and targets are right-closed
over the following 30 samples.

The evidence does **not** approve Tahrir placement. Teleport/relocation touching forecast
collection fails closed; the earlier commanded-lane-change fixture is characterization, not a
complete proof for arbitrary real lane changes. Runtime missing detector, wrong lane/position,
insufficient history, legacy forecast mode, missing checkpoint, lane-sensor failure or physical
phase-query failure invalidates research execution. Synthetic predictors are not performance
evidence.

## Reward, metrics and failure consistency

- Calibration protocol names match implementation fields:
  `delay_vehicle_s` (vehicle-seconds), `co2_mg` (mg), `arrivals_vehicles` (vehicles), and
  `pressure_vehicle_s` (vehicle-seconds). References must be finite and positive.
- Registry v2 verifies all five pools' explicit bytes and canonical physical identities, but
  permits reward traces only for training/calibration. Held-out outcomes cannot be indirect
  calibration sources through the accepted schema.
- The committed 0.95 positive-quantile protocol requires morning, midday, evening and night.
- Horizon metrics include the shared initialization and penalize prevented insertion through
  censored pre-insertion delay. Reward begins only with RL transitions. These are intentionally
  different accounting windows and must be reported.
- Completed-trip means remain conditional secondary measures. Running and undeparted vehicles,
  insertion/completion rates, total network emissions, teleports and collisions remain separate.
- Premature termination, population contradiction, missing sensors/artifacts, invalid physical
  trace, teleport intersection or malformed outputs make a run invalid. Invalid runs cannot be
  dropped and interpreted as favorable; missing-pair reasons and prespecified sensitivity are
  required.

## Manifest preflight: enforced versus documentary

The Phase 3.0C candidate is not an execution manifest. `validate_experiment_candidate()` checks
its structure, versions, seven TLS, one-second steps, warm-up and nominal replicate counts
(`research_protocol.py:216-289`). `evaluate.py` does **not** call this validator. It verifies
individual checkpoints, builds per-run manifests and validates baseline comparability, but it
does not consume the complete candidate, seed-allocation candidate, reward-calibration identity
or five-pool schedule as a single authorization gate.

Therefore these fields are currently documentary: training budget, optimizer freeze at runtime,
five policy replicates, ten-demand crossing, profile set, failure policy, reward-calibration
identity, forecast checkpoint identity and pairing rule. Null hashes and the provisional YAML
are correctly non-executable under the candidate validator, but no authorized publication runner
exists to invoke that gate. Phase 3.1 requires a separate immutable execution manifest and a
runner that verifies all referenced bytes before creating an environment or model. This audit
does not enable it.

## Statistical audit

The descriptive estimator averages four profile means equally and preserves paired absolute
differences (`research_protocol.py:291-325`). The coverage validator requires five indices per
arm, ten demand identities per profile, both arms, consistent checkpoint identity, physical
demand and SUMO seed, and rejects missing/duplicate/failed cells. Its current hard-coded `0..4`
indices are not yet bound to the proposed RNG seeds; this is a blocker, not a reason to reinterpret
existing data.

The proposed hierarchical cluster bootstrap is conceptually aligned with the crossed structure:
trained policy and physical demand are higher-level units, profile is a fixed stratum, and TLS,
steps and overlapping windows are correlated subsamples. With only five trained policies per
arm, tail and variance estimation is fragile. The study should be estimation-first: publish
seed-level effects, bootstrap/model sensitivity, and avoid strong confirmatory claims based only
on nominal cell counts. Synthetic known-answer tests demonstrate arithmetic, not traffic gains.

## Realistic readiness decision

| Scope | Decision | Minimum interpretation |
|---|---|---|
| Bounded software-integrity testing | Ready | Existing fixtures can verify code contracts; they do not validate Tahrir scientifically. |
| Uncalibrated synthetic simulation paper | Conditional | Possible only when explicitly labeled synthetic and after frozen synthetic demands, detector geometry rationale, calibration traces, compatible forecast model, runner and seed schedule exist. No real-world representativeness claim. |
| Field-calibrated Tahrir case study | NO-GO | Requires empirical traffic calibration/validation and independent geometry/signal review not currently available. |
| Real-world deployment | NO-GO | Requires physical controller integration, safety certification, operational governance, sensor validation and field trials far beyond this repository. |

Without a supervisor or traffic engineer, a simulation-only paper may transparently justify
deterministic algorithms, causal splits, strict provenance, synthetic demand assumptions and
negative findings. It cannot claim Tahrir field calibration, signal safety, local demand realism,
deployment readiness or external approval. Empty trust roots must remain empty.

## Stage-specific minimum prerequisites

1. **Training-only diagnostic pilot:** frozen synthetic/training demand registry, provisional but
   explicit reward references generated only from training/calibration traces, effective config,
   compute/time limit, one seed, diagnostic label and no held-out access.
2. **Forecast training:** approved/fail-closed mapping for the claimed scope, dataset/scaler
   manifest, allocated forecast seeds, leakage-free pools, baseline scoring and checkpoint rule.
3. **DQN training:** compatible frozen forecast artifact for the forecast arm, calibrated reward
   references, five DQN seeds, frozen budget/selection rule, complete resume manifests and
   no controller-evaluation demand access.
4. **Held-out evaluation:** immutable execution manifest, authorized full-manifest runner,
   checkpoint hashes, frozen pairing/crossing schedule, ten physical demands/profile, paired SUMO
   seeds, output storage and prespecified failure/statistical plan.

Missing trained checkpoints and outcomes are expected future products, not prerequisites to
starting their properly authorized generation stage. Missing inputs, contracts and allocation
rules are prerequisites.
