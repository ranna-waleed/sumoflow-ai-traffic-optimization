# Phase 3.0A candidate demand-expansion and partition specification

**Status:** engineering candidate, not publication-approved.
**Scope:** deterministic offline preparation only; no training or performance experiment.

## 1. Source contract

The canonical input is one hash-bound SUMO route file, its effective `.sumocfg`, network,
and ordered additional-file list. Simulation demand uses the half-open profile window
`[begin,end)`. Requested/scheduled demand is never called inserted or completed demand;
those populations require summary/tripinfo reconciliation after a successful run.

Contract `sumoflow-number-flow-expansion-spec-v1` supports:

- existing numeric-departure `<vehicle>` records with complete inline or named routes;
- `<flow>` with exactly `id`, `type`, complete named `route`, finite `begin/end`, and a
  positive integer `number`;
- either no optional flow attributes or the source-verified
  `departLane="best"`/`departSpeed="random"` values;
- either no flow children or a bounded `<stop lane="..." duration="..."/>` child; and
- SUMO 1.26.0 equal spacing: for `number=n`, scheduled departure `i` is
  `begin + i*(end-begin)/n`, `i=0,...,n-1`. Thus `begin` is included and `end` excluded.

Generated IDs use SUMO's documented `flowId.runningNumber` convention. Named routes are
resolved into inline routes so the result is compatible with the repository's strict
explicit-demand validator. Existing explicit vehicles, types, routes, attributes, stops,
and departure times are preserved. Output is sorted by numeric departure then ID.

SUMO 1.26 shifts a flow child stop's `until` by each generated vehicle's departure
offset. Contract v1 therefore rejects `until` rather than copying a semantically wrong
absolute value. It also rejects triggered/timetable/waypoint/public-transport stop fields,
`<param>` children, and optional flow values beyond the exact source-observed pair. A
duration-only lane stop is shift-independent and has bounded source-flow/explicit-vehicle
runtime coverage. The four historical flow sets contain no children and use exactly the
supported optional values, so this narrowing does not change their scheduled counts.

The v1 contract rejects `probability`, `period`, exponential/Poisson period,
`vehsPerHour`, incomplete `from/to/via` routing, distributions, triggered departures,
persons, containers, unknown root elements, unverified flow children/attributes, missing
types/routes, unknown or disconnected named and inline routes,
duplicate/generated ID collisions, contradictory inline/named routes, nonfinite times,
and out-of-window demand. These constructs need separate version-pinned semantic proofs;
they are not approximated.

Fixed-count `number` expansion uses no RNG. A different seed cannot create an independent
physical realization and must not be labeled as one. Source `departSpeed="random"` is
preserved and explicitly recorded as runtime randomness requiring a separate SUMO seed.
Future stochastic demand realizations use dedicated demand-generation seeds through the
existing `sumoflow-demand-spec-v1` pathway after expansion and approval. Identical physical
schedules with renamed IDs remain leakage duplicates and are rejected by the split registry.

## 2. Atomic artifacts and verification

`DeepQN/evaluation/demand_expansion.py` writes an isolated directory containing:

- `routes.explicit.rou.xml`, with a deterministic source-identity comment; and
- `expansion_manifest.json`, recording schema, algorithm, SUMO version contract,
  complete input paths/hashes, profile window, counts, runtime-random fields, output hash,
  physical-demand fingerprint, and `publication_approved=false`.

All inputs validate before publication. A staging-directory rename prevents partial
publication; destinations are never overwritten. Strict verification requires externally
supplied expected SHA-256 identities for both the specification and complete manifest,
checks the exact schema/version/input/window/count/output contract, and independently
regenerates the route and semantic manifest from the frozen specification. It then requires
byte-for-byte route and manifest equality. This rejects coordinated route/self-reported-
hash edits, renamed IDs, and changed sources under the original frozen reference. A
candidate without those independently frozen expected identities is untrusted for
scientific use.

Expected hashes must come from a separate immutable review/registry workflow; accepting
fresh hashes from the candidate itself would merely bless a new candidate. Matching local
hashes prove identity relative to that reference, not cryptographic authorship, empirical
validity, traffic-engineering approval, or publication approval.

## 3. Frozen partition design candidate

The existing immutable registry has five top-level partitions and is not silently changed:
`training`, `calibration`, `validation`, `forecast_test`, and
`controller_evaluation`. Six scientific uses map to those partitions through a future
separate, hash-bound usage-allocation manifest:

| Scientific use | Existing partition | Sharing rule |
|---|---|---|
| Forecast training | `training` | Dedicated realization IDs; disjoint from DQN training IDs. |
| DQN training | `training` | Dedicated realization IDs; may consume the already-frozen forecast model, but no shared physical demand with forecast training. |
| Forecast calibration/scaler fit | `calibration` | Fit using this pool only; never controller evaluation. |
| Forecast validation | `validation` | Dedicated subpool, disjoint from DQN checkpoint-selection subpool. |
| DQN checkpoint selection | `validation` | Dedicated subpool; no final evaluation outcomes. |
| Forecast held-out testing | `forecast_test` | Never used for fitting, calibration, hyperparameters, or DQN training. |
| Controller held-out evaluation | `controller_evaluation` | Never seen by either learned model or any selection/calibration procedure. |

Every physical fingerprint, source/spec hash and demand-generation seed must occur in
exactly one usage pool. The future usage manifest must make the two `training` and two
`validation` subpools explicit without altering the frozen registry schema. Forecast
training/calibration/validation/test and controller evaluation are disjoint; the forecast
model must not have seen controller-evaluation physical demand.

Seeds occupy distinct namespaces and roles:

- demand-generation seed: produces an immutable physical schedule;
- SUMO seed: controls preserved runtime randomness such as `departSpeed="random"`;
- five DQN training seeds per learned condition: policy initialization/exploration;
- ten paired controller-evaluation demand realizations per profile; and
- controller/action diagnostic seeds, if any, distinct from all above.

The existing proposal retains DQN seeds 11001–11005 and controller-evaluation demand
seeds 21001–21010 per profile with separate SUMO ranges. Training, calibration,
validation and forecast-test demand seed allocations remain intentionally unassigned.
No proposed seed is an approval or evidence of traffic representativeness.

All four profiles must use the same expansion/generation contract, partition policy,
feature/target definitions, controller arms, and metric definitions. Their distinct
windows, requested volumes, teleport settings, and runtime seeds remain explicit strata;
profiles are not pooled as exchangeable observations without a preregistered model.

## 4. Forecast-dataset feasibility

The code contract can form seven one-second features, 60 consecutive past samples, and
four nonnegative approach-entry counts for the following 30 seconds. `lstm/research_data.py`
requires consecutive timestamps, immutable snapshot/event files, mapping identity,
training-only scaling, and causal targets. Strict inference independently requires a full
60-s history and live E1 installation matching lane/position.

Scientific feasibility is still **NO-GO**:

- `simulation/maps/tahrir.add.xml` does not install the mapped E1 detectors;
- the Tahrir mapping is an unapproved geometry candidate and contains empty directional
  entries for many TLS, so structural zeros and coverage require review;
- actual demand expansion alone cannot validate detector placement or target meaning;
- real multi-detector lane changes, teleport entry, cyclic re-entry, full short-edge
  traversal, relocation, dropped/malformed event rows and simulator interruptions remain
  bounded-test limitations or fail-closed cases; and
- no approved seven-input checkpoint exists.

The target convention must be frozen consistently before collection. Existing code uses
one-second bins and metadata enforcing future samples after the 60-s input; its current
manifest describes `[t,t+30)` whereas the protocol language uses `(t,t+30]`. These are
equivalent only after the precise meaning of timestamp `t` (completed interval versus
boundary instant) is externally fixed. Until reconciled, publication collection remains
blocked rather than silently relabeled.

## 5. Readiness gate

Essential for simulation-only experiments: approved expansion bytes/specs, explicit-ID
realizations, frozen usage allocation and seeds, no physical overlap, detector installation
and mapping review, target-boundary reconciliation, approved reward references, compatible
forecast model, and immutable protocol authorization. Additional requirements for
field-calibrated claims include independent Tahrir counts, OD/type/time distributions,
queues/travel times/saturation flows, calibration and held-out validation targets, and
traffic-engineering signal approval.

Publication trust roots remain empty. This candidate does not authorize publication
demand generation, model training, or Phase 3 performance experiments.

## Authoritative semantic reference

SUMO vehicle/flow documentation:
<https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html>
