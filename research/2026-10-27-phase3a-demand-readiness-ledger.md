# Phase 3.0A demand expansion and dataset-readiness evidence ledger

**Date:** 2026-10-27 (Africa/Cairo)
**Branch:** `research/ieee-reproducibility`
**Starting HEAD:** `38f5416a49f5117fd387f3508c1b81061a6969ff`
**Gate:** publication training and performance experiments remain **NO-GO**.

## Repository state

The branch matched `origin/research/ieee-reproducibility` at the starting HEAD. Phase
2C.3 was committed upstream. Pre-existing unrelated modifications to `.vscode`, tracked
bytecode, `test_metrics_phase1a.py`, backend images, the forecasting initializer and stray
status-named files were outside scope and were not edited as source work.

## Evidence classification

- **Verified source behavior:** directly parsed checked-in files or inspected code.
- **Bounded test evidence:** temporary synthetic XML and offline expansion only.
- **Proposed synthetic-simulation assumption:** candidate partition/seed/model policy.
- **Missing empirical evidence:** no field data establishes demand representativeness.
- **Unresolved scientific/safety question:** requires external domain review.

## Historical source-demand inventory

All configurations use `simulation/maps/tahrirupdated.net.xml`, SHA-256
`a7f25235f208db932735983653e00b3af659d09d04169ef5807265433c2db708`, plus
`tahrir.add.xml` (`87125278...885eac5`) and `vehtype.add.xml`
(`e97ddd34...bd835`). Units are seconds; profile windows are half-open for the primary
demand contract.

| Profile | Window | Configuration SHA-256 | Route SHA-256 | Explicit | Flows / requested | Requested total |
|---|---:|---|---|---:|---:|---:|
| Morning | `[27000,37800)` | `252bf33d...e1aea0` | `e6563d8c...15ce03` | 6,000 | 0 / 0 | 6,000 |
| Midday | `[43200,54000)` | `de5bc9ff...0bdee` | `35d56764...94a9f` | 1,752 | 1,800 / 7,320 | 9,072 |
| Evening | `[54000,72000)` | `f15eeea8...67f33` | `c5190b7e...f7abd` | 4,799 | 3,000 / 15,400 | 20,199 |
| Night | `[79200,86400)` | `9deb687c...b19cc` | `1e84789b...3396a` | 186 | 840 / 1,920 | 2,106 |

There are no `<trip>`, `<person>` or `<container>` definitions. Every one of the 5,640
flows has the same supported attribute set: ID, type, named route, begin/end, integer
number, `departLane="best"`, and `departSpeed="random"`; there are no flow children.
The mixed files contain 20 connected named routes. Their explicit bus/microbus vehicles
retain 2,348 / 6,416 / 242 stop elements for midday/evening/night. Morning is entirely
explicit with 399 OD pairs and near-regular 1.8-s departures.

Vehicle types are defined both locally and in `vehtype.add.xml`. Classes include
passenger, taxi, bus/microbus, truck, motorcycle and bicycle; source IDs and precise
physical attributes differ between local and additional definitions and are preserved,
not harmonized. All source explicit departures are scheduled requests. Flow `number`
is also requested schedule cardinality, never proof of insertion.

SUMO's official documentation says fixed `number` flows are equally distributed, with
the first vehicle at `begin`, and generated IDs `flowId.runningNumber`. Insufficient road
space may delay actual departure; max-depart-delay can remove vehicles. Runtime
`departSpeed="random"` consumes SUMO randomness even though flow departure spacing is
deterministic. Version 1.26.0 is frozen for the candidate contract.

The apparent source generator is `simulation/rush_hour/generate_rush_hour.py`, SHA-256
`040eab6e...754de`. It hardcodes hourly volumes/compositions without cited field sources,
selects from the first 20 unique routes, uses integer allocation/minimum counts, and calls
module-global `random.randint`, `random.choice`, `random.sample` and `random.uniform`
without seeding. Exact historical generation cannot be replayed. File-structure similarity
does not prove these checked-in routes came from that exact invocation.

## Minimal implementation and bounded evidence

Added `DeepQN/evaluation/demand_expansion.py` and
`DeepQN/tests/test_phase3a_demand_expansion.py`. The implementation supports only the
verified fixed-count complete-route subset, uses exact decimal equal spacing, expands
routes inline, preserves supported attributes/stops, writes a manifest, fails closed on
unsupported constructs, and atomically publishes without overwrite. It has no expansion
RNG and cannot manufacture independent realizations by changing a seed.

Nine bounded tests cover byte-identical regeneration, exact interval boundaries,
explicit denominator compatibility, type/route/attribute/stop preservation, stochastic
and incomplete-flow rejection, duplicate IDs, unknown elements, source/config/output
hash changes, self-approval/version/algorithm rejection, overwrite protection, and
failure injection without partial publication. The ninth test ran SUMO 1.26.0 against
the existing tiny signalized network: the four expanded fixed-count vehicles were loaded
and all four arrived by the 40-s horizon. This establishes bounded execution compatibility
for that fixture, not Tahrir insertion feasibility or demand validity.

A read-only/offline compatibility probe expanded all four historical sources in temporary
directories; no artifact was retained and SUMO was not started:

| Profile | Expanded explicit count | Route SHA-256 | Physical fingerprint |
|---|---:|---|---|
| Morning | 6,000 | `2573e62e...6036e5` | `657aacce...c7948` |
| Midday | 9,072 | `4f83090d...5e8ee` | `207a51f0...519c4` |
| Evening | 20,199 | `f9fba33e...89e42` | `1c7c5532...768a4` |
| Night | 2,106 | `3617604e...e62aad` | `8b7fbeb8...d56c5` |

These hashes characterize the current candidate algorithm/source bytes. They are neither
published artifacts nor successful simulations. Morning's output is normalized and
provenance-marked even though it had no flows.

## Partition and forecasting conclusions

The candidate specification maps seven scientific usages onto the existing five frozen
top-level partitions using future disjoint usage subpools; it does not change the registry
schema. Five DQN seeds and ten paired evaluation demands per profile remain proposals.
All demand-generation, SUMO and learning seeds are separate. Controller-evaluation
physical demand is inaccessible to forecast/DQN fitting, calibration and selection.

The causal feature/target software contracts exist, but publication dataset collection is
not ready. Tahrir's production additional file has no E1 detector installation; mapping
approval is absent; many direction slots are empty; `(t,t+30]` versus code-manifest
`[t,t+30)` needs a boundary-time reconciliation; and real lane-change, teleport,
re-entry, short-edge and interruption cases remain unresolved or fail closed. No forecast
model was trained.

## Unsupported constructs and remaining gates

Unsupported: probability/rate/period/Poisson flows, incomplete routes, route/type
distributions, triggered demand, people/containers, flow child routes, unverified SUMO
versions, and any stochastic transformation presented as equivalent to fixed `number`
semantics. None occur in the inspected four historical flow sets, but support is not
general SUMO demand support.

Simulation-only prerequisites still missing: independent approval of the synthetic demand
assumptions and detector boundaries, frozen explicit artifacts and usage allocation,
approved seed schedule/reward calibration/forecast output contract, compatible trained
models, and protocol authorization. Field-calibrated claims additionally require measured
volumes, OD/type/time distributions, queues, travel times, saturation flows and independent
calibration/validation. Trust roots and external decisions remain empty/unresolved.

## Verification record

Executed commands and outcomes:

```text
git ... branch --show-current / rev-parse HEAD / status
research/ieee-reproducibility; HEAD 38f5416a49f5117fd387f3508c1b81061a6969ff;
branch matched origin; only previously excluded unrelated local changes existed.

sumo --version
Eclipse SUMO 1.26.0 Windows release build; exit 0.

python -m unittest DeepQN.tests.test_phase3a_demand_expansion -v
9 run, 9 passed, 0 failed, 0 skipped; 1.650 s; exit 0.

python -m unittest discover -s DeepQN/tests -p "test_*.py" -q
211 run, 211 passed, 0 failed, 1 skipped; 133.444 s; exit 0.

python -c "... compile every DeepQN/lstm Python source in memory ..."
64 Python files compiled; exit 0.

git ... -c core.whitespace=cr-at-eol diff --check
exit 0; only an informational CRLF conversion warning for the pre-existing
test_metrics_phase1a.py worktree modification.
```

The single skip is the explicitly quarantined legacy interactive diagnostic and is not
reported as a pass. Full discovery ran only existing bounded fixtures plus the new 40-s
expansion fixture. Test-created detector XML was deleted and test-modified tracked
bytecode was restored.

Final pre-ledger hashes:

| Artifact | SHA-256 |
|---|---|
| `DeepQN/evaluation/demand_expansion.py` | `8c3914fe1a35185f0de6fca90a9b12fbfee121418313e2c1107cf200898b4f1b` |
| `DeepQN/tests/test_phase3a_demand_expansion.py` | `d551f1cb293d116a9f4386205ccc1f9cbe40aff06d5d0e11317f6db2df1bd866` |
| `research/2026-10-27-phase3a-demand-expansion-candidate.md` | `0b2366ee8b3087c3d1dd8dc6417af3276d1fc7b1cc80a276ab2d5811e58eac70` |

No full Tahrir/Taksim simulation, model training, dependency installation, historical
artifact modification, commit or push occurred. Publication trust roots remain empty and
all external decisions remain unresolved. Phase 3 publication training/performance
experiments remain **NO-GO**; Phase 3.0B and Phase 3.1 were not started.

## Final adversarial correction — 2026-10-27

### Reproduced/source-verified defects

- The prior expander deep-copied every flow child. For the supplied `begin=0`, `end=4`,
  `number=2`, `stop until=10` case this necessarily emitted the same `until=10` for both
  explicit records, while SUMO 1.26 documents that later flow vehicles shift `until` by
  their departure offset. This was a semantic mismatch, not a formatting difference.
- The prior verifier trusted the candidate manifest's output hash and ID-insensitive
  physical fingerprint. Coordinated route/manifest edits could therefore be internally
  self-consistent. The adversarial regression changes `f.0`'s departure and both reported
  values; strict verification rejects it against the frozen expected manifest identity.
- An explicit inline route was previously checked only for a nonempty `edges` string.
  Unknown and disconnected edges were accepted by that local check.

### Corrections and bounded contract

`DeepQN/evaluation/demand_expansion.py` now:

- restricts v1 optional flow fields to absent or the historically observed
  `departLane="best"` and `departSpeed="random"` values;
- permits only shift-independent `<stop lane="..." duration="..."/>` flow children and
  rejects `until`, triggered/timetable/waypoint/transport fields, `<param>`, nonfinite or
  negative duration, and other unverified combinations;
- requires source named routes to contain exactly `id`/`edges` and no child semantics;
- validates unknown and disconnected edges for inline explicit-vehicle routes;
- requires externally supplied expected specification and manifest SHA-256 values;
- validates the exact top-level manifest contract, fixed algorithm/SUMO version, trusted
  specification identity, all source hashes, output hash/count/fingerprint; and
- independently regenerates from the trusted specification in a temporary directory and
  requires identical semantic manifest and route bytes.

This does not create a trust root. The expected hashes must be frozen outside the candidate
artifact; supplying freshly computed hashes from modified files merely defines a different,
unapproved candidate. Hash identity is not authentication, scientific approval, or demand
validity. The current candidates have no externally frozen expected-manifest reference and
therefore remain untrusted for publication use.

### Bounded SUMO 1.26 evidence

The focused suite ran source-flow and expanded-explicit forms on the tiny signalized network
with identical seed 17, one-second timestep, 80-second horizon, four scheduled IDs and a
two-second lane stop. For the deterministic subset, IDs, requested/actual departures,
tripinfo outcomes, loaded/inserted/arrived populations and stop-influenced timings were
byte-value equal in the asserted records; both forms reported 4/4/4 loaded/inserted/arrived.
With preserved `departSpeed="random"`, both forms again reported 4/4/4 and the same IDs, but
the test deliberately does not assert trajectory identity: input representation may change
SUMO RNG consumption. This establishes bounded scheduled-demand and population parity only.

Offline expansion of the unchanged historical inputs again produced explicit counts
6,000 / 9,072 / 20,199 / 2,106. All 5,640 historical flows use the accepted optional pair
and none has children. No historical file was modified and no Tahrir simulation was run.

### Regression and verification record

```text
python -m unittest DeepQN.tests.test_phase3a_demand_expansion -v
15 run, 15 passed, 0 failed, 0 skipped; 11.593 s; exit 0.

python -m unittest discover -s DeepQN/tests -p "test_*.py" -q
217 run, 217 passed, 0 failed, 1 skipped; 159.232 s; exit 0.

python -c "...compile every DeepQN/lstm Python source in memory..."
64 Python files compiled; exit 0.

git ... -c core.whitespace=cr-at-eol diff --check
exit 0; only the pre-existing test_metrics_phase1a.py CRLF warning was emitted.
```

One intermediate focused run had 13 passes and one assertion-message mismatch failure; the
production behavior already failed closed. The test expectation was corrected, and both the
final focused and full runs above passed. The one full-suite skip remains the explicitly
quarantined legacy interactive diagnostic. Test-created detector XML was removed and tracked
bytecode changed by test imports was restored; unrelated worktree changes were untouched.

Final pre-ledger SHA-256 identities:

| Artifact | SHA-256 |
|---|---|
| `DeepQN/evaluation/demand_expansion.py` | `54f39bdd1ddb2877df3b34977faf4f836046e7499a58978c6d9fe40552457f99` |
| `DeepQN/tests/test_phase3a_demand_expansion.py` | `8c60341baec633fb31afa86262be172eabfa882c4378c6ce4c754d02de6300b9` |
| `research/2026-10-27-phase3a-demand-expansion-candidate.md` | `78284f73c1683f8b741e8b0b876e24aa58d1d7a560cd957f7a817eda25f661e4` |

Authoritative semantic references inspected: SUMO's *Definition of Vehicles, Vehicle Types,
and Routes* and *Public Transport* documentation, which state fixed-count equal spacing,
generated ID behavior, delayed actual departures, and the flow-stop `until` offset rule.

Remaining blockers are unchanged: no external expected-manifest registry/approval, no
independent demand realization method, no field calibration, and unresolved detector,
reward-reference, forecast-model, signal-safety and protocol approvals. Phase 3 experiments
and Phase 3.0B remain **NO-GO**.
