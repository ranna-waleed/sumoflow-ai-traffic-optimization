# Phase 2C.2B — demand provenance and frozen-partition workflow

**Date:** 2026-10-26 (Africa/Cairo)  
**Branch:** `research/ieee-reproducibility`  
**Starting commit:** `1de51441101877e949d85cb318677788986fc89e`  
**Scope:** source audit and offline engineering contracts only. No model was
trained, no Tahrir/Taksim simulation was run, and no publication demand was
generated.

## Evidence language

- **Source-verified** means derived from checked-in XML/Python bytes.
- **Test-reproduced** means exercised by the synthetic offline unit suite.
- **Provisional assumption** means a candidate simulation model without empirical
  support.
- **Unresolved** means the repository contains insufficient evidence.

## 1. Historical Tahrir demand audit

All four configurations load the same frozen network, SHA-256
`a7f25235f208db932735983653e00b3af659d09d04169ef5807265433c2db708`,
and the two additional files `tahrir.add.xml` and `vehtype.add.xml`
(`simulation/maps/config_morning_rush.sumocfg:3-12`, with corresponding fields
in the other three configs). The configurations deliberately differ in
`time-to-teleport`: morning/evening 180 s, midday 240 s, and night 300 s
(`config_morning_rush.sumocfg:17-19`, `config_midday.sumocfg:17-19`,
`config_evening_rush.sumocfg:17-19`, `config_night.sumocfg:17-19`). They also set
`ignore-route-errors=true`; the new audit does not rely on that permissive SUMO
setting and instead checks every edge pair against the frozen network.

### Byte-derived inventory

“Requested” below means encoded by explicit vehicles plus `flow@number`. It is
not an observed insertion or arrival count. Actual insertion requires run output.

| Profile | `[begin,end)` s | Route SHA-256 | Explicit vehicles | Flow definitions / requested vehicles | Total requested | Explicit departure min / Q1 / median / Q3 / max (s) | Route/OD structure |
|---|---:|---|---:|---:|---:|---|---|
| morning | `[27000,37800)` | `e6563d8c…15ce03` | 6,000 | 0 / 0 | 6,000 | 27000 / 29699.55 / 32399.10 / 35098.65 / 37798.20 | 399 distinct inline entry→exit pairs; no top-level route table |
| midday | `[43200,54000)` | `35d56764…94a9f` | 1,752 | 1,800 / 7,320 | 9,072 | 43203 / 45443.75 / 47986 / 50819.5 / 53987 | 20 valid top-level routes and 20 resulting entry→exit pairs |
| evening | `[54000,72000)` | `c5190b7e…f7abd` | 4,799 | 3,000 / 15,400 | 20,199 | 54001 / 60098.5 / 63868 / 67431 / 71986 | 20 valid top-level routes and 20 resulting entry→exit pairs |
| night | `[79200,86400)` | `1e84789b…3396a` | 186 | 840 / 1,920 | 2,106 | 79203 / 80564 / 81835 / 83533.25 / 86339 | 20 valid top-level routes and 20 resulting entry→exit pairs |

Midday/evening/night flows cover consecutive 600-second bins. Their requested
bin totals are respectively 360–460, 340–740, and 120–200 vehicles. The exact
per-bin totals are reproducibly returned by
`audit_explicit_demand()` (`DeepQN/evaluation/demand_provenance.py:545`). Morning
explicit departures are almost perfectly spaced by 1.8 seconds; that regularity
is a file property, not evidence of real arrivals.

The encoded vehicle-type counts are:

- morning: `type_car` 3,283; `type_taxi` 937; `type_microbus` 694;
  `type_bus` 466; `type_truck` 302; `type_motorcycle` 250; `type_bicycle` 68.
- midday: flows use passenger 4,320, taxi 840, truck 720, motorcycle 720,
  bicycle 720; explicit vehicles use `type_car` 976, `type_taxi` 249,
  `type_microbus` 221, `type_bus` 128, `type_truck` 87,
  `type_motorcycle` 68, `type_bicycle` 23.
- evening: flows use passenger 9,860, taxi 1,940, truck 1,200, motorcycle
  1,200, bicycle 1,200; explicit counts are `type_car` 2,608,
  `type_taxi` 701, `type_microbus` 615, `type_bus` 403, `type_truck` 247,
  `type_motorcycle` 187, `type_bicycle` 38.
- night: flows use passenger 720, taxi 480, motorcycle 480, truck 240;
  explicit counts are `type_car` 100, `type_taxi` 28, `type_microbus` 21,
  `type_bus` 15, `type_motorcycle` 11, `type_truck` 9, `type_bicycle` 2.

Morning vehicles specify only ID, type and departure; lane/speed therefore use
SUMO defaults (`routes_morning_rush.rou.xml:4-6`). All explicit vehicles and
flows in the other profiles set `departLane="best"` and
`departSpeed="random"`; for example `routes_midday.rou.xml:30`. The latter
therefore require a separately recorded SUMO RNG seed. Midday/evening/night also
contain respectively 2,348 / 6,416 / 242 stop elements, including bus-stop and
lane-stop instructions, so stops are behavior-relevant demand content.

Every inspected inline or top-level route uses existing frozen-network edges and
every adjacent edge pair has a network connection. All demand IDs are unique
within each file. This establishes syntactic/topological validity, not insertion
feasibility under congestion.

### Origin and evidence quality

The repository contains a plausible generator,
`simulation/rush_hour/generate_rush_hour.py` (SHA-256
`040eab6e…754de`). It declares hourly demand without a cited measurement source
(`:6-12`), profile compositions without citations (`:44-78`), samples bus and
microbus behavior through Python's module-global RNG (`:214-297`), and never
calls `random.seed`. It selects the first 20 unique routes from
`tahrir_fixed.rou.xml` (`:114-130,324-330`), whose SHA-256 is
`760563d1…f001`, and allocates flows with integer truncation and a minimum of two
vehicles per route (`:352-409`).

The three mixed flow/vehicle files structurally resemble this generator, but
their explicit vehicles use `type_*` identifiers rather than the generator's
current bus/microbus identifiers; the morning file has a different all-explicit
structure. Git commit `e5df77f` changed all four files together on 2026-06-09,
but its message is only “updated.” No invocation, RNG state, source-count
citation, calibration record, or generator-output manifest is present. The
historical files are therefore **generated/edited simulation inputs of unresolved
exact provenance**, not verified empirical traffic distributions. Existing
metrics CSV/XML can describe outcomes under these inputs; they do not validate
the demand model against field counts.

## 2. Proposed stochastic-demand contract

The implemented `sumoflow-demand-spec-v1` contract is deliberately conservative
(`DeepQN/evaluation/demand_provenance.py:24-38,215-347`):

- input spec, source route, effective profile config, network, and every
  additional file are path- and SHA-256-bound;
- begin/end must exactly equal the config and the config must actually load the
  declared network/source/additional files;
- the source must contain unique, explicit vehicles with valid connected inline
  routes and resolvable types;
- v1 preserves OD routes, vehicle types, stops and departure/insertion attributes;
- the only supported changes are an explicit Bernoulli retention probability and
  bounded integer-second discrete-uniform departure jitter, driven by a local
  `random.Random(seed)` instance;
- each distribution must declare either
  `provisional_simulation_assumption` or `externally_approved` evidence status.

No Tahrir retention probability or jitter bound is proposed as empirically
correct. Volume variability, within-bin arrival process, OD covariance, route
choice, type mix and day-to-day correlation require external counts or an
independently reviewed simulation rationale. Until those are supplied, the
implemented distributions are engineering mechanisms only. OD/type mutation is
rejected rather than guessed.

Three historical profiles still contain `<flow>` elements, and the primary
metric contract requires frozen explicit IDs. Contract v1 therefore refuses
those files as generator sources. A later approved preparation step must expand
flows to explicit vehicles with a documented SUMO version/RNG, freeze the bytes,
and verify counts and constraints before stochastic realization. This limitation
prevents accidental use of unresolved implicit SUMO departures.

Publication-candidate generation is additionally gated by the deliberately empty
`TRUSTED_DEMAND_SPEC_APPROVALS` trust root (`demand_provenance.py:37`). A mutable
boolean in a spec cannot authorize publication data. Approval must be an
independent evidence file bound to the exact spec, source and network hashes.

## 3. Generator and manifest schemas

`generate_demand_realization()` validates all inputs before constructing output,
uses a private RNG, gives each explicit vehicle a realization-qualified ID,
sorts deterministically, and atomically renames a staging directory. Existing
destinations are rejected (`demand_provenance.py:215-342`). It writes:

1. `routes.rou.xml`: deterministic UTF-8 explicit demand;
2. `source_manifest.json` (`sumoflow-demand-source-v1`): immutable input paths,
   hashes, window, evidence status and publication gate;
3. `realization_manifest.json` (`sumoflow-demand-realization-v1`): realization
   ID, seed, profile/window, generation parameters, route hash, a physical-demand
   fingerprint independent of generated vehicle IDs, source-manifest hash,
   and `simulation_status="not_run"`.

Loading re-hashes the route, source manifest, specification, source route,
network, config and additional files (`demand_provenance.py:348-395`). A generated
demand is never itself labelled a successful simulation.

**Test-reproduced synthetic evidence:** seed 17 and realization ID `same`
produced 9 vehicles, route SHA-256
`7e5681d43c568a9cde818c1fc1bffac882e780e0dcf806ce733bfeb14ad09528`.
Regeneration into a fresh directory produced byte-identical route bytes. Its
physical fingerprint was
`1de39480202c3dfc498b7e5d25de474890f32d5d35036456a30cb25eb4993a72`.
These are disposable synthetic fixture identities, not Tahrir data or performance
evidence.

## 4. Frozen split registry and leakage controls

`sumoflow-experiment-split-registry-v1` supports exactly `training`,
`calibration`, `validation`, `forecast_test`, and `controller_evaluation`
(`demand_provenance.py:29-33,397-466`). Assignments are supplied by a
preregistered caller; the code never silently chooses a split. Each row records
the realization and manifest identities, demand seed, profile/window, exact route
and physical fingerprint, source spec, network/config hashes and generation
parameters.

The writer and verifier reject duplicate/conflicting IDs, identical route bytes,
physically identical schedules with renamed IDs, reuse of a spec/seed under a new
ID, invalid partitions, conflicting profile windows or frozen inputs, missing or
modified source artifacts, and registry modification. Output is non-overwriting.
Outcomes from repeated controller/SUMO executions are not new demand identities.

The dry-run `sumoflow-seed-schedule-proposal-v1` retains the accepted design but
explicitly says `proposal_requires_protocol_approval`
(`demand_provenance.py:514-542`):

- DQN training RNG proposal: `11001–11005` for each of `no_forecast` and
  `verified_forecast` (five independent learned-policy seeds, paired by numeric
  ID across conditions);
- controller-evaluation demand RNG proposal: `21001–21010` within each of four
  profile strata, shared across all controllers;
- separate SUMO RNG proposal: `31001–31010`, `31101–31110`, `31201–31210`, and
  `31301–31310` for morning, midday, evening and night;
- 40 unique profile-qualified evaluation replicate IDs.

Training-demand, reward-calibration, validation and forecast-test demand counts
and numeric seed namespaces remain deliberately unallocated. The entire numeric
schedule requires protocol approval before it becomes an immutable registry.

## 5. Evaluation integration

`RunManifest` now has optional demand-realization ID, realization-manifest hash,
partition and split-registry hash fields
(`DeepQN/evaluation/provenance.py:80-83`). If either side of a comparison uses
the new contract, both runs must provide all four fields and they must match
(`provenance.py:255-273`). Legacy/tiny engineering manifests remain readable.

`bind_realization_to_run()` independently verifies registry membership, exact
route bytes and the `[begin,end)` window before populating those fields
(`demand_provenance.py:491-511`). Existing run provenance already records
controller, config/network/additional hashes, SUMO/evaluation seed, output hashes,
completion status and SUMO version. Existing metric parsing still rejects a
premature/contradictory run. Consequently a future completed run can bind all
required identities, while a generated-but-unrun realization remains `not_run`
and a failed evaluation remains non-complete.

## 6. Adversarial verification

`DeepQN/tests/test_phase2c2b_demand_provenance.py` uses only temporary synthetic
XML. It covers byte-identical regeneration, distinct/repeated seeds, output
overwrite, publication gating, invalid/disconnected edges, unknown types,
out-of-window departures, duplicate IDs, changed spec/source/config, duplicate
registry identity, invalid partition, copied demand under a new ID,
seed-to-ID collision, source/manifest/registry tampering, conflicting profile
windows, exact run binding and collision-free dry-run allocation.

Files created or modified by this phase:

- `DeepQN/evaluation/demand_provenance.py` — new offline generator, audit,
  manifest, split-registry, binding and proposed-seed contracts.
- `DeepQN/evaluation/provenance.py` — four optional demand-provenance fields and
  fail-closed paired-comparison validation when the new contract is used.
- `DeepQN/tests/test_phase2c2b_demand_provenance.py` — eight synthetic adversarial
  tests.
- `research/2026-10-26-phase2c2b-demand-provenance.md` — this evidence record.

Pre-existing modifications to `.vscode/settings.json`,
`DeepQN/tests/test_metrics_phase1a.py`, both backend image files,
`DeepQN/forecasting/__init__.py`, and the two stray `tatus*` files were not
edited. Test-created tracked bytecode changes were restored exactly.

Final command record (all exit status 0 unless stated otherwise):

```powershell
python -m unittest DeepQN.tests.test_phase2c2b_demand_provenance -v
python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
python -c "...compile changed Python sources in memory..."
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" -c core.whitespace=cr-at-eol diff --check
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" status --short
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" rev-parse HEAD
```

- Focused suite: **8 passed, 0 failed, 0 skipped**.
- Full discovery: **156 executed; 155 passed, 0 failed, 1 explicitly skipped
  legacy diagnostic**, in 59.977 seconds. Full discovery exercised only the
  repository's pre-existing bounded tiny-SUMO integration tests; no Tahrir or
  Taksim scenario was launched.
- In-memory compilation: **3 changed Python files compiled**, exit 0.
- CRLF-aware `diff --check`: exit 0; Git emitted only line-ending conversion
  notices for pre-existing working-copy files.
- Final branch/HEAD: `research/ieee-reproducibility` at
  `1de51441101877e949d85cb318677788986fc89e`.

## 7. Readiness decision

**NO-GO for Phase 3 training or publication demand generation.** Code-level
determinism and leakage guards are established on synthetic inputs, but the
Tahrir demand distributions have no verified empirical source, three profiles
are not explicit-ID demands, the stochastic parameters are unapproved, the full
partition/seed allocation is not frozen, and forecast-boundary approval remains
NO-GO from Phase 2C.2A.

Promotion requires: recoverable/cited field counts or an independently justified
simulation-demand model; an approved explicit-flow expansion procedure; external
review of volume, temporal, OD/route and type distributions; a signed/hash-bound
demand-spec approval record; approved numeric partition sizes and seeds; frozen
explicit realizations; and a verified registry created before any training,
calibration, validation or test outcome is inspected.

## 8. Final manifest, leakage and evaluation-binding correction

Independent review identified four ways mutable metadata could escape the
intended fail-closed contract. The following correction supersedes the earlier
implementation details where they conflict.

### Reproduced failures and corrected invariants

1. **Physical identity was trusted.** A syntactically valid forged fingerprint
   could hide identical physical demand under different IDs. Generated route
   bytes now contain one canonical provenance comment binding schema,
   realization ID, demand seed, specification hash and purpose
   (`DeepQN/evaluation/demand_provenance.py:34,182-231,340-349`). Loading strips
   vehicle IDs, canonicalizes every remaining vehicle attribute/child element,
   recomputes the physical fingerprint and count from the verified route, and
   validates all generated IDs against the realization prefix. It then performs
   a fresh offline deterministic replay from the hash-verified specification,
   source, seed, ID and purpose and requires byte-identical route/source manifests
   and an exact realization manifest (`demand_provenance.py:400-500`). Forged
   fingerprint, seed, count, ID, parameters, profile and window now fail.
2. **Publication status was mutable.** Loader acceptance no longer follows a
   realization boolean. Purpose and publication state must agree between route,
   source manifest and realization; `research_candidate` replay consults the
   still-empty `TRUSTED_DEMAND_SPEC_APPROVALS`. Engineering fixtures remain false,
   and no approval entry was added.
3. **Registry top-level fields were trusted.** Verification now requires the
   exact five top-level keys, schema, `frozen_assignments` status, fixed supplied-
   preregistration policy, a non-empty row list, and
   `publication_approved=false` before rebuilding all rows from verified
   artifacts (`demand_provenance.py:564-606`). A caller-provided matching hash
   cannot turn the registry into scientific approval.
4. **Run binding trusted selected hashes.** Binding now verifies actual route,
   network, source configuration and every additional-file path and byte hash;
   profile/window; frozen membership; run-purpose/partition eligibility; and an
   actual effective SUMO configuration (`demand_provenance.py:626-760`). The
   effective config must be a derivation that changes only route, output
   destinations/settings and SUMO seed; it must load exactly the registered
   route/network/additional set, retain `[begin,end)`, use the run seed, and enable
   the complete metric outputs. Fields are committed only after all checks pass.

`RunManifest` now additionally records demand usage and the actual effective-
configuration path/file hash (`DeepQN/evaluation/provenance.py:80-86`). Paired
comparability requires matching complete demand identity, partition, registry and
usage (`provenance.py:255-273`). `_patch_sumocfg()` supports an explicit single
route substitution for this verified derivation
(`DeepQN/evaluation/evaluate.py:44-59`). Contract v1 rejects any source config
with additional route files; every demand contributor must be verified rather
than silently entering SUMO.

### Correction regressions

The focused suite now has 12 tests. New adversarial cases cover:

- same physical demand under different seeds/IDs with different route hashes;
- forged physical fingerprint, seed, count, realization ID, parameters, profile,
  window and publication status;
- renamed-ID physical duplicates;
- fabricated registry approval, altered status/policy and unexpected metadata;
- multiple configured source route files;
- incorrect network/config/additional hashes, wrong profile, partition misuse,
  effective-config seed substitution, and atomic failed binding.

The synthetic seed-17 route hash changed because generation provenance is now
embedded in the route bytes:
`7e5681d43c568a9cde818c1fc1bffac882e780e0dcf806ce733bfeb14ad09528`.
Its ID-independent physical fingerprint remains
`1de39480202c3dfc498b7e5d25de474890f32d5d35036456a30cb25eb4993a72`.

Final correction verification:

- Focused: **12 passed, 0 failed, 0 skipped** in 2.400 seconds.
- Full discovery: **160 executed; 159 passed, 0 failed, 1 deliberate legacy
  diagnostic skipped** in 53.219 seconds.
- In-memory compilation: **4 changed Python files compiled**, exit 0.
- CRLF-aware `diff --check`: exit 0; only existing line-ending conversion
  notices were emitted.
- Branch/HEAD remained `research/ieee-reproducibility` at
  `1de51441101877e949d85cb318677788986fc89e`.

### Remaining limitation

Deterministic replay proves consistency with the declared seed and source. It
cannot cryptographically prove when a coordinated rewrite occurred before any
manifest/route hash was placed in an external immutable registry. Scientific use
therefore still requires the separately approved, externally anchored registry
and approval record. Both publication trust roots remain empty, so this code
continues to reject publication demand and scientific run binding. Phase 3
remains **NO-GO**.

