# Phase 2C.2D — Tahrir controller readiness and external review package

Date: 2026-10-26  
Branch: `research/ieee-reproducibility`  
Starting commit: `736517a7735870e007dbdfae60ffbee2759332e8`  
Status: engineering reconstruction complete; every scientific approval remains blank.

## Scope and evidence standard

This phase inspected the frozen network and controller source, created an
unapproved hash-bound review candidate, and ran only existing bounded engineering
fixtures plus contract tests. It did not run Tahrir, select saturation flows or
actuation parameters, implement a new controller, or claim traffic improvement.

The network contains nine static programs at
`simulation/maps/tahrirupdated.net.xml:775-816`. Seven are selected in
`DeepQN/configs/dqn_config.yaml:46-172`; `6288771435` and `96621068` are not.
Controlled links are the network connections at lines 1174-1341, including the
seven controlled sets at 1184-1185, 1203-1207, 1222-1229, 1237-1239,
1268-1271, 1278-1280, and 1322-1341.

Frozen source identities:

| Artifact | SHA-256 |
|---|---|
| Network | `a7f25235f208db932735983653e00b3af659d09d04169ef5807265433c2db708` |
| DQN configuration | `c7f31e298d83f8402fd02df2dcb75d4efe9237e96952f10aaa90635f5833f83d` |
| Forecast mapping candidate | `275d8494f9a408ddee515b63402bdac27bba041730726231d35f428ac04c0f98` |
| Existing additional file | `87125278f5f806a769e3223d70278adbc02c88010c4e80fc32813a6ae885eac5` |
| New review candidate | `ddf2e4dead26bbf45b13489de876c33330e592cbfdfccde3bb5518e1d6f37657` |

## Seven-TLS feasibility matrix

`G/g` is physical service, `y` is yellow, and `r` is red. “Configured safety”
describes current code/configuration, not independent traffic-engineering approval.

| TLS | Physical program and controlled movements | Configured transition limits | Conventional max-pressure | Actuated design status |
|---|---|---|---|---|
| `315744796` | `0: GGGGrrrrrr` serves links 0–3; `1` yellow 6 s; `2: rrrrGGGGGG` serves 4–9; `3` yellow 6 s. Link 0–1 right and 2–3 straight from `28718647#3`; 4–8 straight and 9 left from `690516091#0`. | min green 15 s, max 60 s (`dqn_config.yaml:48-66`) | **Topology eligible only.** Two competing services; exact movement map and saturation weights remain unapproved. | Two-green actuated candidate possible, but detector/gap and min/max values require review. |
| `2031414903` | `0: GGrr` serves links 0–1 from `50211834#1`; 6 s yellow; `2: rrGG` serves links 2–3 from `10873186#5`; 6 s yellow. | min 15 s, max 60 s (`:90-107`) | **Topology eligible only.** Two competing services; saturation weights unapproved. | Two-green candidate possible; no approved detectors or parameters. |
| `96621100` | Phase 0 all red 51 s; phase 1 all yellow 3 s; phase 2 all green 75 s. Links 0–4 originate on `690516091#1`; 5–9 on `10873186#8`; all ten are served together. | one green, red hold 0, min 20 s, max 90 s (`:68-88`) | **Ineligible:** red hold is not a competing service phase. | At most a gate/metering candidate, not phase-selection actuation. Maximum red and demand/gap rules are unapproved. |
| `2031414899` | Four straight links `10873191#3 → #4`; 46 s all green, 3 s yellow, 81 s all red. | one green, red hold 2, min 20 s, max 80 s (`:109-123`) | **Ineligible:** only one service set. | Gate-control candidate only; parameters unapproved. |
| `6288771431` | Four straight links `10873191#4 → #5`; 46 s all green, 3 s yellow, 81 s all red. | one green, red hold 2, min 20 s, max 80 s (`:125-139`) | **Ineligible:** only one service set. | Gate-control candidate only; parameters unapproved. |
| `271064234` | Three straight links from lanes 1–3 of `10873193#1`; 80 s all green, 5 s yellow, 5 s all red. | one green, red hold 2, min 20 s, max 80 s (`:141-155`) | **Ineligible:** only one service set. | Gate-control candidate only; parameters unapproved. |
| `315743335` | Three straight links from lanes 2–4 of `28718874#1`; 80 s all green, 5 s yellow, 5 s all red. | one green, red hold 2, min 20 s, max 80 s (`:157-172`) | **Ineligible:** only one service set. | Gate-control candidate only; parameters unapproved. |

All declared upstream and downstream lanes exist in the frozen network and are
queryable through the TraCI lane API in principle. This phase did not run Tahrir
to establish sensor availability under every demand condition. The DQN
`incoming_lanes` lists are observation inventories and sometimes contain lanes
without a controlled outgoing link; they are not substitutes for a reviewed
movement definition.

## Two unselected TLS

| TLS | Frozen program | Required treatment |
|---|---|---|
| `6288771435` | One 118 s phase, `GGGGGGGG`; all eight links always served. Program hash `d42310d0b58e3a345647910ecea07f160bb8f56f4b31995f0bcefff2a424d64d`. | Remain under this exact native program unless a separate network redesign is approved. |
| `96621068` | One 90 s phase, `GGGG`; all four links always served. Program hash `de861eb091625f89054b679676ed0cc31946aefb6a65ba10dd1f12a0601b1dc3`. | Remain under this exact native program. |

The other canonical program hashes are:

```text
2031414899 25109399c4525cffe6461f4f83349d57a4c128aab24cb284c0186e3ca0ed52a5
2031414903 b0c25c7b01d679c771864e412ecca2921c1e0d7693853b7fe6dff88633023b51
271064234  6c3f5f5f0bee1a0ee5142f25b53c7696b2ec6a9aa4db775d5356cc878be0a037
315743335  f916ecc85b46338566051e530f9be710715748f522600daf081f4fe6d2051ab2
315744796  24f27b2d257987bd454c080c2dc30662a3f2ea1e5c35637fc18328a08d7bef94
6288771431 6191876f6748b279fc3527ccb8b806cfa7a115b339f58e558202249481f8ca1c
96621100   2c57b6d4dfe5abdedc1b28ceef9a48c4b5939676cb5b95180399a47b481e9e09
```

## Max-pressure contract and hybrid decision

For an approved two-green junction, the candidate definition remains

\[
P_p(t)=\sum_{m\in\mathcal{M}_p}s_m\,[q^{up}_m(t)-q^{down}_m(t)],
\qquad p^*(t)=\arg\max_p P_p(t),
\]

with deterministic phase-index tie breaking. Queues are last-step halted vehicle
counts and `s_m` is a positive reviewed saturation weight. Downstream queues and
every link served by every green must be included. No `s_m` is supplied here.

The existing adapter already requires exactly two greens, no red hold, complete
movement service, and target-versus-binary-request separation
(`controller_baselines.py:335-383`). It therefore correctly rejects five of the
seven controlled TLS. `_PhaseController` enforces minimum green and yellow, and
cycles through declared transitions (`sumo_env.py:79-215`); it does not justify a
new service phase.

A possible network controller is separately named
`tahrir_hybrid_pressure_fixed_restricted_candidate_v1`: pressure selection at
the two topology-eligible TLS and an independently approved safe local policy at
the five gate TLS. It is **not** standard all-junction max pressure and is not
currently executable. Sending `keep=0` is not a fixed-program substitute because
the environment's maximum-green override can initiate a switch. A local gate
schedule, red maximum, decision timing, and handoff semantics need approval and
a new reviewed adapter before implementation.

No expanded adapter was implemented because the required reviewed control
contract does not yet exist.

## Actuated-controller design package

The morning effective configuration loads only `tahrir.add.xml` and
`vehtype.add.xml` (`simulation/maps/config_morning_rush.sumocfg:4-8`). Neither
contains an E1/E2 traffic detector. The forecast mapping identifies candidate
approach edges, not a loaded detector installation or approved actuated layout.

Candidate requirements for external design are:

- For `315744796` and `2031414903`: preserve both service phases and intervening
  yellow states; define reviewed approach detectors, gap/passage rules,
  minimum/maximum green, and any all-red clearance. Validate the selected live
  `actuated` program and detector identities through TraCI before execution.
- For the five single-service TLS: decide first whether these are intentional
  meters/gates. Any actuation selects green versus red availability, not competing
  movements. It requires maximum-red protection, queue-spillback analysis, demand
  detection, and coordination review. It must not be called max pressure.
- For the two unselected always-green TLS: no meaningful actuation exists without
  an approved physical program redesign.

No gap, passage-time, detector distance, red maximum, saturation flow, or revised
phase duration is scientifically selected by this package. Execution remains
fail closed. The tiny actuated fixture validates the runner mechanics only and
must not be transplanted.

## Fairness and comparison eligibility

- DQN conditions use the shared 60 s `phase_controller_keep_v1` initialization.
  A future controlled hybrid policy could share it only if every TLS's initial
  physical state and local restricted policy are specified identically.
- The two-green pressure subset can use the controlled evaluator after complete
  mappings and approval, but it cannot currently provide the exact seven-TLS
  action dictionary required by comparative research without an approved policy
  for the other five.
- Native fixed-time and future SUMO-actuated programs operate from simulation
  start. Internal native controller state is not established as transferable
  across a 60 s handoff. They remain `secondary_full_operation`, enforced at
  `baseline.py:48-61`.
- A hybrid candidate must be named and analyzed separately. It must not be pooled
  statistically with a standard all-junction max-pressure claim.
- Within every legitimate comparison, network, all nine TLS programs, frozen
  demand, vehicle types, SUMO version/options, outcome contract, horizon, and
  paired evaluation seed must match. Initialization scope must be a reported
  comparison factor, not hidden.

## Bounded physical-safety evidence

No new Tahrir simulation was run. The focused suite re-executed accepted bounded
tiny-fixture evidence for red-hold/minimum-green behavior, rejection of red-hold
and three-green pressure topologies, target-versus-physical execution, native
actuated program identity, detector activation, and phase schedule fidelity.
These demonstrate implementation mechanics, not Tahrir controller validity.

The Tahrir contract tests independently parse the actual frozen XML and require:
all nine programs present; all seven controlled link indices accounted for;
declared service groups physically green; only the two genuine competing-service
TLS marked topology eligible; saturation weights absent; source hashes exact;
and all review/approval fields blank.

## External review package

The machine-readable candidate is
`DeepQN/configs/controller_contracts/tahrir_controller_review_candidate_v1.json`
(SHA-256 `ddf2e4dead26bbf45b13489de876c33330e592cbfdfccde3bb5518e1d6f37657`).
It binds all nine programs, seven controlled service inventories, restricted-TLS
exceptions, missing detector/actuation decisions, comparison scopes, and sources.

Required reviewer qualifications:

- traffic-signal timing and conflict/clearance design;
- SUMO signal-program and detector configuration;
- independent review of the Tahrir network geometry and intended gate operation.

Decision fields intentionally remain blank:

```text
Decision: ______________________________
Reviewer identity: _____________________
Organization: __________________________
Date: __________________________________
Signature/external record: _____________
Approved mapping SHA-256: ______________
Approved saturation SHA-256: ___________
Approved actuated-program SHA-256: ______
```

A matching hash proves unchanged bytes, not competent approval. Approval must be
an externally anchored record bound to the exact candidate and resulting reviewed
artifacts. Repository booleans cannot authorize publication use. Trust roots are
empty.

## Verification

Focused command:

```text
python -m unittest DeepQN.tests.test_phase2c2d_tahrir_controller_contract DeepQN.tests.test_phase2b3_baselines -v
Ran 31 tests in 14.719s
OK
```

This included only bounded tiny SUMO fixtures already required by the baseline
suite. No full Tahrir/Taksim run occurred. Test file SHA-256:
`00a2276c1534e3413bf81ddbb7bac0458541e7cfba8550faf31872994e21644f`.

Full regression and compilation:

```text
python -c "... compile(...) ..."
compiled 1 changed Python file

python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
Ran 179 tests in 111.601s
OK (skipped=1)
```

The sole skip was the pre-existing quarantined interactive diagnostic; no
Phase 2C.2D test was skipped.

## Remaining limitations and Phase 3 gate

| Requirement | Status |
|---|---|
| Frozen nine-program/link reconstruction | Verified from source and regression tested |
| Two-green topology eligibility | Verified for two TLS only; mapping/saturation approval absent |
| Five gate TLS conventional max pressure | **Ineligible** |
| Hybrid network controller | Candidate design only; adapter and local policy unapproved |
| Tahrir actuated programs/detectors | **NO-GO** |
| Initialization-equivalent primary comparison | **NO-GO pending controller-specific approval** |
| External traffic-engineering review | **Missing** |
| Publication trust roots | **Empty** |
| Phase 3 training/experiments | **NO-GO** |

Stop point: Phase 2C.2D only. Phase 3 was not started.

## Signal-phase sequencing addendum

Independent review identified that phase-state membership alone was
insufficient: native SUMO static programs advance by XML order and wrap from the
last phase to phase 0. The machine-readable candidate now records every ordered
phase, exact `G/g` link inventory, and wraparound transition for all nine TLS.

| TLS | Native cyclic order | Source-derived sequencing observation |
|---|---|---|
| `2031414899` | `0 green → 1 yellow → 2 all-red → 0 green` | Conventional state ordering in source; clearance sufficiency unreviewed. |
| `2031414903` | `0 green(A) → 1 yellow(A) → 2 green(B) → 3 yellow(B) → 0` | No explicit all-red interval between conflicting service groups. Source alone cannot establish that yellow-only clearance is adequate. |
| `271064234` | `0 green → 1 yellow → 2 all-red → 0` | Ordering identifiable; timing safety unreviewed. |
| `315743335` | `0 green → 1 yellow → 2 all-red → 0` | Ordering identifiable; timing safety unreviewed. |
| `315744796` | `0 green(A) → 1 yellow(A) → 2 green(B) → 3 yellow(B) → 0` | No explicit all-red interval between service groups; requires conflict/clearance review. |
| `6288771431` | `0 green → 1 yellow → 2 all-red → 0` | Ordering identifiable; timing safety unreviewed. |
| `6288771435` | `0 permanent green → 0` | No phase transition; safety depends on geometry/right-of-way intent. |
| `96621068` | `0 permanent green → 0` | No phase transition; safety depends on geometry/right-of-way intent. |
| `96621100` | `0 all-red → 1 all-yellow → 2 all-green → 0 all-red` | **Anomaly:** yellow precedes green, while the green returns directly to red without a yellow phase. Intent and acceptability are unresolved. |

The exact `96621100` source is at
`simulation/maps/tahrirupdated.net.xml:813-817`. Native operation therefore
uses `0→1→2→0`, including the last-to-first `2→0` transition.

This is distinct from commanded DQN operation. Configuration declares
`green_to_yellow: 2→1`, `yellow_to_next: 1→0`, and direct
`red_to_green_phase: 2` (`dqn_config.yaml:68-77`). `_PhaseController` uses those
maps rather than allowing the native static timer to choose the next phase
(`sumo_env.py:175-215`). Consequently its commanded cycle is:

```text
green 2 → yellow 1 → red hold 0 → directly green 2
```

Thus the same all-yellow state is traversed in the opposite logical placement:
native red-to-green versus commanded green-to-red. DQN reset also initializes
the controller at configured green phase 2. Neither behavior proves the other
safe, and neither source ordering nor passing tests constitute traffic-engineering
approval.

External decisions required for `96621100` are:

1. confirm whether the XML order is intentional or an authoring error;
2. establish required green-to-red and red-to-green clearance from conflict
   geometry, speeds, and local signal practice;
3. decide whether the commanded `2→1→0→2` sequence is physically acceptable;
4. approve a corrected, hash-bound program externally if change is necessary.

The alternating two-green TLS also require explicit review of their lack of an
all-red interval. The permanent-green programs require confirmation that no
conflicting controlled service exists. These are review flags, not findings that
the programs are unsafe. Safety cannot be established from source alone.

The review-candidate hash changed from
`d740f039f2e07400b06d76c0cf5f66c60071324c8bbf7f8ab1c124977e754e1b`
to `ddf2e4dead26bbf45b13489de876c33330e592cbfdfccde3bb5518e1d6f37657`.
Every approval field remains empty and `publication_approved` remains false.

Addendum verification:

```text
python -m unittest DeepQN.tests.test_phase2c2d_tahrir_controller_contract -v
Ran 6 tests in 0.070s
OK

python -c "... compile(...) ..."
compiled 1 changed Python file

python -m unittest discover -s DeepQN/tests -p "test_*.py" -v
Ran 181 tests in 106.389s
OK (skipped=1)
```

The single skip remains the quarantined legacy interactive diagnostic. No
addendum test was skipped, and no Tahrir simulation was executed.
