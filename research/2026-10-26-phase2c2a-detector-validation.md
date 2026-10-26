# Phase 2C.2A — E1 edge-case characterization and Tahrir boundary review

**Date:** 2026-10-26 (Africa/Cairo)  
**Branch:** `research/ieee-reproducibility`  
**Starting commit:** `f20b9534b96f5b72d64aad70ecbb190dd71852f1`  
**Scope:** bounded detector physics and review preparation only. No Tahrir
traffic run, model training, mapping approval, or performance experiment occurred.

## 1. Reproducible characterization harness

`DeepQN/forecasting/characterize_e1_edge_cases.py` generates five separate
isolated SUMO fixtures with seed 17 and a 0.1-second simulation step. Each uses
real E1 `getVehicleData()` records as the signal under test and independently
records 0.1-second TraCI lane/position coordinates plus SUMO FCD. Network,
demand, detector definition, effective configuration, and FCD bytes are hashed.
Every external generator call has a 15-second timeout, TraCI startup has no retry
loop, the simulation is step-bounded, and the test subprocess has a 45-second
wall-clock limit.

Executed command (exit 0):

```powershell
python -m DeepQN.forecasting.characterize_e1_edge_cases D:\temp\phase2c2a-run4
```

SUMO reported `Eclipse SUMO sumo 1.26.0`. The complete result is outside the
repository at `D:/temp/phase2c2a-run4/result.json`, SHA-256
`6e2c2e1918f8f15fb8f5005a02638de60dade87e5ef9f4f959297ac829930823`.
It is bounded synthetic engineering evidence, not Tahrir traffic.

### Observed cases

| Case | Independent reference and intervention | E1 observation | Collector interpretation |
|---|---|---|---|
| Adjacent-lane change | At 5.3 s and 14 m on `B0C0_0`, TraCI requested lane 1; the 0.1 s trajectory contains both lanes near the 20 m plane | One entry on destination detector `e1_1` at 5.9 s; no source-lane event | One physical longitudinal plane crossing. No duplicate occurred in this tested geometry. Existing same-TLS/same-vehicle/same-time multi-detector rejection remains appropriate for ambiguous duplicates. |
| Controlled discontinuity | `moveTo` jumped `probe` from 10 m to 25 m across the 20 m plane at 4.9 s | No E1 event; SUMO starting/ending teleport lists were also empty | Correctly **not** a physical arrival, but proves `moveTo` cannot be detected from E1 or standard teleport lists. Publication collection must prohibit/log all relocation commands and invalidate a source containing one. |
| Genuine cyclic re-entry | Closed real route traversed `A0B0` twice; 0.1 s trajectory independently crosses the 20 m plane twice | Entries at 1.89 s and 21.0372 s for the same vehicle/detector | Legitimate repeated crossings have distinct entry times and remain two events. The existing identity includes entry time and preserves both. |
| Complete 5 m edge traversal | Integer snapshots show `e0_0` at 2 s and `e2_0` at 3 s; 0.1 s trajectory occupies `e1_0` between them | One entry at 2.41 s, leave 2.61 s | E1 retains a crossing that one-second snapshots miss. Snapshot inference is not an acceptable substitute. |
| Complete 0.20 m edge traversal | Integer snapshots again omit `e1_0`; fine trace establishes occupancy between them | One entry at 2.41 s, leave 2.61 s | SUMO can install/report a 0.10 m E1 on a 0.20 m lane. This demonstrates software behavior only, not suitability of the Tahrir boundary. |

The 0.20 m detector remains occupied for 0.20 seconds because the 2 m vehicle
body spans the plane; it does not spend 0.20 seconds wholly inside that edge.

### Fixture hashes

| Scenario | Network | Demand | Detector definition | SUMO config | FCD reference |
|---|---|---|---|---|---|
| lane change | `597299f91fa3554fae8ab93615037484a0dbb0917410f6ef8fe13a7d7da313ce` | `44cc4abdcc592190b267e385f82e292e769410906086ca47d619e140d18a85cd` | `3e26059fb8b01376cf4ac8bd1a412f6549f59b3ec90273c559233a94b031db1a` | `3265c0d61d645087d28e35825eea977e28372fa7c59727886c66976cb670fb61` | `2579eef93bb3aae9e3a07ce1d1c2a86b1777c679224d5bbb8488e297d32854ba` |
| controlled `moveTo` | `136fad9243157a76b281dbb5badeccf5c8030385bca50ef5bcd2f5e54b1c24ea` | `44cc4abdcc592190b267e385f82e292e769410906086ca47d619e140d18a85cd` | `2f0dd474567738a45d635910d67d3f270f885c4f129ed28bc9c8bdf4291a6ab1` | `3265c0d61d645087d28e35825eea977e28372fa7c59727886c66976cb670fb61` | `38b1d59f18e27e2e6232f666e14050447f0af58f7a12a00760d3a359283fd8d1` |
| cyclic re-entry | `7977ca60dff39c0f7c1a80c31f91c7e5194451fa4c52b1f45d3cb392f28a6d7d` | `cd5610996e87e1332fe8cb9e4ab3348152a715f483827013a3a5ea53dbc55230` | `d72b5cb9cb717f169d4d462f9ca037c5e77b3d589aeb3f77be1060922e738bcb` | `3265c0d61d645087d28e35825eea977e28372fa7c59727886c66976cb670fb61` | `96ae29f859319d4affedafb8168cf7531ae9ed9866fbfed15d6c0388a49d7b9b` |
| 5 m edge | `b6857e9f3d98d96f5421328b4141760a9077b93283e96901f4fcd421c16ea1bf` | `c6f3109d83ff1db79a46e4b8350be523e01dc3001db5c907e92d1b65a0b6b750` | `7b818800f1f6b22f4fccce61077f557e55342d755060f90fa40dba23540a65bb` | `3265c0d61d645087d28e35825eea977e28372fa7c59727886c66976cb670fb61` | `bea62358241996782c92e5f679e8fc21943dc2409736f8fdbdbd0aa9944fba77` |
| 0.20 m edge | `e817f5103e83e49033e757c09dc4b45181a1ac6ed5bbfb5a9830059acb84e531` | `c6f3109d83ff1db79a46e4b8350be523e01dc3001db5c907e92d1b65a0b6b750` | `7b818800f1f6b22f4fccce61077f557e55342d755060f90fa40dba23540a65bb` | `3265c0d61d645087d28e35825eea977e28372fa7c59727886c66976cb670fb61` | `ee766d37c13c57c76edf23371c360f2656b5b0157022d8d6be645a42ac40bd1a` |

## 2. Validity-rule consequence

The event schema itself did not change: detector/vehicle/exact-entry-time remains
the correct identity, and the real cyclic test supports retaining later re-entry.
No version was bumped merely to fit a fixture.

One fail-closed runtime rule was added at
`DeepQN/forecasting/inference.py:174-185`: verified-model collection rejects any
vehicle reported by SUMO's starting or ending teleport lists before E1 data is
accepted. This detects ordinary SUMO teleport diagnostics. The controlled
`moveTo` counterexample is deliberately documented as outside those lists;
therefore publication collection must also establish that no API/client can
relocate vehicles, and must retain a command/audit log. Neither E1 nor snapshot
inference can reconstruct the skipped physical path.

No detector event is fabricated for a discontinuity. Deterministic event
serialization and the empty approval trust root are unchanged.

## 3. Exact Tahrir `10873186#8` assessment

The frozen network remains
`simulation/maps/tahrirupdated.net.xml`, SHA-256
`a7f25235f208db932735983653e00b3af659d09d04169ef5807265433c2db708`.
The approach mapping remains
`DeepQN/configs/forecast_mappings/tahrir_v1.json`, SHA-256
`275d8494f9a408ddee515b63402bdac27bba041730726231d35f428ac04c0f98`.
The generated 46-detector mapping remains
`afd17243b1d154ecf073cca645c3a58dfc4b7e4875e105206be6d44b8fed4a06`
and unapproved.

### Geometry and connectivity

- `10873186#8` is a 0.20 m, five-lane `highway.primary` edge from ordinary node
  `3105658889` to controlled TLS `96621100`; every lane has speed 27.78 m/s
  (`tahrirupdated.net.xml:484-490`). It is a very short connector immediately
  before the signal, not an internal `:` edge.
- Upstream `10873186#7` is 72.38 m with eight lanes
  (`tahrirupdated.net.xml:474-483`). Its lanes 3–7 connect one-to-one through
  internal node paths to `10873186#8` lanes 0–4
  (`tahrirupdated.net.xml:1198-1202`).
- The five short-edge lanes then form signal-controlled straight links 5–9 to
  `10873186#9` lanes 1–5 at TLS `96621100`
  (`tahrirupdated.net.xml:1203-1207`). The downstream edge is 5.41 m and is an
  exit from the controlled junction, so it is not a candidate arrival plane.
- At 27.78 m/s, 0.20 m corresponds to about 7.2 ms of front-bumper travel.
  Detector behavior at that scale is supported by the synthetic 0.20 m test,
  but geometric meaning and numerical robustness require human review.

### Candidate decisions for independent review

1. **Retain current plane:** five E1s at 0.10 m on `10873186#8`. This measures a
   vehicle after it has passed node `3105658889` and committed to one of the five
   lanes feeding TLS `96621100`. Advantage: exact ownership and proximity to the
   controlled junction. Concern: negligible edge length and limited physical
   separation from adjacent junction/internal geometry.
2. **Candidate upstream plane:** five E1s near the ends of `10873186#7` lanes
   3–7 (for example, 0.10 m before the lane end), with explicit one-to-one
   ownership to `10873186#8` lanes 0–4. Advantage: meaningful 72.38 m lane
   geometry. Concerns: a measurement occurs before the internal connector;
   lane changes/route choices before or across the node can change which vehicles
   actually enter the short edge. Lanes 0–2 must not be included because they do
   not connect to `10873186#8`.
3. **Reject downstream plane:** `10873186#9` is after TLS service and would
   measure departures, not arrivals.

No candidate is selected here. Changing to option 2 requires a new approach-
mapping schema capable of lane-specific boundaries, a new mapping file/hash,
new detector mapping/hash, new collected data, new scaler/model, and retraining.
Approval of the present hash cannot transfer to a replacement.

### Sequential counting implications

Some mapped approaches are downstream of another controlled TLS—for example,
`690516091#1` follows `315744796` and feeds `96621100`, while `10873191#4`
follows `2031414899` and feeds `6288771431`. A vehicle is intentionally counted
once for each local junction it approaches. These are distinct local targets,
not duplicates within one TLS, but counts must not be summed into a network-wide
unique-vehicle arrival metric without reconciling vehicle identity.

## 4. Proposed seven-TLS boundary specification (not approved)

Every current boundary is an E1 at 0.10 m from the start of each listed lane,
owned by exactly one `(TLS, direction, edge, lane)`. Direction means SUMO travel
heading, not the geographic origin of demand.

| TLS | Direction / edge | Covered lane indices | Count | Geometric basis and review issue |
|---|---|---:|---:|---|
| `315744796` | north `690516091#0`; west `28718647#3` | 0–4; 0–4 | 10 | Both edges end at TLS; reviewer must confirm edge-start catchment and inclusion of physically uncontrolled lane `28718647#3_4` as demand context. |
| `96621100` | north `690516091#1`; north `10873186#8` | 0–4; 0–4 | 10 | Both enter TLS; second edge is the 0.20 m issue above. Two same-heading approaches are deliberately aggregated. |
| `2031414903` | south `10873186#5`; east `50211834#1` | 0–4; 0–3 | 9 | Both enter TLS; several lanes are incoming observations but not signal-controlled links. Confirm they belong in approach demand. |
| `2031414899` | north `10873191#3` | 0–3 | 4 | Edge enters TLS; local sequential counting with downstream TLS must be accepted explicitly. |
| `6288771431` | north `10873191#4` | 0–3 | 4 | This is immediately downstream of the prior controlled TLS and is a separate local target. |
| `271064234` | north `10873193#1` | 0–3 | 4 | Edge enters TLS; lane 0 is not a controlled-from lane and needs justification. |
| `315743335` | east `28718874#1` | 0–4 | 5 | Edge enters TLS; lanes 0–1 are not controlled-from lanes and need justification. |
| **Total** | 10 edges | all generated lane-specific E1s | **46** | Structural coverage only. |

## 5. Independent reviewer acceptance record

The reviewer should inspect the frozen network visually and topologically, not
merely verify hashes. Required checklist:

- [ ] Qualifications include traffic-signal/network engineering and SUMO network interpretation.
- [ ] All seven TLS and all 46 proposed lane planes were inspected.
- [ ] Each edge/lane truly represents entry to the intended local approach.
- [ ] Travel-heading direction labels are correct and scientifically meaningful.
- [ ] Uncontrolled incoming lanes are deliberately included or removed with justification.
- [ ] Sequential cross-TLS counts are accepted for local targets and prohibited for unique-network totals.
- [ ] Lane-change ambiguity/exclusion policy is acceptable.
- [ ] Teleport and external-relocation invalidation policy is acceptable.
- [ ] The `10873186#8` decision explicitly chooses current, upstream replacement, or rejection, with rationale.
- [ ] Reviewed network, approach mapping, and generated detector mapping hashes exactly match the record.
- [ ] Any proposed change creates a new version/hash and invalidates earlier collected data.

Approval record fields (intentionally blank):

| Field | Reviewer entry |
|---|---|
| Reviewer identity | |
| Professional role / qualifications | |
| Organization / independence relationship | |
| Review date | |
| Network SHA-256 reviewed | |
| Approach-mapping SHA-256 reviewed | |
| Detector-mapping SHA-256 reviewed | |
| `10873186#8` decision and rationale | |
| Exclusions/conditions | |
| Decision: approve / reject / revise | |
| Signature or verifiable approval reference | |

This blank template is not approval and is not added to
`TRUSTED_MAPPING_APPROVALS`.

## 6. GO/NO-GO

**NO-GO for publication detector-backed Tahrir collection.** The bounded tests
resolve the tested lane-change, cyclic, and short-edge software behavior, and
they establish a concrete relocation counterexample. They do not supply an
independent Tahrir boundary decision, do not characterize a naturally triggered
SUMO teleport across the plane, and do not approve the 0.20 m boundary.

Engineering collection may continue only in explicitly non-publication fixtures.
Publication readiness still requires the completed independent review above,
a hash-bound approved mapping, a production no-relocation/teleport audit trail,
and regeneration under the approved boundary version.

## 7. Verification record

Final commands and outcomes:

```powershell
python -m unittest DeepQN.tests.test_phase2b2a_detector_events DeepQN.tests.test_phase2b2b_methodology.StrictForecastTests DeepQN.tests.test_phase2c2a_detector_characterization -v
python -m unittest discover -s DeepQN/tests -p "test_*.py"
python -c "...compile changed Python files in memory..."
git -c safe.directory="D:/Courses/University zewail/Year4/grad_project/sumoflow-ai-traffic-optimization" -c core.whitespace=cr-at-eol diff --check
```

- Focused detector/strict-inference suite: **27 passed, 0 failed, 0 skipped**
  in 21.038 seconds, exit 0.
- Final full discovery after strengthening the trajectory reference: **148
  executed; 147 passed, 0 failed, 1 explicitly skipped legacy diagnostic** in
  64.753 seconds, exit 0.
- In-memory compilation: all three changed/created Python files compiled, exit 0.
- CRLF-aware `diff --check`: exit 0; only line-ending conversion notices for
  existing working-copy files were emitted.
- HEAD remained `f20b9534b96f5b72d64aad70ecbb190dd71852f1`.
- The first exploratory invocation used a nonexistent SUMO option
  `--fcd-output.period`, failed before simulation, and entered TraCI's retry
  behavior. The owned Python/SUMO processes were stopped, the option was removed,
  retry was disabled, and only the successful `run4` evidence above is retained.
  No failed run is represented as evidence.
- A strengthened short-edge assertion initially produced one `StopIteration`
  error because the 0.20 m edge can be traversed between adjacent 0.1-second
  lane samples. The ground-truth check was corrected to use the global detector
  plane coordinate (`x=25.1 m`), the individual regression passed, and the final
  full suite above passed. This is recorded as a characterization finding, not
  hidden as test success.

Files created or changed in this phase:

- `DeepQN/forecasting/characterize_e1_edge_cases.py`
- `DeepQN/forecasting/inference.py`
- `DeepQN/tests/test_phase2c2a_detector_characterization.py`
- `research/2026-10-26-phase2c2a-detector-validation.md`

Historical data, models, mappings, checkpoints, reward/replay logic, and unrelated
worktree files were not changed. No commit or push was performed.

