# Night synthetic v5 route-integrity correction — 2026-10-27

## Classification and scope

This is an uncalibrated, nonpublication synthetic-input correction. No SUMO
simulation or DQN training was run during this correction. The failed run
`night-v4-3ep-seed41001` and all v4 inputs were left unchanged.

## Reproduced blocker

Historical `simulation/maps/routes_night.rou.xml` defines `microbus_982` at
depart 86261 with type `type_bicycle` and route:

`28718647#2 28718647#3 690516091#1 10873186#9 10873186#0 10873186#1 10873191#0`

`simulation/maps/vehtype.add.xml` maps that type to `vClass="bicycle"`. The
network contains two geometric connections from `28718647#3` to
`690516091#1`, but every lane on `690516091#1` disallows bicycles. Thus there
is no permitted lane-to-lane connection for this vehicle class. The record is
unchanged in the expanded v4 intermediate and is not introduced by stop
correction, rerouting, or dynamic stop handling; the vehicle has no stops.

The v5 candidate changes only this record's type from `type_bicycle` to
`type_microbus` (`vClass="bus"`). ID, departure, route, and stop count remain
unchanged. This explicit correction is recorded in `input-manifest.json`; it
is not a general automatic route-repair rule.

## Candidate identities

- Input manifest SHA-256: `bfcec6f42ef9d6167f95528b1c95484a8f9a275596a7ded37028248b2adad02c`
- Corrected explicit route SHA-256: `1c9f8cb824f158f3d56ef0d0d1d5ff60f2a53df09b6eb0755dd012d888becf94`
- Physical-demand fingerprint: `afdd090bc71a624ab1d06889b04ae55a28b8c748c68cdd90ea8c194438ed1afa`
- Network SHA-256: `6b6bd5390b7b4797db15e7f5c0d305516f3ed8e7a0b5f3cbba856f0fe3bbd9ce`
- Full SUMO configuration SHA-256: `285e6578139d1f025f7adb9837613cac3f30c184ac98d889d74f04c9f01c4408`
- Vehicles: 2,106; required stops checked: 228; explicit class corrections: 1.

## Verification performed

The source validator rejects v4 at the exact class-incompatible pair. All v5
vehicles pass known-edge, ordered connectivity, lane-permission, vClass, stop
order, and stop-lane permission checks. All nine identities referenced by the
v5 manifest were rehashed successfully. Twenty-seven focused unit tests passed
and five changed Python files compiled from source.

Connectivity checking does not prove dynamic lane reachability in congestion.
The full-night fixed-time probe is deliberately unexecuted and remains the
required runtime gate before restarting training.

## Existing adverse-event evidence

The failed v4 DQN episode reports wrong-lane, yield, and jam teleports,
four detailed emergency-braking warnings, and a collision at time 84668 on
internal lane `:96621100_0_3` between passenger `flow_817.0` and motorcycle
`flow_868.0`. The episode did not persist decision-by-decision physical TLS
traces, so the existing artifacts cannot establish action-level causality.
The fixed-time v5 full-night probe must determine whether these outcomes also
occur without DQN control. A repeated collision or skipped required stop is a
scientific/network-integrity blocker requiring review, not a value to suppress.

## Runtime addendum: shared full-night defects

The subsequently supplied fixed-time v5 probe reached the complete 7,200-second
horizon, loaded all 2,106 vehicles, and produced valid reconciled metrics, but
reported one collision, five teleports, two skipped stops, and emergency
braking. The v5 DQN episode reached the horizon with related outcomes. The
absence of the former `microbus_982` fatal error confirms only that correction.

Source and runtime evidence now supersede the earlier v5 route-integrity claim:

- `microbus_21` and `microbus_634` stop at the effective end of lane
  `28718874#0_0`, while their next edge `10873193#0` is reachable only from
  lane 2. Their stop/turn sequence is structurally contradictory.
- `microbus_532` approaches from `10873186#8`, whose connections enter
  `10873186#9` only in lanes 1..5. The turn to `28718874#0` later requires
  lane 0. The intervening `10873186#9` and `10873186#0` edges total only
  11.64 m, explaining the observed mandatory lane change and its failure under
  congestion, but not establishing an approved geometry correction.
- Both recorded collisions occurred on internal lanes for the same movement
  `690516091#1 -> 10873186#9`: fixed-time lane `:96621100_0_4` and DQN lane
  `:96621100_0_3`. They are rear-end/same-movement evidence, not proof of a
  protected-versus-permissive crossing conflict. Internal-lane emergency
  braking remains unresolved.

The source validator now rejects an end-of-edge stop whose lane cannot serve
the next route edge, and the exploratory trainer independently reruns that
validation. Therefore v5 is blocked from any new training. No v6 candidate was
created: moving a curb-side stop, changing route geometry, or changing TLS
priority without traffic-engineering evidence would manufacture an unsupported
input. Collision and teleport warnings are classified as recorded traffic
outcomes; skipped stops remain execution-integrity failures. Collisions still
flag a snapshot and trigger the existing mandatory review stop when no other
integrity defect is present.
