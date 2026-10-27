# Night synthetic v6 candidate — 2026-10-27

> **Superseded implementation evidence:** the first compiled-XML v6 network
> crashed SUMO 1.26.0. Its identity and isolation evidence are preserved in
> `2026-10-27-v6-native-crash-isolation.md`. The current v6 is rebuilt from
> SUMO plain XML with `netconvert`; hashes and assumptions below from the first
> draft must not be used as current artifact identities.

## Scope

This is an explicitly uncalibrated, Tahrir-derived synthetic scenario. Its
effective lane lengths, service-stop lanes, and protected signal alternation
are engineering assumptions for internal simulation coherence, not surveyed
Tahrir geometry or traffic-engineering approval. No DQN training or full-night
SUMO run was performed while preparing it.

## Versioned corrections

- All 2,106 explicit departures are retained.
- All 242 source stop obligations are retained. Ten vehicle stop lists that
  were historically ordered against their route direction are reordered by
  route edge; none are removed.
- `microbus_21` and `microbus_634` retain their stop edge, positions,
  durations, identities, departures, types, and routes. Their stop lane changes
  from `28718874#0_0` to `28718874#0_2`, the lane serving their next left-turn
  edge. Lane 2 is declared a synthetic in-lane service position.
- The reviewed `microbus_982` class correction from v5 is retained.
- Effective lengths for the schematic short/service edges are set as follows:
  `10873186#8/#9/#0 = 30 m`, `28718874#0 = 60 m`,
  `10873193#0 = 30 m`, and `192591565#0/#1 = 30 m`. Coordinate shapes remain
  schematic. The route contract permits lateral movement only where at least
  25 m of effective edge length is available.
- The secondary approach `10873186#8` maps lanes `0,1,2,3,4` to downstream
  `10873186#9` lanes `0,1,2,3,5`. This gives lane 0 direct access to the later
  right turn while retaining an incoming connection for every downstream lane.
- TLS `96621100` retains controlled-link indices 0..9 but separates approaches:
  `GGGGGrrrrr -> yyyyyrrrrr -> rrrrrGGGGG -> rrrrryyyyy`.
  The DQN controller contract is `[green 0, green 2]`, with yellow mappings
  `0->1->2` and `2->3->0`. No protected and secondary merge movement is green
  simultaneously.

## Verification

The full route-aware contract validates known edges, vehicle-class permission,
ordered connectivity, every stop lane, end-of-edge outgoing reachability, and
a continuous lane chain using the disclosed 25 m lateral-movement rule. It
reports 2,106 vehicles and 242 required stops. `netconvert` accepted the network;
its remaining warnings concern pre-existing turnaround ambiguity and turning-
radius speed reductions outside the changed corridor. A bounded SUMO 1.26 load
check exited 0 without route, stop, signal, or network warnings. It advanced one
step (79200 to 79201), so it is not claimed as full-horizon validation.

Focused result: 32 tests passed. The full 7,200-second fixed-time probe remains
the runtime gate before exploratory DQN training.

## Artifact identities

- Input manifest: `43cea319def1c0e5dbcaf40f8e9fd04795740b4717ddd8c97a62dd0ea36c7bf2`
- Network: `5739ffd84764579afd1672af5d00681daaa427544b9beb0bcfb898b1b5eddd71`
- Explicit route: `692a6b302bc1aad6ac3debe9fa8337f8b9e9c19689a01e4ac51ccbf63081595f`
- Physical-demand fingerprint: `5448f00a62297592b83c44482ea4a80083b5a957fd77d0f8c7e81570d47ef671`
- Full-night SUMO config: `8b03204746a11196c184bdcbdf7528c80f26643d33088905ed41b44b1795d75c`
- 900-second SUMO config: `6e2a084423465e7cbde9be7a831594fe9d2b95dd467becdf38ee856b3d9be368`
- Expansion manifest: `122f95c8bc05042a7ef9a9d0f78d75014fdb71cbf9c5605b6d9340fb5475d54c`
- Three-episode exploratory YAML: `c5bf8c698b4eb61c286b7cb5a4df2a2e912409aaefd07828849b14658970bc7d`

All publication approval fields remain false. The full-night probe must report
complete demand reconciliation and no route, stop, physical-signal, or telemetry
integrity defect. Collisions, teleports, and emergency braking remain measured
traffic outcomes and must not be suppressed.
