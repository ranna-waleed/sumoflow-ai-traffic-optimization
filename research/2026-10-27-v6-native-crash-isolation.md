# V6 native SUMO crash isolation — 2026-10-27

## Scope and reproduced failure

This correction used Eclipse SUMO 1.26.0 and only the bounded interval
`[79200,79300)`, one-second steps, seed 42001, and the unchanged v5 explicit
demand. No DQN training or full-night simulation was run.

The machine-readable ablation is
`DeepQN/diagnostics/ablations/v6-native-crash-20261027-r2/ablation-report.json`.
It preserves the original crashing v6 network identity
`5739ffd84764579afd1672af5d00681daaa427544b9beb0bcfb898b1b5eddd71`
and a detailed v5/v6 lane, connection, and TLS change inventory.

| Variant | SHA-256 | Exit | Complete steps / last time |
|---|---|---:|---|
| unchanged v5 | `6b6bd5390b7b4797db15e7f5c0d305516f3ed8e7a0b5f3cbba856f0fe3bbd9ce` | 0 | 100 / 79299 |
| TLS edits only | `dc4187415614022b3c359c926a2dacaf3e6b57275e71633888a37471005574d9` | 0 | 100 / 79299 |
| connection edits only | `09983c54ffde7f196a724c6e2822aee8dcae33875f50e69071206e24ca7b5e09` | timed out/native fault | 59 / 79258 |
| length edits only | `ee0df38cd3b41817c5e00d1029906f2e01a6927d0f5a9da9d505f27f4bd6fddc` | 0 | 100 / 79299 |
| TLS + connections | `4c43e3cded30eef1da34e7f18da0408389f27055a0826270b187b0aa946fbc28` | timed out/native fault | 44 / 79243 |
| TLS + lengths | `d222090c3ef7200d5841bd487f627cccccd0cff6c56882a33c8ba3b8ba07e0fe` | 0 | 100 / 79299 |
| connections + lengths | `9d71eff410cca6da57e1a9d86f1244fc32a010aa9b96e127b98b0140ba37c8ab` | timed out/native fault | 59 / 79258 |
| all compiled edits | `f40b6d68ad1bfd615c1322b3c8584545e32cf23cd5b0fadc0caed492129e810b` | 3221225477 (`0xC0000005`) | 59 / 79258 |

The tests prove that the directly edited compiled connection is necessary for
the reproduced failure. TLS and length changes alone do not reproduce it.
The precise invalid pointer inside the closed Windows binary is unknown, but
the malformed-network mechanism is supported: changing compiled external
connection endpoints left precompiled `via` internal lanes, junction requests,
foes, and right-of-way data unchanged.

## Supported repair

The v6 generator now exports v5 through `netconvert --plain-output-prefix`,
edits the plain connection and TLS files, and rebuilds the compiled network
with `netconvert`. This recomputes internal lanes, `via` mappings, junction
logic, and right-of-way data. It no longer changes compiled lane lengths.

The direct corridor mapping provides a lane-0 chain without invented length:
`10873186#8_0 -> 10873186#9_0 -> 10873186#0_0`. Stops on
`28718874#0` are corrected route-wide: services followed by the left turn use
lane 2; over-bound 10 m stop extents are shifted within the unchanged 20.69 m
edge. All 2,106 departures and 242 obligations remain.

The repaired network completed the same bounded interval with exit code 0 and
100 summary steps. This is crash isolation, not full-night behavioral proof.

## Repaired artifact identities

- v6 manifest: `10570192e72ea90be23643b9b6c5803816e75ca1484dbcbb49621d71a3bb8479`
- rebuilt network: `9e9d1c19c2d9e7b2ab1093722ce10457633e1dfc8574e81f74e5dd8244f4fe45`
- corrected demand: `8ac9a27f77e1c45b8ff03115071975c14ae92d6b05b215e37cd69e0460bbb5ed`
- physical-demand fingerprint: `1109c838553c6e2b64e8212952f90b6da195d512c98e32a4676747f73cff3ff7`

Publication approval remains false. Full-night stop service, collisions,
teleports, and emergency braking remain unresolved runtime outcomes.

## Verification

- Focused crash-reporting regressions: 2 passed.
- Combined synthetic-input and persistent-trainer regressions: 34 passed in
  6.788 seconds.
- Changed-source compilation: passed.
- Targeted CRLF-aware `git diff --check`: passed.
- Repaired 100-second SUMO run: exit 0, 100 complete summary steps.
