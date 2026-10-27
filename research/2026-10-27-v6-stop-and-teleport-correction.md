# V6 stopping-place and teleport correction — 2026-10-27

The immutable source probe is
`DeepQN/diagnostics/probes/fixed-night-v6-rebuilt-seed42001`.
It completed 7,200 steps with 2,106 inserted, 2,105 arrived, one running,
zero collisions and 13 teleports. Its only non-traffic integrity warning was
that `bs_5` was shorter than `bus_160`.

## Stopping places

The source `bs_5` occupied 0.2–2.5 m on a 3.34 m lane and cannot contain a
12 m bus. All ten `bs_5` users share the next edge `28718647#3`; v6 therefore
places `bs_5` at 0.1–12.1 m there and places `bs_3` at 13.1–25.1 m. The service
order is position-aware. The inventory also found used stops `bs_2`, `bs_3`,
and `bs_9` shorter than their longest scheduled 12 m vehicle, so their legal
extents are 1–13 m, 13.1–25.1 m, and 1–13 m respectively. No vehicle, scheduled
stop, vehicle dimension, or network geometry was removed or changed.

## Teleports

`teleport-analysis.json` records 13 unique starts: six wrong-lane, six jam and
one yield. All six wrong-lane cases were on `10873186#0` lanes 2/3 while their
routes required `28718874#0`; the source offered that turn only from lane 0.
V6 plain connection XML now exposes the turn from all six approach lanes and
is rebuilt with `netconvert`. The six jam cases cluster on internal lanes at
`3105658889` or `10873186#8`; the yield case is on `10873186#8_0`. Those seven
are congestion/right-of-way outcomes, not proven topology defects, and no
timeout or signal behavior was changed.

A bounded run through time 80500 exited 0 without stop warnings and without
the former first wrong-lane teleport at 80434. Full-night effects remain to be
verified.

## Current hashes

- Manifest: `7a6c9cefb9e258404f0e08d60f0401c30368be44976fc047957ce05393dccfc7`
- Network: `4d65197847de59ee645047dd6ccc7fd2ea0d7c49818c9f00a6788ceb1e704e53`
- Explicit demand: `8ac9a27f77e1c45b8ff03115071975c14ae92d6b05b215e37cd69e0460bbb5ed`
- Local stopping places: `e6b3703b82d8c4b8ea797540029c945cd980fcd98f724268858abaaab5633c41`
- Full SUMO configuration: `d52401d3263b37ddaa18ccfef655e23c67f2d2a451fc0232d68a3ae6d37e1ec9`
- Physical-demand fingerprint: `1109c838553c6e2b64e8212952f90b6da195d512c98e32a4676747f73cff3ff7`

Focused regression result: 36 tests passed in 9.181 seconds. Compilation and
targeted diff checks passed. Publication approval remains false.
