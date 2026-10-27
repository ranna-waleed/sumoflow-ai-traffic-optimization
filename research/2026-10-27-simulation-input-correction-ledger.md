# SUMOFlowAI simulation-input correction evidence ledger

Date: 2026-10-27  
Branch: `research/ieee-reproducibility`  
Scope: bounded synthetic engineering inputs and diagnostics only  
Publication approval: **false**

## Corrections

- The historical `simulation/maps/routes_night.rou.xml` remains unchanged. Its 840
  deterministic `<flow number=...>` declarations were expanded through the existing
  demand-expansion contract into 1,920 explicit vehicles and combined with the 186
  existing explicit vehicles. The corrected candidate contains 2,106 unique,
  nondecreasing explicit departures and no flow elements.
- SUMO eager loading exposed invalid historical stop geometry/order. Candidate v4
  records 121 explicit corrections: position-invalid lane stops use SUMO
  `friendlyPos=true`; topologically impossible or out-of-order stop instructions are
  removed while retaining their vehicle and route. These are synthetic corrections,
  not field calibration. Earlier v1-v3 candidates are retained as rejected evidence.
- The historical network remains unchanged. A separate synthetic signal candidate
  makes the merge at TLS `96621068` permissive (`gggg`) and preserves the controller's
  phase indices at `96621100` while defining the native cycle red -> green -> yellow ->
  red. At `96621100`, trunk links 0-4 remain protected and merging primary links 5-9
  are permissive. This is an engineering hypothesis, not traffic-engineering approval.
- Observation contract v4 retains the 37-dimensional, full-lane layout. Finite,
  nonnegative lane occupancy above 1.0 is saturated at 1.0 and counted/reported;
  contract v3 remains fail-closed above 1.0. No lane is filtered or silently replaced.
- The diagnostic runner now aligns the effective SUMO begin/end with its configured
  horizon, captures SUMO messages, reports warning categories and per-agent plus
  aggregate optimizer diagnostics, preserves the full physical action trace, separates
  accepted keep/switch requests from forced max-green overrides, and publishes checkpoint
  hashes only after a completed behavior-valid run. Any SUMO warning/error, ignored
  departure, emergency braking, unsafe-signal warning, teleport, or collision invalidates
  the diagnostic.
- A separate fixed-time probe applies the same frozen candidate inputs and produces an
  isolated manifest plus output hashes. Existing output directories are never reused.

## Artifact identity

- Historical routes: `1e84789b2fac27aaff9d336a22cb0fc60b83ad08fc032046bf1671003223396a`
- Historical network: `a7f25235f208db932735983653e00b3af659d09d04169ef5807265433c2db708`
- Synthetic network: `6b6bd5390b7b4797db15e7f5c0d305516f3ed8e7a0b5f3cbba856f0fe3bbd9ce`
- Corrected explicit routes: `76f30d0812617c9916e72a69b543194de7fbe9b3d86983dc8df3b012eceb640c`
- Physical-demand fingerprint: `7e8d74a865d270c02339eca5ddfc0dad6d19bc01fbca62504eb76546158f0172`
- 900-second SUMO config: `0dc524d78ed6a850c2310786b45c3b0d0beaee44ae0952deb470cc1e03c3c99d`
- Full-night SUMO config: `b7ee5c399dcee8a62bcd4a6e52ef998359c59faec1bc73b2181b3d229d3ab174`

The authoritative local identity and the complete correction inventory are in
`DeepQN/diagnostics/inputs/night_synthetic_v4/input-manifest.json`. Paths in this local
candidate manifest are machine-specific and must be made portable before publication.

## Executed verification

1. A one-second SUMO 1.26 eager-load check of candidate v4, using begin 79200, end
   79201 and `--route-steps 0`, exited 0 with no emitted warnings. This establishes
   parse/startup compatibility for all routes, not dynamic 900-second safety.
2. Focused synthetic/input tests: 20 passed.
3. Broader focused regression set: 38 passed.
4. Final full discovery: 245 tests executed in 110.656 seconds; 244 passed, one
   intentionally quarantined legacy interactive diagnostic skipped, zero failures.
5. Source compilation and CRLF-aware diff checks are recorded in the handoff response.

No 900-second fixed-time or DQN probe, training, publication experiment, commit, or
push was performed in this correction.

## Remaining restrictions

- The synthetic priorities and phase sequence lack independent traffic-engineering
  approval and field calibration.
- Startup success does not establish absence of emergency braking, teleports,
  collisions, or ignored insertions over 900 seconds; both proposed probes must pass.
- Saturating occupancy preserves the full observation shape but changes the research
  observation contract. Historical v2/v3 checkpoints are not compatible with v4.
- The 121 stop corrections materially define a new synthetic demand artifact and must
  not be represented as the untouched historical demand.
- All generated candidates remain exploratory and `publication_approved=false`.
