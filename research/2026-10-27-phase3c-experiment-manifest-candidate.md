# Phase 3.0C frozen experiment-manifest candidate

The machine-readable candidate is
`research/2026-10-27-phase3c-experiment-manifest-candidate.json`. It is structurally
validated but explicitly unauthorized. Null artifact hashes are blockers, not wildcards.

The candidate binds the v3 37-value observation, forecast inference v2, right-closed dataset
v4, interval reward v1, metric definition v5, seven controlled TLS, frozen network, one-second
SUMO step, ten-second decisions and common 60-second `phase_controller_keep_v1` warm-up.
Warm-up demand remains inside the frozen denominator and outcomes. Learned arms differ only
by four explicit-zero forecast slots versus a strictly loaded verified forecast.

Five independent policy-training seeds per learned arm and ten paired held-out demand
realizations per profile are required. Demand-generation, SUMO execution and policy RNG seeds
are separate. Training budget remains unresolved until a training-only runtime/learning-curve
pilot; neither the historical 200 episodes nor YAML's three evaluation episodes is adopted.
Checkpoint selection is validation-only after a frozen budget. Optimizer settings must be
copied into the future effective configuration and hash-bound before execution.

The policy-seed pairing/crossing rule remains null. Publication inference is fail-closed until
that rule is frozen without consulting held-out outcomes. Reward calibration is bound to the
immutable 0.95 protocol artifact; all five pools retain verified demand identities, while only
training/calibration rows may reference calibration traces.

All policies use byte-identical explicit demand within a paired cell. Four profiles retain
their distinct windows. Each run must write tripinfo with unfinished/undeparted records,
summary, statistic and emissions outputs plus action/physical-trace diagnostics and hashes.
Sensor failure, missing detector/model/checkpoint, teleport crossing, contradictory physical
trace, premature termination or population inconsistency invalidates the run. Unfinished and
undeparted demand at a valid horizon is measured, not treated as failure. Native fixed-time is
a secondary full-operation comparator and is not claimed to share phase-synchronized warm-up.

The effective publication configuration SHA is currently unresolved. The checked-in YAML SHA
`eb60241f8433c2f72667c3f15c2f7df4c91d78b2ce6d1a64a989ca46efd3a8f4`
is source evidence only because it still contains provisional budget/evaluation defaults.
The candidate cannot pass the execution gate while approval/authorization are false or any
required artifact identity is null.
