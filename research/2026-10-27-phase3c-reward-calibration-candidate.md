# Phase 3.0C reward-calibration candidate

**Status:** versioned software contract; no publication constants were calculated.

The active research reward is `interval-global-reward-v1`:

\[
r_k=\operatorname{clip}_{[c_{min},c_{max}]}\left(
-w_D D_k/D_{ref}-w_E E_k/E_{ref}+w_A A_k/A_{ref}-w_P P_k/P_{ref}
\right).
\]

`D_k` integrates incoming-lane halting populations over actual step duration in
vehicle-seconds. `E_k` integrates unique controlled-edge TraCI CO₂ rates in mg/s into mg.
`A_k` sums nonnegative reported arrivals at every simulator step. `P_k` integrates the mean
within-TLS population standard deviation of incoming-lane queues in vehicle-seconds. One
global scalar is copied to every agent. Requested or applied actions do not enter the reward
directly; they affect subsequent physical observations. Clipping occurs after normalization,
weighting and summation, so it can collapse large outcome differences and its saturation rate
must be reported during training diagnostics.

Warm-up traffic is included in evaluation outcomes but reward collection begins with the
first policy decision. Each decision calls `begin_interval`, including a partial final
interval, and every completed simulator step contributes its actual duration. The current
YAML values (`100 vehicle-s`, `10000 mg`, `1 vehicle`, `10 vehicle-s`) and weights are
engineering defaults, not calibrated publication values.

## Candidate calibration v3

`calibrate_publication_reward_references` requires a separately hash-bound five-pool split
registry and immutable per-interval traces. Every row binds the original template hash,
explicit-demand bytes, canonical physical fingerprint, demand-generation seed, SUMO seed,
profile and partition. Common historical templates are permitted; repeated
explicit bytes or physical fingerprints under renamed IDs are rejected. Only `training` and
`calibration` rows may contain trace paths/hashes. Held-out rows must use null trace identities,
so calibration can finish before evaluation and cannot dereference held-out outcomes. The
fingerprint is independently recomputed from explicit XML using Phase 3.0A's ID-insensitive
canonical expansion algorithm rather than trusting the registry declaration.

The prespecified estimator is the per-profile 0.95 positive quantile using NumPy's linear
method. Zeros remain in source traces but cannot define a positive normalization scale.
Empty traces, absent positive values, nonfinite/negative quantities, invalid durations,
altered sources or held-out-only registries fail. Extreme positive observations remain in
the quantile calculation; they are not silently trimmed. Changing estimator, pooling or
outlier treatment requires a new contract version before viewing held-out outcomes.

The immutable estimator specification is
`research/2026-10-27-phase3c-reward-calibration-protocol.json`, SHA-256
`0bf45ff5bcb2f265227178b9f3aa425a7948fc1366173079c4f9e3ae1ae51740`. It binds the
estimator, quantile, linear method, separate-profile pooling, zero policy, four-profile
coverage, eligible partitions and non-evaluation diagnostic trace-generation protocol.

Output publication is atomic and non-overwriting, contains source paths/hashes and always
sets `publication_approved=false`. No actual Tahrir calibration traces or approved split
registry exist, so no reference values are presented here. Registry schema v2 verifies demand
identities across all five pools while separating the eligible trace inventory.
