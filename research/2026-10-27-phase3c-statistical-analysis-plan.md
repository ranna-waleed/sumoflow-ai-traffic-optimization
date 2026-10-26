# Phase 3.0C statistical analysis plan candidate

## Primary estimand

For each profile, the primary estimand is the mean paired difference

`verified-forecast DQN minus forecast-free DQN`

in demand-normalized fixed-horizon waiting seconds per due vehicle. Negative values favor the
forecast condition. Both arms must use identical explicit physical demand, intended departures,
SUMO seed, horizon and metric contract. Absolute differences are primary; percentages are not
used when the baseline is zero.

The design has five independent policy-initialization seeds per learned arm and ten paired
held-out demand realizations per profile. The pairing schedule must be frozen before evaluation.
Where policies are not naturally paired across training seeds, a prespecified balanced crossing
or matched-seed rule must be fixed without inspecting evaluation outcomes. All eligible policies
are evaluated on the same held-out demands and SUMO seeds.

## Dependence and inference

Training-policy initialization and physical-demand realization are independent higher-level
units; repeated controller arms on the same demand/SUMO seed are paired. Profile is a fixed
stratum. TLS measurements, simulator steps, episodes and overlapping forecast windows are
correlated subsamples—not independent replicates.

Report profile-specific paired means and a preregistered equally weighted overall profile mean.
The descriptive implementation averages the four profile means, not all available cells, so
unequal cell counts cannot alter profile weights.
Primary uncertainty uses a hierarchical cluster bootstrap resampling training policies and then
demand realizations, retaining paired arms and profile strata. With only five training seeds,
also report all seed-level effects, median and range, and sensitivity intervals from a
small-sample hierarchical model or cluster-robust analysis. Do not use asymptotic normality alone
or count 50 demand-policy cells as 50 independent trained policies.

Secondary outcomes include insertion/completion rates, arrivals, backlog, completed-trip
conditional measures, CO₂, teleports, collisions, intervention rates and forecast scores.
Control the secondary family with Holm correction within each declared outcome family; report
unadjusted effect sizes and adjusted confidence decisions. Forecast accuracy is analyzed by
independent demand realization, not overlapping windows.

## Failures and missingness

Before execution, classify failures as infrastructure, contract-invalid, or valid adverse
traffic outcomes. Valid horizon runs with unfinished demand remain included. Contract-invalid
runs are never imputed as successful; reruns use the same frozen identity and are reported.
Primary analysis requires complete learned-arm pairs. Report missing-pair counts/reasons and a
worst-case sensitivity analysis bounded by observed profile outcomes. No post-hoc deletion,
seed substitution or selective rerun is allowed.

The implemented known-answer function verifies pair keys and absolute differences only. A
separate publication coverage validator requires four profiles, five checkpoint-bound policy
identities per arm, ten distinct physical demands per profile, matched SUMO seeds and physical
demand, and no missing, duplicate, failed or unbalanced cells. The candidate pairing rule is
still null; therefore publication inference remains unavailable rather than assuming matched
seed pairing. Neither descriptive function claims that a simple mean supplies publication
uncertainty; hierarchical inference awaits genuine runs and a frozen pairing schedule.
