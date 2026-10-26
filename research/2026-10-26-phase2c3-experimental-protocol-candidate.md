# Phase 2C.3 experimental protocol candidate

**Candidate only — not approved and not execution authorization.**

## Question and estimand

Primary question: under frozen Tahrir-derived simulation demand, does a strictly causal
30-second local forecast alter demand-normalized horizon waiting relative to the same
v3 Dueling Double DQN with four forecast slots fixed to zero?

For due vehicles with intended departure in the half-open evaluation interval `[B,H)`,
the primary outcome is the Phase 2A/2B.1 demand-normalized horizon waiting burden:

`W_H = sum_i(departure_delay_i,censored + on_network_wait_i(H)) / N_due`.

It includes censored pre-insertion delay for due vehicles not inserted by `H`. Required
complementary outcomes are inserted/due, arrived/due, arrived/inserted, arrivals per
simulated hour, running and undeparted at `H`, conditional completed-trip waiting,
duration and time loss, removals/vaporization, teleports, collisions, total network CO2,
and network CO2 per reported arrival. Undefined denominators remain unavailable.

## Frozen arms and common treatment

1. Native fixed-time (secondary full-operation reference).
2. Native actuated only if an actual loaded and hash-bound program exists.
3. Topology-qualified max-pressure: conventional claims only for the verified two-TLS
   competing-phase subset; any seven-TLS gate policy is separately named.
4. Forecast-free v3 Dueling Double DQN (four forecast slots exactly zero).
5. Forecast-conditioned v3 Dueling Double DQN with an approved seven-feature model.

Random control is diagnostic, not a scientific baseline. Learned arms share the exact
37-dimensional v3 observation, reward, architecture, optimizer budget, controller safety
logic, 60-second `phase_controller_keep_v1` observed warm-up, first-decision timestamp,
demand, and evaluation horizon. Fixed-time/native comparisons are full-operation
secondary comparisons and must not be described as phase-synchronized warm-up matches.

## Forecast and observation contracts

Forecast history is 60 consecutive one-second samples ending at decision time. Features
are `[a_N,a_S,a_E,a_W,a_total,mean_speed,mean_accumulated_waiting]`; targets are four
junction-specific approach-entry counts over `(t,t+30]`. Training-only scalers and
disjoint demand realizations are mandatory. Mappings, E1 source files, network,
additional files, demand, schema, and configuration are hash-bound. No proxy, zero-fill,
snapshot inference, or legacy adapter is allowed in strict model mode.

The DQN lane feature is the finite raw SUMO lane occupancy fraction in `[0,1]`, not E1
occupancy percent. The effective v3 observation configuration hash is part of every
publication resume contract. Any modification invalidates resume compatibility.

## Partitions, seeds, selection, and analysis

- Freeze independent training, validation/calibration, and held-out evaluation demand
  registries before training; explicit vehicle IDs and intended departures are required.
- Train five independent seeds for each learned condition. Select checkpoints only by a
  preregistered validation rule after a fixed budget; evaluation demand cannot affect
  model, hyperparameter, reference-value, or checkpoint selection.
- Evaluate ten paired held-out demand seeds for every eligible controller and profile.
  A demand realization is byte-identical across paired arms. Record Python, NumPy,
  Torch/CUDA, SUMO, controller, and demand seeds separately.
- Learned-controller inference treats training seed as the independent unit. Use a
  hierarchical paired analysis retaining training-seed and demand-seed levels. Report
  effect sizes and confidence intervals; apply a preregistered multiplicity correction
  to secondary outcomes. Report all failed runs and their predefined handling.
- Forecasting is evaluated separately against zero, persistence, historical mean, and
  moving-average baselines on held-out demand.

## Run validity and provenance

Every run records Git SHA/dirty state; SUMO/Python/Torch versions; OS/hardware; frozen
net, route, additional-file, program, mapping, dataset, model/checkpoint, and effective
configuration hashes; simulated `[B,H)`; timestep; wall-clock limit; all seeds; exact
command; raw output paths/hashes; completion/failure state; sensor and physical-trace
validity. Controller-specific output destinations are permitted; behavior-affecting
inputs are not. Historical SUMO 1.24 and new SUMO 1.26 outputs are not equivalent arms.

## Known mismatch with current repository defaults

The current YAML's three evaluation episodes conflict with the proposed ten paired
demand seeds. Its 200-episode training budget is historical/provisional, its reward
references are engineering values, its forecast mode is `no_forecast`, and its legacy
flow-based profiles do not provide the explicit-ID primary denominator. A future,
separate immutable publication manifest must resolve these discrepancies; this document
does not mutate defaults or authorize an override.

## Stop conditions

Abort rather than substitute when any mapping, detector, lane sensor, physical TLS
trace, demand identity, output, checkpoint, forecast, observation hash, or population
reconciliation is missing or contradictory. The `96621100` sequence and detector
placement require external review. Taksim is excluded until its eight-agent compatibility
is independently established. Phase 3 remains **NO-GO**.

