# Phase 2C.3 external-review dossier

**Status:** candidate evidence package; no scientific approval is encoded.  
**Branch / audited base:** `research/ieee-reproducibility` / `75557e668fb9fb968810040e1e6b8882e59f9f2d`.  
**Gate:** Phase 3 **NO-GO**.

## Claim boundary

The defensible prospective claim is simulation-only: under frozen, synthetic and not
field-calibrated Tahrir-derived demand, test whether causal 30-second junction forecasts
change demand-normalized fixed-horizon waiting for an otherwise matched v3 Dueling
Double DQN. No result presently establishes improvement, deployment readiness, Tahrir
traffic calibration, signal safety, or transfer to Taksim.

## Evidence independently checked in the repository

- The research observation has 37 values and now declares
  `sumoflow-dqn-observation-v3-lane-occupancy-fraction` in
  `DeepQN/configs/dqn_config.yaml`. Strict research evaluation checkpoints carry that
  identifier; historical v2 weights are rejected even where tensor shapes match.
- Research-training snapshots already bind semantic contracts, dataset, demand split,
  environment, Git identity, RNG state, optimizer, replay, and ensemble membership.
  Phase 2C.3 adds a publication-only hash of the effective observation configuration;
  `train_research()` validates it before constructing agents. Generic snapshot tests
  remain engineering interruption-equivalence tests, not publication authorization.
- SUMO 1.26.0's lane API defines occupancy as vehicle length divided by lane length.
  Its `MSLane::getNettoOccupancy()` does not cap the quotient, unlike brutto occupancy.
  Therefore the research contract accepts only finite `[0,1]` observations and fails
  closed above one; it does not clip anomalous values or reinterpret E1 percentages.
- Publication training remains blocked by empty external trust roots. Passing tests do
  not approve the detector mapping, demand, signal safety, reward references, or models.

## Related-work comparison

Fawzy, Mohamed, Amer, and Ata, “Scenario based traffic optimization in Egypt
performance gains through simulation modeling,” *Scientific Reports* 16, 10812 (2026),
<https://doi.org/10.1038/s41598-026-41535-x>, studies Umm Kulthum Square in Mansoura.
It jointly changes geometry, lane connections, turning movements, signal locations,
phase timing, and states. Its reported effects therefore concern that combined scenario,
not forecast-augmented RL and not SUMOFlowAI's Tahrir-derived network.

Tables 6–7 are an important censoring lesson. At 3,599 seconds both scenarios report
6,252 loaded vehicles, more than 5,000 awaiting insertion (5,078 in Scenario 1), 500
running vehicles, and unequal arrivals/completions (672 versus 715). Conditional
trip-level waiting, duration, and emissions consequently describe different selected
populations. SUMOFlowAI must freeze identical demand bytes and horizon across arms and
report due, inserted, arrived, running, undeparted, removed, and teleported populations;
it must not infer superiority from completed-trip means alone.

The article describes satellite/OSM/Google Earth construction, not independent field
demand calibration. The reviewed text does not identify a SUMO version, random seeds,
independent replications, or uncertainty intervals. Those omissions are methodological
lessons, not criticisms converted into evidence for SUMOFlowAI. Its performance values
remain the article's results and are not reproduced here.

## Simulation-only blockers

1. No compatible, approved seven-input/four-output BiLSTM checkpoint exists; the
   historical artifact is incompatible with the strict contract.
2. The current count-prediction head is unconstrained linear; a nonnegative count-output
   contract must be frozen before training.
3. Historical `<flow>` demand has not been expanded into immutable, explicit vehicle
   identities required by the primary denominator.
4. Publication reward-calibration reference values do not exist; current YAML values
   remain engineering defaults.
5. The future protocol specifies five training seeds and ten paired evaluation demand
   seeds, while the current YAML still declares three evaluation episodes.
6. The detector mapping lacks independent traffic-engineering approval. Publication
   forecast training is prohibited until evidence is bound to its exact mapping hash.
7. Only two controlled TLS have a conventional competing-phase max-pressure topology;
   the seven-TLS system includes gate/red-hold behavior and must be labeled separately.
8. TLS `96621100` has a source-derived all-red → all-yellow → all-green native cycle;
   source inspection cannot establish that this is an acceptable safety sequence.

## Additional blockers to real-world claims

There are no independent field turning counts, queue/travel-time observations,
saturation-flow estimates, or calibration/validation targets for Tahrir. Detector
placement and signal sequencing lack external traffic-engineering approval; evaluated
control is not YOLO-driven; and the eight-agent Taksim topology/state/mapping/policy
compatibility remains unverified. Accordingly, “Tahrir-derived simulation” is permitted;
“calibrated Tahrir,” deployment, safety, or generalization claims are not.

## Required external decisions and missing artifacts

The machine-readable open decisions are in
`research/2026-10-26-phase2c3-external-decision-register.json`; reviewer prompts are in
`research/2026-10-26-phase2c3-reviewer-questions.md`. Missing artifacts include approved
detector/mapping evidence, explicit-ID frozen demand registries, leakage-safe forecast
splits, an approved model output contract, reward-calibration sources and constants, a
frozen publication protocol manifest, and independent traffic-engineering review.

## Authoritative references

- SUMO TraCI lane value retrieval:
  <https://sumo.dlr.de/docs/TraCI/Lane_Value_Retrieval.html>
- SUMO 1.26.0 `MSLane.cpp`, especially `getBruttoOccupancy()` and
  `getNettoOccupancy()`:
  <https://github.com/eclipse-sumo/sumo/blob/v1_26_0/src/microsim/MSLane.cpp#L3286-L3312>
- Fawzy et al. (2026): <https://doi.org/10.1038/s41598-026-41535-x>

