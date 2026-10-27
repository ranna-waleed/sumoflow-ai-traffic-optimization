# IEEE Methods outline — verified contracts, results pending

## Study scope

Describe the work as an uncalibrated, Tahrir-derived synthetic SUMO study unless independent field
calibration is later supplied. Separate software-integrity evidence from controller-effectiveness
results. State that the present exploratory run did not complete and contributes no performance
result.

## Simulation and demand

Record SUMO version, frozen network/config/additional/route hashes, one-second timestep, horizon,
teleport/collision settings and every seed. Generate explicit, physically disjoint demand
realizations assigned before training to training, calibration, validation, forecast-test and
controller-evaluation pools. Report scheduled, inserted, arrived, running, undeparted and removed
populations.

## Controllers

Compare fixed-time, feasible native actuated control, topology-supported max pressure,
forecast-free Dueling Double DQN and strictly verified forecast-conditioned DQN. Use identical
physical demand and paired SUMO seeds. Restrict max pressure to validated service topologies. Keep
random policy as an engineering diagnostic. Report requested and physically observed TLS actions.

## Observation and forecast contracts

Use the 37-value v3 DQN observation: ten lanes × three features, two junction features and five
global features. Lane occupancy is the SUMO lane-domain fraction. Fail on unavailable/nonfinite
sensors. Both learned arms receive identical 60-second initial control when used comparatively.
Forecast inputs use the seven-feature causal, junction-specific contract and 30-second directional
count targets; training-only scaling and disjoint partitions are mandatory.

## Learning and reproducibility

Use five independently seeded policy replicates per learned arm after the seed rule, reward
references, training budget and checkpoint-selection rule are frozen. Record Python, NumPy, Torch,
CUDA and SUMO seeds. Preserve full resume state and publish only hash-verified whole-ensemble v2
research checkpoints compatible with observation v3. Exploratory legacy diagnostic checkpoints
are excluded.

## Outcomes and analysis

Primary outcome: demand-normalized horizon-censored waiting burden, including insertion backlog.
Complementary outcomes: insertion/completion rates, arrival throughput, running and undeparted
counts, departure delay, completed-trip waiting/duration/time loss, teleports, collisions and total
network CO2. Use paired evaluation demands, training seed as the learned-policy unit, hierarchical
paired uncertainty estimates and preregistered multiplicity handling.

## Results

**Pending.** No traffic improvement, convergence, baseline superiority, forecast benefit or
real-world validity is claimed. The first requested exploratory episode was stopped before its first
decision after SUMO exposed unresolved native signal-program safety warnings.
