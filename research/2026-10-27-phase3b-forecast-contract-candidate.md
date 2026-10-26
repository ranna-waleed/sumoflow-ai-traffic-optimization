# Phase 3.0B causal forecast and nonnegative-count contract candidate

**Status:** simulation-only software candidate; not publication-approved and not trained.

## 1. Causal time coordinate

Contract versions are `sumo-e1-post-step-right-closed-v1` and
`causal-junction-arrivals-60s-to-30s-v4-right-closed`. Calling
`traci.simulationStep()` advances from the prior boundary to the next boundary;
`simulation.getTime()` then returns that boundary. TraCI E1 vehicle data describes the
last completed simulator step. Consequently a feature row labelled `s` represents the
one-second reporting interval `(s-1,s]`. The first recorded interval alone includes its
left endpoint so an entry exactly at simulation begin is not discarded.

For a decision boundary `t`, the model input is the ordered matrix

\[
X_t=[x_{t-59},\ldots,x_t]\in\mathbb{R}_{\ge0}^{60\times7},
\]

representing 60 complete one-second bins through `t`. Its target is

\[
y_t^d=\#\{e:\operatorname{direction}(e)=d,\ t<e_{entry}\le t+30\},
\quad d\in\{N,S,E,W\}.
\]

The 30 target rows are labelled `t+1,...,t+30`; thus `[row t+1,row t+30]`
and the physical interval `(t,t+30]` are the same contract in two coordinates. The prior
v3 manifest text `[t,t+30)` was not correct at exact integer boundaries. The old
`floor(entry)+1` binning moved an event exactly at `t` into the following row; v4 uses the
post-step right-closed convention. Events just after `t` enter target row `t+1`; an event
exactly at `t` remains in history; an event exactly at `t+30` is in the target; one just
after is not. Missing/reordered/nonfinite samples, events outside recorded bins, incomplete
60/30 windows and inconsistent immutable event sources fail closed.

In the environment, each simulation step completes before the forecast adapter's `tick()`
records its E1/snapshot row (`DeepQN/env/sumo_env.py:543,568` and `:640,672`). State and
action selection occur only after the history has been collected (`:510-526`, `:690,765`).
No target event is available at decision time. This claim is a software timing contract,
not evidence that the unapproved Tahrir detector geometry is scientifically valid.

## 2. Seven input features

Each TLS row is ordered exactly as

`[a_N, a_S, a_E, a_W, a_total, mean_speed, mean_accumulated_waiting]`.

- `a_d` is the number of unique, validated E1 approach-plane entry events in direction `d`
  during the completed one-second bin; its unit is vehicles/one-second bin.
- `a_total` must equal the exact sum of the four directional values.
- `mean_speed` is the arithmetic mean m/s over vehicles presently on mapped approach edges
  at the observation boundary.
- `mean_accumulated_waiting` is the arithmetic mean seconds from TraCI's rolling
  `getAccumulatedWaitingTime`, with explicitly verified 100-second memory.

Every value uses information available by its row boundary. Duplicate/ambiguous same-TLS
events, conflicting ownership, invalid detectors, nonfinite values, altered snapshot/event
files and inconsistent mapping/network hashes fail closed. An empty direction produces a
structural zero in software but is not evidence that real traffic is absent or that the
mapping is approved. Snapshot inference remains an engineering mode, never a substitute
for missing publication E1 evidence.

## 3. Target, architecture and likelihood

The target shape is `[4]`, with integer observations measured in vehicles per following
30-second interval. Model outputs are continuous nonnegative expected counts, not rounded
observations, occupancy, queue length, global demand, or a 30-by-4 future sequence.

The strict research architecture is frozen as: seven inputs; 60 history steps; two-layer
bidirectional LSTM; hidden size 128 per direction; dropout 0.2; final-state linear four-unit
head; softplus activation. Bidirectionality sees only the already-observed history and is
therefore causal with respect to the target. The training objective candidate is Poisson
negative log likelihood parameterized by expected counts. Tests establish finite loss and
gradients for zero and large synthetic counts. A negative-binomial likelihood is a reasonable
alternative only if training-pool dispersion diagnostics support it; adopting it would
require a new frozen contract and an additional dispersion output/parameterization. No such
empirical selection was made here.

Strict inference requires the v2 inference manifest, v4 dataset contract, exact feature/time
schemas, mapping/detector/network hashes, architecture, softplus activation, Poisson-loss
declaration, scaler and checkpoint hashes. Loading uses restricted `weights_only=True` and
strict tensor matching. The legacy trainer instead emits a `[batch,30,4]` sequence through
one linear layer (`lstm/train_lstm.py:52-58`) and uses MSE (`:153`); it is not this model.
Historical files remain unchanged and cannot silently load as this research contract.

## 4. Leakage, baselines and scoring

The registry contract requires physical disjointness across `training`, `calibration`,
`validation`, `forecast_test` and `controller_evaluation`, checking realization identity,
source bytes and canonical physical-demand fingerprint. Membership is fixed before window
creation; windows never cross source boundaries. Timestamp-shifting or ID renaming cannot
legitimately create independence because the frozen physical fingerprint/realization
registry remains authoritative. The existing dataset scaler is fit only from training
windows (`lstm/research_data.py:301,473`). Validation may select a model; forecast-test and
controller-evaluation data remain unavailable to fitting, scaling and selection.

Leakage-safe baselines use only the 60-row history and frozen training quantities:

- zero: four zeros;
- persistence: direction counts in the preceding 30 seconds;
- moving average: mean per-second direction count over a declared history lookback times 30;
- historical mean: four target means fit from the training pool only.

Scoring provides window-level MAE, RMSE, directional bias and Poisson deviance, then groups
results by immutable independent demand realization. Percentage errors are explicitly
unavailable because zero count denominators are valid. Statistical inference must use the
independent realization, not overlapping windows, as its unit.

## 5. Evidence boundary and blockers

Synthetic CPU tests establish shapes, nonnegativity, finite gradients, strict serialization,
boundary binning, baseline causality and leakage rejection. They provide no Tahrir forecast
accuracy, calibration, overdispersion estimate or model-performance result. No model was
fit. Publication training remains blocked by the empty approval trust root, unapproved
Tahrir detector mapping (including the 0.20 m edge decision), absent publication demand
realizations/split registry, unresolved relocation/teleport audit policy, and lack of an
approved dataset/model manifest. Phase 3 performance experiments remain NO-GO.

## Authoritative SUMO references

- <https://sumo.dlr.de/docs/TraCI/Control-related_commands.html>
- <https://sumo.dlr.de/docs/TraCI/Induction_Loop_Value_Retrieval.html>
- <https://sumo.dlr.de/daily/pydoc/traci/_simulation.html>
