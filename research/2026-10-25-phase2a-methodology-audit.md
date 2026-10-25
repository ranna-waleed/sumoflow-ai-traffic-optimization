# SUMOFlowAI Phase 2A methodology audit

**Date:** 2026-10-25 (Africa/Cairo)  
**Branch:** `research/ieee-reproducibility`  
**Starting commit:** `8f84635a374c9e94e66e5d3cc2321be0564beb89`  
**Scope:** evaluation preflight, scientific-method audit, and experiment-contract design. No retraining or publication-scale simulation was performed.

## Evidence labels

- **Source-verified:** established directly from the checked-out source at the starting commit plus the Phase 2A diff.
- **Artifact-verified:** established by inspecting a local artifact and recording its hash or structure.
- **Test-reproduced:** established by an executed deterministic test.
- **SUMO-documented:** established by SUMO 1.26.0 help or official SUMO documentation.
- **Proposed contract:** a Phase 2B design; it does not describe current behavior.
- **Unresolved:** the repository does not contain sufficient evidence.

## 1. Evaluation preflight

### Fixed-horizon semantics

- **Source-verified:** `SumoEnv` retains its historical default `terminate_on_empty=True` at `DeepQN/env/sumo_env.py:271`.
- **Source-verified:** controlled evaluation now creates an isolated deep-copied configuration and sets `terminate_on_empty=False` at `DeepQN/evaluation/evaluate.py:106-111`; the source configuration is not mutated.
- **Source-verified:** `SumoEnv.step()` records interval start time and executed steps at `DeepQN/env/sumo_env.py:371-376`, checks the horizon inside the step loop at `DeepQN/env/sumo_env.py:385-394`, and reports `interval_steps` and `interval_duration_s` at `DeepQN/env/sumo_env.py:414-415`.
- **Test-reproduced:** a decision beginning at 20 s with a 23 s horizon and ten-step decision interval executes exactly three steps and stops at 23 s (`DeepQN/tests/test_phase2a_methodology.py:38-71`).
- **Test-reproduced:** fixed-horizon evaluation continues after demand becomes empty, whereas a separately constructed environment retains the training default (`DeepQN/tests/test_phase2a_methodology.py:17-36,73-103`).

The environment still returns one replay transition for a shortened final interval. It does not yet distinguish `terminated` from `truncated`; that is a Phase 2B change requiring replay/checkpoint compatibility decisions.

### Baseline wall-clock timeout

- **Source-verified:** the fixed-time runner accepts `wall_clock_timeout_s: float | None`, validates finite positive values, and forwards it to `subprocess.run` at `DeepQN/evaluation/baseline.py:15-30,42-47,74`.
- **Source-verified:** timeout failures preserve the command and captured diagnostic output at `DeepQN/evaluation/baseline.py:31-39`.
- **Source-verified:** the tiny validation explicitly uses a 30 s wall limit at `DeepQN/evaluation/tiny_sumo_validation.py:40-44`; simulated duration remains defined by the SUMO configuration.
- **Test-reproduced:** forwarding, invalid values, and diagnostic timeout behavior are covered at `DeepQN/tests/test_phase2a_methodology.py:105-126`.

Publication runners must choose and record a wall-clock timeout. It must never be derived silently from simulated seconds.

## 2. Emissions semantics and units

- **SUMO-documented:** installed `sumo --version` reports Eclipse SUMO 1.26.0. Its `--help` says `--emission-output.step-scaled` writes values scaled to the step length rather than per-second values.
- **SUMO-documented:** TraCI edge CO2 is the sum on an edge in mg/s during the last timestep: <https://sumo.dlr.de/docs/TraCI/Edge_Value_Retrieval.html>.
- **SUMO-documented:** vehicle TraCI CO2 is mg/s and must be multiplied by step length for a per-step amount: <https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html>.
- **SUMO-documented:** tripinfo nested `emissions@CO2_abs` is cumulative mg over the vehicle journey: <https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html>.
- **Source-verified:** the Phase 1 XML parser multiplies unscaled emission-output values by configured step length and does not multiply step-scaled values at `DeepQN/evaluation/metrics.py:142-159`.
- **Source-verified:** the current reward reads endpoint TraCI edge rates at `DeepQN/env/reward.py:161-169` but subtracts consecutive endpoint rates at `DeepQN/env/reward.py:113-116`; this is a change in mg/s, not interval emissions in mg.
- **Test-reproduced:** `DeepQN/tests/test_phase2a_methodology.py:157-161` verifies rates 100 and 200 mg/s integrate to 150 mg for two 0.5 s steps, 300 mg for two 1 s steps, and 600 mg for two 2 s steps.

**Proposed contract:** for an interval containing steps (I_k),

\[
E_k=\sum_{j\in I_k}\Delta t_j\sum_e c_e(j),
\]

where (c_e(j)) is mg/s and (E_k) is mg. XML `CO2_abs` is already an integrated mass and must not be multiplied again.

## 3. Horizon-censored traffic outcome

### Population definitions

For a frozen realized demand and horizon (H):

- (N_{due}): vehicles with intended departure no later than (H).
- (N_{inserted}): due vehicles admitted to the network by (H).
- (N_{arrived}): inserted vehicles reported arrived by (H).
- (N_{running}): inserted vehicles still on-network at (H).
- (N_{undeparted}): due vehicles not inserted by (H).
- Vehicles scheduled after (H) are excluded from all denominators.

For each due vehicle,

\[
d_i^{pre}=\begin{cases}
t_i^{depart}-t_i^{intended},&\text{inserted},\\
H-t_i^{intended},&\text{undeparted at }H,
\end{cases}
\]

and (w_i(H)) is accumulated involuntary on-network waiting through (H), zero for an undeparted vehicle. The proposed primary measure is

\[
W_H=\frac{\sum_{i=1}^{N_{due}}(d_i^{pre}+w_i(H))}{N_{due}}
\quad\text{seconds per due vehicle}.
\]

This denominator prevents a controller from looking better merely by suppressing insertion.

- **SUMO-documented:** `tripinfo-output.write-unfinished` writes records for vehicles that have not arrived; `tripinfo-output.write-undeparted` writes vehicles unable to depart because of departure delay. Both switches are present in installed SUMO 1.26.0 help and documented at <https://sumo.dlr.de/docs/sumo.html>.
- **Source-verified limitation:** `_patch_sumocfg` currently enables only `write-unfinished` at `DeepQN/evaluation/evaluate.py:55`; it does not enable `write-undeparted`, enumerate scheduled demand, or calculate (W_H).
- **Unresolved:** two bounded attempts to inspect a deliberately backlogged tiny SUMO output did not complete within their command sessions and were manually terminated. No repository or historical artifact was modified. Therefore, the exact per-record fields emitted by SUMO 1.26 for this case were not claimed as empirically verified.
- **Decision:** retain (W_H) as the intended metric, but mark it **not implemented**. Phase 2B must validate actual unfinished/undeparted XML against frozen demand IDs or use a stepwise accumulator before making it the primary outcome.

Required complementary reporting: insertion rate (N_{inserted}/N_{due}), completion rate (N_{arrived}/N_{due}), conditional completion (N_{arrived}/N_{inserted}), arrivals per simulated hour, running and undeparted counts, observed departure delay, censored pre-insertion backlog, completed-trip waiting/duration/time loss, teleports, non-arrival removals, collisions, total network CO2, and network CO2 per reported arrival.

## 4. BiLSTM audit

### Current training data contract

- **Source-verified:** feature order is `north, south, east, west, total, avg_speed, avg_waiting`; targets are the first four directions (`lstm/train_lstm.py:13-16`).
- **Source-verified:** angles map north at [315,360) or [0,45), east [45,135), south [135,225), and west [225,315) (`lstm/prepare_data.py:17-25`).
- **Source-verified:** each FCD timestep counts every network vehicle by heading; `total` is their sum, `avg_speed` is their arithmetic mean in m/s, and `avg_waiting` is summary `meanWaitingTime` at the same timestamp (`lstm/prepare_data.py:47-81`). These are network-wide occupancy snapshots, not approach arrivals.
- **Source-verified:** sliding samples use 60 history rows and the following 30 rows of the four direction columns (`lstm/train_lstm.py:13-16,88-93`). Output units are therefore vehicle snapshot counts for each predicted second.
- **Source-verified:** sequences are constructed before an 80/20 split (`lstm/train_lstm.py:91-98`). Adjacent train and validation sequences can share source timesteps, so temporal leakage is possible.
- **Source-verified:** min/max values are derived from flattened training windows and applied to train and validation data (`lstm/train_lstm.py:112-125`). Repeated overlapping timesteps receive repeated weight. There is no independent test split.
- **Source-verified:** the training loader shuffles and drops the last batch (`lstm/train_lstm.py:133`), but the training script sets no Python, NumPy, Torch, or CUDA seed.
- **Unresolved:** `lstm/data/sequences.csv`, the exact source dataset split, run command, logs, and checkpoint-generation manifest are absent.

### Current model and inference path

- **Artifact-verified:** hashes are:
  - `lstm/models/config.json`: `cb074e2355b1c721ea786b46b804e640b027a0fab6092d65ac5f5ae20b230b52`
  - `lstm/models/scaler.pkl`: `7f8375c95fa9be7e117957b3440bc6086015a75b36a078eebbb8127c8b16af8e`
  - `lstm/models/lstm_best.pth`: `44d08cd7b1bef9bd4c5fabb317dbce95b12db138a08c4ae71f20bbceada11274`
- **Artifact-verified and test-reproduced:** restricted `torch.load(..., weights_only=True)` shows `lstm.weight_ih_l0` and its reverse have shape `(512, 6)`, while `config.json` declares input size 7. The output layer has shape `(120, 256)`, consistent with 30 x 4 outputs. Regression: `DeepQN/tests/test_phase2a_methodology.py:163-178`.
- **Source-verified:** `lstm/predict.py:82` currently calls unrestricted `torch.load`; the configured seven-input model cannot strictly load the six-input checkpoint.
- **Source-verified:** histories shorter than 60 are left-padded with zeros, missing fields default to zero, predictions are inverse-scaled, clamped, rounded, and reduced to integer means over 30 predicted snapshots (`lstm/predict.py:101-129`).
- **Source-verified:** the adapter hardcodes `lstm/models/{config.json,scaler.pkl,lstm_best.pth}` (`DeepQN/integration/bilstm_adapter.py:108-110`) and hardcodes Tahrir direction edges (`DeepQN/integration/bilstm_adapter.py:57-62`).
- **Source-verified:** the adapter marks loading successful after reading configuration/scaler without loading the weight tensors; inference exceptions switch silently to proxy values (`DeepQN/integration/bilstm_adapter.py:131-155,195-207`).
- **Source-verified:** directionless `total`, `avg_speed`, and `avg_waiting` receive zero because unmatched features are assigned zero (`DeepQN/integration/bilstm_adapter.py:232-245`). Direction names fall through to vehicle counts on hardcoded edges.
- **Source-verified:** proxy predictions are used when the model is unavailable and while history length is below five (`DeepQN/integration/bilstm_adapter.py:183-192`), although the predictor itself expects 60 samples.
- **Source-verified:** `lstm/evaluate.py:50-67` evaluates rolling windows from the available rows and does not enforce a held-out split or compare against a naive baseline.

**Conclusion:** the committed artifacts cannot establish a valid seven-feature real-BiLSTM inference path. This is confirmed by source, tensor shapes, and a reproducible restricted-load test—not inferred from historical performance.

### Intended Phase 2B forecast contract

At decision time (t):

\[
x_t=[a_N(t),a_S(t),a_E(t),a_W(t),a_\Sigma(t),\bar v(t),\bar w(t)]
\]

- (a_d(t)): unique vehicles crossing the declared approach-entry boundary during ((t-1,t]), vehicles/s.
- (a_\Sigma(t)=\sum_d a_d(t)).
- (\bar v(t)): vehicle-weighted mean speed on the monitored approaches, m/s; zero only if none are present.
- (\bar w(t)): mean accumulated waiting time on those approaches, s; zero only if none are present.
- Input: 60 samples ending at (t).
- Target: (y_d(t)=\sum_{\tau=t+1}^{t+30}a_d(\tau)), vehicles per next 30 s, for four directions.
- Every direction-to-edge/detector mapping, dataset, split, scaler, model, and configuration is versioned and hashed.

This new feature/target definition is incompatible with the historical forecast checkpoint and with forecast-conditioned historical DQN weights.

## 5. Reinforcement-learning semantics

### State and action

- **Source-verified:** each per-agent state has up to ten lanes with normalized halting count, lane waiting time, and occupancy; normalized phase and elapsed time; four forecast/proxy values; and time of day (`DeepQN/env/sumo_env.py:449-482`). Total dimension is 37.
- **Source-verified:** action 0 requests keeping the current phase and action 1 requests switching (`DeepQN/env/sumo_env.py:99-114`). Yellow state and minimum green can ignore a request; maximum green forces a switch. Yellow transitions are ticked every SUMO step (`DeepQN/env/sumo_env.py:119-170`).
- **Source-verified:** training stores the requested action returned by the policy, not a separately logged physical execution (`DeepQN/training/train.py:128-133`, `DeepQN/agent/dqn_agent.py:313-325`). This is coherent only if the action is formally defined as a request and safety overrides are environment dynamics.

### Current reward

The implemented endpoint-difference equation is:

\[
r_k=\operatorname{clip}(-\alpha[W_k-W_{k-1}]-\beta[\dot C_k-\dot C_{k-1}]
+\gamma[a_k-a_{k-1}]-\delta\,\operatorname{std}_l(q_l(t_k))).
\]

- (W_k) is an endpoint sum of lane waiting-time values.
- (\dot C_k) is an endpoint sum of last-step edge CO2 rates in mg/s.
- (a_k) is the arrival count from the last SUMO step, not the interval.
- The optional pressure is the population standard deviation of controlled-lane halting counts, not upstream-minus-downstream pressure.

Evidence: `DeepQN/env/reward.py:94-140,151-188`. The comments at `DeepQN/env/reward.py:5-20` instead describe interval quantities and upstream/downstream pressure.

- **Source-verified:** the environment already accumulates arrivals at every internal SUMO step (`DeepQN/env/sumo_env.py:370-378`) but the reward does not consume `step_arrived`.
- **Source-verified:** one scalar reward is copied to every agent (`DeepQN/env/sumo_env.py:397-398`): decentralized observations/actions with a shared global reward.
- **Test-reproduced:** with endpoint wait 20 to 5, rate 100 to 50 mg/s, and last-step arrivals 2 to 0, the current reward is +14.05 before clipping even though the interval may contain positive waiting, emissions, and arrivals (`DeepQN/tests/test_phase2a_methodology.py:129-155`).

### Proposed Phase 2B reward

\[
D_k=\sum_{j\in I_k}\Delta t_j\sum_l h_l(j),\quad
E_k=\sum_{j\in I_k}\Delta t_j\sum_e c_e(j),\quad
A_k=\sum_{j\in I_k}a(j)
\]

\[
r_k=\operatorname{clip}(-\alpha D_k/D_{ref}-\beta E_k/E_{ref}
+\gamma A_k/A_{ref}-\delta P_k/P_{ref}).
\]

Here (D_k) is vehicle-seconds, (E_k) is mg, (A_k) is vehicles, and (P_k) must be a declared per-junction pressure/imbalance definition. References are fitted only on training/calibration demand and frozen before test evaluation. No part of this equation was implemented in Phase 2A.

### Replay, termination, seeding, and resume

- **Source-verified:** replay stores `(state, requested_action, reward, next_state, done)` with no truncation flag (`DeepQN/agent/replay_buffer.py:28-42,63-99`).
- **Source-verified:** Double-DQN selection/evaluation is correctly separated (`DeepQN/agent/dqn_agent.py:209-218`).
- **Source-verified:** training seeds Python and NumPy only (`DeepQN/training/train.py:84-92`); it does not seed Torch/CUDA or explicitly seed SUMO.
- **Source-verified:** replay constructors mutate global Python or NumPy RNG state (`DeepQN/agent/replay_buffer.py:58-62,129-137`), coupling agents.
- **Source-verified:** checkpoints contain online/target weights, episode, epsilon, and total steps (`DeepQN/agent/dqn_agent.py:251-265`). Optimizer, replay, RNG, profile position, configuration, and provenance are absent.
- **Source-verified:** `load_latest()` remains a permissive training-resume interface (`DeepQN/agent/dqn_agent.py:337-348`); strict atomic evaluation loading is separate at `DeepQN/agent/dqn_agent.py:350-414`.

## 6. Retraining requirements

Retraining is required after Phase 2B for:

1. The BiLSTM, because the intended approach-arrival inputs/targets differ from the historical network-occupancy task and the committed checkpoint is six-input while configuration is seven-input.
2. Forecast-conditioned DQN, because forecast values, scaling, timing, and fallback rules change.
3. No-forecast DQN, to provide a matched architecture/training-budget ablation under the corrected reward, termination, and seed contract.
4. Any policy using the proposed interval reward or terminated/truncated distinction; historical replay and weights cannot retroactively validate those semantics.

No model was retrained in Phase 2A.

## 7. Phase 2B implementation plan by file

- `lstm/prepare_data.py`: replace heading-based snapshots with versioned approach-boundary arrivals; emit a dataset manifest and leakage-safe partition identifiers.
- `lstm/train_lstm.py` and `lstm/predict.py`: implement the four-count target, deterministic seeds, training-only scaler fitting, strict schema/checkpoint validation, and held-out metrics.
- `DeepQN/integration/bilstm_adapter.py`: consume configured per-network mappings; require explicit `model`, `proxy`, or `none` mode; prohibit fallback in strict research mode; log prediction provenance.
- `DeepQN/env/reward.py`: add interval accumulators with explicit units and a separately reviewed pressure definition.
- `DeepQN/env/sumo_env.py`: return requested/executed action diagnostics and separate termination from horizon truncation.
- `DeepQN/agent/replay_buffer.py`, `DeepQN/agent/dqn_agent.py`, and `DeepQN/training/train.py`: propagate truncation, isolate RNGs, seed Torch/SUMO, and checkpoint optimizer/replay/RNG/profile state atomically.
- `DeepQN/evaluation/metrics.py` and `evaluate.py`: enable and validate unfinished/undeparted output, reconcile frozen demand IDs, and implement (W_H) only after the SUMO 1.26 record contract is fixture-tested.
- New controller modules: implement actuated and max-pressure baselines using identical phase/safety constraints and common manifests.

Phase 2B must begin with contract fixtures and a compute-cost pilot; it must not begin training without separate approval.

## 8. Proposed peer-reviewed experiment matrix

| Controller | Training seeds | Paired evaluation demand seeds/profile | Forecast |
|---|---:|---:|---|
| Native fixed-time | n/a | 10 | none |
| SUMO actuated, where feasible | n/a | 10 | none |
| Max-pressure | n/a | 10 | none |
| Dueling Double DQN | 5 | 10 per policy | four declared zero slots |
| Dueling Double DQN + verified BiLSTM | 5 | 10 per policy | strict real model only |

- Use disjoint training, validation, and final test demand sets. Freeze and hash every realized demand file.
- Select checkpoints by a preregistered validation-only rule after a fixed budget.
- Primary contrast: forecast DQN versus matched no-forecast DQN. Baseline comparisons are secondary; no superiority is assumed.
- Treat training seed as the independent learned-policy unit. Use hierarchical paired confidence intervals over training and demand seeds, paired effect sizes, and multiplicity correction for secondary outcomes.
- Forecast evaluation is separate and held out, with MAE/RMSE by direction and total against zero, persistence, historical mean, and moving average.
- At four profiles, the minimum evaluation matrix is 400 learned-policy runs plus 120 non-learning-controller runs, excluding validation, forecast evaluation, and ten training runs. Runtime remains unresolved until a Phase 2B training-only pilot.
- Taksim is conditional. Its eight TLS entries use `yellow_to_green` while the controller reads `yellow_to_next`, two TLS entries have empty incoming lanes, DQN keys differ from those consumed by the agent, and reward keys/signs differ (`DeepQN/configs/dqn_taksim_config.yaml:22-112`).

## 9. Test and command evidence

Executed successfully:

```text
sumo --version
```

Exit 0: Eclipse SUMO 1.26.0 Windows release build.

```text
sumo --help
```

Exit 0. Confirmed step length, emission step scaling, and unfinished/undeparted switches.

```text
python -m compileall -q DeepQN/env/sumo_env.py DeepQN/evaluation DeepQN/tests/test_phase2a_methodology.py
```

Exit 0.

The first broad discovery command executed 43 passing test cases but exited 1 because legacy `DeepQN/tests/test_ai_behavior.py` executes at import time and its Unicode arrow cannot be encoded by the Windows CP1252 console. This script was not part of the 34-test Phase 1 suite and was not modified during this focused phase.

Final authoritative command:

```text
python -m unittest DeepQN.tests.test_metrics_phase1a DeepQN.tests.test_checkpoints_phase1a DeepQN.tests.test_tiny_sumo_phase1b DeepQN.tests.test_phase2a_methodology -v
```

Result before the final artifact-shape regression was added: **43 passed in 13.672 s**, exit 0 (34 existing plus 9 new).

Final result after adding the restricted-load regression: **44 passed in 13.552 s**, exit 0 (34 existing plus 10 new). Compilation exited 0. `git diff --check` reported CRLF-at-EOL warnings for newly added lines in the existing CRLF-formatted `sumo_env.py`; no semantic whitespace error was introduced.

## 10. Remaining limitations and approval gate

- Passing tests establish code behavior, not traffic-control effectiveness or scientific validity.
- The censored primary outcome is specified but not yet implemented or empirically validated against an undeparted SUMO fixture.
- Reward and forecast semantics remain historically incompatible with the intended contracts.
- Historical checkpoint training provenance remains incomplete.
- No actuated/max-pressure implementation, training, Tahrir/Taksim evaluation, or statistical result was produced.
- `test_ai_behavior.py` remains a legacy executable/import hazard and can falsely describe absent checkpoints as loaded; it is not publication evidence.
- Existing `.vscode/settings.json` and backend image changes were not modified, staged, or reverted.

**Stop:** Phase 2A ends here. Phase 2B requires explicit approval.
