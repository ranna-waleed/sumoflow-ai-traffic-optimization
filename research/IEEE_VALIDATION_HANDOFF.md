# SUMOFlowAI — Codex research and IEEE submission handoff

**Prepared:** 2026-10-25

**Repository:** https://github.com/ranna-waleed/sumoflow-ai-traffic-optimization
**Code revision cited in prior audits:** `5b93cd21502e09deef3c4a8100a1c7d4b0e49c2b`
**Google Drive artifacts:** https://drive.google.com/drive/u/1/folders/1ULhxaaJfYKngSDmra5949AVY5koPnaeH

> This document transfers the context of an earlier evidence-based audit. It is a **handoff**, not a substitute for rereading the repository. Verify each path and line against the checked-out commit before editing. Descriptions explicitly differentiate verified code behavior, observed artifacts, and uncertainties. Historical metrics are not publishable evidence until independently validated.

## Objective and recommended scope

**Primary paper hypothesis:** Does short-horizon BiLSTM traffic-flow forecasting improve safety-constrained decentralized Dueling Double DQN traffic control at an irregular urban intersection compared with fixed-time, actuated and/or max-pressure baselines, under matched SUMO evaluation conditions? Are any benefits robust across traffic profiles and generalize to a second network?

**Default scope:** Train/evaluate the Tahrir Square multi-agent traffic controller and forecast ablation. Use Taksim only if transfer or comparison provenance can be established. Vehicle-detector benchmarking can be a separate auxiliary result; do not claim YOLO inputs drove the DQN unless the full path is implemented and rigorously tested.

**Do not prioritize:** frontend UI polish, adding additional AI frameworks, or claiming a deployed production system.

## Verified code observations (recheck against checked-out revision)

### 1. Incompatible baseline and DQN metric collection
- `DeepQN/evaluation/evaluate.py:65-77`: baseline metrics loaded from SUMO XML through `load_metrics_from_outputs`.
- `DeepQN/evaluation/evaluate.py:101-157`: DQN evaluator samples `traci.vehicle.getWaitingTime` and `getCO2Emission`, accumulates `getArrivedNumber()` on decision iterations, and constructs `EpisodeMetrics` only from `avg_waiting_time_s`, `total_co2_mg`, and `throughput`.
- `DeepQN/configs/dqn_config.yaml:9`: `decision_interval: 10` SUMO steps, so the DQN emissions collection omits intervening timesteps; baseline XML includes all timesteps.
- `DeepQN/evaluation/metrics.py:66-102`: baseline average waiting time is mean tripinfo `waitingTime` across completed trips.
- `DeepQN/evaluation/metrics.py:105-133`: baseline XML emissions summed over all vehicle-timesteps.
- `DeepQN/evaluation/metrics.py:28-40`: unpopulated metrics default to zero.
- `DeepQN/evaluation/evaluate_fair.py:186-261`: alternative evaluator creates DQN tripinfo/emission XML. `:264-359` builds `evaluation_report_fair.json`; however its aggregated `EpisodeMetrics` also discards travel time and trip count by creating a new object with only wait/CO2/throughput fields, so do not treat default-zero metrics as measured zeros.

**Implication:** the existing `evaluation_report.json` reported 90%-scale CO2 reductions under incompatible collection schemes; the improvements are **not verified**. Correct the procedure and rebuild results before publication.

### 2. Trained-checkpoint verification bug
- `DeepQN/agent/dqn_agent.py:300-311`: `MultiAgentDQN.load_latest` logs/returns on absent directory and logs missing per-agent weights without raising.
- `DeepQN/evaluation/evaluate.py:179-188`, `DeepQN/evaluation/evaluate_fair.py:285-293`: assume any nonexception means trained checkpoints loaded; random fallback only if exception raised.
- This can mislabel **partially or entirely randomly initialized DQN networks** as trained DQN.
- `DeepQN/agent/dqn_agent.py:220-242`: individual checkpoint stores online/target weights, episode, epsilon, steps, not a full run provenance manifest.

**Required behavior:** check exact expected traffic-light IDs, architecture compatibility, all seven checkpoints, file hashes and desired episode; fail clearly on missing weights. Random control should be explicit, not a silent fallback. Preserve provenance in run manifest.

### 3. BiLSTM inference and feature provenance
- `DeepQN/env/sumo_env.py:314-315, 327-350`: creates `BiLSTMAdapter` and loads it on `reset()`.
- `DeepQN/integration/bilstm_adapter.py:103-159`: attempts to load `lstm/models/config.json`, `scaler.pkl`, `lstm_best.pth`; silently falls back to TraCI proxy if missing or load fails.
- `DeepQN/integration/bilstm_adapter.py:174-209`: proxy before sufficient history and on inference exception; active model returns `next_30s` directional output normalized by saved scaler max values.
- `DeepQN/integration/bilstm_adapter.py:232-245`: history collection matches direction-name features; **unmatched `total`, `avg_speed`, and `avg_waiting` fields are filled with 0.0**.
- `lstm/predict.py:111-130`: 30×4 outputs are inverse-scaled and rounded, then each direction's `next_30s` field is the integer-truncated average across 30 predicted timesteps.
- `DeepQN/env/sumo_env.py:430-474`: per-agent 37-dimensional state contains lane features, two phase features, four forecast/proxy channels, time of day.
- `backend/services/dqn_runner.py` uses a different feature composition, angle-derived global flow and hardcoded scaling. Not the evaluated training environment. Separate backend has an `UnboundLocalError` involving `_total_co2_accumulated` in `_get_traffic_metrics()`; fixing frontend/live runner is **not** necessary for a simulation-only paper, but prohibit claims of validated deployment.

**Required behavior:** record model load success and time of each prediction; validate seven feature definitions/order, scaling, history length, prediction timing, and missing edge handling; make proxy-vs-model mode explicit. For an ablation, train/evaluate *both* no-forecast DQN and forecast-augmented DQN under matched conditions. Forecast quality itself needs a held-out test against persistence or moving-average baselines.

### 4. Actions and safety overrides
- `DeepQN/training/train.py:127-136`: `actions=multi_agent.act(...)`, `env.step(actions)`, and `multi_agent.push(obs, actions,...)`.
- `DeepQN/env/sumo_env.py:101-172`: `_PhaseController.apply_action` can ignore switch requests in yellow/min-green or force a switch at max-green; phase controller ticks during simulation.
- `DeepQN/agent/dqn_agent.py:276-288`: pushes **policy-requested** actions to replay.

This is **not necessarily a bug** if the action semantically means "request keep/switch" and safety constraints are part of the environment dynamics. Do not blindly relabel replay actions! First formalize action semantics and test edge cases; optionally log requested and executed behavior, override reason, and active phase.

### 5. Reward and research design
- `DeepQN/env/sumo_env.py:386-408`: computes one `global_reward` shared by all traffic-light agents, advances 10 simulation steps per action, and ticks the BiLSTM history during each SUMO step.
- `DeepQN/agent/dqn_agent.py:165-185`: online network selects the next action, target network evaluates it (Double-DQN target).
- The saved DQN training CSV had **200** recorded episodes and `epsilon_mean=0.05` from episode 1, whereas the committed YAML specifies `epsilon_start=1.0` and `epsilon_decay=0.99992`. The exact historical training config or resume behavior is **not established**.
- Test reward normalization/penalties and record collisions, teleports, and throughput; note that 200-episode training logs do not alone prove statistically robust policy performance.

### 6. Live backend and CV claims
- `backend/services/dqn_runner.py:433-457`: live directional vehicle counts from TraCI `traci.vehicle.getIDList()` and heading `traci.vehicle.getAngle()`, **not YOLO detections**.
- YOLO screenshot/class counts are separate and feed logging/dashboard only. Do not say camera detections drive live DQN.
- Live backend uses different state values (including two constant zeros as junction features and constant time-of-day) than training, so report training/evaluation results separately.

## Verified historical artifacts and dates

- Drive `THE FINAL DQN VERSION (OUTPUTS)/results/evaluation_report.json`: file **modified May 21, 2026**, contains fixed-time/DQN comparison; DQN `n_trips=0`, average travel time `0.0`, average speed `0.0` due to default/unpopulated fields, so travel improvements in report are invalid.
- Drive DQN logs: `training_20260520_160300.csv` has 200 episode rows for four profiles. Seven `*_ep0200.pt` checkpoints exist in Drive, file content metadata last modified May 21 (many uploaded June 9). Their existence proves checkpoint files, not exact training code, forecast mode, or their use during evaluation.
- Fixed-time baseline four-profile `tripinfo_{morning,midday,evening,night}.xml` from Drive last modified May 21. Independent calculations matched report's fixed-time trip counts and average waiting time (morning 3482/626.618s; midday 2937/629.051s; evening 5711/632.061s; night 882/24.281s).
- Drive `BI-LSTM FILES/models.zip`: archive modified June 9 21:48 UTC (**June 10, 00:48 Egypt time**) and contains `config.json`, `scaler.pkl`, `lstm_best.pth`; the archive upload date does NOT mean weights trained June 10.
- Drive `3 MODELS RESULTS (YOLO, FASTER-RCNN, RETINANET)` folder created June 9 21:43 UTC (**June 10, 00:43 Egypt time**). YOLO `tahrir_yolo_v3/results.csv` modified May 1; 60th epoch reported precision .81828, recall .41124, mAP@0.5 .48319, mAP@0.5:.95 .34408. Faster R-CNN `eval_results_run3 (1).json` modified June 9 21:43 UTC (Egypt June 10) reports mAP@0.5 .469683, mAP@0.5:.95 .341592, precision .945926, recall .373519, 109.332ms avg inference; class-level metrics may not simply average to reported aggregate—validate evaluator convention. RetinaNet folder listed empty during last audit.
- Drive `TAKSIM SQUARE MAP (OUTPUTS)(FOR TESTING THE PIPELINE)` has original `baseline_outputs` and `outputs` from **June 9**; independent tripinfo comparison: 1206 vs 1218 trips, mean wait 1319.92s vs 1269.29s, mean travel duration 1536.91 vs 1488.08s. Which policy ran `outputs` was NOT conclusively established. Repo Taksim evaluation report shows different numbers (DQN wait 99.18s, throughput 116) under an incompatible evaluator; do NOT merge these as a single experiment or claim a transfer improvement.
- Historical thesis table reportedly gave unweighted four-profile mean ~25.2% waiting and ~17.9% CO2 reductions; separate May evaluation JSON has ~83.43% and ~90.45% unweighted percentage averages (methodologically invalid comparison). Don't use either as verified publication evidence without matched reruns.

## Phased roadmap and acceptance gates

### Phase 0 — Read-only audit and baselines inventory (FIRST)
1. Verify checked-out commit and `git status`; inventory branch, code, existing tests, SUMO configs and output directory structure.
2. Map exact code paths (train, evaluation, rewards, adapters, phase controller, checkpoint loading, XML parser); identify any repository changes since the cited audit commit.
3. Make a run/artifact provenance table with each source file's original modified date, download/upload date, experiment scenario, control mode, checksum where possible, and evidence quality.
4. Reproduce existing baseline tripinfo metrics *read-only* with independent parser; flag missing/incompatible data without fabricating substitute metrics.
5. Produce `research/phase0-audit.md` with minimal-fix proposal, test plan, expected computational cost, and unresolved questions. **Stop for user approval.**

### Phase 1 — Correct evaluator and test its metrics
- Explicitly require trained checkpoint manifests for trained evaluation, prevent silent uninitialized policies.
- Use the same SUMO tripinfo/emission/summary methods for baseline and adaptive policy, or justifiably equivalent per-step instrumentation; test reference values against known tiny fixtures.
- Fix zero-default metrics being falsely interpreted as measured zeros; preserve n_trips, travel time, time loss, speed and run variability in JSON/CSV. Missing input must be reported as missing/invalid.
- Ensure throughput is counted at every simulation step (or directly from tripinfo XML). Pair route input, start/end, teleport settings, SUMO binary/version, random seed and traffic demand; explicitly track uncompleted vehicles.
- Run fast parser/unit tests and a tiny SUMO smoke test if environment permits; no large experiments yet. **Stop and review.**

### Phase 2 — BiLSTM and training semantics
- Validate and test exact model load and failure modes: explicit prediction/proxy flags, all seven features, history collection, direction mappings, scaling, caching and timing.
- Verify prediction dataset split to avoid temporal leakage and establish held-out MAE/RMSE versus simple persistence baseline.
- Formalize requested-vs-executed action semantics; test min-green, max-green and yellow phase; log safety intervention rates.
- Record a training run manifest with code/model/config hashes, RNG seeds, agent checkpoint IDs, forecasting mode and fallback counts. **Stop and review.**

### Phase 3 — Experiments and comparisons
- Implement/check fixed-time, actuated and max-pressure comparators and **no-forecast DQN versus forecast-augmented DQN** under same constraints and demand.
- Select independent training seeds and paired evaluation seeds; at least five eval seeds per traffic profile as an initial design, and preferably several training seeds. Report mean, uncertainty, completed/unfinished trips, CO2, wait, duration, time loss, teleport count, crashes and reward.
- Save full per-run raw outputs and metadata in versioned external artifact storage, not gigantic Git history. No cherry-picking or tuning on the final held-out evaluation seeds.
- Run planned experiments only after user confirms compute costs and available SUMO/Python/GPU setup. **Stop and review.**

### Phase 4 — Taksim and robustness (conditional)
- Specify a precise protocol: source Tahrir checkpoints, target mapping, frozen transfer vs fine-tune vs retrain; do not call it 'transfer' without frozen/shared weights.
- Check 7-vs-8-controlled-junction discrepancy and state/action compatibility. Reuse the same metric pipeline.
- Evaluate unseen demand profiles or perturbations, capacity, or sensor-noise conditions as time permits.

### Phase 5 — Paper readiness
- Freeze source SHA, artifact manifests, scripts and exact tables. Preserve negative findings.
- Define primary hypothesis and contribution supported by results, draw diagrams only of measured paths, list constraints and limitations.
- Prepare IEEE two-column LaTeX manuscript with methods, baselines, ablations, uncertainty, original figures and grounded citations; ensure coauthor/supervisor agreement and verify relevant target-conference rules.
- An IEEE-formatted arXiv preprint is not itself IEEE peer review or acceptance.

## Acceptance checklist for the first publishable results table

- [ ] All evaluated weights loaded explicitly, matched to requested agent IDs and hashed.
- [ ] Same measurement definitions, matched scenarios and complete emissions sampling for every method.
- [ ] Both baseline and DQN have actual nonzero/valid trip counts or flagged missing values—not silent defaults.
- [ ] All BiLSTM features and forecast/proxy switches recorded; no false real-model labels.
- [ ] Multiple evaluation seeds, mean/uncertainty, complete trip and teleport statistics saved.
- [ ] No-forecast vs forecast ablation completed and interpreted; clear training budgets.
- [ ] At least one strong non-RL adaptive comparator, preferably max-pressure, included.
- [ ] Every figure/table has source files, script, git SHA, config and seeds.
- [ ] Reviewers can re-run metrics from a documented recipe; large outputs stored outside Git.

## Sources to revisit

- GitHub reference SHA: https://github.com/ranna-waleed/sumoflow-ai-traffic-optimization/tree/5b93cd21502e09deef3c4a8100a1c7d4b0e49c2b
- `DeepQN/evaluation/evaluate.py`, `DeepQN/evaluation/evaluate_fair.py`, `DeepQN/evaluation/metrics.py`
- `DeepQN/agent/dqn_agent.py`, `DeepQN/env/sumo_env.py`, `DeepQN/training/train.py`
- `DeepQN/integration/bilstm_adapter.py`, `lstm/predict.py`, `lstm/models/config.json`
- `DeepQN/configs/dqn_config.yaml`, `DeepQN/configs/dqn_taksim_config.yaml`
- `backend/services/dqn_runner.py`, `backend/services/yolo_detect.py`
- Drive DQN report: https://drive.google.com/file/d/1wWClU4cAfm-fC06-yNEh3hHwCvFxDpsK/view
- Drive baseline outputs: https://drive.google.com/drive/folders/1wHFHoSMK6BIzKNZ__zOBZXOIs87x1BeU
- Drive DQN checkpoints: https://drive.google.com/drive/folders/1Zxjo_qrzUa0FUMli9FL7XXwgBk-L-BqQ
- Drive model ZIP: https://drive.google.com/file/d/195nEN_XTDVegA8aGxWcofCGsOopQZ-C9/view
- Drive June 10 model-results folder: https://drive.google.com/drive/folders/1nhocPiUGAtG84UDPMpfKNotZZfIOS7k_
- Drive Taksim paired outputs: https://drive.google.com/drive/folders/1yKGnuwPgDOaPPtt1sOrQuQ5sIV3txFpN

## First Codex instruction (paste into IDE chat)

"We are preparing SUMOFlowAI for a reproducible academic study, not optimizing for impressive-looking numbers. Read AGENTS.md and research/IEEE_VALIDATION_HANDOFF.md and inspect the real repository at its current commit. Perform **Phase 0 only, read-only**: confirm or correct the previously audited findings with file and line references, inventory tests and experiment artifacts, and propose the smallest changes and tests for Phase 1. Clearly separate verified code behavior, artifact observations, and unknown historical run provenance. Do not modify source code, create commits, download massive files, run long simulations, or start the next phase. Present the audit and seek approval."
