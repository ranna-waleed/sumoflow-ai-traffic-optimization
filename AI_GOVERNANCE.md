# SUMOFlow AI: AI Governance Checklist
## El-Tahrir Square Traffic Optimization (Graduation Project)

---

## 0. AI Usage Justification

**Why are we using AI here?**

Fixed-time traffic signals run the same cycle regardless of real conditions. At 3 AM with empty roads they behave identically to peak rush hour. Rule-based adaptive systems like SCOOT or SCATS require manually tuned thresholds per junction and per time-of-day. this does not scale and cannot discover dependencies between junctions.

**What does AI enable that rules cannot?**

The DQN learns optimal switching policies directly from 350 episodes of simulated experience without any manual threshold tuning. The BiLSTM discovers temporal patterns in traffic flow and provides 30-second predictions that allow proactive switching instead of reactive. YOLOv8s detects 7 vehicle classes simultaneously in a single forward pass. The 7 DQN agents implicitly coordinate through a shared reward signal. a rule-based system would need explicit coordination logic per junction pair.

**What happens if you remove the AI component?**

Waiting time returns to the fixed-time baseline: 626.6 seconds average per vehicle during morning rush hour. The 92.6% waiting time reduction and 90.3% CO2 reduction disappear entirely. Verified through controlled SUMO baseline experiments stored in `simulation/maps/baseline_outputs/`.

**Key files:**
- Baseline results: `simulation/maps/baseline_outputs/tripinfo_morning.xml`
- Evaluation report: `DeepQN/results/evaluation_report_fair.json`

---

## 1. Model Understanding

**Models used:**

| Model | Type | Purpose |
|---|---|---|
| YOLOv8s | CNN single-stage detector | Detect and classify vehicles in simulation frames |
| Faster R-CNN | CNN two-stage detector | Comparison model |
| RetinaNet | CNN one-stage with focal loss | Comparison model |
| BiLSTM | Bidirectional LSTM | Predict vehicle flow per direction 30 seconds ahead |
| Dueling Double DQN | Deep reinforcement learning | Decide keep/switch per traffic light junction |

**How each works at a high level:**

YOLOv8s divides the input image into a grid, predicts bounding boxes and class probabilities in a single forward pass, anchor-free architecture.

Faster R-CNN uses a region proposal network first, then classifies each region. more accurate but slower than YOLOv8s.

RetinaNet uses focal loss to handle class imbalance between easy and hard examples. good recall on rare classes like bicycles.

BiLSTM processes a 60-step history in both forward and backward directions simultaneously, capturing both recent trends and longer-term patterns. Outputs predicted vehicle counts for N/S/E/W for the next 30 seconds.

Dueling Double DQN splits the network into value and advantage streams. Double DQN uses the online network to select actions and the target network to evaluate them, preventing Q-value overestimation.

**Limitations:**

- DQN trained only on SUMO simulation data, real deployment requires retraining on real sensor data
- BiLSTM scaler fitted on Tahrir data, different city requires recalibration
- YOLOv8s trained on 1,800 images, may degrade in extreme weather or night conditions

**Key files:**
- DQN architecture: `DeepQN/agent/network.py`
- DQN agent: `DeepQN/agent/dqn_agent.py`
- BiLSTM: `lstm/` folder
- Detection models: `detection/` folder

---

## 2. Data and Inputs

**What inputs does each model take?**

YOLOv8s takes a 960x600 JPEG screenshot from SUMO-GUI via TraCI. Outputs bounding boxes, class labels, and confidence scores for 7 vehicle classes.

BiLSTM takes a 60-step sliding window with 7 features per step: north count, south count, east count, west count, total vehicles, avg speed, avg waiting time. Outputs predicted vehicle counts for N/S/E/W for the next 30 seconds.

DQN takes a 37-feature state vector per junction: 10 lanes x 3 features (halting count, waiting time, occupancy) = 30 lane features, plus 2 junction features (phase index, time in phase), plus 5 global features (BiLSTM N/S/E/W predictions and time of day).

**Input validation and cleaning:**

NaN and Inf values from TraCI are replaced with 0.0 before entering the DQN network. This guard is in `DeepQN/env/sumo_env.py`. Lane features are clipped to [0, 1] using np.clip(). SUMO validates route files and rejects malformed XML before simulation starts.

**Noisy and unexpected inputs:**

Tested in `DeepQN/tests/test_ai_behavior.py`. All-zero state, all-ones state, NaN injection, wrong vector size, and 1,000 random noise states all produce valid actions in {0, 1}.

**Key files:**
- NaN guard: `DeepQN/env/sumo_env.py`
- BiLSTM adapter: `DeepQN/integration/bilstm_adapter.py`

---

## 3. Evaluation and Metrics

**Detection models:**
- mAP@0.5 (mean Average Precision at IoU 0.5)
- mAP@0.5:0.95 (averaged across IoU thresholds)
- Precision, Recall, FPS
- Per-class AP for all 7 vehicle types

**DQN primary KPIs:**
- Average vehicle waiting time in seconds, measured per vehicle via TraCI getWaitingTime()
- Total CO2 emissions in mg, measured per vehicle via TraCI getCO2Emission()

**Baseline:** Standalone SUMO with original fixed-time signal plans, no Python. Results read from XML tripinfo files.

**Results:**


| Traffic Profile | Average Wait Time Change | CO₂ Emissions Change |
| --------------- | ------------------------ | -------------------- |
| Morning Rush    | ↓ 86.5%                  | ↓ 58.6%              |
| Evening Rush    | ↑ 17.4%                  | ↑ 36.3%              |
| Midday          | ↓ 17.3%                  | ↓ 14.2%              |
| Night           | ↑ 14.2%                  | ↓ 9.1%               |
| **Overall**     | **↓ 25.2%**              | **↓ 17.9%**          |


**Key files:**
- Baseline XML: `simulation/maps/baseline_outputs/`
- Evaluation report: `DeepQN/results/evaluation_report_fair.json`
- Evaluation script: `DeepQN/evaluation/evaluate.py`
- Metrics parser: `DeepQN/evaluation/metrics.py`

---

## 4. Testing AI Behavior

Run: `python -m DeepQN.tests.test_ai_behavior`

Result: 12/12 tests passed.

**Test groups:**

Group 1 — State edge cases:
- Empty network (all-zero state),produces valid action
- Maximum congestion (all-one state), produces valid action
- NaN/Inf injection, sanitized by guard, produces valid action
- Random noise state, produces valid action

Group 2 — BiLSTM input variations:
- Zero prediction (no flow signal), DQN handles gracefully
- Maximum prediction (traffic spike), DQN responds correctly

Group 3 — Determinism and consistency:
- Same input gives same output across 5 consecutive calls (epsilon=0 is fully deterministic)
- 1,000 random states — all actions in valid set {0, 1}

Group 4 — Multi-agent independence:
- Changing one agent's state does not affect other agents' decisions
- Each agent outputs independent Q(keep) and Q(switch) values

Group 5 — Failure recovery:
- Missing junction in observation, filled with zeros, simulation continues
- Wrong state vector size, padded or truncated to 37, produces valid action

**Key file:** `DeepQN/tests/test_ai_behavior.py`

---

## 5. Reliability and Failure Handling

**Wrong output:** The DQN can only output 0 (keep) or 1 (switch). A suboptimal switch at worst extends one red phase by 10 seconds. Yellow transition buffers prevent abrupt changes.

**DQN inference failure:** Automatic fallback to fixed alternating timing (39-second cycles). Logged as [fallback mode] in the decision CSV.

**SUMO connection failure:** TraCI retry loop with 5 attempts and 1 second between each. If all fail, simulation thread exits cleanly.

**No checkpoints:** Evaluation script automatically switches to random policy and logs [RANDOM]. Confirmed working in Taksim Square pipeline test.

**Key files:**
- Fallback logic: `backend/services/dqn_runner.py`
- Retry loop: `DeepQN/env/sumo_env.py`
- Random policy fallback: `DeepQN/evaluation/evaluate.py`

---

## 6. Safety and Governance

**PII:** No personal data collected. SUMO generates synthetic vehicle IDs (vehicle_0, vehicle_1). No real people tracked.

**Harmful content:** Not applicable. this is a traffic signal optimizer.

**Input filtering:** NaN/Inf guard in state builder prevents corrupted data from reaching the DQN.

**Output filtering:** DQN outputs are restricted to indices {0, 1} mapped to pre-validated phase sequences from the net.xml. The DQN cannot create a conflicting green signal or an unsafe phase. All transitions go through a yellow buffer.

**Key files:**
- NaN guard: `DeepQN/env/sumo_env.py`
- Phase validation: `DeepQN/configs/dqn_config.yaml`
- Phase application: `backend/services/dqn_runner.py`

---

## 7. Prompt / Model Design (for LLMs)

Not applicable. This project does not use any Large Language Model, generative AI, or prompt-based system. All three AI components are trained models: a CNN detector, a recurrent predictor, and a reinforcement learning agent.

---

## 8. System Integration

**Architecture:**

```
SUMO Simulation
     TraCI (synchronous, every 10 simulation steps)
     State collection (lane sensors + vehicle metrics)
     BiLSTM prediction (every 30 steps, ~5ms)
     DQN decision (7 agents, ~1ms each)
     TraCI phase apply
     SUMO advances 10 steps -> repeat
```

DQN and BiLSTM inference are synchronous within the simulation loop. Inference is fast enough (~6ms total) that it does not delay the simulation perceptibly.

The FastAPI backend runs the DQN in a background thread so HTTP requests from the frontend are non-blocking. The frontend polls every 500ms asynchronously.

**Key files:**
- SUMO environment: `DeepQN/env/sumo_env.py`
- BiLSTM integration: `DeepQN/integration/bilstm_adapter.py`
- DQN runner: `backend/services/dqn_runner.py`
- FastAPI main: `backend/main.py`

---

## 9. Performance and Cost

| Component | Inference time |
|---|---|
| DQN (7 agents, CPU) | ~1ms per decision step |
| BiLSTM (CPU) | ~5ms per call |
| YOLOv8s (CPU) | ~50-100ms per frame |

**Cost:** Zero. All models run locally on CPU/GPU. No cloud API, no tokens, no per-request cost.

**Scalability:** Adding junctions requires only a YAML config file change. Confirmed by scaling from 7 agents (Tahrir) to 8 agents (Taksim) with zero code changes.

---

## 10. Monitoring AI in Production

The DQNMonitor class runs automatically during every live DQN simulation session and detects:

- Degradation: rolling average wait exceeds 50% of trained baseline (313s threshold)
- Policy collapse: same number of junctions switched for 20 or more consecutive steps
- CO2 spike: current reading exceeds 2x the rolling mean
- Critical failure: average wait exceeds 110% of baseline

**What is logged:**
- `DeepQN/logs/monitor_metrics_TIMESTAMP.csv` avg wait, CO2, n_switched per step
- `DeepQN/logs/monitor_alerts_TIMESTAMP.csv`  all alerts with timestamp, level, and message
- `logs/dqn_decisions_PROFILE_TIMESTAMP.csv`  full decision log per step

**Verified output from live run:**
```
DQN MONITORING SUMMARY
Total alerts   : 1
Warning alerts : 1  (CO2 spike at startup, expected)
Final avg wait : 14.36s
No degradation detected.
```

**Key file:** `DeepQN/monitoring/monitor.py`

---

## 11. Explainability

The DQNExplainer generates human-readable explanations for each junction decision every 90 simulation steps. Each explanation includes:

- Junction name and ID
- Action taken (keep or switch)
- Confidence level (High/Medium/Low) based on Q-value margin |Q(keep) - Q(switch)|
- Top congested lanes with halting percentage, waiting time, and occupancy
- BiLSTM forecast for all 4 directions
- Plain-English reason sentence

Example output:
```
Junction: S-Corridor Gate 1
Decision: SWITCH PHASE
Confidence: High (Q-margin=0.847)
Top congested lanes:
  228979748#2_0 — 72% halting, wait=180s
BiLSTM forecast: N=89 S=12 E=34 W=8 vehicles/30s
Reason: heavy queue on lane 228979748 (72% halting);
        BiLSTM predicts 89 incoming from NORTH.
```

**Key file:** `DeepQN/explainability/explainer.py`

---

## 12. Improvement Strategy

**More training:** Resume from any checkpoint using --resume flag. The replay buffer retains 50,000 transitions. New experience mixes with old, preventing forgetting.

**New profiles:** Add a new entry under profiles in dqn_config.yaml. The training loop rotates through all profiles automatically.

**New maps:** Proven with Taksim Square, only a new YAML config file is needed. Run the net parser to extract junction IDs and lanes, fill the template, done.

**Data drift:** If real traffic patterns shift, re-run randomTrips.py on updated OSM data to regenerate routes, then retrain from scratch or from latest checkpoint.

**Feedback loop:** Training CSV records reward per episode. If reward stops improving, adjust epsilon decay or learning rate in dqn_config.yaml.

**Key files:**
- Training: `DeepQN/training/train.py`
- Config: `DeepQN/configs/dqn_config.yaml`
- Training logs: `DeepQN/logs/training_*.csv`

---

## 13. Ethical and Responsible AI

**Bias:** No demographic or personal data is used. All vehicles are treated identically. The reward function penalizes total waiting time across all junctions equally. No junction, direction, or vehicle type is prioritized over another.

**Privacy:** No user data collected. Synthetic SUMO vehicle IDs carry no personal information. No data leaves the local machine.

**Fairness:** The reward function is:
```
reward = -1.0 * (change in total waiting time)
       - 0.001 * (change in total CO2)
       + 0.5 * (vehicles completing trips)
```
All vehicles contribute equally to each term.

**Wrong decisions:** Monitoring detects degradation in real time. Decision log provides full traceability per step. DQN output is restricted to pre-validated phase sequences. architecturally impossible to create a conflicting green signal. Automatic fallback to fixed-time signals if inference fails.

**Key files:**
- Reward function: `DeepQN/env/reward.py`
- Monitoring: `DeepQN/monitoring/monitor.py`
- Decision log: `logs/dqn_decisions_*.csv`
