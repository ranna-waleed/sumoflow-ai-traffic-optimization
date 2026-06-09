# backend/services/dqn_runner.py
import os, sys, time, threading, csv, yaml
from datetime import datetime
import numpy as np

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS_DIR = os.path.join(BASE_DIR, "simulation", "maps")

# ── Runtime state ─────────────────────────────────────────────
_running      = False
_profile      = None
_step         = 0
_view_id      = "View #0"
_multi_agent  = None
_tls_config   = None
_metrics      = {}
_lock         = threading.Lock()

_pipeline_state = {
    "sumo_vehicles":    0,
    "yolo_counts":      {},
    "lstm_prediction":  {"north": 0, "south": 0, "east": 0, "west": 0},
    "dqn_action":       0,
    "dqn_action_name":  "—",
    "avg_wait_s":       0.0,
}

FRAME_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dqn_frame.jpg"
)

# ── Logging ───────────────────────────────────────────────────
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
_decision_log_path   = None
_decision_log_file   = None
_decision_log_writer = None


def _init_decision_log(profile: str):
    global _decision_log_path, _decision_log_file, _decision_log_writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _decision_log_path = os.path.join(
        LOGS_DIR, f"dqn_decisions_{profile}_{timestamp}.csv"
    )
    _decision_log_file   = open(_decision_log_path, "w", newline="")
    _decision_log_writer = csv.writer(_decision_log_file)
    _decision_log_writer.writerow([
        "timestamp", "step", "profile",
        "n_switched", "action_name",
        "avg_wait_s", "vehicles", "total_co2_mg",
        "lstm_north", "lstm_south", "lstm_east", "lstm_west",
        "yolo_car", "yolo_bus", "yolo_taxi", "yolo_microbus",
        "yolo_truck", "yolo_motorcycle", "yolo_bicycle",
    ])
    _decision_log_file.flush()
    print(f"[DQN Logger] Logging -> {_decision_log_path}")


def _log_decision(step, profile, n_switched, action_name,
                  avg_wait, n_vehicles, total_co2,
                  yolo_counts, lstm_pred):
    global _decision_log_writer, _decision_log_file
    if _decision_log_writer is None:
        return
    try:
        _decision_log_writer.writerow([
            datetime.now().strftime("%H:%M:%S"),
            step, profile,
            n_switched, action_name,
            round(avg_wait, 2), n_vehicles, round(total_co2, 1),
            lstm_pred.get("north", 0), lstm_pred.get("south", 0),
            lstm_pred.get("east",  0), lstm_pred.get("west",  0),
            yolo_counts.get("car",        0),
            yolo_counts.get("bus",        0),
            yolo_counts.get("taxi",       0),
            yolo_counts.get("microbus",   0),
            yolo_counts.get("truck",      0),
            yolo_counts.get("motorcycle", 0),
            yolo_counts.get("bicycle",    0),
        ])
        _decision_log_file.flush()
    except Exception as e:
        print(f"[DQN Logger] Log error: {e}")


def _close_decision_log():
    global _decision_log_file, _decision_log_writer
    try:
        if _decision_log_file:
            _decision_log_file.close()
            print(f"[DQN Logger] Log saved -> {_decision_log_path}")
    except Exception:
        pass
    _decision_log_file   = None
    _decision_log_writer = None


# ── Config ────────────────────────────────────────────────────
CONFIG_PATHS = {
    "morning_rush": os.path.join(MAPS_DIR, "config_morning_rush.sumocfg"),
    "evening_rush": os.path.join(MAPS_DIR, "config_evening_rush.sumocfg"),
    "midday":       os.path.join(MAPS_DIR, "config_midday.sumocfg"),
    "night":        os.path.join(MAPS_DIR, "config_night.sumocfg"),
}

BEGIN_TIMES = {
    "morning_rush": "27000",
    "evening_rush": "54000",
    "midday":       "43200",
    "night":        "79200",
}

MAX_STEPS_PER_PROFILE = {
    "morning_rush": 10800,
    "evening_rush": 18000,
    "midday":       10800,
    "night":        7200,
}

STEP_DELAY = 0.05
_lstm_history = []


# ── Agent loading ─────────────────────────────────────────────

def _load_agent() -> bool:
    global _multi_agent, _tls_config
    if _multi_agent is not None:
        return True
    try:
        config_path = os.path.join(
            BASE_DIR, "DeepQN", "configs", "dqn_config.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        _tls_config = cfg["tls_junctions"]

        sys.path.insert(0, BASE_DIR)
        from DeepQN.agent.dqn_agent import MultiAgentDQN

        tls_ids      = list(cfg["tls_junctions"].keys())
        _multi_agent = MultiAgentDQN(tls_ids, cfg["dqn"])
        _multi_agent.load_latest(
            os.path.join(BASE_DIR, "DeepQN", "checkpoints")
        )
        print(f"[DQN Runner] MultiAgentDQN loaded - {len(tls_ids)} agents")
        return True
    except Exception as e:
        print(f"[DQN Runner] Failed to load MultiAgentDQN: {e}")
        return False


# ── Observation builder ───────────────────────────────────────

def _build_observations(lstm_pred: dict) -> dict:
    if _tls_config is None:
        return {}

    MAX_LANES = 10
    FPERLANE  = 3
    MAX_QUEUE = 50.0
    MAX_WAIT  = 300.0
    MAX_FLOW  = 136.0

    global_feats = np.array([
        min(lstm_pred.get("north", 0) / MAX_FLOW, 1.0),
        min(lstm_pred.get("south", 0) / MAX_FLOW, 1.0),
        min(lstm_pred.get("east",  0) / MAX_FLOW, 1.0),
        min(lstm_pred.get("west",  0) / MAX_FLOW, 1.0),
        0.5,
    ], dtype=np.float32)

    observations = {}
    for tid, tls_info in _tls_config.items():
        lanes      = tls_info.get("incoming_lanes", [])
        lane_feats = np.zeros(MAX_LANES * FPERLANE, dtype=np.float32)

        for i, lane in enumerate(lanes[:MAX_LANES]):
            try:
                halt = traci.lane.getLastStepHaltingNumber(lane) / MAX_QUEUE
                wait = traci.lane.getWaitingTime(lane)           / MAX_WAIT
                occ  = traci.lane.getLastStepOccupancy(lane)
            except Exception:
                halt, wait, occ = 0.0, 0.0, 0.0

            base = i * FPERLANE
            lane_feats[base]     = float(np.clip(halt, 0.0, 1.0))
            lane_feats[base + 1] = float(np.clip(wait, 0.0, 1.0))
            lane_feats[base + 2] = float(np.clip(occ,  0.0, 1.0))

        junc_feats = np.array([0.0, 0.0], dtype=np.float32)
        obs = np.concatenate([lane_feats, junc_feats, global_feats])

        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)

        observations[tid] = obs

    return observations


# ── Build Q-value display dict ────────────────────────────────

def _build_q_display(per_junction_actions: dict, observations: dict) -> dict:
    """
    Build a per-junction action summary for frontend display.
    Shows which junctions kept (0) vs switched (1).
    Format matches what BeforeAfter.jsx q_values panel expects:
      { "TLS_id (keep)": 0.0, "TLS_id (switch)": 1.0, ... }
    but simplified to just show junction decisions.
    """
    q_display = {}
    short_names = {
        "315744796":  "N-trunk entry",
        "96621100":   "Ring N-entry",
        "2031414903": "Ring W-entry",
        "2031414899": "S-gate 1",
        "6288771431": "S-gate 2",
        "271064234":  "E-exit 1",
        "315743335":  "E-exit 2",
    }
    for tid, action in per_junction_actions.items():
        label = short_names.get(tid, tid)
        # Use 1.0 for switch (active decision), 0.2 for keep (passive)
        q_display[label] = 1.0 if action == 1 else 0.2
    return q_display


# ── Apply DQN decisions ───────────────────────────────────────

def _apply_dqn_actions(per_junction_actions: dict):
    if _tls_config is None:
        return
    for tid, action in per_junction_actions.items():
        if action == 1:
            tls_info = _tls_config.get(tid, {})
            g2y = {
                int(k): v
                for k, v in tls_info.get("green_to_yellow", {}).items()
            }
            try:
                cur = traci.trafficlight.getPhase(tid)
                if cur in g2y:
                    traci.trafficlight.setPhase(tid, g2y[cur])
            except Exception:
                pass


# ── YOLO ─────────────────────────────────────────────────────

def _run_yolo_on_frame() -> dict:
    if not os.path.exists(FRAME_PATH):
        return {}
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from services.yolo_detect import detect_image
        with open(FRAME_PATH, "rb") as f:
            image_bytes = f.read()
        if len(image_bytes) < 1000:
            return {}
        result = detect_image(image_bytes)
        return result.get("class_counts", {})
    except Exception as e:
        print(f"[YOLO] Skipped: {e}")
        return {}


# ── LSTM ──────────────────────────────────────────────────────

def _run_lstm_prediction() -> dict:
    if len(_lstm_history) < 10:
        return {"north": 0, "south": 0, "east": 0, "west": 0}
    try:
        sys.path.insert(0, BASE_DIR)
        from lstm.predict import predict
        result = predict(_lstm_history)
        return result.get(
            "next_30s",
            {"north": 0, "south": 0, "east": 0, "west": 0}
        )
    except Exception as e:
        print(f"[LSTM] Prediction skipped: {e}")
        return {"north": 0, "south": 0, "east": 0, "west": 0}


# ── Traffic metrics ───────────────────────────────────────────

def _angle_to_direction(angle: float) -> str:
    a = float(angle) % 360
    if a < 45 or a >= 315: return "north"
    if 45  <= a < 135:     return "east"
    if 135 <= a < 225:     return "south"
    return "west"


def _get_traffic_metrics():
    vehicle_ids = traci.vehicle.getIDList()
    dir_counts  = {"north": 0, "south": 0, "east": 0, "west": 0}
    waiting, co2_vals = [], []

    for v in vehicle_ids:
        try:
            angle = float(traci.vehicle.getAngle(v))
            wait  = traci.vehicle.getWaitingTime(v)
            co2_vals.append(traci.vehicle.getCO2Emission(v))
            dir_counts[_angle_to_direction(angle)] += 1
            waiting.append(wait)
        except Exception:
            continue

    avg_wait  = sum(waiting)  / len(waiting)  if waiting  else 0.0
    total_co2 = sum(co2_vals) if co2_vals else 0.0
    return avg_wait, len(vehicle_ids), dir_counts, total_co2


# ── Main start ────────────────────────────────────────────────

def start(profile: str) -> bool:
    global _running, _profile, _step, _view_id, _metrics, _lstm_history

    if _running:
        stop()
        time.sleep(2.0)

    if not _load_agent():
        return False

    config = CONFIG_PATHS.get(profile)
    if not config:
        return False

    _init_decision_log(profile)

    try:
        traci.start([
            "sumo-gui", "-c", config,
            "--no-warnings",
            "--no-step-log",
            "--begin", BEGIN_TIMES.get(profile, "0"),
            "--start",
            "--delay", "50",
        ], port=8813)

        _running      = True
        _profile      = profile
        _step         = 0
        _metrics      = {}
        _lstm_history = []

        # ── GUI setup: wait longer + retry to avoid black window ──
        time.sleep(3.0)
        for attempt in range(5):
            try:
                views    = traci.gui.getIDList()
                _view_id = views[0] if views else "View #0"
                traci.gui.setOffset(_view_id, 3694.5, 1539.5)
                traci.gui.setZoom(_view_id, 1500)
                # Force a render step so window is not black
                traci.simulationStep()
                _step += 1
                print(f"[DQN Runner] GUI ready (attempt {attempt+1})")
                break
            except Exception as e:
                print(f"[DQN Runner] GUI attempt {attempt+1}: {e}")
                time.sleep(1.0)

        max_steps = MAX_STEPS_PER_PROFILE.get(profile, 10800)
        print(f"[DQN Runner] Started {profile}")

        # ── Simulation thread ─────────────────────────────────────
        def run():
            global _running, _step, _metrics, _pipeline_state

            action_timer   = 0
            lstm_counter   = 0
            lstm_pred      = [{"north": 0, "south": 0, "east": 0, "west": 0}]
            _fallback_mode = False

            # Point 10: Monitoring
            try:
                from DeepQN.monitoring.monitor import DQNMonitor
                _monitor = DQNMonitor(
                    baseline_avg_wait=626.6,
                    log_dir=os.path.join(BASE_DIR, "DeepQN", "logs"),
                )
                print("[DQN Runner] Monitor active")
            except Exception as _me:
                _monitor = None

            # Point 11: Explainability
            try:
                import yaml as _eyaml
                with open(os.path.join(BASE_DIR, "DeepQN", "configs", "dqn_config.yaml")) as _ef:
                    _ecfg = _eyaml.safe_load(_ef)
                from DeepQN.explainability.explainer import DQNExplainer
                _explainer = DQNExplainer(_ecfg)
                print("[DQN Runner] Explainer active")
            except Exception as _ee:
                _explainer = None

            while _running:
                try:
                    if traci.simulation.getMinExpectedNumber() == 0:
                        break

                    # ── Every 30 steps: full pipeline cycle ───────
                    if action_timer % 30 == 0:

                        # Step 1: Screenshot
                        try:
                            traci.gui.screenshot(
                                _view_id, FRAME_PATH,
                                width=960, height=600
                            )
                            time.sleep(0.1)
                        except Exception:
                            pass

                        # Step 2: YOLO
                        yolo_counts = _run_yolo_on_frame()

                        # Step 3: LSTM history entry
                        vehicle_ids = traci.vehicle.getIDList()
                        dir_c = {"north":0,"south":0,"east":0,"west":0}
                        spds, waits = [], []
                        for v in vehicle_ids:
                            try:
                                a = float(traci.vehicle.getAngle(v))
                                dir_c[_angle_to_direction(a)] += 1
                                spds.append(traci.vehicle.getSpeed(v))
                                waits.append(traci.vehicle.getWaitingTime(v))
                            except Exception:
                                continue

                        avg_speed   = sum(spds)  / len(spds)  if spds  else 0.0
                        avg_waiting = sum(waits) / len(waits) if waits else 0.0

                        _lstm_history.append({
                            "north":       dir_c["north"],
                            "south":       dir_c["south"],
                            "east":        dir_c["east"],
                            "west":        dir_c["west"],
                            "total":       len(vehicle_ids),
                            "avg_speed":   avg_speed,
                            "avg_waiting": avg_waiting,
                        })
                        if len(_lstm_history) > 120:
                            _lstm_history.pop(0)

                        # Step 4: LSTM prediction
                        lstm_counter += 1
                        if lstm_counter % 3 == 0:
                            lstm_pred[0] = _run_lstm_prediction()

                        # Step 5: Traffic metrics
                        avg_wait, n_vehicles, dir_counts, total_co2 = \
                            _get_traffic_metrics()

                        # Step 6: DQN decisions (7 agents)
                        try:
                            observations         = _build_observations(lstm_pred[0])
                            per_junction_actions = _multi_agent.act(
                                observations, eval_mode=True
                            )

                            # Step 7: Apply to traffic lights
                            _apply_dqn_actions(per_junction_actions)

                            # Point 11: Explain decisions every 90 steps
                            if _explainer and action_timer % 90 == 0:
                                try:
                                    exps = _explainer.explain_all(
                                        observations       = observations,
                                        actions            = per_junction_actions,
                                        agents             = _multi_agent,
                                        lstm_pred          = lstm_pred[0],
                                        step               = _step,
                                    )
                                    switched_exps = [e for e in exps if e.action == 1]
                                    if switched_exps:
                                        print(f"[Explainer] Step {_step} — {len(switched_exps)} junctions switched:")
                                        for exp in switched_exps[:3]:
                                            print(f"  {exp.junction_name}: {exp._switch_reason()} (confidence={exp.confidence_label})")
                                except Exception as _ex:
                                    pass

                            # Build display values for frontend
                            n_switched  = sum(
                                1 for a in per_junction_actions.values()
                                if a == 1
                            )
                            action_name = f"{n_switched}/7 junctions switched"
                            q_display   = _build_q_display(
                                per_junction_actions, observations
                            )
                            _fallback_mode = False

                        except Exception as dqn_err:
                            print(f"[DQN] Inference failed: {dqn_err}")
                            n_switched     = 0
                            action_name    = "fallback (fixed timing)"
                            q_display      = {}
                            _fallback_mode = True

                        # Step 8: Log decision + monitor
                        _log_decision(
                            step        = _step,
                            profile     = profile,
                            n_switched  = n_switched,
                            action_name = action_name,
                            avg_wait    = avg_wait,
                            n_vehicles  = n_vehicles,
                            total_co2   = total_co2,
                            yolo_counts = yolo_counts,
                            lstm_pred   = lstm_pred[0],
                        )
                        if _monitor:
                            alerts = _monitor.record(
                                step       = _step,
                                avg_wait   = avg_wait,
                                total_co2  = total_co2,
                                n_switched = n_switched,
                                throughput = n_vehicles,
                            )
                            for alert in alerts:
                                print(f"[Monitor] {alert.level}: {alert.code} — {alert.message}")

                        # Step 9: Update pipeline state for API
                        with _lock:
                            _pipeline_state = {
                                "sumo_vehicles":   n_vehicles,
                                "yolo_counts":     yolo_counts,
                                "lstm_prediction": lstm_pred[0],
                                "dqn_action":      n_switched,
                                "dqn_action_name": action_name,
                                "avg_wait_s":      round(avg_wait, 2),
                                "fallback_mode":   _fallback_mode,
                                "q_values":        q_display,
                            }

                    # ── Every step ────────────────────────────────
                    traci.simulationStep()
                    _step        += 1
                    action_timer += 1

                    time.sleep(STEP_DELAY)

                    # ── Every 10 steps: update metrics ────────────
                    if _step % 10 == 0:
                        vehicle_ids          = traci.vehicle.getIDList()
                        waiting, co2_vals, speeds = [], [], []
                        queued = 0
                        for v in vehicle_ids:
                            try:
                                w = traci.vehicle.getWaitingTime(v)
                                waiting.append(w)
                                co2_vals.append(traci.vehicle.getCO2Emission(v))
                                speeds.append(traci.vehicle.getSpeed(v))
                                if w > 1.0:
                                    queued += 1
                            except Exception:
                                continue

                        with _lock:
                            _metrics = {
                                "step":           _step,
                                "vehicles":       len(vehicle_ids),
                                "queued":         queued,
                                "avg_wait_s":     round(
                                    sum(waiting)/len(waiting) if waiting else 0.0, 2
                                ),
                                "avg_speed":      round(
                                    sum(speeds)/len(speeds) if speeds else 0.0, 2
                                ),
                                "total_co2_mg":   round(
                                    sum(co2_vals) if co2_vals else 0.0, 1
                                ),
                                "current_action": _pipeline_state.get(
                                    "dqn_action_name", "—"
                                ),
                                "profile":        _profile,
                                "lstm_pred":      _pipeline_state.get(
                                    "lstm_prediction", {}
                                ),
                                # q_values for the frontend panel
                                "q_values": _pipeline_state.get("q_values", {}),
                                "fallback_mode": _fallback_mode,
                            }

                    if _step >= max_steps:
                        break

                except Exception as e:
                    print(f"[DQN Runner] Step error: {e}")
                    break

            _running = False
            _close_decision_log()
            if _monitor:
                _monitor.print_summary()
            try:
                traci.close()
            except Exception:
                pass
            print("[DQN Runner] Simulation ended")

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return True

    except Exception as e:
        print(f"[DQN Runner] Start error: {e}")
        _running = False
        _close_decision_log()
        return False


# ── Stop ──────────────────────────────────────────────────────

def stop():
    global _running
    _running = False
    time.sleep(0.5)
    _close_decision_log()
    try:
        traci.close()
    except Exception:
        pass


# ── Screenshot ────────────────────────────────────────────────

def get_screenshot() -> bytes | None:
    if not _running:
        return None
    try:
        traci.gui.screenshot(_view_id, FRAME_PATH)
        with open(FRAME_PATH, "rb") as f:
            return f.read()
    except Exception:
        return None


# ── Status ────────────────────────────────────────────────────

def get_status() -> dict:
    with _lock:
        return {
            "running":  _running,
            "profile":  _profile,
            "step":     _step,
            "metrics":  _metrics,
            "log_path": _decision_log_path,
        }


def get_pipeline_state() -> dict:
    with _lock:
        return dict(_pipeline_state)


def get_log_path() -> str | None:
    return _decision_log_path