# backend/services/dqn_runner.py
import os, sys, time, threading, csv
from datetime import datetime
import numpy as np

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS_DIR = os.path.join(BASE_DIR, "simulation", "maps")

#  State 
_running  = False
_profile  = None
_step     = 0
_view_id  = "View #0"
_agent    = None
_metrics  = {}
_lock     = threading.Lock()

# Pipeline state : exposed via /api/dqn/cycle/status
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

# Logging 
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
_decision_log_path = None
_decision_log_file = None
_decision_log_writer = None


def _init_decision_log(profile: str):
    """Initialize CSV log file for this simulation session."""
    global _decision_log_path, _decision_log_file, _decision_log_writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _decision_log_path = os.path.join(
        LOGS_DIR, f"dqn_decisions_{profile}_{timestamp}.csv"
    )
    _decision_log_file = open(_decision_log_path, "w", newline="")
    _decision_log_writer = csv.writer(_decision_log_file)
    # Header
    _decision_log_writer.writerow([
        "timestamp", "step", "profile",
        # DQN decision
        "action", "action_name",
        # Q-values for all 4 actions
        "q_value_0", "q_value_1", "q_value_2", "q_value_3",
        # State features
        "north_ratio", "south_ratio", "east_ratio", "west_ratio",
        "north_queue", "south_queue", "east_queue", "west_queue",
        "lstm_north", "lstm_south", "lstm_east", "lstm_west",
        "current_phase", "avg_wait_normalized",
        # Traffic metrics
        "vehicles", "avg_wait_s", "total_co2_mg",
        # YOLO counts
        "yolo_car", "yolo_bus", "yolo_taxi", "yolo_microbus",
        "yolo_truck", "yolo_motorcycle", "yolo_bicycle",
        # LSTM predictions
        "pred_north", "pred_south", "pred_east", "pred_west",
    ])
    _decision_log_file.flush()
    print(f"[DQN Logger] Logging decisions → {_decision_log_path}")


def _log_decision(step, profile, action, q_values, state,
                  avg_wait, n_vehicles, total_co2,
                  yolo_counts, lstm_pred):
    """Log one DQN decision to CSV."""
    global _decision_log_writer, _decision_log_file
    if _decision_log_writer is None:
        return
    try:
        _decision_log_writer.writerow([
            datetime.now().strftime("%H:%M:%S"),
            step, profile,
            # DQN decision
            action, ACTION_NAMES[action],
            # Q-values : shows WHY the action was chosen
            round(float(q_values[0]), 4),
            round(float(q_values[1]), 4),
            round(float(q_values[2]), 4),
            round(float(q_values[3]), 4),
            # State features (14)
            round(float(state[0]),  4),   # north_ratio
            round(float(state[1]),  4),   # south_ratio
            round(float(state[2]),  4),   # east_ratio
            round(float(state[3]),  4),   # west_ratio
            round(float(state[4]),  4),   # north_queue
            round(float(state[5]),  4),   # south_queue
            round(float(state[6]),  4),   # east_queue
            round(float(state[7]),  4),   # west_queue
            round(float(state[8]),  4),   # lstm_north
            round(float(state[9]),  4),   # lstm_south
            round(float(state[10]), 4),   # lstm_east
            round(float(state[11]), 4),   # lstm_west
            round(float(state[12]), 4),   # current_phase
            round(float(state[13]), 4),   # avg_wait_normalized
            # Traffic metrics
            n_vehicles,
            round(avg_wait, 2),
            round(total_co2, 1),
            # YOLO counts
            yolo_counts.get("car",        0),
            yolo_counts.get("bus",        0),
            yolo_counts.get("taxi",       0),
            yolo_counts.get("microbus",   0),
            yolo_counts.get("truck",      0),
            yolo_counts.get("motorcycle", 0),
            yolo_counts.get("bicycle",    0),
            # LSTM predictions
            lstm_pred.get("north", 0),
            lstm_pred.get("south", 0),
            lstm_pred.get("east",  0),
            lstm_pred.get("west",  0),
        ])
        _decision_log_file.flush()
    except Exception as e:
        print(f"[DQN Logger] Log error: {e}")


def _close_decision_log():
    """Close the log file cleanly."""
    global _decision_log_file, _decision_log_writer
    try:
        if _decision_log_file:
            _decision_log_file.close()
            print(f"[DQN Logger] Log saved → {_decision_log_path}")
    except Exception:
        pass
    _decision_log_file   = None
    _decision_log_writer = None


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
    "morning_rush": 3600,
    "evening_rush": 3600,
    "midday":       3600,
    "night":        3600,
}

STEP_DELAY = 0.05

MAIN_TL = "315744796"
SECONDARY_TL_PHASES = {
    0: {"2031414903": 0, "96621100": 2, "2031414899": 0, "6288771431": 0,
        "96621068": 0, "271064234": 0, "315743335": 0, "6288771435": 0},
    1: {"2031414903": 2, "96621100": 2, "2031414899": 2, "6288771431": 2,
        "96621068": 0, "271064234": 0, "315743335": 0, "6288771435": 0},
}
SECONDARY_TL_PHASES[2] = SECONDARY_TL_PHASES[0]
SECONDARY_TL_PHASES[3] = SECONDARY_TL_PHASES[1]

YELLOW_PHASE    = 1
YELLOW_DURATION = 3

ACTION_NAMES = [
    "N-S Green (39s)",
    "E-W Green (39s)",
    "N-S Green (20s)",
    "E-W Green (20s)",
]

_lstm_history = []


def _load_agent():
    global _agent
    if _agent is not None:
        return True
    try:
        sys.path.insert(0, BASE_DIR)
        from dqn.agent import DQNAgent
        _agent = DQNAgent(state_size=14, action_size=4)
        _agent.load()
        _agent.epsilon = 0.0
        print("[DQN Runner] Agent loaded ")
        return True
    except Exception as e:
        print(f"[DQN Runner] Failed to load agent: {e}")
        return False


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


def _run_lstm_prediction() -> dict:
    if len(_lstm_history) < 10:
        return {"north": 0, "south": 0, "east": 0, "west": 0}
    try:
        sys.path.insert(0, BASE_DIR)
        from lstm.predict import predict
        result = predict(_lstm_history)
        return result.get("next_30s",
            {"north": 0, "south": 0, "east": 0, "west": 0}
        )
    except Exception as e:
        print(f"[LSTM] Prediction skipped: {e}")
        return {"north": 0, "south": 0, "east": 0, "west": 0}


def _get_state(current_phase, lstm_pred):
    """Returns (state_array, avg_wait, n_vehicles, dir_counts, total_co2)"""
    vehicle_ids = traci.vehicle.getIDList()
    dir_counts  = {"north": 0, "south": 0, "east": 0, "west": 0}
    dir_queues  = {"north": 0, "south": 0, "east": 0, "west": 0}
    waiting     = []
    co2_vals    = []

    for v in vehicle_ids:
        try:
            a    = float(traci.vehicle.getAngle(v)) % 360
            wait = traci.vehicle.getWaitingTime(v)
            co2_vals.append(traci.vehicle.getCO2Emission(v))

            if a < 45 or a >= 315:    direction = "north"
            elif 45  <= a < 135:      direction = "east"
            elif 135 <= a < 225:      direction = "south"
            else:                     direction = "west"

            dir_counts[direction] += 1
            if wait > 1.0:
                dir_queues[direction] += 1
            waiting.append(wait)
        except Exception:
            continue

    total     = max(len(vehicle_ids), 1)
    avg_wait  = sum(waiting)  / len(waiting)  if waiting  else 0.0
    total_co2 = sum(co2_vals) if co2_vals else 0.0
    max_queue = 50.0
    max_pred  = 100.0

    state = np.array([
        dir_counts["north"] / total,
        dir_counts["south"] / total,
        dir_counts["east"]  / total,
        dir_counts["west"]  / total,
        min(dir_queues["north"] / max_queue, 1.0),
        min(dir_queues["south"] / max_queue, 1.0),
        min(dir_queues["east"]  / max_queue, 1.0),
        min(dir_queues["west"]  / max_queue, 1.0),
        min(lstm_pred.get("north", 0) / max_pred, 1.0),
        min(lstm_pred.get("south", 0) / max_pred, 1.0),
        min(lstm_pred.get("east",  0) / max_pred, 1.0),
        min(lstm_pred.get("west",  0) / max_pred, 1.0),
        current_phase / 3.0,
        min(avg_wait / 300.0, 1.0),
    ], dtype=np.float32)

    # NaN/Inf guard
    # If TraCI returns bad values (sensor glitch, division error)
    # replace NaN/Inf with 0 to prevent silent DQN corruption
    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
        print(f"[DQN] Bad state detected (NaN/Inf) at step {_step} — replacing with zeros")
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)

    return state, avg_wait, len(vehicle_ids), dir_counts, total_co2

def _apply_yellow(tl_ids, phase_counts):
    for tl in tl_ids:
        try:
            n = phase_counts.get(tl, 1)
            if n > YELLOW_PHASE:
                traci.trafficlight.setPhase(tl, YELLOW_PHASE)
        except Exception:
            pass
    for _ in range(YELLOW_DURATION):
        try:
            traci.simulationStep()
        except Exception:
            break


def _apply_action(action, tl_ids, phase_counts, prev_action=None):
    if prev_action is not None and action != prev_action:
        _apply_yellow(tl_ids, phase_counts)

    if MAIN_TL in tl_ids:
        main_phase = 0 if action in [0, 2] else 2
        try:
            n = phase_counts.get(MAIN_TL, 1)
            traci.trafficlight.setPhase(MAIN_TL, main_phase % n)
        except Exception:
            pass

    secondary = SECONDARY_TL_PHASES.get(action, SECONDARY_TL_PHASES[0])
    for tl, phase in secondary.items():
        if tl in tl_ids:
            try:
                n = phase_counts.get(tl, 1)
                traci.trafficlight.setPhase(tl, phase % n)
            except Exception:
                pass


def _angle_to_direction(angle: float) -> str:
    a = float(angle) % 360
    if a < 45 or a >= 315:  return "north"
    if 45  <= a < 135:      return "east"
    if 135 <= a < 225:      return "south"
    return "west"


def start(profile: str) -> bool:
    global _running, _profile, _step, _view_id, _metrics, _lstm_history

    if _running:
        stop()
        time.sleep(2.0)

    if not _load_agent():
        return False

    config    = CONFIG_PATHS.get(profile)
    max_steps = MAX_STEPS_PER_PROFILE.get(profile, 3600)

    if not config:
        return False

    # Initialize decision log for this session
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

        try:
            views    = traci.gui.getIDList()
            _view_id = views[0] if views else "View #0"
            time.sleep(2.0)
            traci.gui.setOffset(_view_id, 3694.5, 1539.5)
            traci.gui.setZoom(_view_id, 1500)
            print(f"[DQN Runner] GUI ready ")
        except Exception as e:
            print(f"[DQN Runner] GUI setup: {e}")

        tl_ids       = list(traci.trafficlight.getIDList())
        phase_counts = {}
        for tl in tl_ids:
            try:
                logics = traci.trafficlight.getAllProgramLogics(tl)
                phase_counts[tl] = len(logics[0].phases)
            except Exception:
                phase_counts[tl] = 1

        current_phase = [0]
        prev_action   = [0]
        lstm_pred     = [{"north": 0, "south": 0, "east": 0, "west": 0}]

        _apply_action(0, tl_ids, phase_counts)
        print(f"[DQN Runner] Started {profile} ")

        def run():
            global _running, _step, _metrics, _pipeline_state

            action_timer = 0
            lstm_counter = 0
            _fallback_mode = False

            while _running:
                try:
                    if traci.simulation.getMinExpectedNumber() == 0:
                        break

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

                        # Step 3: LSTM history
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

                        # Step 5: Build DQN state
                        result = _get_state(current_phase[0], lstm_pred[0])
                        state, avg_wait, n_vehicles, dir_counts, total_co2 = result

                        # Step 6: DQN decides , get Q-values too for logging
                        # Fallback to fixed timing if DQN fails
                        try:
                            action, q_values = _agent.act_with_q(state)
                            _fallback_mode = False
                        except Exception as dqn_err:
                            print(f"[DQN] Inference failed: {dqn_err} — using fallback timing")
                            # Fallback: cycle through phases like fixed timing
                            # 0=N-S Green, 1=E-W Green alternating every 39s
                            action     = 0 if (action_timer // 39) % 2 == 0 else 1
                            q_values   = [0.0, 0.0, 0.0, 0.0]
                            _fallback_mode = True

                        # Step 7: Apply signal
                        _apply_action(
                            action, tl_ids, phase_counts,
                            prev_action=prev_action[0]
                        )
                        prev_action[0]   = action
                        current_phase[0] = action

                        #  LOG THE DECISION 
                        _log_decision(
                            step        = _step,
                            profile     = profile,
                            action      = action,
                            q_values    = q_values,
                            state       = state,
                            avg_wait    = avg_wait,
                            n_vehicles  = n_vehicles,
                            total_co2   = total_co2,
                            yolo_counts = yolo_counts,
                            lstm_pred   = lstm_pred[0],
                        )

                        with _lock:
                            _pipeline_state = {
                                "sumo_vehicles":   n_vehicles,
                                "yolo_counts":     yolo_counts,
                                "lstm_prediction": lstm_pred[0],
                                "dqn_action":      action,
                                "dqn_action_name": ACTION_NAMES[action],
                                "avg_wait_s":      round(avg_wait, 2),
                                "fallback_mode":   _fallback_mode,
                                # Q-values exposed for frontend
                                "q_values": {
                                    "N-S Green (39s)": round(float(q_values[0]), 4),
                                    "E-W Green (39s)": round(float(q_values[1]), 4),
                                    "N-S Green (20s)": round(float(q_values[2]), 4),
                                    "E-W Green (20s)": round(float(q_values[3]), 4),
                                },
                            }

                    traci.simulationStep()
                    _step        += 1
                    action_timer += 1

                    time.sleep(STEP_DELAY)

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
                                "avg_wait_s":     round(sum(waiting)/len(waiting) if waiting else 0.0, 2),
                                "avg_speed":      round(sum(speeds)/len(speeds)   if speeds  else 0.0, 2),
                                "total_co2_mg":   round(sum(co2_vals) if co2_vals else 0.0, 1),
                                "current_action": ACTION_NAMES[current_phase[0]],
                                "profile":        _profile,
                                "lstm_pred":      lstm_pred[0],
                                "q_values":       _pipeline_state.get("q_values", {}),
                            }

                    if _step >= max_steps:
                        break

                except Exception as e:
                    print(f"[DQN Runner] Step error: {e}")
                    break

            _running = False
            _close_decision_log()
            try:
                traci.close()
            except Exception:
                pass
            print("[DQN Runner] Simulation ended ")

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return True

    except Exception as e:
        print(f"[DQN Runner] Start error: {e}")
        _running = False
        _close_decision_log()
        return False


def stop():
    global _running
    _running = False
    time.sleep(0.5)
    _close_decision_log()
    try:
        traci.close()
    except Exception:
        pass


def get_screenshot() -> bytes | None:
    if not _running:
        return None
    try:
        traci.gui.screenshot(_view_id, FRAME_PATH)
        with open(FRAME_PATH, "rb") as f:
            return f.read()
    except Exception:
        return None


def get_status() -> dict:
    with _lock:
        return {
            "running":      _running,
            "profile":      _profile,
            "step":         _step,
            "metrics":      _metrics,
            "log_path":     _decision_log_path,
        }


def get_pipeline_state() -> dict:
    with _lock:
        return dict(_pipeline_state)


def get_log_path() -> str | None:
    """Return path to current decision log file."""
    return _decision_log_path