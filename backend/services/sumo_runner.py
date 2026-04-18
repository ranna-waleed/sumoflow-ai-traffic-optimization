# backend/services/sumo_runner.py
import os, sys, time, base64
import traci

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise EnvironmentError("SUMO_HOME not set.")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS_DIR   = os.path.join(BASE_DIR, "simulation", "maps")
FRAME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_frame.jpg")

PROFILES = {
    "morning_rush": os.path.join(MAPS_DIR, "config_morning_rush.sumocfg"),
    "evening_rush": os.path.join(MAPS_DIR, "config_evening_rush.sumocfg"),
    "midday":       os.path.join(MAPS_DIR, "config_midday.sumocfg"),
    "night":        os.path.join(MAPS_DIR, "config_night.sumocfg"),
    "custom":       os.path.join(MAPS_DIR, "config_file.sumocfg"),
}

_running       = False
_profile       = None
_step          = 0
_gui           = False
_port          = 8813
_view_id       = None
_last_ss_error = None

# LSTM history buffer
# Now stores 7 features (matching new train_lstm.py FEATURES):
# north, south, east, west, total, avg_speed, avg_waiting
_lstm_history = []


def _angle_to_direction(angle: float) -> str:
    a = float(angle) % 360
    if a < 45 or a >= 315:  return "north"
    if 45  <= a < 135:      return "east"
    if 135 <= a < 225:      return "south"
    return "west"


def _run_yolo_on_frame() -> dict:
    """
    Run YOLO inference on the latest screenshot.
    Returns vehicle type counts from the frame.
    Falls back gracefully if YOLO not available.
    """
    if not os.path.exists(FRAME_PATH):
        return {}
    try:
        from services.yolo_detect import detect_image
        with open(FRAME_PATH, "rb") as f:
            image_bytes = f.read()
        if len(image_bytes) < 1000:   # too small = blank frame
            return {}
        result = detect_image(image_bytes)
        return result.get("class_counts", {})
    except Exception as e:
        print(f"[YOLO] Inference skipped: {e}")
        return {}


def start(profile: str = "morning_rush", gui: bool = True) -> dict:
    global _running, _profile, _step, _gui, _view_id, _lstm_history

    if _running:
        return {"status": "already_running", "profile": _profile, "step": _step}

    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'.")

    config_path = PROFILES[profile]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    try:
        traci.close()
    except Exception:
        pass

    binary   = "sumo-gui" if gui else "sumo"
    sumo_cmd = [
        binary, "-c", config_path,
        "--no-warnings",  
        "--start", "--delay", "50",
        "--window-size", "1920,1080",
        "--window-pos", "0,0",
    ]

    traci.start(sumo_cmd, port=_port)
    _running      = True
    _profile      = profile
    _gui          = gui
    _step         = 0
    _lstm_history = []

    time.sleep(2.0)

    if gui:
        try:
            views    = traci.gui.getIDList()
            _view_id = views[0] if views else "View #0"
            boundary = traci.simulation.getNetBoundary()
            cx = (boundary[0][0] + boundary[1][0]) / 2
            cy = (boundary[0][1] + boundary[1][1]) / 2
            traci.gui.setOffset(_view_id, cx, cy)
            traci.gui.setZoom(_view_id, 1500)
            print(f"[SUMO] View={_view_id}, zoomed to center")
        except Exception as e:
            _view_id = "View #0"
            print(f"[SUMO] GUI setup: {e}")

    return {"status": "started", "profile": profile, "gui": gui}


def stop() -> dict:
    global _running, _profile, _step, _gui, _view_id, _lstm_history
    if not _running:
        return {"status": "not_running"}
    try:
        traci.close()
    except Exception:
        pass
    _running      = False
    _step         = 0
    _gui          = False
    _view_id      = None
    _lstm_history = []
    profile       = _profile
    _profile      = None
    return {"status": "stopped", "profile": profile}


def run_steps_with_screenshot(n: int = 30) -> dict:
    global _step, _last_ss_error

    if not _running:
        raise RuntimeError("Simulation not running.")

    for _ in range(n):
        if traci.simulation.getMinExpectedNumber() == 0:
            break
        traci.simulationStep()
        _step += 1

    # Screenshot
    image_b64 = None
    if _gui and _view_id:
        try:
            traci.gui.screenshot(_view_id, FRAME_PATH, width=960, height=600)
            time.sleep(0.3)
            if os.path.exists(FRAME_PATH) and os.path.getsize(FRAME_PATH) > 0:
                with open(FRAME_PATH, "rb") as f:
                    image_b64 = "data:image/jpeg;base64," + base64.b64encode(
                        f.read()).decode()
                print(f"[screenshot] done length={len(image_b64)}")
        except Exception as e:
            _last_ss_error = str(e)
            print(f"[screenshot] error: {e}")

    state = _get_state_safe()

    # Run YOLO on the frame to get vehicle type counts
    yolo_counts = _run_yolo_on_frame()

    # Build richer LSTM history entry
    # 7 features: north, south, east, west, total, avg_speed, avg_waiting
    # YOLO enriches vehicle counts; TraCI provides speed + waiting
    lstm_entry = {
        "north":       state.get("direction_counts", {}).get("north", 0),
        "south":       state.get("direction_counts", {}).get("south", 0),
        "east":        state.get("direction_counts", {}).get("east",  0),
        "west":        state.get("direction_counts", {}).get("west",  0),
        "total":       state["vehicles"],
        "avg_speed":   state.get("avg_speed", 0.0),
        "avg_waiting": state.get("avg_wait_s", 0.0),   # ← NEW: waiting time
    }

    # If YOLO detected vehicles, cross-check total with TraCI
    # (YOLO gives type breakdown; TraCI gives exact positions)
    if yolo_counts:
        yolo_total = sum(yolo_counts.values())
        # Blend: use TraCI direction counts but log YOLO type counts
        lstm_entry["yolo_total"] = yolo_total   # stored but not used as feature

    _lstm_history.append(lstm_entry)
    if len(_lstm_history) > 120:
        _lstm_history.pop(0)

    return {
        "steps_run":        n,
        "latest":           state,
        "image":            image_b64,
        "yolo_counts":      yolo_counts,
        "lstm_history_len": len(_lstm_history),
        "screenshot_error": _last_ss_error if image_b64 is None else None,
    }


def get_lstm_history() -> list:
    return _lstm_history


def _get_state_safe() -> dict:
    vehicle_ids  = traci.vehicle.getIDList()
    num_vehicles = len(vehicle_ids)
    waiting_times, co2_values, speeds = [], [], []
    type_counts = {}
    dir_counts  = {"north": 0, "south": 0, "east": 0, "west": 0}

    for v in vehicle_ids:
        try:
            waiting_times.append(traci.vehicle.getWaitingTime(v))
            co2_values.append(traci.vehicle.getCO2Emission(v))
            speeds.append(traci.vehicle.getSpeed(v))
            readable = _map_type(traci.vehicle.getTypeID(v))
            type_counts[readable] = type_counts.get(readable, 0) + 1
            angle = traci.vehicle.getAngle(v)
            dir_counts[_angle_to_direction(angle)] += 1
        except traci.exceptions.TraCIException:
            continue

    avg_wait  = round(sum(waiting_times) / len(waiting_times), 2) if waiting_times else 0.0
    max_wait  = round(max(waiting_times), 2)                      if waiting_times else 0.0
    total_co2 = round(sum(co2_values), 2)
    avg_speed = round(sum(speeds) / len(speeds), 2)               if speeds        else 0.0

    tl_states = {}
    for tl in traci.trafficlight.getIDList():
        try:
            tl_states[tl] = traci.trafficlight.getRedYellowGreenState(tl)
        except Exception:
            continue

    return {
        "step":             _step,
        "profile":          _profile,
        "time_s":           round(_step * 1.0, 1),
        "vehicles":         num_vehicles,
        "avg_wait_s":       avg_wait,
        "max_wait_s":       max_wait,
        "total_co2_mg":     total_co2,
        "avg_speed":        avg_speed,
        "type_counts":      type_counts,
        "direction_counts": dir_counts,
        "traffic_lights":   tl_states,
        "simulation_done":  traci.simulation.getMinExpectedNumber() == 0,
    }


def get_state() -> dict:
    if not _running:
        return {"status": "not_running"}
    return _get_state_safe()

def is_running() -> bool:
    return _running

def get_status() -> dict:
    return {
        "running":               _running,
        "profile":               _profile,
        "step":                  _step,
        "gui":                   _gui,
        "view_id":               _view_id,
        "lstm_history_len":      len(_lstm_history),
        "last_screenshot_error": _last_ss_error,
    }

def _map_type(vtype_id: str) -> str:
    v = vtype_id.lower()
    if "bus"        in v: return "bus"
    if "truck"      in v: return "truck"
    if "taxi"       in v: return "taxi"
    if "microbus"   in v: return "microbus"
    if "motorcycle" in v: return "motorcycle"
    if "bicycle"    in v: return "bicycle"
    return "car"