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

# Tahrir Square center coordinates (from net.xml boundary)
MAP_CENTER_X = 3694.5
MAP_CENTER_Y = 1539.5
MAP_ZOOM     = 1500

_running       = False
_profile       = None
_step          = 0
_gui           = False
_port          = 8813
_view_id       = "View #0"
_last_ss_error = None
_first_frame   = None   # cached first frame for instant display
_lstm_history  = []


def _angle_to_direction(angle: float) -> str:
    a = float(angle) % 360
    if a < 45 or a >= 315: return "north"
    if 45  <= a < 135:     return "east"
    if 135 <= a < 225:     return "south"
    return "west"


def _run_yolo_on_frame() -> dict:
    if not os.path.exists(FRAME_PATH):
        return {}
    try:
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


def _take_screenshot() -> str | None:
    """Take screenshot and return as base64 data URL."""
    global _last_ss_error
    try:
        traci.gui.screenshot(_view_id, FRAME_PATH, width=960, height=600)
        time.sleep(0.15)
        if os.path.exists(FRAME_PATH) and os.path.getsize(FRAME_PATH) > 1000:
            with open(FRAME_PATH, "rb") as f:
                data = f.read()
            _last_ss_error = None
            return "data:image/jpeg;base64," + base64.b64encode(data).decode()
    except Exception as e:
        _last_ss_error = str(e)
    return None


def _setup_gui_view():
    """
    Set up GUI camera and force SUMO to render the map.
    Returns True when the map is visible (non-black frame).
    """
    global _view_id, _first_frame

    # Wait for GUI to be ready
    time.sleep(2.5)

    # Get view ID
    for attempt in range(8):
        try:
            views = traci.gui.getIDList()
            if views:
                _view_id = views[0]
                break
        except Exception:
            pass
        time.sleep(0.5)

    # Set camera to Tahrir Square center
    try:
        traci.gui.setOffset(_view_id, MAP_CENTER_X, MAP_CENTER_Y)
        traci.gui.setZoom(_view_id, MAP_ZOOM)
    except Exception as e:
        print(f"[SUMO] Camera setup: {e}")

    # Force render by stepping simulation and retrying screenshot
    # until we get a non-black frame
    for attempt in range(10):
        try:
            traci.simulationStep()
            time.sleep(0.3)
            frame = _take_screenshot()
            if frame and len(frame) > 5000:  # real image, not black
                _first_frame = frame
                print(f"[SUMO] Map rendered on attempt {attempt + 1}")
                return True
        except Exception as e:
            print(f"[SUMO] Render attempt {attempt + 1}: {e}")
        time.sleep(0.5)

    print("[SUMO] Could not confirm map render — continuing anyway")
    return False


def start(profile: str = "morning_rush", gui: bool = True) -> dict:
    global _running, _profile, _step, _gui, _view_id, _first_frame, _lstm_history

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

    # Always use sumo-gui so user can see the simulation
    sumo_cmd = [
        "sumo-gui", "-c", config_path,
        "--no-warnings",
        "--no-step-log",
        "--start",
        "--delay", "50",
        "--window-size", "1280,720",
        "--error-log", "NUL",
    ]

    traci.start(sumo_cmd, port=_port)
    _running      = True
    _profile      = profile
    _gui          = True
    _step         = 0
    _first_frame  = None
    _lstm_history = []

    # Set up GUI view and get first frame BEFORE returning
    # This is what prevents the black window on frontend
    _setup_gui_view()

    print(f"[SUMO] Started {profile} — GUI ready")
    return {"status": "started", "profile": profile, "gui": True}


def stop() -> dict:
    global _running, _profile, _step, _gui, _view_id, _first_frame, _lstm_history
    if not _running:
        return {"status": "not_running"}
    try:
        traci.close()
    except Exception:
        pass
    _running      = False
    _step         = 0
    _gui          = False
    _first_frame  = None
    _lstm_history = []
    profile       = _profile
    _profile      = None
    return {"status": "stopped", "profile": profile}


def run_steps_with_screenshot(n: int = 30) -> dict:
    global _step, _first_frame

    if not _running:
        raise RuntimeError("Simulation not running.")

    for _ in range(n):
        if traci.simulation.getMinExpectedNumber() == 0:
            break
        traci.simulationStep()
        _step += 1

    # Take screenshot
    image_b64 = _take_screenshot()

    # If screenshot failed but we have first frame, use it
    if not image_b64 and _first_frame:
        image_b64 = _first_frame

    state = _get_state_safe()
    yolo_counts = _run_yolo_on_frame()

    # Update LSTM history
    dir_c = state.get("direction_counts", {})
    _lstm_history.append({
        "north":       dir_c.get("north", 0),
        "south":       dir_c.get("south", 0),
        "east":        dir_c.get("east",  0),
        "west":        dir_c.get("west",  0),
        "total":       state["vehicles"],
        "avg_speed":   state.get("avg_speed", 0.0),
        "avg_waiting": state.get("avg_wait_s", 0.0),
    })
    if len(_lstm_history) > 120:
        _lstm_history.pop(0)

    return {
        "steps_run":        n,
        "latest":           state,
        "image":            image_b64,
        "yolo_counts":      yolo_counts,
        "lstm_history_len": len(_lstm_history),
        "screenshot_error": _last_ss_error if not image_b64 else None,
    }


def get_first_frame() -> str | None:
    """Return cached first frame for immediate display."""
    return _first_frame


def get_lstm_history() -> list:
    return _lstm_history


def _get_state_safe() -> dict:
    vehicle_ids  = traci.vehicle.getIDList()
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

    avg_wait  = round(sum(waiting_times)/len(waiting_times), 2) if waiting_times else 0.0
    max_wait  = round(max(waiting_times), 2)                    if waiting_times else 0.0
    total_co2 = round(sum(co2_values), 2)
    avg_speed = round(sum(speeds)/len(speeds), 2)               if speeds        else 0.0

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
        "vehicles":         len(vehicle_ids),
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
        "first_frame_ready":     _first_frame is not None,
    }


def _map_type(vtype_id: str) -> str:
    v = vtype_id.lower()
    if "bicycle"    in v: return "bicycle"
    if "motorcycle" in v: return "motorcycle"
    if "microbus"   in v: return "microbus"
    if "bus"        in v: return "bus"
    if "truck"      in v: return "truck"
    if "taxi"       in v: return "taxi"
    return "car"