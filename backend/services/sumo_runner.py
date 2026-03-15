# backend/services/sumo_runner.py
import os
import sys
import time
import base64
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
    "custom":       os.path.join(MAPS_DIR, "config_file.sumocfg"),   # ← custom traffic
}

_running       = False
_profile       = None
_step          = 0
_gui           = False
_port          = 8813
_view_id       = None
_last_ss_error = None


def start(profile: str = "morning_rush", gui: bool = True) -> dict:
    global _running, _profile, _step, _gui, _view_id

    if _running:
        return {"status": "already_running", "profile": _profile, "step": _step}

    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Choose from: {list(PROFILES.keys())}")

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
        "--no-warnings", "true",
        "--start",
        "--delay", "50",
        "--window-size", "1920,1080",
        "--window-pos", "0,0",
    ]

    traci.start(sumo_cmd, port=_port)
    _running = True
    _profile = profile
    _gui     = gui
    _step    = 0

    time.sleep(2.0)

    if gui:
        try:
            views    = traci.gui.getIDList()
            _view_id = views[0] if views else "View #0"
            print(f"[SUMO] Available views: {views}")
            print(f"[SUMO] Using view: {_view_id}")

            traci.gui.setSize(_view_id, 1920, 1080)

            boundary = traci.simulation.getNetBoundary()
            min_x, min_y = boundary[0]
            max_x, max_y = boundary[1]
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            traci.gui.setOffset(_view_id, center_x, center_y)
            traci.gui.setZoom(_view_id, 1500)

            print(f"[SUMO] Auto-zoomed: center=({center_x:.1f},{center_y:.1f}) zoom=1500")

        except Exception as e:
            _view_id = "View #0"
            print(f"[SUMO] GUI setup failed: {e}")

    return {"status": "started", "profile": profile, "gui": gui, "view_id": _view_id}


def stop() -> dict:
    global _running, _profile, _step, _gui, _view_id
    if not _running:
        return {"status": "not_running"}
    try:
        traci.close()
    except Exception:
        pass
    _running  = False
    _step     = 0
    _gui      = False
    _view_id  = None
    profile   = _profile
    _profile  = None
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

    image_b64 = None
    if _gui and _view_id:
        try:
            traci.gui.screenshot(_view_id, FRAME_PATH, width=960, height=600)
            time.sleep(0.3)
            if os.path.exists(FRAME_PATH):
                size = os.path.getsize(FRAME_PATH)
                if size > 0:
                    with open(FRAME_PATH, "rb") as f:
                        image_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
                    print(f"[screenshot] done: length={len(image_b64)}")
                else:
                    _last_ss_error = "File empty"
            else:
                _last_ss_error = "File not written"
        except Exception as e:
            _last_ss_error = str(e)
            print(f"[screenshot] failed: {e}")

    state = _get_state_safe()

    return {
        "steps_run":        n,
        "latest":           state,
        "image":            image_b64,
        "screenshot_error": _last_ss_error if image_b64 is None else None,
    }


def _get_state_safe() -> dict:
    vehicle_ids  = traci.vehicle.getIDList()
    num_vehicles = len(vehicle_ids)
    waiting_times, co2_values, type_counts = [], [], {}

    for v in vehicle_ids:
        try:
            waiting_times.append(traci.vehicle.getWaitingTime(v))
            co2_values.append(traci.vehicle.getCO2Emission(v))
            readable = _map_type(traci.vehicle.getTypeID(v))
            type_counts[readable] = type_counts.get(readable, 0) + 1
        except traci.exceptions.TraCIException:
            continue

    avg_wait  = round(sum(waiting_times) / len(waiting_times), 2) if waiting_times else 0.0
    max_wait  = round(max(waiting_times), 2) if waiting_times else 0.0
    total_co2 = round(sum(co2_values), 2)

    tl_states = {}
    for tl in traci.trafficlight.getIDList():
        try:
            tl_states[tl] = traci.trafficlight.getRedYellowGreenState(tl)
        except Exception:
            continue

    return {
        "step":            _step,
        "profile":         _profile,
        "time_s":          round(_step * 1.0, 1),
        "vehicles":        num_vehicles,
        "avg_wait_s":      avg_wait,
        "max_wait_s":      max_wait,
        "total_co2_mg":    total_co2,
        "type_counts":     type_counts,
        "traffic_lights":  tl_states,
        "simulation_done": traci.simulation.getMinExpectedNumber() == 0,
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