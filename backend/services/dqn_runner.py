# backend/services/dqn_runner.py
import os, sys, time, threading
import numpy as np

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS_DIR = os.path.join(BASE_DIR, "simulation", "maps")

#  State 
_running     = False
_profile     = None
_step        = 0
_view_id     = "View #0"
_agent       = None
_metrics     = {}
_lock        = threading.Lock()

CONFIG_PATHS = {
    "morning_rush": os.path.join(MAPS_DIR, "config_morning_rush.sumocfg"),
    "evening_rush": os.path.join(MAPS_DIR, "config_evening_rush.sumocfg"),
    "midday":       os.path.join(MAPS_DIR, "config_midday.sumocfg"),
    "night":        os.path.join(MAPS_DIR, "config_night.sumocfg"),
}

BEGIN_TIMES = {
    "morning_rush": "28800",   # 8:00 AM
    "evening_rush": "57600",   # 4:00 PM
    "midday":       "43200",   # 12:00 PM
    "night":        "79200",   # 10:00 PM
}

MAX_STEPS    = 7200   # 2 hours simulation
STEP_DELAY   = 0.05   # 50ms per step — visible speed

MAIN_TL = "315744796"
SECONDARY_TL_PHASES = {
    0: {"2031414903": 0, "96621100": 2, "2031414899": 0, "6288771431": 0,
        "96621068": 0, "271064234": 0, "315743335": 0, "6288771435": 0},
    1: {"2031414903": 2, "96621100": 2, "2031414899": 2, "6288771431": 2,
        "96621068": 0, "271064234": 0, "315743335": 0, "6288771435": 0},
}
SECONDARY_TL_PHASES[2] = SECONDARY_TL_PHASES[0]
SECONDARY_TL_PHASES[3] = SECONDARY_TL_PHASES[1]

ACTION_NAMES = ["N-S Green (39s)", "E-W Green (39s)", "N-S Green (20s)", "E-W Green (20s)"]


def _load_agent():
    global _agent
    if _agent is not None:
        return True
    try:
        sys.path.insert(0, BASE_DIR)
        from dqn.agent import DQNAgent
        _agent = DQNAgent(state_size=6, action_size=4)
        _agent.load()
        _agent.epsilon = 0.0
        print("[DQN Runner] Agent loaded successfully")
        return True
    except Exception as e:
        print(f"[DQN Runner] Failed to load agent: {e}")
        return False


def _get_state(current_phase):
    vehicle_ids = traci.vehicle.getIDList()
    dir_counts  = {"north": 0, "south": 0, "east": 0, "west": 0}
    waiting     = []
    for v in vehicle_ids:
        try:
            a = float(traci.vehicle.getAngle(v)) % 360
            if a < 45 or a >= 315:   dir_counts["north"] += 1
            elif 45  <= a < 135:     dir_counts["east"]  += 1
            elif 135 <= a < 225:     dir_counts["south"] += 1
            else:                    dir_counts["west"]  += 1
            waiting.append(traci.vehicle.getWaitingTime(v))
        except Exception:
            continue
    total    = max(len(vehicle_ids), 1)
    avg_wait = sum(waiting) / len(waiting) if waiting else 0.0
    return np.array([
        dir_counts["north"] / total,
        dir_counts["south"] / total,
        dir_counts["east"]  / total,
        dir_counts["west"]  / total,
        current_phase / 3.0,
        min(avg_wait / 120.0, 1.0),
    ], dtype=np.float32)


def _apply_action(action, tl_ids, phase_counts):
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


def start(profile: str) -> bool:
    global _running, _profile, _step, _view_id, _metrics

    if _running:
        stop()
        time.sleep(2.0)

    if not _load_agent():
        return False

    config = CONFIG_PATHS.get(profile)
    if not config:
        return False

    try:
        traci.start([
            "sumo-gui", "-c", config,
            "--no-warnings", "--no-step-log",
            "--begin", BEGIN_TIMES.get(profile, "0"),  # start at correct time
            "--start",        # ← auto-starts simulation without clicking play
            "--delay", "50",
        ], port=8813)

        _running = True
        _profile = profile
        _step    = 0
        _metrics = {}

        # Setup GUI zoom
        try:
            views    = traci.gui.getIDList()
            _view_id = views[0] if views else "View #0"
            time.sleep(2.0)
            traci.gui.setOffset(_view_id, 3694.5, 1539.5)
            traci.gui.setZoom(_view_id, 1500)
            print(f"[DQN Runner] GUI ready ")
        except Exception as e:
            print(f"[DQN Runner] GUI setup: {e}")

        # Discover TLs
        tl_ids       = list(traci.trafficlight.getIDList())
        phase_counts = {}
        for tl in tl_ids:
            try:
                logics = traci.trafficlight.getAllProgramLogics(tl)
                phase_counts[tl] = len(logics[0].phases)
            except Exception:
                phase_counts[tl] = 1

        current_phase = [0]   # mutable for thread
        _apply_action(0, tl_ids, phase_counts)

        print(f"[DQN Runner] Started {profile} ")

        def run():
            global _running, _step, _metrics
            action_timer = 0

            while _running:
                try:
                    if traci.simulation.getMinExpectedNumber() == 0:
                        break

                    # DQN decides every 30 steps
                    if action_timer % 30 == 0:
                        state  = _get_state(current_phase[0])
                        action = _agent.act(state)
                        _apply_action(action, tl_ids, phase_counts)
                        current_phase[0] = action

                    traci.simulationStep()
                    _step        += 1
                    action_timer += 1

                    # Slow down so SUMO-GUI shows vehicles moving
                    time.sleep(STEP_DELAY)

                    # Update metrics every 10 steps
                    if _step % 10 == 0:
                        vehicle_ids = traci.vehicle.getIDList()
                        waiting, co2_vals, speeds = [], [], []
                        for v in vehicle_ids:
                            try:
                                waiting.append(traci.vehicle.getWaitingTime(v))
                                co2_vals.append(traci.vehicle.getCO2Emission(v))
                                speeds.append(traci.vehicle.getSpeed(v))
                            except Exception:
                                continue

                        with _lock:
                            _metrics = {
                                "step":           _step,
                                "vehicles":       len(vehicle_ids),
                                "avg_wait_s":     round(sum(waiting)  / len(waiting)  if waiting  else 0.0, 2),
                                "avg_speed":      round(sum(speeds)   / len(speeds)   if speeds   else 0.0, 2),
                                "total_co2_mg":   round(sum(co2_vals) if co2_vals else 0.0, 1),
                                "current_action": ACTION_NAMES[current_phase[0]],
                                "profile":        _profile,
                            }

                    # Stop after MAX_STEPS
                    if _step >= MAX_STEPS:
                        break

                except Exception as e:
                    print(f"[DQN Runner] Step error: {e}")
                    break

            _running = False
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
        return False


def stop():
    global _running
    _running = False
    time.sleep(0.5)
    try:
        traci.close()
    except Exception:
        pass


def get_screenshot() -> bytes | None:
    if not _running:
        return None
    try:
        import tempfile
        path = os.path.join(tempfile.gettempdir(), "dqn_frame.png")
        traci.gui.screenshot(_view_id, path)
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def get_status() -> dict:
    with _lock:
        return {
            "running": _running,
            "profile": _profile,
            "step":    _step,
            "metrics": _metrics,
        }