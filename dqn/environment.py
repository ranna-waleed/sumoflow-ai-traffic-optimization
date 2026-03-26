# dqn/environment.py - Simple 6-feature version 
import os, sys
import numpy as np

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise EnvironmentError("SUMO_HOME not set.")

import traci

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(BASE_DIR, "simulation", "maps")

ACTION_PERIOD = 30
MAX_STEPS     = 3600
ACTION_NAMES  = ["N-S Green (39s)", "E-W Green (39s)", "N-S Green (20s)", "E-W Green (20s)"]

MAIN_TL = "315744796"

SECONDARY_TL_PHASES = {
    0: {"2031414903": 0, "96621100": 2, "2031414899": 0, "6288771431": 0,
        "96621068": 0, "271064234": 0, "315743335": 0, "6288771435": 0},
    1: {"2031414903": 2, "96621100": 2, "2031414899": 2, "6288771431": 2,
        "96621068": 0, "271064234": 0, "315743335": 0, "6288771435": 0},
}
SECONDARY_TL_PHASES[2] = SECONDARY_TL_PHASES[0]
SECONDARY_TL_PHASES[3] = SECONDARY_TL_PHASES[1]


class TahrirEnv:
    def __init__(self, profile="morning_rush", gui=False, port=8814, verbose=False):
        self.profile = profile
        self.gui     = gui
        self.port    = port
        self.verbose = verbose

        self.config_paths = {
            "morning_rush": os.path.join(MAPS_DIR, "config_morning_rush.sumocfg"),
            "evening_rush": os.path.join(MAPS_DIR, "config_evening_rush.sumocfg"),
            "midday":       os.path.join(MAPS_DIR, "config_midday.sumocfg"),
            "night":        os.path.join(MAPS_DIR, "config_night.sumocfg"),
        }

        self.state_size  = 6
        self.action_size = 4

        self._step          = 0
        self._current_phase = 0
        self._prev_wait     = 0.0
        self._tl_ids        = []
        self._phase_counts  = {}

        self.episode_waits = []
        self.episode_co2   = []

    def reset(self, profile=None):
        if profile:
            self.profile = profile

        try:
            traci.close()
        except Exception:
            pass

        config = self.config_paths.get(self.profile)
        binary = "sumo-gui" if self.gui else "sumo"

        traci.start([
            binary, "-c", config,
            "--no-warnings", "--no-step-log",
        ], port=self.port)

        self._step          = 0
        self._current_phase = 0
        self._prev_wait     = 0.0
        self.episode_waits  = []
        self.episode_co2    = []
        self._tl_ids        = list(traci.trafficlight.getIDList())

        self._phase_counts = {}
        for tl in self._tl_ids:
            try:
                logics = traci.trafficlight.getAllProgramLogics(tl)
                self._phase_counts[tl] = len(logics[0].phases)
            except Exception:
                self._phase_counts[tl] = 1

        self._apply_action(0)
        return self._get_state()

    def _apply_action(self, action: int):
        self._current_phase = action

        if MAIN_TL in self._tl_ids:
            main_phase = 0 if action in [0, 2] else 2
            try:
                n = self._phase_counts.get(MAIN_TL, 1)
                traci.trafficlight.setPhase(MAIN_TL, main_phase % n)
            except Exception:
                pass

        secondary = SECONDARY_TL_PHASES.get(action, SECONDARY_TL_PHASES[0])
        for tl, phase in secondary.items():
            if tl in self._tl_ids:
                try:
                    n = self._phase_counts.get(tl, 1)
                    traci.trafficlight.setPhase(tl, phase % n)
                except Exception:
                    pass

    def step(self, action: int):
        self._apply_action(action)

        duration = 20 if action in [2, 3] else ACTION_PERIOD

        if self.verbose:
            n_veh = len(traci.vehicle.getIDList())
            print(f"  [DQN] Step {self._step:4d} | "
                  f"Action: {action} ({ACTION_NAMES[action]:<18}) | "
                  f"Vehicles: {n_veh:3d}")

        for _ in range(duration):
            if self._step >= MAX_STEPS:
                break
            if traci.simulation.getMinExpectedNumber() == 0:
                break
            self._apply_action(action)
            traci.simulationStep()
            self._step += 1

        state  = self._get_state()
        reward = self._get_reward()
        done   = (self._step >= MAX_STEPS or
                  traci.simulation.getMinExpectedNumber() == 0)

        vehicle_ids = traci.vehicle.getIDList()
        waiting, co2_vals = [], []
        for v in vehicle_ids:
            try:
                waiting.append(traci.vehicle.getWaitingTime(v))
                co2_vals.append(traci.vehicle.getCO2Emission(v))
            except Exception:
                continue

        avg_wait  = sum(waiting)  / len(waiting)  if waiting  else 0.0
        avg_co2   = sum(co2_vals) / len(co2_vals) if co2_vals else 0.0
        total_co2 = sum(co2_vals) if co2_vals else 0.0

        self.episode_waits.append(avg_wait)
        co2_per_step = total_co2 / max(self._step, 1)
        self.episode_co2.append(co2_per_step)

        info = {
            "step":                     self._step,
            "phase":                    self._current_phase,
            "action_name":              ACTION_NAMES[action],
            "waiting":                  avg_wait,
            "system_mean_waiting_time": avg_wait,
            "total_co2_mg":             total_co2,
            "avg_co2_mg":               avg_co2,
        }

        return state, reward, done, info

    def _get_state(self):
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
            self._current_phase / max(self.action_size - 1, 1),
            min(avg_wait / 120.0, 1.0),
        ], dtype=np.float32)

    def _get_reward(self):
        vehicle_ids = traci.vehicle.getIDList()
        waiting = []
        for v in vehicle_ids:
            try:
                waiting.append(traci.vehicle.getWaitingTime(v))
            except Exception:
                continue
        avg_wait        = sum(waiting) / len(waiting) if waiting else 0.0
        reward          = self._prev_wait - avg_wait
        self._prev_wait = avg_wait
        return reward

    def get_episode_summary(self):
        return {
            "avg_wait_s":   float(np.mean(self.episode_waits))  if self.episode_waits else 0.0,
            "avg_co2_mg":   float(np.mean(self.episode_co2))    if self.episode_co2   else 0.0,
            "total_co2_mg": float(np.sum(self.episode_co2))     if self.episode_co2   else 0.0,
        }

    def close(self):
        try:
            traci.close()
        except Exception:
            pass