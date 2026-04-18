# dqn/environment.py
import os, sys
import numpy as np

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise EnvironmentError("SUMO_HOME not set.")

import traci

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(BASE_DIR, "simulation", "maps")

ACTION_PERIOD   = 30
YELLOW_DURATION = 3

MAX_STEPS_PER_PROFILE = {
    "morning_rush": 10800,
    "evening_rush": 18000,
    "midday":       10800,
    "night":        7200,
}

ACTION_NAMES = [
    "N-S Green (39s)",
    "E-W Green (39s)",
    "N-S Green (20s)",
    "E-W Green (20s)",
]

MAIN_TL = "315744796"

SECONDARY_TL_PHASES = {
    0: {"2031414903": 0, "96621100": 2, "2031414899": 0, "6288771431": 0,
        "96621068": 0, "271064234": 0, "315743335": 0, "6288771435": 0},
    1: {"2031414903": 2, "96621100": 2, "2031414899": 2, "6288771431": 2,
        "96621068": 0, "271064234": 0, "315743335": 0, "6288771435": 0},
}
SECONDARY_TL_PHASES[2] = SECONDARY_TL_PHASES[0]
SECONDARY_TL_PHASES[3] = SECONDARY_TL_PHASES[1]

YELLOW_PHASE = 1
MAX_PRED     = 100.0   # normalization cap for LSTM predictions


class TahrirEnv:
    def __init__(self, profile="morning_rush", gui=False, port=8814, verbose=False):
        self.profile  = profile
        self.gui      = gui
        self.port     = port
        self.verbose  = verbose

        self.config_paths = {
            "morning_rush": os.path.join(MAPS_DIR, "config_morning_rush.sumocfg"),
            "evening_rush": os.path.join(MAPS_DIR, "config_evening_rush.sumocfg"),
            "midday":       os.path.join(MAPS_DIR, "config_midday.sumocfg"),
            "night":        os.path.join(MAPS_DIR, "config_night.sumocfg"),
        }

        # State: 14 features
        # [N_ratio, S_ratio, E_ratio, W_ratio,        (4) live direction ratios
        #  N_queue, S_queue, E_queue, W_queue,         (4) queue lengths
        #  N_pred,  S_pred,  E_pred,  W_pred,          (4) LSTM predictions
        #  current_phase_norm, avg_wait_norm]           (2) phase + wait
        self.state_size  = 14
        self.action_size = 4

        self._step          = 0
        self._max_steps     = MAX_STEPS_PER_PROFILE.get(profile, 7200)
        self._current_phase = 0
        self._prev_wait     = 0.0
        self._tl_ids        = []
        self._phase_counts  = {}

        self.episode_waits = []
        self.episode_co2   = []

        # LSTM predictions — injected externally from dqn_runner
        self._lstm_pred = {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0}

    def set_lstm_prediction(self, pred: dict):
        """
        Called by dqn_runner after each LSTM inference.
        pred = {"north": int, "south": int, "east": int, "west": int}
        """
        self._lstm_pred = {
            "north": float(pred.get("north", 0)),
            "south": float(pred.get("south", 0)),
            "east":  float(pred.get("east",  0)),
            "west":  float(pred.get("west",  0)),
        }

    def reset(self, profile=None):
        if profile:
            self.profile    = profile
            self._max_steps = MAX_STEPS_PER_PROFILE.get(profile, 7200)

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
        self._lstm_pred     = {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0}
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

    def _apply_yellow(self):
        """Yellow transition between phases — realistic signal behaviour."""
        for tl in self._tl_ids:
            try:
                n = self._phase_counts.get(tl, 1)
                if n > YELLOW_PHASE:
                    traci.trafficlight.setPhase(tl, YELLOW_PHASE)
            except Exception:
                pass
        for _ in range(YELLOW_DURATION):
            if traci.simulation.getMinExpectedNumber() == 0:
                break
            traci.simulationStep()
            self._step += 1

    def _apply_action(self, action: int):
        if action != self._current_phase:
            self._apply_yellow()

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
                  f"Vehicles: {n_veh:3d} | "
                  f"LSTM_N={self._lstm_pred['north']:.0f}")

        for _ in range(duration):
            if self._step >= self._max_steps:
                break
            if traci.simulation.getMinExpectedNumber() == 0:
                break
            traci.simulationStep()
            self._step += 1

        state  = self._get_state()
        reward = self._get_reward()
        done   = (self._step >= self._max_steps or
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
        self.episode_co2.append(total_co2 / max(self._step, 1))

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
        dir_queues  = {"north": 0, "south": 0, "east": 0, "west": 0}
        waiting     = []

        for v in vehicle_ids:
            try:
                a    = float(traci.vehicle.getAngle(v)) % 360
                wait = traci.vehicle.getWaitingTime(v)

                if a < 45 or a >= 315:   direction = "north"
                elif 45  <= a < 135:     direction = "east"
                elif 135 <= a < 225:     direction = "south"
                else:                    direction = "west"

                dir_counts[direction] += 1
                if wait > 1.0:
                    dir_queues[direction] += 1
                waiting.append(wait)
            except Exception:
                continue

        total     = max(len(vehicle_ids), 1)
        avg_wait  = sum(waiting) / len(waiting) if waiting else 0.0
        max_queue = 50.0

        return np.array([
            # Live ratios (4)
            dir_counts["north"] / total,
            dir_counts["south"] / total,
            dir_counts["east"]  / total,
            dir_counts["west"]  / total,
            # Queue lengths normalized (4)
            min(dir_queues["north"] / max_queue, 1.0),
            min(dir_queues["south"] / max_queue, 1.0),
            min(dir_queues["east"]  / max_queue, 1.0),
            min(dir_queues["west"]  / max_queue, 1.0),
            # LSTM predictions normalized (4)
            min(self._lstm_pred["north"] / MAX_PRED, 1.0),
            min(self._lstm_pred["south"] / MAX_PRED, 1.0),
            min(self._lstm_pred["east"]  / MAX_PRED, 1.0),
            min(self._lstm_pred["west"]  / MAX_PRED, 1.0),
            # Phase + wait (2)
            self._current_phase / max(self.action_size - 1, 1),
            min(avg_wait / 300.0, 1.0),
        ], dtype=np.float32)

    def _get_reward(self):
        vehicle_ids = traci.vehicle.getIDList()
        waiting     = []
        queued      = 0
        for v in vehicle_ids:
            try:
                w = traci.vehicle.getWaitingTime(v)
                waiting.append(w)
                if w > 1.0:
                    queued += 1
            except Exception:
                continue

        avg_wait        = sum(waiting) / len(waiting) if waiting else 0.0
        wait_reward     = self._prev_wait - avg_wait
        queue_penalty   = -0.01 * queued
        reward          = wait_reward + queue_penalty
        self._prev_wait = avg_wait
        return reward

    def get_episode_summary(self):
        return {
            "avg_wait_s":   float(np.mean(self.episode_waits)) if self.episode_waits else 0.0,
            "avg_co2_mg":   float(np.mean(self.episode_co2))   if self.episode_co2   else 0.0,
            "total_co2_mg": float(np.sum(self.episode_co2))    if self.episode_co2   else 0.0,
        }

    def close(self):
        try:
            traci.close()
        except Exception:
            pass