"""
dqn/env/sumo_env.py
OpenAI-Gym-compatible SUMO environment for the SUMOFlow AI DQN controller.

Architecture:
- 7 independent DQN agents, one per controllable TLS junction.
- Each agent sees a LOCAL state (its own incoming lanes) + GLOBAL state
  (BiLSTM flow predictions + time-of-day).  All states are zero-padded to
  ``state_dim = 37`` so agents can share the same Q-network architecture.
- Actions are binary: 0 = keep current phase, 1 = switch to next green phase.
- Yellow transitions are managed internally; the agent is blocked from
  switching again until the yellow clears AND min_green seconds have elapsed.

Usage:
    env = SumoEnv(config, profile="morning_rush")
    obs = env.reset()               # dict: {tls_id: np.ndarray(37,)}
    while True:
        actions = {tid: agent.act(obs[tid]) for tid, agent in agents.items()}
        obs, rewards, done, info = env.step(actions)
        if done: break
    env.close()
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# TraCI import, SUMO_HOME must be set in the environment
try:
    import traci
    import traci.constants as tc
except ImportError:
    raise ImportError(
        "TraCI not found. Set SUMO_HOME and add "
        "$SUMO_HOME/tools to PYTHONPATH."
    )

from DeepQN.env.reward import RewardFunction, build_reward_function
from DeepQN.integration.bilstm_adapter import BiLSTMAdapter

logger = logging.getLogger(__name__)


#  Phase controller per junction 

class _PhaseController:
    """
    Manages green/yellow/red phase transitions for a single TLS junction.

    Rules:
    - Action 1 (switch) is only honoured after ``min_green`` seconds in
      the current green phase.
    - Switching triggers a yellow phase for ``yellow_duration`` SUMO steps
      before advancing to the next green phase.
    - Action 0 (keep) simply extends the current phase.
    """

    def __init__(self, tls_id: str, cfg: dict, step_length: float = 1.0):
        self.tls_id       = tls_id
        self.cfg          = cfg
        self.step_length  = step_length

        self.green_phases: List[int]      = cfg["green_phases"]
        self.green_to_yellow: Dict[int, int] = {
            int(k): v for k, v in cfg.get("green_to_yellow", {}).items()
        }
        self.yellow_to_next: Dict[int, int] = {
            int(k): v for k, v in cfg.get("yellow_to_next", {}).items()
        }
        self.yellow_duration: int = cfg.get("yellow_duration", 3)
        self.min_green:       int = cfg.get("min_green", 15)
        self.max_green:       int = cfg.get("max_green", 90)
        self.red_hold_phase: Optional[int] = cfg.get("red_hold_phase")
        self.red_to_green:   Optional[int] = cfg.get("red_to_green_phase")

        # Runtime state
        self.current_green_phase: int  = self.green_phases[0]
        self.phase_elapsed_steps: int  = 0   # steps in current green phase
        self.in_yellow:           bool = False
        self.yellow_elapsed:      int  = 0
        self.in_red_hold:         bool = False
        self.forced_switch:       bool = False   # max_green exceeded

    # Seconds helpers 

    @property
    def phase_elapsed_s(self) -> float:
        return self.phase_elapsed_steps * self.step_length

    # Core logic 

    def apply_action(self, action: int) -> None:
        """
        Call ONCE per decision interval.
        action: 0 = keep, 1 = request switch.
        """
        if self.in_yellow:
            return  # mid-transition , ignore action

        switch_requested = (action == 1)
        # Enforce max-green override
        if self.phase_elapsed_s >= self.max_green:
            switch_requested = True
            self.forced_switch = True

        if switch_requested and self.phase_elapsed_s >= self.min_green:
            self._start_yellow_transition()

    def tick(self) -> None:
        """
        Call ONCE per SUMO simulation step.
        Updates internal counters and advances yellow transitions.
        """
        if self.in_yellow:
            self.yellow_elapsed += 1
            if self.yellow_elapsed >= self.yellow_duration:
                self._finish_transition()
        elif self.in_red_hold:
            self.phase_elapsed_steps += 1
        else:
            self.phase_elapsed_steps += 1

    def _start_yellow_transition(self) -> None:
        """Begin yellow phase for current green."""
        yellow_idx = self.green_to_yellow.get(self.current_green_phase)
        if yellow_idx is None:
            # No yellow defined , switch directly (e.g., red_hold -> green)
            self._finish_transition()
            return
        traci.trafficlight.setPhase(self.tls_id, yellow_idx)
        self.in_yellow      = True
        self.yellow_elapsed = 0
        self.phase_elapsed_steps = 0
        logger.debug("%s  yellow start (phase %d)", self.tls_id, yellow_idx)

    def _finish_transition(self) -> None:
        """Advance to next green (or red-hold) phase after yellow."""
        if self.in_yellow:
            prev_yellow = self.green_to_yellow.get(self.current_green_phase)
            if prev_yellow is not None:
                next_green = self.yellow_to_next.get(prev_yellow)
            else:
                next_green = None

            if next_green is not None:
                self.current_green_phase = next_green
            elif self.red_hold_phase is not None:
                # e.g. 96621100 cycles green → yellow → red_hold
                self.current_green_phase = self.red_hold_phase
                self.in_red_hold = True

        else:
            # Coming from red_hold: go directly to green
            if self.red_to_green is not None:
                self.current_green_phase = self.red_to_green
                self.in_red_hold = False

        traci.trafficlight.setPhase(self.tls_id, self.current_green_phase)
        self.in_yellow           = False
        self.yellow_elapsed      = 0
        self.phase_elapsed_steps = 0
        self.forced_switch       = False
        logger.debug("%s  switched to phase %d", self.tls_id, self.current_green_phase)

    def phase_index_norm(self) -> float:
        """Phase index normalised to [0, 1]."""
        n = max(len(self.green_phases), 1)
        try:
            idx = self.green_phases.index(self.current_green_phase)
        except ValueError:
            idx = 0
        return idx / n

    def phase_elapsed_norm(self, max_s: float = 120.0) -> float:
        return min(self.phase_elapsed_s / max_s, 1.0)


#  BiLSTM proxy 

class _BiLSTMProxy:
    """
    Provides [N, S, E, W] flow predictions normalised to [0, 1].

    If ``use_real_model=True``, loads the actual BiLSTM .pt weights.
    Otherwise falls back to real-time TraCI vehicle counts as proxy
    (sufficient during training; replace with real model for production).
    """

    DIRECTION_EDGES = {
        "north": ["690516091#0", "690516091#1"],
        "south": ["10873191#3",  "10873191#4",  "10873191#5"],
        "east":  ["28718647#1",  "28718647#2",  "28718647#3"],
        "west":  ["50211834#0",  "50211834#1",  "192591565#0"],
    }

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.use_real = cfg.get("use_real_model", False)
        self.max_flow = cfg.get("max_flow_per_edge", 50)
        self._model = None

        if self.use_real:
            self._load_model(cfg.get("model_path", ""))

    def _load_model(self, path: str):
        try:
            import torch
            self._model = torch.load(path, map_location="cpu")
            self._model.eval()
            logger.info("BiLSTM model loaded from %s", path)
        except Exception as exc:
            logger.warning("Could not load BiLSTM model (%s). Using proxy.", exc)
            self.use_real = False

    def predict(self) -> np.ndarray:
        """Returns np.ndarray shape (4,): [N, S, E, W]."""
        if self.use_real and self._model is not None:
            return self._predict_real()
        return self._predict_proxy()

    def _predict_proxy(self) -> np.ndarray:
        flows = []
        for direction in ("north", "south", "east", "west"):
            edges = self.DIRECTION_EDGES[direction]
            total = 0
            for e in edges:
                try:
                    total += traci.edge.getLastStepVehicleNumber(e)
                except traci.TraCIException:
                    pass
            flows.append(min(total / (self.max_flow * len(edges)), 1.0))
        return np.array(flows, dtype=np.float32)

    def _predict_real(self) -> np.ndarray:
        # TODO: feed real sensor / camera input through actual BiLSTM
        return self._predict_proxy()


#  Main environment 

class SumoEnv:
    """
    Multi-agent SUMO environment for DQN traffic signal control.

    Observation space (per junction): Box(37,) in [0, 1]
    Action space (per junction):      Discrete(2)  — {keep=0, switch=1}
    """

    # Junctions deliberately skipped (always-green)
    SKIP_TLS = {"6288771435", "96621068"}

    def __init__(self, config: dict, profile: str = "morning_rush", port: int = 8813):
        """
        Parameters:
        config  : full dqn_config.yaml loaded as dict
        profile : one of morning_rush | midday | evening_rush | night
        port    : TraCI TCP port (increment when running parallel envs)
        """
        self.config         = config
        self.profile        = profile
        self.port           = port
        self.step_length: float = config["simulation"].get("step_length", 1.0)
        self.decision_int:  int = config["simulation"].get("decision_interval", 10)

        sim_cfg             = config["simulation"]
        self.sumo_binary    = sim_cfg.get("sumo_binary", "sumo")
        self.net_file       = sim_cfg["net_file"]
        self.add_file       = sim_cfg.get("add_file", "")

        # Profile config
        profile_map         = config["profiles"]
        if profile not in profile_map:
            raise ValueError(f"Unknown profile '{profile}'. Valid: {list(profile_map)}")
        self.profile_cfg    = profile_map[profile]
        self.sim_begin: int = self.profile_cfg["begin"]
        self.sim_end:   int = self.profile_cfg["end"]

        # TLS configuration
        tls_yaml: dict = config["tls_junctions"]
        self.tls_ids: List[str] = [
            tid for tid in tls_yaml if tid not in self.SKIP_TLS
        ]

        # Phase controllers
        self._phase_controllers: Dict[str, _PhaseController] = {
            tid: _PhaseController(tid, tls_yaml[tid], self.step_length)
            for tid in self.tls_ids
        }

        # State config
        st_cfg = config["state"]
        self.state_dim:  int   = st_cfg.get("state_dim", 37)
        self.max_lanes:  int   = st_cfg.get("max_lanes", 10)
        self.fperlane:   int   = st_cfg.get("features_per_lane", 3)
        self.max_queue: float  = st_cfg.get("max_queue",   50.0)
        self.max_wait:  float  = st_cfg.get("max_wait",    300.0)
        self.max_elapsed: float = st_cfg.get("max_phase_elapsed", 120.0)

        # Per-junction incoming lanes (from config)
        self._lanes: Dict[str, List[str]] = {
            tid: tls_yaml[tid].get("incoming_lanes", [])
            for tid in self.tls_ids
        }

        # BiLSTM : uses detection/lstm/predict.py when model files are present
        self._bilstm = BiLSTMAdapter(config)

        # Reward function
        self._reward_fn = build_reward_function(tls_yaml, config.get("reward", {}))

        # Runtime state
        self._connected:  bool  = False
        self._step_count: int   = 0
        self._episode_done: bool = False

    #  Gym-like API 

    def reset(self) -> Dict[str, np.ndarray]:
        """Start (or restart) SUMO and return the initial observations."""
        if self._connected:
            self.close()
        self._launch_sumo()
        self._connected = True
        self._step_count = 0
        self._episode_done = False

        # Reset phase controllers to SUMO's current state
        for tid, ctrl in self._phase_controllers.items():
            ctrl.phase_elapsed_steps = 0
            ctrl.in_yellow = False
            ctrl.yellow_elapsed = 0
            ctrl.current_green_phase = ctrl.green_phases[0]
            try:
                traci.trafficlight.setPhase(tid, ctrl.current_green_phase)
            except traci.TraCIException as exc:
                logger.warning("Could not set initial phase for %s: %s", tid, exc)

        self._reward_fn.reset()
        self._bilstm.reset()
        self._bilstm.load()       # no-op if already loaded; safe to call each episode
        return self._get_all_states()

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, dict]:
        """
        Apply actions to all TLS, advance ``decision_interval`` SUMO steps,
        return (next_obs, rewards, done, info).

        Parameters:
        actions : {tls_id: 0_or_1}
        """
        if not self._connected:
            raise RuntimeError("Call reset() before step().")

        #  1. Apply requested phase switches 
        for tid, action in actions.items():
            if tid in self._phase_controllers:
                self._phase_controllers[tid].apply_action(action)

        #  2. Advance simulation step-by-step 
        step_arrived = 0
        for _ in range(self.decision_int):
            traci.simulationStep()
            self._step_count += 1
            # Accumulate arrivals every SUMO step (not just the last one)
            step_arrived += traci.simulation.getArrivedNumber()
            # Tick every phase controller once per SUMO step
            for ctrl in self._phase_controllers.values():
                ctrl.tick()
            # Update BiLSTM history buffer every step
            self._bilstm.tick()
            # Exit early if simulation ends mid-interval
            if traci.simulation.getMinExpectedNumber() == 0:
                break

        #  3. Collect reward 
        global_reward = self._reward_fn.compute()
        rewards = {tid: global_reward for tid in self.tls_ids}

        # 4. Collect next observations 
        next_obs = self._get_all_states()

        # 5. Termination check 
        sim_time = traci.simulation.getTime()
        done = (
            traci.simulation.getMinExpectedNumber() == 0
            or sim_time >= self.sim_end
        )

        info = {
            "sim_time":     sim_time,
            "step_count":   self._step_count,
            "step_arrived": step_arrived,   # arrivals this decision interval
            "profile":      self.profile,
            **self._reward_fn.get_metrics(),
        }

        return next_obs, rewards, done, info

    def close(self):
        """Close the TraCI connection and terminate SUMO."""
        if self._connected:
            try:
                traci.close()
            except Exception:
                pass
            self._connected = False
            logger.info("SUMO environment closed.")

    #  State construction 

    def _get_all_states(self) -> Dict[str, np.ndarray]:
        """Build the state vector for every controlled TLS junction."""
        global_feats = self._global_features()
        return {
            tid: self._junction_state(tid, global_feats)
            for tid in self.tls_ids
        }

    def _global_features(self) -> np.ndarray:
        """
        Returns a (5,) vector:
          [bilstm_N, bilstm_S, bilstm_E, bilstm_W, time_of_day]
        """
        bilstm = self._bilstm.predict()   # (4,)
        sim_time = traci.simulation.getTime()
        tod = (sim_time - self.sim_begin) / max(self.sim_end - self.sim_begin, 1)
        tod = float(np.clip(tod, 0.0, 1.0))
        return np.concatenate([bilstm, [tod]]).astype(np.float32)

    def _junction_state(self, tid: str, global_feats: np.ndarray) -> np.ndarray:
        """
        Builds the (state_dim,) observation for a single junction:
          [lane_0_halt, lane_0_wait, lane_0_occ, ...,  <- max_lanes * 3 (zero-padded)
           phase_idx_norm, phase_elapsed_norm,           <- 2
           bilstm_N, bilstm_S, bilstm_E, bilstm_W, tod] <- 5
        Total = 37
        """
        lanes      = self._lanes[tid]
        ctrl       = self._phase_controllers[tid]
        lane_feats = np.zeros(self.max_lanes * self.fperlane, dtype=np.float32)

        for i, lane in enumerate(lanes[:self.max_lanes]):
            try:
                halt = traci.lane.getLastStepHaltingNumber(lane) / self.max_queue
                wait = traci.lane.getWaitingTime(lane)            / self.max_wait
                occ  = traci.lane.getLastStepOccupancy(lane)
            except traci.TraCIException:
                halt, wait, occ = 0.0, 0.0, 0.0
            base = i * self.fperlane
            lane_feats[base]     = float(np.clip(halt, 0.0, 1.0))
            lane_feats[base + 1] = float(np.clip(wait, 0.0, 1.0))
            lane_feats[base + 2] = float(np.clip(occ,  0.0, 1.0))

        junc_feats = np.array([
            ctrl.phase_index_norm(),
            ctrl.phase_elapsed_norm(self.max_elapsed),
        ], dtype=np.float32)

        state = np.concatenate([lane_feats, junc_feats, global_feats])
        assert state.shape[0] == self.state_dim, (
            f"State dim mismatch for {tid}: {state.shape[0]} != {self.state_dim}"
        )
        return state

    #  SUMO launch 

    def _launch_sumo(self):
        profile_config = self.profile_cfg["config"]
        cmd = [
            self.sumo_binary,
            "-c", profile_config,
            "--start",
            "--quit-on-end",
            "--no-step-log",
            "--remote-port", str(self.port),
        ]
        logger.info("Launching SUMO: %s", " ".join(cmd))
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3.0)  # let SUMO initialise
        traci.init(self.port)
        logger.info(
            "TraCI connected (port %d)  profile=%s  t=[%d, %d]",
            self.port, self.profile, self.sim_begin, self.sim_end,
        )

    #  Properties 

    @property
    def n_agents(self) -> int:
        return len(self.tls_ids)

    @property
    def action_space_n(self) -> int:
        return 2  # keep / switch

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        return (self.state_dim,)


#  Factory 

def make_env(config_path: str = "dqn/configs/dqn_config.yaml",
             profile: str = "morning_rush",
             port: int = 8813) -> SumoEnv:
    """Convenience factory that loads the YAML and returns a ready SumoEnv."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return SumoEnv(cfg, profile=profile, port=port)