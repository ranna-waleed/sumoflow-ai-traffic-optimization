"""
dqn/integration/dqn_controller.py:
Drop-in controller that the FastAPI backend imports to run the trained DQN
agents inside an active SUMO/TraCI session.

Design principle:
The backend already manages the TraCI lifecycle.  This module adds a
``DQNController`` singleton that:
  1. Loads the trained agent checkpoints on startup.
  2. Exposes a ``tick(traci_conn)`` method called every ``decision_interval``
     simulation steps from the existing backend step-loop.
  3. Exposes ``get_status()`` for the REST API ``GET /dqn/status`` endpoint.
  4. Exposes ``set_mode(mode)`` for toggling DQN / fixed-time from the dashboard.

Usage in backend (pseudo-code):
    from DeepQN.integration.dqn_controller import get_controller

    controller = get_controller()          # singleton
    controller.load("DeepQN/checkpoints")

    # Inside your TraCI step loop:
    for step in range(sim_steps):
        traci.simulationStep()
        controller.tick(step)

    # REST endpoint:
    @app.get("/dqn/status")
    def dqn_status():
        return controller.get_status()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

#  Lazy imports , backend may not have torch in the same venv 
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — DQNController will run in stub mode.")

try:
    import traci
    _TRACI_AVAILABLE = True
except ImportError:
    _TRACI_AVAILABLE = False


# Controller 

class DQNController:
    """
    Stateful controller that runs DQN agents inside an active TraCI session.

    Parameters:
    config_path : path to dqn_config.yaml
    """

    MODE_DQN        = "dqn_adaptive"
    MODE_FIXED      = "fixed_time"
    MODE_EVAL       = "evaluation"   # DQN, epsilon=0

    def __init__(self, config_path: str = "DeepQN/configs/dqn_config.yaml"):
        self.config_path = config_path
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.tls_ids:        List[str] = list(self.cfg["tls_junctions"].keys())
        self.decision_int:   int       = self.cfg["simulation"].get("decision_interval", 10)
        self.state_dim:      int       = self.cfg["state"].get("state_dim", 37)

        self.mode:           str  = self.MODE_DQN
        self._loaded:        bool = False
        self._step_counter:  int  = 0
        self._agents         = None   # MultiAgentDQN (lazy import)
        self._env_state      = None   # last observation dict
        self._last_actions:  Dict[str, int] = {}
        self._tick_times:    List[float] = []

        # Stats
        self._total_switches: Dict[str, int] = {tid: 0 for tid in self.tls_ids}
        self._total_keeps:    Dict[str, int] = {tid: 0 for tid in self.tls_ids}
        self._episode_start:  float = time.time()

        # BiLSTM adapter — loaded lazily on first reset()
        from DeepQN.integration.bilstm_adapter import BiLSTMAdapter
        self._bilstm = BiLSTMAdapter(self.cfg)

    # Lifecycle 

    def load(self, checkpoint_dir: str = "DeepQN/checkpoints") -> bool:
        """
        Load trained DQN agent weights.
        Returns True on success, False if torch is unavailable.
        """
        if not _TORCH_AVAILABLE:
            logger.error("Cannot load DQN: PyTorch is not installed.")
            return False

        from DeepQN.agent.dqn_agent import MultiAgentDQN
        self._agents = MultiAgentDQN(self.tls_ids, self.cfg["dqn"])
        self._agents.load_latest(checkpoint_dir)
        self._loaded = True
        logger.info(
            "DQNController loaded %d agents from '%s'",
            len(self.tls_ids), checkpoint_dir,
        )
        return True

    def reset(self):
        """Call at the start of each new SUMO simulation run."""
        self._step_counter  = 0
        self._last_actions  = {}
        self._env_state     = None
        self._episode_start = time.time()
        self._total_switches = {tid: 0 for tid in self.tls_ids}
        self._total_keeps    = {tid: 0 for tid in self.tls_ids}
        self._bilstm.reset()
        self._bilstm.load()   # loads model files; no-op if already loaded
        logger.info("DQNController reset. Mode=%s", self.mode)

    # Main tick 

    def tick(self, sumo_step: int) -> Optional[Dict[str, int]]:
        """
        Call this once per SUMO step from the backend's step loop.

        Returns:
        actions : {tls_id: 0_or_1} if a decision was made this step, else None
        """
        self._step_counter += 1

        # Always update BiLSTM history buffer every SUMO step
        self._bilstm.tick()

        if self.mode == self.MODE_FIXED:
            return None   # let SUMO run its static timing

        if not self._loaded or self._agents is None:
            logger.debug("DQN not loaded; skipping tick.")
            return None

        # Only decide every decision_interval steps
        if self._step_counter % self.decision_int != 0:
            return None

        t0 = time.perf_counter()

        # Build state for each junction
        obs = self._build_observations()

        # Select actions (eval_mode=True in production)
        eval_mode = (self.mode == self.MODE_EVAL)
        actions = self._agents.act(obs, eval_mode=eval_mode)

        # Apply via TraCI
        self._apply_actions(actions)

        self._last_actions = actions
        self._env_state    = obs

        # Accumulate stats
        for tid, a in actions.items():
            if a == 1:
                self._total_switches[tid] += 1
            else:
                self._total_keeps[tid] += 1

        dt = time.perf_counter() - t0
        self._tick_times.append(dt)

        logger.debug(
            "Tick %d | actions=%s | dt=%.3fms",
            sumo_step, actions, dt * 1000,
        )
        return actions

    #  Observation builder 

    def _build_observations(self) -> Dict[str, np.ndarray]:
        """
        Build the (state_dim,) observation array for each TLS junction.
        Mirrors the logic in SumoEnv._junction_state().
        """
        tls_cfg   = self.cfg["tls_junctions"]
        st_cfg    = self.cfg["state"]
        max_lanes = st_cfg.get("max_lanes", 10)
        fperlane  = st_cfg.get("features_per_lane", 3)
        max_queue = st_cfg.get("max_queue", 50.0)
        max_wait  = st_cfg.get("max_wait", 300.0)

        # Global features
        try:
            bilstm_preds = self._bilstm_predictions()
        except Exception:
            bilstm_preds = np.zeros(4, dtype=np.float32)

        sim_time  = traci.simulation.getTime() if _TRACI_AVAILABLE else 0.0
        profile   = self._current_profile()
        tod = np.clip(
            (sim_time - profile["begin"]) / max(profile["end"] - profile["begin"], 1),
            0.0, 1.0,
        )
        global_feats = np.concatenate([bilstm_preds, [tod]]).astype(np.float32)

        obs: Dict[str, np.ndarray] = {}
        for tid in self.tls_ids:
            lanes       = tls_cfg[tid].get("incoming_lanes", [])
            lane_feats  = np.zeros(max_lanes * fperlane, dtype=np.float32)

            for i, lane in enumerate(lanes[:max_lanes]):
                if not _TRACI_AVAILABLE:
                    break
                try:
                    halt = traci.lane.getLastStepHaltingNumber(lane) / max_queue
                    wait = traci.lane.getWaitingTime(lane)           / max_wait
                    occ  = traci.lane.getLastStepOccupancy(lane)
                except Exception:
                    halt, wait, occ = 0.0, 0.0, 0.0
                base = i * fperlane
                lane_feats[base]     = float(np.clip(halt, 0, 1))
                lane_feats[base + 1] = float(np.clip(wait, 0, 1))
                lane_feats[base + 2] = float(np.clip(occ,  0, 1))

            # Phase features (approximation — full phase tracking lives in SumoEnv)
            phase_idx = 0.0
            elapsed   = 0.0
            if _TRACI_AVAILABLE:
                try:
                    raw_phase = traci.trafficlight.getPhase(tid)
                    green_phases = tls_cfg[tid].get("green_phases", [0])
                    if raw_phase in green_phases:
                        phase_idx = green_phases.index(raw_phase) / max(len(green_phases), 1)
                except Exception:
                    pass

            junc_feats = np.array([phase_idx, elapsed], dtype=np.float32)
            obs[tid] = np.concatenate([lane_feats, junc_feats, global_feats])

        return obs

    def _bilstm_predictions(self) -> np.ndarray:
        """Delegate to BiLSTMAdapter — uses lstm/predict.py or proxy fallback."""
        return self._bilstm.predict()

    def _current_profile(self) -> dict:
        """Guess current profile from simulation time."""
        if not _TRACI_AVAILABLE:
            return {"begin": 0, "end": 86400}
        try:
            t = traci.simulation.getTime()
        except Exception:
            return {"begin": 0, "end": 86400}
        for p in self.cfg["profiles"].values():
            if p["begin"] <= t <= p["end"]:
                return p
        return {"begin": 0, "end": 86400}

    def _apply_actions(self, actions: Dict[str, int]):
        """
        Apply DQN phase decisions via TraCI.
        Phase transitions (yellow handling) are managed by SumoEnv._PhaseController
        when using SumoEnv.  In standalone backend mode, this applies directly.
        """
        if not _TRACI_AVAILABLE:
            return
        tls_cfg = self.cfg["tls_junctions"]
        for tid, action in actions.items():
            if action == 1:
                cfg = tls_cfg.get(tid, {})
                try:
                    current = traci.trafficlight.getPhase(tid)
                    g2y = {int(k): v for k, v in cfg.get("green_to_yellow", {}).items()}
                    if current in g2y:
                        traci.trafficlight.setPhase(tid, g2y[current])
                except Exception as exc:
                    logger.debug("Phase switch failed for %s: %s", tid, exc)

    # Status API 

    def get_status(self) -> Dict[str, Any]:
        """Serialisable status dict for the REST ``GET /dqn/status`` endpoint."""
        avg_tick_ms = (
            float(np.mean(self._tick_times[-100:])) * 1000
            if self._tick_times else 0.0
        )
        return {
            "mode":          self.mode,
            "loaded":        self._loaded,
            "step_counter":  self._step_counter,
            "last_actions":  self._last_actions,
            "total_switches": self._total_switches,
            "total_keeps":    self._total_keeps,
            "avg_tick_ms":    round(avg_tick_ms, 3),
            "uptime_s":       round(time.time() - self._episode_start, 1),
            "agents": {
                tid: {
                    "epsilon":  round(agent.epsilon, 4),
                    "updates":  agent.total_updates,
                    "buffer":   len(agent.replay),
                }
                for tid, agent in (self._agents.agents.items() if self._agents else {})
            },
        }

    def set_mode(self, mode: str) -> str:
        """Toggle between DQN adaptive and fixed-time. Returns new mode."""
        valid = {self.MODE_DQN, self.MODE_FIXED, self.MODE_EVAL}
        if mode not in valid:
            raise ValueError(f"Invalid mode '{mode}'. Valid: {valid}")
        self.mode = mode
        logger.info("DQNController mode set to: %s", mode)
        return self.mode

    #  FastAPI endpoint helpers 

    def get_phase_summary(self) -> List[Dict]:
        """Returns current TLS phase state for the dashboard."""
        if not _TRACI_AVAILABLE:
            return []
        summary = []
        for tid in self.tls_ids:
            try:
                phase = traci.trafficlight.getPhase(tid)
                state = traci.trafficlight.getRedYellowGreenState(tid)
            except Exception:
                phase, state = -1, "unknown"
            summary.append({
                "tls_id":        tid,
                "current_phase": phase,
                "state_string":  state,
                "last_action":   self._last_actions.get(tid, 0),
                "n_switches":    self._total_switches.get(tid, 0),
            })
        return summary


# Singleton 

_controller_instance: Optional[DQNController] = None


def get_controller(
    config_path: str = "DeepQN/configs/dqn_config.yaml",
) -> DQNController:
    """
    Return the global DQNController singleton.
    Instantiates on first call.
    """
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = DQNController(config_path)
        logger.info("DQNController singleton created.")
    return _controller_instance


def reset_controller():
    """Force re-creation of the singleton (e.g. after config change)."""
    global _controller_instance
    _controller_instance = None