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
import hashlib
import json
import math
import socket
import tempfile
from dataclasses import asdict, dataclass
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

from DeepQN.env.reward import IntervalRewardFunction, RewardFunction, build_reward_function
from DeepQN.forecasting.inference import NoForecastAdapter, StrictForecastAdapter
from DeepQN.integration.bilstm_adapter import BiLSTMAdapter

logger = logging.getLogger(__name__)
DQN_OBSERVATION_CONTRACT_VERSION = "sumoflow-dqn-observation-v2-occupancy-percent"


#  Phase controller per junction 

@dataclass(frozen=True)
class ActionDecision:
    requested_action: int
    accepted: bool
    reason: str
    phase_before: int
    transition_started: bool


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

    def reset(self) -> None:
        """Restore every episode-local controller field to its initial state."""
        self.current_green_phase = self.green_phases[0]
        self.phase_elapsed_steps = 0
        self.in_yellow = False
        self.yellow_elapsed = 0
        self.in_red_hold = False
        self.forced_switch = False

    # Seconds helpers 

    @property
    def phase_elapsed_s(self) -> float:
        return self.phase_elapsed_steps * self.step_length

    # Core logic 

    def apply_action(self, action: int) -> ActionDecision:
        """
        Call ONCE per decision interval.
        action: 0 = keep, 1 = request switch.
        """
        if type(action) is not int or action not in (0, 1):
            raise ValueError("TLS action must be keep=0 or switch=1")
        before = self.current_green_phase
        if self.in_yellow:
            return ActionDecision(action, False, "deferred_yellow", before, False)

        switch_requested = (action == 1)
        # Enforce max-green override
        if self.phase_elapsed_s >= self.max_green:
            switch_requested = True
            self.forced_switch = True

        if switch_requested and self.phase_elapsed_s >= self.min_green:
            forced = self.forced_switch
            self._start_yellow_transition()
            return ActionDecision(action, True,
                                  "forced_max_green" if forced else "accepted_switch",
                                  before, True)
        if switch_requested:
            return ActionDecision(action, False, "deferred_min_green", before, False)
        return ActionDecision(action, True, "accepted_keep", before, False)

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
                self.in_red_hold = (next_green == self.red_hold_phase)
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
        self.terminate_on_empty: bool = config["simulation"].get("terminate_on_empty", True)
        self.research_evaluation: bool = bool(
            config["simulation"].get("research_evaluation", False)
        )
        comparative = config["simulation"].get("comparative_research")
        if self.research_evaluation and type(comparative) is not bool:
            raise ValueError(
                "research evaluation requires explicit boolean simulation.comparative_research"
            )
        self.comparative_research = bool(comparative) if self.research_evaluation else False
        if self.research_evaluation and "physical_trace_mode" not in config["simulation"]:
            raise ValueError("research evaluation requires explicit physical_trace_mode")
        self.physical_trace_mode: str = config["simulation"].get(
            "physical_trace_mode", "legacy_estimate"
        )
        if self.physical_trace_mode not in {"research_fail_closed", "legacy_estimate"}:
            raise ValueError("physical_trace_mode must be research_fail_closed or legacy_estimate")
        if self.research_evaluation and self.physical_trace_mode != "research_fail_closed":
            raise ValueError("research evaluation requires physical_trace_mode=research_fail_closed")
        self.lane_observation_mode = config["simulation"].get(
            "lane_observation_mode", "legacy_zero_fill"
        )
        if self.lane_observation_mode not in {"research_fail_closed", "legacy_zero_fill"}:
            raise ValueError(
                "lane_observation_mode must be research_fail_closed or legacy_zero_fill"
            )
        if self.research_evaluation and self.lane_observation_mode != "research_fail_closed":
            raise ValueError(
                "research evaluation requires lane_observation_mode=research_fail_closed"
            )

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
        self.observation_contract = st_cfg.get("observation_contract")
        if (self.research_evaluation
                and self.observation_contract != DQN_OBSERVATION_CONTRACT_VERSION):
            raise ValueError(
                "research evaluation requires observation_contract="
                f"{DQN_OBSERVATION_CONTRACT_VERSION}"
            )
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

        # Historical mode is retained explicitly. Research modes never fall
        # back from a missing model to occupancy or zeros.
        forecast_cfg = config.get("bilstm", {})
        if self.research_evaluation and "mode" not in forecast_cfg:
            raise ValueError("research evaluation requires explicit bilstm.mode")
        forecast_mode = forecast_cfg.get("mode", "legacy_adapter")
        if self.research_evaluation and forecast_mode == "legacy_adapter":
            raise ValueError("research evaluation cannot use legacy_adapter")
        if forecast_mode == "no_forecast":
            self._bilstm = NoForecastAdapter(self.tls_ids)
        elif forecast_mode == "verified_model":
            manifest = config.get("bilstm", {}).get("manifest_path")
            if not manifest:
                raise ValueError("verified_model mode requires bilstm.manifest_path")
            self._bilstm = StrictForecastAdapter.from_manifest(manifest, self.tls_ids)
        elif forecast_mode == "legacy_adapter":
            self._bilstm = BiLSTMAdapter(config)
        else:
            raise ValueError(f"unknown BiLSTM mode {forecast_mode!r}")
        self._forecast_mode = forecast_mode
        self._warmup_seconds = float(forecast_cfg.get("history_warmup_seconds", 0))
        self._initial_control_protocol = forecast_cfg.get("initial_control_protocol")
        if not math.isfinite(self._warmup_seconds) or self._warmup_seconds < 0:
            raise ValueError("bilstm.history_warmup_seconds must be finite and nonnegative")
        if self._warmup_seconds and self._initial_control_protocol != "phase_controller_keep_v1":
            raise ValueError(
                "forecast warm-up requires initial_control_protocol=phase_controller_keep_v1"
            )
        if forecast_mode == "verified_model":
            if self.step_length != 1.0 or self._warmup_seconds != 60.0:
                raise ValueError(
                    "verified_model requires one-second steps and exactly 60 seconds of live warm-up"
                )
        matched_warmup = config["simulation"].get("require_matched_forecast_warmup")
        if self.comparative_research and matched_warmup is not True:
            raise ValueError(
                "comparative research requires require_matched_forecast_warmup=true"
            )
        if not self.comparative_research and matched_warmup not in (None, False):
            raise ValueError(
                "non-comparative diagnostics cannot claim matched forecast warm-up"
            )
        if matched_warmup is True:
            if (self.step_length != 1.0 or self._warmup_seconds != 60.0
                    or self._initial_control_protocol != "phase_controller_keep_v1"):
                raise ValueError(
                    "research DQN comparisons require the shared 60-second initial-control protocol"
                )
        waiting_memory = config["simulation"].get("waiting_time_memory_s")
        if self.research_evaluation and waiting_memory != 100:
            raise ValueError(
                "research evaluation requires explicit simulation.waiting_time_memory_s=100"
            )
        self._waiting_time_memory_s = waiting_memory

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

        try:
            # Reset every controller field, command the initial phase, and
            # confirm the physical state before collecting any observation.
            self._initial_tls_states = {}
            for tid, ctrl in self._phase_controllers.items():
                ctrl.reset()
                try:
                    traci.trafficlight.setPhase(tid, ctrl.current_green_phase)
                    physical = self._read_physical_tls_state(tid, ctrl)
                    if (not physical["available"]
                            or physical["phase"] != ctrl.current_green_phase
                            or not physical["state"]
                            or physical["state"] == "unavailable"):
                        raise RuntimeError(
                            f"initial physical TLS state mismatch for {tid}: "
                            f"expected phase {ctrl.current_green_phase}, observed {physical}"
                        )
                    self._initial_tls_states[tid] = physical
                except Exception as exc:
                    if self.research_evaluation:
                        raise RuntimeError(
                            f"failed to initialize and confirm physical TLS {tid}: {exc}"
                        ) from exc
                    logger.warning("Could not confirm initial phase for %s: %s", tid, exc)

            self._reward_fn.reset()
            self._bilstm.reset()
            if self._forecast_mode == "legacy_adapter":
                self._bilstm.load()
            if self._forecast_mode == "verified_model":
                self._detector_preflight = self._bilstm.validate_runtime_installation(
                    traci, self.net_file
                )
            else:
                self._detector_preflight = None
            self._warmup_diagnostics = self._run_initial_control_warmup()
            if self._forecast_mode == "verified_model" and not self._bilstm.history_ready():
                raise RuntimeError("verified forecast warm-up did not produce 60 live samples")
            # Warm-up traffic remains part of the XML outcomes and full demand
            # window, but it is not an RL transition or reward.
            self._reward_fn.reset()
            return self._get_all_states()
        except Exception:
            self.close()
            raise

    def _run_initial_control_warmup(self) -> dict:
        """Collect real history under a declared controller before decision one.

        The simulation clock, demand, XML outputs, arrivals, waiting, and
        emissions all advance normally and remain in the evaluation outcome.
        Only policy decisions and replay/reward transitions begin afterwards.
        """
        if self._warmup_seconds == 0:
            self._warmup_trace = []
            return {
                "protocol": "none", "duration_s": 0.0,
                "first_forecast_decision_s": float(traci.simulation.getTime()),
                "arrivals": 0,
            }
        start = float(traci.simulation.getTime())
        arrivals = 0
        samples = 0
        trace = []
        while float(traci.simulation.getTime()) - start < self._warmup_seconds:
            previous = float(traci.simulation.getTime())
            decisions = {
                tid: asdict(ctrl.apply_action(0))
                for tid, ctrl in self._phase_controllers.items()
            }
            execution_start = {
                tid: self._read_physical_tls_state(tid, ctrl)
                for tid, ctrl in self._phase_controllers.items()
            }
            traci.simulationStep()
            current = float(traci.simulation.getTime())
            if current - previous != 1.0:
                raise RuntimeError("forecast warm-up requires consecutive one-second SUMO steps")
            if current > self.sim_end:
                raise RuntimeError("forecast warm-up exceeds the configured simulation horizon")
            arrivals += int(traci.simulation.getArrivedNumber())
            execution_end = {
                tid: self._read_physical_tls_state(tid, ctrl)
                for tid, ctrl in self._phase_controllers.items()
            }
            for ctrl in self._phase_controllers.values():
                ctrl.tick()
            after_update = {
                tid: self._read_physical_tls_state(tid, ctrl)
                for tid, ctrl in self._phase_controllers.items()
            }
            trace.append({
                "start_s": previous, "end_s": current,
                "requested_actions": {tid: 0 for tid in self.tls_ids},
                "action_decisions": decisions,
                "execution_start": execution_start,
                "execution_end_before_controller": execution_end,
                "after_controller_update": after_update,
            })
            self._bilstm.tick()
            self._step_count += 1
            samples += 1
        duration = float(traci.simulation.getTime()) - start
        if duration != self._warmup_seconds:
            raise RuntimeError("forecast warm-up duration is not exactly configured")
        self._warmup_trace = trace
        trace_sha256 = hashlib.sha256(json.dumps(
            trace, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        return {
            "protocol": self._initial_control_protocol,
            "duration_s": duration,
            "samples": samples,
            "arrivals": arrivals,
            "requested_keep_actions": samples * len(self.tls_ids),
            "physical_trace_sha256": trace_sha256,
            "first_forecast_decision_s": float(traci.simulation.getTime()),
            "included_in_evaluation_outputs": True,
            "demand_window_begin_s": float(self.sim_begin),
            "demand_window_end_s": float(self.sim_end),
        }

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

        self._validate_action_dictionary(actions)

        #  1. Apply requested phase switches 
        action_decisions = {}
        for tid, action in actions.items():
            if tid in self._phase_controllers:
                action_decisions[tid] = self._phase_controllers[tid].apply_action(action)

        #  2. Advance simulation step-by-step 
        step_arrived = 0
        interval_start_time = float(traci.simulation.getTime())
        interval_steps = 0
        action_trace = {tid: [] for tid in self.tls_ids}
        if isinstance(self._reward_fn, IntervalRewardFunction):
            self._reward_fn.begin_interval()
        for _ in range(self.decision_int):
            prior_time = float(traci.simulation.getTime())
            pre_step = {
                tid: self._read_physical_tls_state(tid, ctrl)
                for tid, ctrl in self._phase_controllers.items()
            }
            traci.simulationStep()
            current_time = float(traci.simulation.getTime())
            duration_s = current_time - prior_time
            self._step_count += 1
            interval_steps += 1
            # Accumulate arrivals every SUMO step (not just the last one)
            arrivals = int(traci.simulation.getArrivedNumber())
            step_arrived += arrivals
            if isinstance(self._reward_fn, IntervalRewardFunction):
                self._reward_fn.observe_step(duration_s, arrivals)
            # Read the simulator endpoint before controller bookkeeping can
            # issue a setPhase that applies to the following interval.
            post_step = {
                tid: self._read_physical_tls_state(tid, ctrl)
                for tid, ctrl in self._phase_controllers.items()
            }
            for tid, ctrl in self._phase_controllers.items():
                before = pre_step[tid]; after = post_step[tid]
                executed = before if before == after and before["available"] else None
                ctrl.tick()
                next_boundary = self._read_physical_tls_state(tid, ctrl)
                action_trace[tid].append({
                    "start_s": prior_time, "end_s": current_time,
                    "start_boundary": before, "end_boundary_before_controller": after,
                    "after_controller_update": next_boundary,
                    "executed_state": executed,
                    "execution_evidence": (
                        "stable_physical_boundaries" if executed is not None
                        else "boundary_changed_or_unavailable"
                    ),
                })
            # Update BiLSTM history buffer every step
            self._bilstm.tick()
            # Avoid advancing a full decision interval beyond the configured
            # horizon. Empty-demand termination remains the training default;
            # fixed-horizon evaluation disables it explicitly.
            if (
                traci.simulation.getTime() >= self.sim_end
                or (
                    self.terminate_on_empty
                    and traci.simulation.getMinExpectedNumber() == 0
                )
            ):
                break

        #  3. Collect reward 
        global_reward = self._reward_fn.compute()
        rewards = {tid: global_reward for tid in self.tls_ids}

        # 4. Collect next observations 
        next_obs = self._get_all_states()

        # 5. Termination check 
        sim_time = traci.simulation.getTime()
        terminated = bool(self.terminate_on_empty and
                          traci.simulation.getMinExpectedNumber() == 0 and sim_time < self.sim_end)
        truncated = bool(sim_time >= self.sim_end)
        done = (
            (self.terminate_on_empty and traci.simulation.getMinExpectedNumber() == 0)
            or sim_time >= self.sim_end
        )

        info = {
            "sim_time":     sim_time,
            "step_count":   self._step_count,
            "step_arrived": step_arrived,   # arrivals this decision interval
            "interval_steps": interval_steps,
            "interval_duration_s": sim_time - interval_start_time,
            "profile":      self.profile,
            "terminated": terminated,
            "truncated": truncated,
            "transition_valid": True,
            "requested_actions": dict(actions),
            "action_decisions": {tid: asdict(value) for tid, value in action_decisions.items()},
            "action_trace": action_trace,
            "action_trace_sha256": hashlib.sha256(json.dumps(
                action_trace, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
            **self._reward_fn.get_metrics(),
        }

        return next_obs, rewards, done, info

    def _validate_action_dictionary(self, actions: Dict[str, int]) -> None:
        if not isinstance(actions, dict):
            raise ValueError("actions must be a dictionary keyed by TLS ID")
        supplied = set(actions)
        configured = set(self.tls_ids)
        unknown = sorted(supplied - configured)
        missing = sorted(configured - supplied)
        if unknown:
            raise ValueError(f"action dictionary contains unknown TLS IDs: {unknown}")
        if getattr(self, "comparative_research", False) and missing:
            raise ValueError(f"comparative research action dictionary is missing TLS IDs: {missing}")
        invalid = {
            tid: action for tid, action in actions.items()
            if type(action) is not int or action not in (0, 1)
        }
        if invalid:
            raise ValueError(f"action dictionary contains non-binary integer actions: {invalid}")

    def _read_physical_tls_state(self, tid: str, ctrl: _PhaseController) -> dict:
        """Read a boundary sample; never infer research evidence from controller state."""
        try:
            return {"available": True, "phase": int(traci.trafficlight.getPhase(tid)),
                    "state": str(traci.trafficlight.getRedYellowGreenState(tid)),
                    "source": "traci_physical_query"}
        except Exception as exc:
            if self.physical_trace_mode == "research_fail_closed":
                raise RuntimeError(f"physical TLS trace unavailable for {tid}: {exc}") from exc
            return {"available": False, "phase": ctrl.current_green_phase,
                    "state": "unavailable", "source": "legacy_controller_estimate"}

    def close(self):
        """Close the TraCI connection and terminate SUMO."""
        if self._connected:
            try:
                traci.close(wait=True)
            except Exception as exc:
                logger.warning("Failed to close TraCI/SUMO cleanly: %s", exc)
            self._connected = False
            logger.info("SUMO environment closed.")

    #  State construction 

    def _get_all_states(self) -> Dict[str, np.ndarray]:
        """Build the state vector for every controlled TLS junction."""
        return {
            tid: self._junction_state(tid, self._global_features(tid))
            for tid in self.tls_ids
        }

    def _global_features(self, tid: str | None = None) -> np.ndarray:
        """
        Returns a (5,) vector:
          [bilstm_N, bilstm_S, bilstm_E, bilstm_W, time_of_day]
        """
        if hasattr(self._bilstm, "predict_for"):
            bilstm = self._bilstm.predict_for(tid)
        else:
            bilstm = self._bilstm.predict()
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
            halt = self._lane_measurement(
                tid, lane, "halting_number", traci.lane.getLastStepHaltingNumber
            ) / self.max_queue
            wait = self._lane_measurement(
                tid, lane, "waiting_time", traci.lane.getWaitingTime
            ) / self.max_wait
            raw_occ_percent = self._lane_measurement(
                tid, lane, "occupancy", traci.lane.getLastStepOccupancy
            )
            if raw_occ_percent < 0 or raw_occ_percent > 100:
                if self.lane_observation_mode == "research_fail_closed":
                    raise RuntimeError(
                        f"lane observation failed: TLS={tid}, lane={lane}, "
                        f"measurement=occupancy: percentage outside [0, 100]: "
                        f"{raw_occ_percent}"
                    )
            occ = raw_occ_percent / 100.0
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

    def _lane_measurement(self, tid: str, lane: str, measurement: str, query) -> float:
        try:
            value = float(query(lane))
        except Exception as exc:
            if self.lane_observation_mode == "research_fail_closed":
                raise RuntimeError(
                    f"lane observation failed: TLS={tid}, lane={lane}, "
                    f"measurement={measurement}: {exc}"
                ) from exc
            return 0.0
        if not math.isfinite(value):
            if self.lane_observation_mode == "research_fail_closed":
                raise RuntimeError(
                    f"lane observation failed: TLS={tid}, lane={lane}, "
                    f"measurement={measurement}: nonfinite value {value}"
                )
            return 0.0
        return value

    #  SUMO launch 

    def _launch_sumo(self):
        profile_config = self.profile_cfg["config"]
        cmd = [
            self.sumo_binary,
            "-c", profile_config,
            "--no-step-log",
        ]
        if self.research_evaluation:
            cmd += ["--waiting-time-memory", str(self._waiting_time_memory_s)]
        if "seed" in self.config["simulation"]:
            seed = self.config["simulation"]["seed"]
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError("simulation.seed must be a nonnegative integer")
            cmd += ["--seed", str(seed)]
        # Fail before spawning when the explicitly assigned port is occupied.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", self.port))
        except OSError as exc:
            raise RuntimeError(f"TraCI port {self.port} is unavailable: {exc}") from exc
        finally:
            probe.close()

        diagnostic = tempfile.NamedTemporaryFile(
            prefix="sumoflow-sumo-", suffix=".log", mode="w+", encoding="utf-8", delete=False
        )
        self._sumo_diagnostic_path = Path(diagnostic.name)
        logger.info("Launching headless SUMO: %s (diagnostic log: %s)", " ".join(cmd), self._sumo_diagnostic_path)
        try:
            # traci.start owns the child process and retries the connection rather
            # than relying on a fixed startup sleep. The configured port ties this
            # connection to this exact process.
            traci.start(cmd, port=self.port, numRetries=30, stdout=diagnostic)
        except Exception as exc:
            diagnostic.flush()
            diagnostic.seek(0)
            output = diagnostic.read().strip()
            raise RuntimeError(
                f"SUMO failed to start on TraCI port {self.port}: {exc}. "
                f"Diagnostic output: {output or '<empty>'}"
            ) from exc
        finally:
            diagnostic.close()
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
