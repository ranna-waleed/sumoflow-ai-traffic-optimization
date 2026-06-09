"""
dqn/env/reward.py
-----------------
Reward signal for the DQN traffic controller.

Design
------
The reward at each decision step is:

    R_t = - w_wait  * ΔW_t
          - w_co2   * ΔC_t
          + w_thru  * ΔT_t
          - w_press * P_t

Where:
  ΔW_t  = change in total halting-vehicle-seconds across all controlled lanes
  ΔC_t  = change in total CO2 emissions (mg) across all controlled edges
  ΔT_t  = number of vehicles that completed their trip this interval
  P_t   = queue pressure = |queue_upstream - queue_downstream| (optional)

This formulation:
  • Penalises increases in waiting time (primary KPI)
  • Lightly penalises CO2 spikes (secondary KPI)
  • Rewards vehicles clearing the network (throughput)
  • Clips to [clip_min, clip_max] for training stability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import traci

logger = logging.getLogger(__name__)


# Configuration 

@dataclass
class RewardConfig:
    wait_time_weight:  float = 1.0
    co2_weight:        float = 0.001
    throughput_weight: float = 0.5
    pressure_weight:   float = 0.0
    clip_min:          float = -200.0
    clip_max:          float =  10.0

    @classmethod
    def from_yaml(cls, cfg: dict) -> "RewardConfig":
        return cls(
            wait_time_weight  = cfg.get("wait_time_weight",  1.0),
            co2_weight        = cfg.get("co2_weight",        0.001),
            throughput_weight = cfg.get("throughput_weight", 0.5),
            pressure_weight   = cfg.get("pressure_weight",   0.0),
            clip_min          = cfg.get("clip_min",          -200.0),
            clip_max          = cfg.get("clip_max",           10.0),
        )


#  Reward Function 

class RewardFunction:
    """
    Stateful reward calculator.

    Call ``reset()`` at the start of each episode, then ``compute()``
    after every simulation decision step.

    Parameters
    ----------
    all_controlled_lanes : list of str
        Every incoming lane ID across all 7 TLS junctions.
        Used to compute the global waiting-time signal.
    all_controlled_edges : list of str
        Upstream edges leading into the controlled junctions.
        Used for CO2 observation via TraCI.
    config : RewardConfig
    """

    def __init__(
        self,
        all_controlled_lanes: List[str],
        all_controlled_edges: List[str],
        config: RewardConfig | None = None,
    ):
        self.lanes  = all_controlled_lanes
        self.edges  = all_controlled_edges
        self.config = config or RewardConfig()

        # State carried between steps
        self._prev_total_wait: float = 0.0
        self._prev_total_co2:  float = 0.0
        self._prev_departed:   int   = 0
        self._prev_arrived:    int   = 0

    #  Public API 

    def reset(self):
        """Call at the beginning of every episode."""
        self._prev_total_wait = self._get_total_wait()
        self._prev_total_co2  = self._get_total_co2()
        self._prev_departed   = traci.simulation.getDepartedNumber()
        self._prev_arrived    = traci.simulation.getArrivedNumber()

    def compute(self) -> float:
        """
        Compute and return the scalar reward for the current decision step.
        Must be called *after* advancing the simulation.
        """
        cfg = self.config

        #  Waiting-time delta 
        curr_wait = self._get_total_wait()
        delta_wait = curr_wait - self._prev_total_wait
        self._prev_total_wait = curr_wait

        #  CO2 delta 
        curr_co2 = self._get_total_co2()
        delta_co2 = curr_co2 - self._prev_total_co2
        self._prev_total_co2 = curr_co2

        #  Throughput delta 
        curr_arrived  = traci.simulation.getArrivedNumber()
        delta_arrived = curr_arrived - self._prev_arrived
        self._prev_arrived = curr_arrived

        #  Pressure (optional) 
        pressure = self._get_queue_pressure() if cfg.pressure_weight > 0 else 0.0

        # ─ Compose reward 
        reward = (
            - cfg.wait_time_weight  * delta_wait
            - cfg.co2_weight        * delta_co2
            + cfg.throughput_weight * delta_arrived
            - cfg.pressure_weight   * pressure
        )

        reward = float(max(cfg.clip_min, min(cfg.clip_max, reward)))

        logger.debug(
            "Reward=%.3f  ΔWait=%.1f  ΔCO2=%.1f  ΔArrived=%d",
            reward, delta_wait, delta_co2, delta_arrived,
        )
        return reward

    def get_metrics(self) -> Dict[str, float]:
        """Return the raw (un-weighted) metrics for logging."""
        return {
            "total_wait_s":   self._prev_total_wait,
            "total_co2_mg":   self._prev_total_co2,
        }

    #  Private helpers 

    def _get_total_wait(self) -> float:
        """Sum of waiting-time (seconds) across all controlled lanes."""
        total = 0.0
        for lane in self.lanes:
            try:
                total += traci.lane.getWaitingTime(lane)
            except traci.TraCIException:
                pass  # lane not yet loaded in sim
        return total

    def _get_total_co2(self) -> float:
        """Sum of CO2 emissions (mg/s) across all controlled edges."""
        total = 0.0
        for edge in self.edges:
            try:
                total += traci.edge.getCO2Emission(edge)
            except traci.TraCIException:
                pass
        return total

    def _get_queue_pressure(self) -> float:
        """
        Queue-pressure heuristic: mean absolute difference between
        upstream and downstream vehicle counts per lane-pair.
        (Optional; disabled by default via pressure_weight=0.)
        """
        pressures = []
        for lane in self.lanes:
            try:
                halt = traci.lane.getLastStepHaltingNumber(lane)
                pressures.append(float(halt))
            except traci.TraCIException:
                pass
        if not pressures:
            return 0.0
        mean_p = sum(pressures) / len(pressures)
        variance = sum((p - mean_p) ** 2 for p in pressures) / len(pressures)
        return variance ** 0.5


#  Convenience builder 

def build_reward_function(tls_config: dict, reward_yaml: dict) -> RewardFunction:
    """
    Build a RewardFunction from the tls_junctions block of dqn_config.yaml.

    Parameters
    ----------
    tls_config  : dict  , the ``tls_junctions`` sub-dict from yaml
    reward_yaml : dict  , the ``reward`` sub-dict from yaml
    """
    all_lanes: List[str] = []
    all_edges: set = set()

    for tid, cfg in tls_config.items():
        lanes = cfg.get("incoming_lanes", [])
        all_lanes.extend(lanes)
        # Derive edge IDs from lane IDs (strip trailing _N)
        for lane in lanes:
            edge_id = "_".join(lane.split("_")[:-1])
            all_edges.add(edge_id)

    rc = RewardConfig.from_yaml(reward_yaml)
    return RewardFunction(
        all_controlled_lanes=all_lanes,
        all_controlled_edges=list(all_edges),
        config=rc,
    )