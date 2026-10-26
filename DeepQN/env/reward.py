"""
dqn/env/reward.py:
Reward signal for the DQN traffic controller.

Design:
The reward at each decision step is:

    R_t = - w_wait  * ΔW_t - w_co2   * ΔC_t + w_thru  * ΔT_t - w_press * P_t

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
import math
from typing import Dict, List, Mapping

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
    delay_reference_vehicle_s: float = 100.0
    co2_reference_mg: float = 10000.0
    arrivals_reference_vehicles: float = 1.0
    pressure_reference_vehicle_s: float = 10.0

    @classmethod
    def from_yaml(cls, cfg: dict) -> "RewardConfig":
        return cls(
            wait_time_weight  = cfg.get("wait_time_weight",  1.0),
            co2_weight        = cfg.get("co2_weight",        0.001),
            throughput_weight = cfg.get("throughput_weight", 0.5),
            pressure_weight   = cfg.get("pressure_weight",   0.0),
            clip_min          = cfg.get("clip_min",          -200.0),
            clip_max          = cfg.get("clip_max",           10.0),
            delay_reference_vehicle_s=cfg.get("delay_reference_vehicle_s", 100.0),
            co2_reference_mg=cfg.get("co2_reference_mg", 10000.0),
            arrivals_reference_vehicles=cfg.get("arrivals_reference_vehicles", 1.0),
            pressure_reference_vehicle_s=cfg.get("pressure_reference_vehicle_s", 10.0),
        )


@dataclass(frozen=True)
class RewardBreakdown:
    delay_vehicle_s: float
    co2_mg: float
    arrivals: int
    pressure_vehicle_s: float
    weighted_unclipped: float
    reward: float
    interval_duration_s: float


class IntervalRewardFunction:
    """Phase 2B interval reward with explicit units.

    Delay is the integral of incoming-lane halting vehicles (vehicle-seconds).
    CO2 is the integral of unique controlled-edge TraCI rates (mg/s -> mg).
    Arrivals are summed per simulation step. Pressure is the time integral of
    the mean, across TLSs, of the population standard deviation of local
    incoming-lane queues (vehicle-seconds). The global quantities are computed
    once and shared once; they are not summed again per agent.
    """
    contract_version = "interval-global-reward-v1"

    def __init__(self, lanes_by_tls: Mapping[str, List[str]], edges: List[str],
                 config: RewardConfig | None = None):
        self.lanes_by_tls = {key: tuple(value) for key, value in lanes_by_tls.items()}
        flattened = [lane for lanes in self.lanes_by_tls.values() for lane in lanes]
        if len(flattened) != len(set(flattened)):
            raise ValueError("controlled lanes must have one TLS owner for reward allocation")
        self.lanes = tuple(flattened); self.edges = tuple(sorted(set(edges)))
        self.config = config or RewardConfig()
        references = (self.config.delay_reference_vehicle_s, self.config.co2_reference_mg,
                      self.config.arrivals_reference_vehicles,
                      self.config.pressure_reference_vehicle_s)
        if any(not math.isfinite(value) or value <= 0 for value in references):
            raise ValueError("reward normalisation references must be finite and positive")
        self.begin_interval()

    def reset(self):
        self.begin_interval()

    def begin_interval(self):
        self._delay = self._co2 = self._pressure = self._duration = 0.0
        self._arrivals = 0
        self._last = None

    def observe_step(self, duration_s: float, arrivals: int) -> None:
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("reward timestep duration must be finite and positive")
        if not isinstance(arrivals, int) or isinstance(arrivals, bool) or arrivals < 0:
            raise ValueError("interval arrivals must be a nonnegative integer")
        queues = {}
        for tls_id, lanes in self.lanes_by_tls.items():
            values = [float(traci.lane.getLastStepHaltingNumber(lane)) for lane in lanes]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError("invalid lane halting count")
            queues[tls_id] = values
        total_halts = sum(sum(values) for values in queues.values())
        local_pressure = [
            (sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values)) ** .5
            for values in queues.values() if values
        ]
        co2_rate = sum(float(traci.edge.getCO2Emission(edge)) for edge in self.edges)
        if not math.isfinite(co2_rate) or co2_rate < 0:
            raise ValueError("invalid TraCI CO2 rate")
        self._delay += duration_s * total_halts
        self._co2 += duration_s * co2_rate
        self._pressure += duration_s * (sum(local_pressure) / len(local_pressure) if local_pressure else 0)
        self._arrivals += arrivals; self._duration += duration_s

    def compute(self) -> float:
        cfg = self.config
        raw = (-cfg.wait_time_weight * self._delay / cfg.delay_reference_vehicle_s
               -cfg.co2_weight * self._co2 / cfg.co2_reference_mg
               +cfg.throughput_weight * self._arrivals / cfg.arrivals_reference_vehicles
               -cfg.pressure_weight * self._pressure / cfg.pressure_reference_vehicle_s)
        value = float(max(cfg.clip_min, min(cfg.clip_max, raw)))
        self._last = RewardBreakdown(self._delay, self._co2, self._arrivals,
                                     self._pressure, raw, value, self._duration)
        return value

    def get_metrics(self) -> Dict[str, float]:
        row = self._last or RewardBreakdown(0, 0, 0, 0, 0, 0, 0)
        return {"reward_contract": self.contract_version,
                "interval_delay_vehicle_s": row.delay_vehicle_s,
                "interval_co2_mg": row.co2_mg, "interval_arrivals": row.arrivals,
                "interval_pressure_vehicle_s": row.pressure_vehicle_s,
                "reward_unclipped": row.weighted_unclipped,
                "reward_clipped": row.reward}


#  Reward Function 

class RewardFunction:
    """
    Stateful reward calculator.

    Call ``reset()`` at the start of each episode, then ``compute()``
    after every simulation decision step.

    Parameters:
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

        # Compose reward 
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

    Parameters:
    tls_config  : dict  , the ``tls_junctions`` sub-dict from yaml
    reward_yaml : dict  , the ``reward`` sub-dict from yaml
    """
    all_lanes: List[str] = []
    lanes_by_tls: Dict[str, List[str]] = {}
    all_edges: set = set()

    for tid, cfg in tls_config.items():
        lanes = cfg.get("incoming_lanes", [])
        lanes_by_tls[tid] = list(lanes)
        all_lanes.extend(lanes)
        # Derive edge IDs from lane IDs (strip trailing _N)
        for lane in lanes:
            edge_id = "_".join(lane.split("_")[:-1])
            all_edges.add(edge_id)

    rc = RewardConfig.from_yaml(reward_yaml)
    contract = reward_yaml.get("contract", "legacy_endpoint_v1")
    if contract == "interval_v1":
        return IntervalRewardFunction(lanes_by_tls, list(all_edges), rc)
    if contract != "legacy_endpoint_v1":
        raise ValueError(f"unknown reward contract {contract!r}")
    return RewardFunction(
        all_controlled_lanes=all_lanes,
        all_controlled_edges=list(all_edges),
        config=rc,
    )
