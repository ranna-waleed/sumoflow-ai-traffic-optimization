"""Episode-transaction orchestration for future authorized research training."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from DeepQN.agent.dqn_agent import MultiAgentDQN
from DeepQN.training.research_resume import (ResumeContracts, TrainingCursor,
    publish_training_snapshot)


@dataclass(frozen=True)
class ResearchSchedule:
    profiles: tuple[str, ...]
    demand_realization_ids: tuple[str, ...]
    sumo_seeds: tuple[int, ...]
    schedule_contract: str = "sumoflow-training-schedule-v1"

    def validate(self) -> None:
        if not self.profiles or not self.demand_realization_ids or not self.sumo_seeds:
            raise ValueError("research schedule sequences must be nonempty")
        if any(not value for value in (*self.profiles, *self.demand_realization_ids)):
            raise ValueError("research schedule identities must be nonempty")
        if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in self.sumo_seeds):
            raise ValueError("SUMO seeds must be nonnegative integers")

    @property
    def sha256(self) -> str:
        self.validate()
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                            separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class EpisodePlan:
    episode: int
    profile: str
    demand_realization_id: str
    sumo_seed: int
    run_index: int


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    finalized: bool
    metrics: dict


def plan_next_episode(cursor: TrainingCursor, schedule: ResearchSchedule) -> EpisodePlan:
    cursor.validate(); schedule.validate()
    if cursor.profile_cursor >= len(schedule.profiles):
        raise ValueError("profile cursor exceeds the frozen schedule")
    if cursor.demand_cursor >= len(schedule.demand_realization_ids):
        raise ValueError("demand cursor exceeds the frozen schedule")
    if cursor.sumo_run_index >= len(schedule.sumo_seeds):
        raise ValueError("SUMO run cursor exceeds the frozen schedule")
    seed = schedule.sumo_seeds[cursor.sumo_run_index]
    if cursor.next_sumo_seed != seed:
        raise ValueError("next SUMO seed disagrees with the frozen schedule")
    return EpisodePlan(cursor.next_episode, schedule.profiles[cursor.profile_cursor],
                       schedule.demand_realization_ids[cursor.demand_cursor], seed,
                       cursor.sumo_run_index)


def _advanced_cursor(plan: EpisodePlan, cursor: TrainingCursor,
                     schedule: ResearchSchedule) -> TrainingCursor:
    next_run = plan.run_index + 1
    if next_run >= len(schedule.sumo_seeds):
        next_seed = schedule.sumo_seeds[-1]  # terminal sentinel; no further plan is legal
    else:
        next_seed = schedule.sumo_seeds[next_run]
    return TrainingCursor(plan.episode, plan.episode + 1,
                          (cursor.profile_cursor + 1) % len(schedule.profiles),
                          (cursor.demand_cursor + 1) % len(schedule.demand_realization_ids),
                          next_run, next_seed, True)


def run_research_episode_transactions(*, agents: MultiAgentDQN,
        initial_cursor: TrainingCursor, schedule: ResearchSchedule,
        contracts: ResumeContracts, snapshot_root: str | Path,
        episode_runner: Callable[[EpisodePlan, MultiAgentDQN], EpisodeResult],
        episode_count: int,
        snapshot_publisher: Callable = publish_training_snapshot) -> tuple[TrainingCursor, list[dict]]:
    """Run completed-episode transactions with injected episode execution.

    A failed/unfinalized episode publishes neither a cursor nor an authoritative
    log.  The episode record lives inside the immutable snapshot manifest; any
    external derived log is recoverable/idempotent from that snapshot.
    """
    if not isinstance(episode_count, int) or isinstance(episode_count, bool) or episode_count < 0:
        raise ValueError("episode_count must be a nonnegative integer")
    schedule.validate(); contracts.validate(); cursor = initial_cursor; records = []
    for _ in range(episode_count):
        plan = plan_next_episode(cursor, schedule)
        result = episode_runner(plan, agents)
        if not isinstance(result, EpisodeResult) or not result.success or not result.finalized:
            raise RuntimeError(f"episode {plan.episode} did not finalize; cursor was not advanced")
        next_cursor = _advanced_cursor(plan, cursor, schedule)
        record = {"schema": "sumoflow-training-episode-record-v1",
                  "schedule_sha256": schedule.sha256, "plan": asdict(plan),
                  "metrics": result.metrics}
        published = snapshot_publisher(snapshot_root, agents, next_cursor, contracts,
                                       episode_record=record)
        records.append({**record, "snapshot": published})
        cursor = next_cursor
    return cursor, records
