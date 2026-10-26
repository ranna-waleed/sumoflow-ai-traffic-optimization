"""Fresh-process CPU probe for research snapshot equivalence (engineering only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from DeepQN.agent.dqn_agent import MultiAgentDQN
from DeepQN.training.research_resume import (ResumeContracts, TrainingCursor,
    publish_training_snapshot, restore_training_snapshot)
from DeepQN.training.research_orchestration import (
    EpisodeResult, ResearchSchedule, run_research_episode_transactions,
)


def _cfg(per: bool) -> dict:
    return {"state_dim": 4, "action_dim": 2, "hidden_dims": [8], "learning_rate": .001,
            "replay_buffer_size": 32, "min_replay_size": 2, "batch_size": 2,
            "gamma": .9, "epsilon_start": .8, "epsilon_end": .1,
            "epsilon_decay": .95, "tau": .2, "seed": 17, "prioritised_replay": per}


def _contracts() -> ResumeContracts:
    return ResumeContracts("fresh-process-probe", "1"*64, "2"*64, "3"*64, "4"*64,
                           "obs-v1", "reward-v1", "no-forecast-v1", git_commit="probe",
                           environment_sha256="5"*64)


def _schedule() -> ResearchSchedule:
    return ResearchSchedule(("morning", "evening"), ("demand-a", "demand-b"),
                            (101, 202, 303, 404))


def _runner(trace: list):
    def run(plan, model):
        step_trace = []
        for offset in range(2):
            observations = {}
            for tls_index, tls_id in enumerate(model.agents):
                index = plan.episode * 10 + offset * 2 + tls_index
                observations[tls_id] = np.array([index, index+1, -index, .5], dtype=np.float32) / 10
            actions = model.act(observations)
            model.push(observations, actions, {tid: float((plan.episode + offset) % 3)
                       for tid in model.agents}, {tid: state + .1 for tid, state in observations.items()},
                       terminated=False, interval_duration_s=10)
            losses = model.update()
            step_trace.append({"actions": actions,
                "epsilon": {tid: model.agents[tid].epsilon for tid in model.agents},
                "losses": losses})
        trace.append({"plan": plan.__dict__, "steps": step_trace})
        return EpisodeResult(True, True, {"synthetic_steps": 2})
    return run


def _digest(model: MultiAgentDQN) -> str:
    digest = hashlib.sha256()
    for tls_id, agent in sorted(model.agents.items()):
        digest.update(tls_id.encode())
        for network in (agent.online_net, agent.target_net):
            for name, tensor in sorted(network.state_dict().items()):
                digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
        replay = agent.replay.state_dict()
        for item in replay.get("transitions", replay.get("buffer", [])):
            if item is not None:
                digest.update(item.state.tobytes()); digest.update(item.next_state.tobytes())
                digest.update(repr((item.action, item.reward, item.terminated, item.truncated,
                                    item.interval_duration_s)).encode())
        if "priorities" in replay: digest.update(replay["priorities"].tobytes())
        optimizer = agent.optimiser.state_dict()
        for parameter_id, state in sorted(optimizer["state"].items()):
            digest.update(str(parameter_id).encode())
            for name, value in sorted(state.items()):
                digest.update(name.encode())
                if torch.is_tensor(value): digest.update(value.cpu().numpy().tobytes())
                else: digest.update(repr(value).encode())
        digest.update(json.dumps(optimizer["param_groups"], sort_keys=True).encode())
        digest.update(repr((agent.epsilon, agent.total_steps, agent.total_updates,
                            agent._episode_losses, replay.get("rng_state"),
                            replay.get("beta"), replay.get("ptr"))).encode())
    digest.update(repr(random.getstate()).encode())
    digest.update(repr(np.random.get_state()).encode())
    digest.update(torch.get_rng_state().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("full", "save", "resume"))
    parser.add_argument("root"); parser.add_argument("--per", action="store_true")
    args = parser.parse_args(); root = Path(args.root)
    random.seed(9); np.random.seed(9); torch.manual_seed(9); torch.use_deterministic_algorithms(True)
    model = MultiAgentDQN(["tls-a", "tls-b"], _cfg(args.per)); trace = []
    initial = TrainingCursor(0, 1, 0, 0, 0, 101)
    if args.mode == "full":
        cursor, records = run_research_episode_transactions(agents=model, initial_cursor=initial,
            schedule=_schedule(), contracts=_contracts(), snapshot_root=root / "snapshots",
            episode_runner=_runner(trace), episode_count=4)
    elif args.mode == "save":
        cursor, records = run_research_episode_transactions(agents=model, initial_cursor=initial,
            schedule=_schedule(), contracts=_contracts(), snapshot_root=root / "snapshots",
            episode_runner=_runner(trace), episode_count=2)
        (root / "snapshot.json").write_text(json.dumps(records[-1]["snapshot"]), encoding="utf-8")
    else:
        info = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
        initial = restore_training_snapshot(info["path"], model, _contracts(), info["manifest_sha256"],
                                            expected_schedule_sha256=_schedule().sha256)
        cursor, records = run_research_episode_transactions(agents=model, initial_cursor=initial,
            schedule=_schedule(), contracts=_contracts(), snapshot_root=root / "snapshots",
            episode_runner=_runner(trace), episode_count=2)
    record_ids = [{"plan": item["plan"], "metrics": item["metrics"],
                   "schedule_sha256": item["schedule_sha256"]} for item in records]
    (root / f"{args.mode}.json").write_text(json.dumps({"trace": trace, "digest": _digest(model),
        "cursor": cursor.__dict__, "records": record_ids},
                                                       sort_keys=True), encoding="utf-8")


if __name__ == "__main__": main()
