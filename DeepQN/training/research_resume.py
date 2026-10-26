"""Atomic, episode-boundary research-training snapshots.

This is deliberately separate from historical/evaluation checkpoints.  A local
SHA-256 establishes integrity relative to an externally supplied expected
manifest hash; it does not make untrusted PyTorch bytes trustworthy.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from DeepQN.agent.dqn_agent import MultiAgentDQN, RESEARCH_OBSERVATION_CONTRACT
from DeepQN.agent.replay_buffer import Transition

SNAPSHOT_SCHEMA = "sumoflow-research-training-state-v1"
AGENT_SCHEMA = "sumoflow-research-agent-state-v1"
MARKER_SCHEMA = "sumoflow-research-training-complete-v1"


class ResearchResumeError(RuntimeError):
    """A research snapshot is incomplete, untrusted, or incompatible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value.lower()
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ResumeContracts:
    run_id: str
    run_manifest_sha256: str
    dataset_manifest_sha256: str
    demand_registry_sha256: str
    split_registry_sha256: str
    observation_contract: str
    reward_contract: str
    forecast_contract: str
    checkpoint_contract: str = SNAPSHOT_SCHEMA
    git_commit: str = ""
    environment_sha256: str = ""
    # Optional for generic interruption-equivalence fixtures. Publication-mode
    # validation requires this binding to the effective observation config.
    observation_config_sha256: str = ""

    def validate(self) -> None:
        identities = (self.run_id, self.observation_contract, self.reward_contract,
                      self.forecast_contract)
        if not all(isinstance(value, str) and value and value == value.strip()
                   for value in identities):
            raise ValueError("run and semantic contract identities are required")
        if self.checkpoint_contract != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported research-training checkpoint contract")
        if not self.git_commit:
            raise ValueError("Git commit identity is required")
        for name in ("run_manifest_sha256", "dataset_manifest_sha256",
                     "demand_registry_sha256", "split_registry_sha256"):
            if not _valid_hash(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256")
        if not _valid_hash(self.environment_sha256):
            raise ValueError("environment_sha256 must be a SHA-256")


def observation_config_payload(config: dict) -> dict:
    """Return the behavior-relevant v3 observation configuration.

    This is deliberately narrower than a whole YAML hash: output destinations
    and training budgets may change without redefining an observation, while
    lane inventories, profile clocks, feature normalization, forecasts, and
    simulator timing may not.
    """
    if not isinstance(config, dict):
        raise ValueError("research configuration must be a mapping")
    required = ("simulation", "profiles", "tls_junctions", "state", "bilstm", "dqn")
    if any(not isinstance(config.get(name), dict) for name in required):
        raise ValueError("research configuration lacks observation-defining sections")
    simulation = config["simulation"]
    return {
        "schema": "sumoflow-observation-config-fingerprint-v1",
        "simulation": {
            "step_length": simulation.get("step_length"),
            "waiting_time_memory_s": simulation.get("waiting_time_memory_s"),
            "lane_observation_mode": simulation.get("lane_observation_mode"),
        },
        "profiles": config["profiles"],
        "tls_junctions": config["tls_junctions"],
        "state": config["state"],
        "bilstm": config["bilstm"],
        "dqn_state_dim": config["dqn"].get("state_dim"),
    }


def observation_config_sha256(config: dict) -> str:
    return _canonical_hash(observation_config_payload(config))


def validate_publication_observation_contract(config: dict,
                                              contracts: ResumeContracts) -> str:
    """Bind future publication training/resume to the exact v3 observation.

    Generic snapshot machinery intentionally supports synthetic contract names;
    this gate is mandatory in the production research-training entry point.
    """
    contracts.validate()
    state = config.get("state") if isinstance(config, dict) else None
    identifier = state.get("observation_contract") if isinstance(state, dict) else None
    if identifier != RESEARCH_OBSERVATION_CONTRACT:
        raise ResearchResumeError(
            f"publication training requires {RESEARCH_OBSERVATION_CONTRACT}"
        )
    if contracts.observation_contract != identifier:
        raise ResearchResumeError("resume and effective observation contracts differ")
    if not _valid_hash(contracts.observation_config_sha256):
        raise ResearchResumeError("publication resume lacks an observation-config SHA-256")
    actual = observation_config_sha256(config)
    if actual != contracts.observation_config_sha256:
        raise ResearchResumeError("effective observation configuration hash mismatch")

    simulation = config["simulation"]
    forecast = config["bilstm"]
    dqn = config["dqn"]

    layout = {
        "max_lanes": 10,
        "features_per_lane": 3,
        "junction_features": 2,
        "global_features": 5,
        "state_dim": 37,
    }
    for name, expected in layout.items():
        value = state.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value != expected:
            raise ResearchResumeError(
                f"v3 publication observation requires state.{name}={expected}"
            )
    calculated = (state["max_lanes"] * state["features_per_lane"]
                  + state["junction_features"] + state["global_features"])
    if calculated != state["state_dim"]:
        raise ResearchResumeError("v3 publication observation feature arithmetic is inconsistent")
    if state["state_dim"] != dqn.get("state_dim"):
        raise ResearchResumeError("state and DQN dimensions differ")

    if simulation.get("lane_observation_mode") != "research_fail_closed":
        raise ResearchResumeError(
            "publication observation requires lane_observation_mode=research_fail_closed"
        )
    step_length = simulation.get("step_length")
    if (not isinstance(step_length, (int, float)) or isinstance(step_length, bool)
            or not math.isfinite(step_length) or step_length != 1):
        raise ResearchResumeError("publication observation requires a one-second timestep")
    waiting_memory = simulation.get("waiting_time_memory_s")
    if (not isinstance(waiting_memory, (int, float)) or isinstance(waiting_memory, bool)
            or not math.isfinite(waiting_memory) or waiting_memory != 100):
        raise ResearchResumeError("publication observation requires 100-second waiting memory")

    for name in ("max_queue", "max_wait", "max_occupancy", "max_phase_elapsed"):
        value = state.get(name)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            raise ResearchResumeError(
                f"publication observation normalization state.{name} must be finite and positive"
            )

    mode = forecast.get("mode")
    if mode not in {"no_forecast", "verified_model"}:
        raise ResearchResumeError(
            "publication comparison forecast mode must be no_forecast or verified_model"
        )
    warmup = forecast.get("history_warmup_seconds")
    if (not isinstance(warmup, (int, float)) or isinstance(warmup, bool)
            or not math.isfinite(warmup) or warmup != 60):
        raise ResearchResumeError("publication comparison requires a 60-second shared warm-up")
    if forecast.get("initial_control_protocol") != "phase_controller_keep_v1":
        raise ResearchResumeError(
            "publication comparison requires initial_control_protocol=phase_controller_keep_v1"
        )
    # Selecting verified_model is only a configuration identity. This validator
    # does not approve, load, or authenticate any forecast checkpoint.
    return actual


@dataclass(frozen=True)
class TrainingCursor:
    completed_episode: int
    next_episode: int
    profile_cursor: int
    demand_cursor: int
    sumo_run_index: int
    next_sumo_seed: int
    episode_complete: bool = True

    def validate(self) -> None:
        values = (self.completed_episode, self.next_episode, self.profile_cursor,
                  self.demand_cursor, self.sumo_run_index, self.next_sumo_seed)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
            raise ValueError("training cursors must be nonnegative integers")
        if not self.episode_complete or self.next_episode != self.completed_episode + 1:
            raise ValueError("snapshots are allowed only after a complete episode")


def _pack_transition(item: Transition | None) -> dict | None:
    if item is None:
        return None
    return {"state": torch.as_tensor(item.state).clone(), "action": item.action,
            "reward": item.reward, "next_state": torch.as_tensor(item.next_state).clone(),
            "terminated": item.terminated, "truncated": item.truncated,
            "interval_duration_s": item.interval_duration_s}


def _unpack_transition(item: dict | None) -> Transition | None:
    if item is None:
        return None
    required = {"state", "action", "reward", "next_state", "terminated", "truncated",
                "interval_duration_s"}
    if not isinstance(item, dict) or set(item) != required:
        raise ResearchResumeError("invalid serialized replay transition")
    return Transition(item["state"].cpu().numpy().copy(), int(item["action"]),
                      float(item["reward"]), item["next_state"].cpu().numpy().copy(),
                      bool(item["terminated"]), bool(item["truncated"]),
                      float(item["interval_duration_s"]))


def _pack_replay(state: dict) -> dict:
    packed = copy.deepcopy(state)
    key = "transitions" if "transitions" in packed else "buffer"
    packed[key] = [_pack_transition(item) for item in state[key]]
    if isinstance(packed.get("priorities"), np.ndarray):
        packed["priorities"] = torch.as_tensor(packed["priorities"]).clone()
    return packed


def _unpack_replay(state: dict) -> dict:
    unpacked = copy.deepcopy(state)
    key = "transitions" if "transitions" in unpacked else "buffer"
    if not isinstance(unpacked.get(key), list):
        raise ResearchResumeError("serialized replay entries are missing")
    unpacked[key] = [_unpack_transition(item) for item in unpacked[key]]
    if torch.is_tensor(unpacked.get("priorities")):
        unpacked["priorities"] = unpacked["priorities"].cpu().numpy().copy()
    return unpacked


def _numpy_global_state() -> dict:
    state = np.random.get_state()
    return {"name": state[0], "keys": torch.as_tensor(state[1].astype(np.int64)),
            "position": state[2], "has_gauss": state[3], "cached_gaussian": state[4]}


def _restore_numpy_state(state: dict) -> tuple:
    return (state["name"], state["keys"].cpu().numpy().astype(np.uint32),
            int(state["position"]), int(state["has_gauss"]), float(state["cached_gaussian"]))


def _agent_payload(tls_id: str, agent: Any, episode: int, contracts_hash: str) -> dict:
    return {"schema": AGENT_SCHEMA, "tls_id": tls_id, "episode": episode,
            "contracts_sha256": contracts_hash,
            "architecture_sha256": _canonical_hash(agent.cfg),
            "optimizer_class": agent.optimiser.__class__.__module__ + "." + agent.optimiser.__class__.__qualname__,
            "online": agent.online_net.state_dict(), "target": agent.target_net.state_dict(),
            "optimizer": agent.optimiser.state_dict(), "replay": _pack_replay(agent.replay.state_dict()),
            "epsilon": agent.epsilon, "total_steps": agent.total_steps,
            "total_updates": agent.total_updates,
            "episode_losses": list(agent._episode_losses)}


def publish_training_snapshot(directory: str | Path, agents: MultiAgentDQN,
                              cursor: TrainingCursor, contracts: ResumeContracts,
                              *, cuda_required: bool | None = None,
                              fail_after_files: int | None = None,
                              episode_record: dict | None = None) -> dict:
    """Publish a complete immutable snapshot; the directory rename is atomic."""
    cursor.validate(); contracts.validate()
    agent_devices = {tid: str(agent.device) for tid, agent in agents.agents.items()}
    actual_cuda_required = any(torch.device(device).type == "cuda" for device in agent_devices.values())
    if cuda_required is not None and bool(cuda_required) != actual_cuda_required:
        raise ResearchResumeError("CUDA requirement contradicts actual agent devices")
    if actual_cuda_required and not torch.cuda.is_available():
        raise ResearchResumeError("CUDA agents exist but CUDA state is unavailable")
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{contracts.run_id}-episode-{cursor.completed_episode:06d}"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite research snapshot: {destination}")
    contracts_dict = asdict(contracts); contracts_hash = _canonical_hash(contracts_dict)
    staging = Path(tempfile.mkdtemp(prefix=".training-state-", dir=root))
    writes = 0
    try:
        rows = []
        for tls_id in sorted(agents.agents):
            filename = f"agent-{tls_id}.pt"; path = staging / filename
            torch.save(_agent_payload(tls_id, agents.agents[tls_id], cursor.completed_episode,
                                      contracts_hash), path)
            rows.append({"tls_id": tls_id, "filename": filename, "sha256": _sha256(path)})
            writes += 1
            if fail_after_files is not None and writes >= fail_after_files:
                raise OSError("injected snapshot publication failure")
        global_payload = {"schema": SNAPSHOT_SCHEMA, "python_rng": random.getstate(),
                          "numpy_rng": _numpy_global_state(),
                          "torch_cpu_rng": torch.get_rng_state(),
                          "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                          "cuda_required": actual_cuda_required,
                          "cuda_device_count": torch.cuda.device_count() if actual_cuda_required else 0,
                          "agent_devices": agent_devices,
                          "cuda_device_names": ([torch.cuda.get_device_name(index)
                              for index in range(torch.cuda.device_count())]
                              if actual_cuda_required else []),
                          "cursor": asdict(cursor)}
        global_path = staging / "global-state.pt"; torch.save(global_payload, global_path)
        manifest = {"schema": SNAPSHOT_SCHEMA, "contracts": contracts_dict,
                    "contracts_sha256": contracts_hash, "cursor": asdict(cursor),
                    "tls_ids": sorted(agents.agents), "agents": rows,
                    "agent_devices": agent_devices,
                    "episode_record": episode_record,
                    "global_state": {"filename": global_path.name, "sha256": _sha256(global_path)},
                    "versions": {"python": os.sys.version, "numpy": np.__version__,
                                 "torch": torch.__version__, "cuda": torch.version.cuda}}
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_hash = _sha256(manifest_path)
        marker = {"schema": MARKER_SCHEMA, "manifest_sha256": manifest_hash}
        (staging / "COMPLETE.json").write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
        staging.replace(destination)
        return {"path": str(destination.resolve()), "manifest_sha256": manifest_hash,
                "semantic_identity": _canonical_hash(manifest)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def restore_training_snapshot(snapshot: str | Path, agents: MultiAgentDQN,
                              expected_contracts: ResumeContracts,
                              expected_manifest_sha256: str,
                              *, expected_schedule_sha256: str | None = None) -> TrainingCursor:
    """Validate the whole snapshot and candidates before one ensemble swap."""
    expected_contracts.validate()
    if not _valid_hash(expected_manifest_sha256):
        raise ResearchResumeError("an external expected manifest SHA-256 is required")
    root = Path(snapshot); manifest_path = root / "manifest.json"; marker_path = root / "COMPLETE.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise ResearchResumeError("snapshot is incomplete (manifest or completion marker missing)")
    actual_manifest_hash = _sha256(manifest_path)
    if actual_manifest_hash != expected_manifest_sha256:
        raise ResearchResumeError("snapshot manifest does not match the external trust reference")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchResumeError(f"snapshot metadata is unreadable: {exc}") from exc
    if marker != {"schema": MARKER_SCHEMA, "manifest_sha256": actual_manifest_hash}:
        raise ResearchResumeError("completion marker is invalid")
    expected_dict = asdict(expected_contracts)
    if (manifest.get("schema") != SNAPSHOT_SCHEMA or manifest.get("contracts") != expected_dict
            or manifest.get("contracts_sha256") != _canonical_hash(expected_dict)):
        raise ResearchResumeError("snapshot semantic/data/run contracts are incompatible")
    if expected_schedule_sha256 is not None:
        record = manifest.get("episode_record")
        if (not _valid_hash(expected_schedule_sha256) or not isinstance(record, dict)
                or record.get("schedule_sha256") != expected_schedule_sha256):
            raise ResearchResumeError("snapshot schedule identity is incompatible")
    cursor = TrainingCursor(**manifest.get("cursor", {})); cursor.validate()
    expected_ids = sorted(agents.agents)
    rows = manifest.get("agents")
    if manifest.get("tls_ids") != expected_ids or not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ResearchResumeError("snapshot ensemble membership is incomplete")
    if {row.get("tls_id") for row in rows} != set(expected_ids):
        raise ResearchResumeError("snapshot contains mixed or missing agents")
    live_devices = {tid: str(agent.device) for tid, agent in agents.agents.items()}
    if manifest.get("agent_devices") != live_devices:
        raise ResearchResumeError("snapshot agent-device assumptions are incompatible")

    prepared = {}
    for row in rows:
        tls_id = row["tls_id"]; path = root / row.get("filename", "")
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            raise ResearchResumeError(f"agent state hash mismatch for {tls_id}")
        try:
            payload = torch.load(path, map_location=agents.agents[tls_id].device, weights_only=True)
        except Exception as exc:
            raise ResearchResumeError(f"cannot safely load agent state for {tls_id}: {exc}") from exc
        live = agents.agents[tls_id]
        required = {"schema", "tls_id", "episode", "contracts_sha256", "architecture_sha256",
                    "optimizer_class", "online", "target", "optimizer", "replay", "epsilon",
                    "total_steps", "total_updates", "episode_losses"}
        optimizer_name = live.optimiser.__class__.__module__ + "." + live.optimiser.__class__.__qualname__
        if (set(payload) != required or payload["schema"] != AGENT_SCHEMA
                or payload["tls_id"] != tls_id or payload["episode"] != cursor.completed_episode
                or payload["contracts_sha256"] != manifest["contracts_sha256"]
                or payload["architecture_sha256"] != _canonical_hash(live.cfg)
                or payload["optimizer_class"] != optimizer_name):
            raise ResearchResumeError(f"agent state identity is incompatible for {tls_id}")
        candidate = copy.deepcopy(live)
        try:
            candidate.online_net.load_state_dict(payload["online"], strict=True)
            candidate.target_net.load_state_dict(payload["target"], strict=True)
            candidate.optimiser.load_state_dict(payload["optimizer"])
            candidate.replay.load_state_dict(_unpack_replay(payload["replay"]))
            candidate.epsilon = float(payload["epsilon"])
            candidate.total_steps = int(payload["total_steps"])
            candidate.total_updates = int(payload["total_updates"])
            candidate._episode_losses = [float(value) for value in payload["episode_losses"]]
            if (not np.isfinite(candidate.epsilon) or candidate.epsilon < 0
                    or candidate.total_steps < 0 or candidate.total_updates < 0
                    or any(not np.isfinite(value) for value in candidate._episode_losses)):
                raise ValueError("nonfinite or negative training counters")
        except Exception as exc:
            raise ResearchResumeError(f"agent state is incompatible for {tls_id}: {exc}") from exc
        prepared[tls_id] = candidate

    global_row = manifest.get("global_state", {}); global_path = root / global_row.get("filename", "")
    if not global_path.is_file() or _sha256(global_path) != global_row.get("sha256"):
        raise ResearchResumeError("global state hash mismatch")
    try:
        global_state = torch.load(global_path, map_location="cpu", weights_only=True)
        if global_state.get("schema") != SNAPSHOT_SCHEMA or global_state.get("cursor") != asdict(cursor):
            raise ResearchResumeError("global state identity is incompatible")
        if global_state.get("agent_devices") != live_devices:
            raise ResearchResumeError("global device identity is incompatible")
        py_probe = random.Random(); py_probe.setstate(global_state["python_rng"])
        np_probe = np.random.RandomState(); np_probe.set_state(_restore_numpy_state(global_state["numpy_rng"]))
        cpu_rng = global_state["torch_cpu_rng"]
        if not torch.is_tensor(cpu_rng) or cpu_rng.dtype != torch.uint8:
            raise ResearchResumeError("Torch CPU RNG state is invalid")
        torch.Generator(device="cpu").set_state(cpu_rng)
        cuda_states = global_state["torch_cuda_rng"]
        expected_cuda = any(torch.device(device).type == "cuda" for device in live_devices.values())
        if bool(global_state["cuda_required"]) != expected_cuda:
            raise ResearchResumeError("CUDA requirement does not match the live ensemble")
        if global_state["cuda_required"] and (not torch.cuda.is_available()
                or global_state.get("cuda_device_count") != torch.cuda.device_count()
                or len(cuda_states) != torch.cuda.device_count()
                or global_state.get("cuda_device_names") != [torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())]):
            raise ResearchResumeError("required CUDA RNG state cannot be restored")
        if global_state["cuda_required"]:
            for index, state in enumerate(cuda_states):
                torch.Generator(device=f"cuda:{index}").set_state(state)
    except ResearchResumeError:
        raise
    except Exception as exc:
        raise ResearchResumeError(f"global RNG state is invalid: {exc}") from exc

    # Validation is complete. Swapping the dict prevents a partially restored ensemble.
    agents.agents = prepared
    random.setstate(global_state["python_rng"])
    np.random.set_state(_restore_numpy_state(global_state["numpy_rng"]))
    torch.set_rng_state(cpu_rng)
    if global_state["cuda_required"]:
        torch.cuda.set_rng_state_all(cuda_states)
    return cursor
