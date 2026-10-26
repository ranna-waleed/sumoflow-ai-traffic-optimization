"""
dqn/agent/dqn_agent.py:
DQN Agent for one TLS junction.

Features:
- Double DQN (DDQN): online net selects action, target net evaluates it.
  Reduces Q-value overestimation.
- Soft or hard target-network update.
- Epsilon-greedy exploration with linear or exponential decay.
- Works with both UniformReplayBuffer and PrioritisedReplayBuffer.
- Each configured TLS junction gets its own DQNAgent instance that shares
  the same DuelingQNetwork *architecture* but has independent weights.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DeepQN.agent.network import DuelingQNetwork, build_network
from DeepQN.agent.replay_buffer import (
    Batch,
    PrioritisedReplayBuffer,
    UniformReplayBuffer,
    build_replay_buffer,
)

logger = logging.getLogger(__name__)


class CheckpointValidationError(RuntimeError):
    """The requested trained ensemble is absent, inconsistent, or incompatible."""


@dataclass(frozen=True)
class CheckpointRecord:
    tls_id: str
    episode: int
    path: str
    sha256: str


@dataclass(frozen=True)
class CheckpointManifest:
    episode: int
    checkpoints: tuple[CheckpointRecord, ...]
    schema_version: str = "legacy-strict-ensemble-v1"
    ensemble_manifest_path: str | None = None
    ensemble_manifest_sha256: str | None = None
    run_manifest_sha256: str | None = None

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "checkpoints": [asdict(item) for item in self.checkpoints],
            "schema_version": self.schema_version,
            "ensemble_manifest_path": self.ensemble_manifest_path,
            "ensemble_manifest_sha256": self.ensemble_manifest_sha256,
            "run_manifest_sha256": self.run_manifest_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DQNAgent:
    """
    One DQN agent controlling one TLS junction.

    Parameters:
    tls_id    : str , e.g. "315744796"
    cfg       : dict from dqn_config.yaml[dqn] section
    device    : torch device
    """

    def __init__(self, tls_id: str, cfg: dict, device: Optional[torch.device] = None):
        self.tls_id = tls_id
        self.cfg    = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #  Networks 
        self.online_net: DuelingQNetwork = build_network(cfg, self.device)
        self.target_net: DuelingQNetwork = copy.deepcopy(self.online_net)
        self.target_net.eval()

        #  Optimiser 
        self.optimiser = optim.Adam(
            self.online_net.parameters(),
            lr=cfg.get("learning_rate", 5e-4),
        )

        # Replay buffer 
        self.replay: UniformReplayBuffer | PrioritisedReplayBuffer = (
            build_replay_buffer(cfg)
        )
        self.min_replay = cfg.get("min_replay_size", 1000)
        self.batch_size = cfg.get("batch_size", 64)

        #  Hyper-params 
        self.gamma      = cfg.get("gamma", 0.95)
        self.discount_mode = cfg.get("discount_mode", "decision_step_v1")
        if self.discount_mode != "decision_step_v1":
            raise ValueError(
                "unsupported discount_mode; Phase 2B.2B uses one gamma per decision transition"
            )
        self.grad_clip  = cfg.get("grad_clip", 10.0)
        self.tau        = cfg.get("tau", 0.005)
        self.target_update_freq = cfg.get("target_update_freq", 200)

        #  Exploration 
        self.epsilon:       float = cfg.get("epsilon_start", 1.0)
        self.epsilon_end:   float = cfg.get("epsilon_end",   0.05)
        self.epsilon_decay: float = cfg.get("epsilon_decay", 0.9995)

        # Counters 
        self.total_steps:    int = 0
        self.total_updates:  int = 0
        self._episode_losses: list = []

        logger.info(
            "DQNAgent created for TLS '%s' on device %s", tls_id, self.device
        )

    # Action selection

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Epsilon-greedy action selection.

        Parameters:
        state     : np.ndarray (state_dim,)
        eval_mode : if True, always greedy (epsilon = 0)

        Returns:
        action : 0 (keep) or 1 (switch)
        """
        eps = 0.0 if eval_mode else self.epsilon

        if np.random.random() < eps:
            return np.random.randint(0, self.cfg.get("action_dim", 2))

        state_t = torch.FloatTensor(state).to(self.device)
        return self.online_net.get_action(state_t)

    #  Learning 

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool | None = None,
        *, terminated: bool | None = None, truncated: bool = False,
        valid: bool = True, interval_duration_s: float = 0.0,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay.push(state, action, reward, next_state, done,
                         terminated=terminated, truncated=truncated, valid=valid,
                         interval_duration_s=interval_duration_s)
        self.total_steps += 1

        # Decay epsilon after each step
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def update(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient step.
        Returns the scalar loss (float) or None if buffer is not ready.
        """
        if len(self.replay) < self.min_replay:
            return None

        batch = self.replay.sample(self.batch_size)
        loss, td_errors = self._compute_loss(batch)

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimiser.step()

        # Update PER priorities if applicable
        if isinstance(self.replay, PrioritisedReplayBuffer):
            self.replay.update_priorities(
                batch.indices, td_errors.detach().cpu().numpy()
            )

        # Target network update (soft or hard)
        self.total_updates += 1
        if self.tau < 1.0:
            self._soft_update()
        elif self.total_updates % self.target_update_freq == 0:
            self._hard_update()

        loss_val = loss.item()
        self._episode_losses.append(loss_val)
        return loss_val

    def _compute_loss(self, batch: Batch):
        """Double DQN TD loss with optional IS weighting."""
        s  = torch.FloatTensor(batch.states).to(self.device)
        a  = torch.LongTensor(batch.actions).to(self.device)
        r  = torch.FloatTensor(batch.rewards).to(self.device)
        s2 = torch.FloatTensor(batch.next_states).to(self.device)
        d  = torch.FloatTensor(batch.terminated).to(self.device)
        w  = torch.FloatTensor(batch.weights).to(self.device)

        # Current Q-values
        q_values = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target: online selects, target evaluates
        with torch.no_grad():
            next_actions = self.online_net(s2).argmax(dim=1, keepdim=True)
            next_q = self.target_net(s2).gather(1, next_actions).squeeze(1)
            # Preregistered convention: one gamma per policy decision. Actual
            # interval duration remains recorded for audit/truncation but does
            # not exponentiate gamma, including the shortened final interval.
            targets = r + (1.0 - d) * self.gamma * next_q

        td_errors = targets - q_values
        loss = (w * td_errors.pow(2)).mean()
        return loss, td_errors

    # Target network updates

    def _soft_update(self):
        """Polyak averaging: θ_target ← τ·θ_online + (1-τ)·θ_target"""
        for t_p, o_p in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            t_p.data.copy_(self.tau * o_p.data + (1.0 - self.tau) * t_p.data)

    def _hard_update(self):
        """Copy online weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        logger.debug("%s  target network hard-updated", self.tls_id)

    # Episode helpers 

    def episode_stats(self) -> Dict[str, float]:
        """Return and reset per-episode training stats."""
        losses = self._episode_losses
        stats = {
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "epsilon":   self.epsilon,
            "updates":   self.total_updates,
            "buffer":    len(self.replay),
        }
        self._episode_losses = []
        return stats

    #  Persistence 

    def save(self, directory: str, episode: int):
        """Save online & target network weights to ``directory``."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online":  self.online_net.state_dict(),
                "target":  self.target_net.state_dict(),
                "episode": episode,
                "epsilon": self.epsilon,
                "steps":   self.total_steps,
            },
            path / f"{self.tls_id}_ep{episode:04d}.pt",
        )
        logger.info("Saved %s checkpoint (ep %d)", self.tls_id, episode)

    def load(self, checkpoint_path: str):
        """Load weights from a previously saved checkpoint."""
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:  # PyTorch versions before weights_only was introduced
            ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.epsilon     = ckpt.get("epsilon", self.epsilon_end)
        self.total_steps = ckpt.get("steps", 0)
        logger.info(
            "Loaded checkpoint for %s from %s (ep %s)",
            self.tls_id, checkpoint_path, ckpt.get("episode", "?"),
        )


# Multi-agent manager

class MultiAgentDQN:
    """
    Container for the configured per-junction DQN agents.
    Wraps act / push / update / save / load for convenience.
    """

    def __init__(self, tls_ids: list, cfg: dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agents: Dict[str, DQNAgent] = {
            tid: DQNAgent(tid, cfg, device) for tid in tls_ids
        }
        logger.info(
            "MultiAgentDQN: %d agents on %s", len(self.agents), device
        )

    # Delegation helpers 

    def act(
        self,
        observations: Dict[str, np.ndarray],
        eval_mode: bool = False,
    ) -> Dict[str, int]:
        return {
            tid: agent.act(observations[tid], eval_mode=eval_mode)
            for tid, agent in self.agents.items()
            if tid in observations
        }

    def push(
        self,
        obs:      Dict[str, np.ndarray],
        actions:  Dict[str, int],
        rewards:  Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        done:     bool | None = None,
        *, terminated: bool | None = None, truncated: bool = False,
        valid: bool = True, interval_duration_s: float = 0.0,
    ):
        for tid, agent in self.agents.items():
            if tid in obs and tid in actions:
                agent.push(
                    obs[tid], actions[tid], rewards[tid], next_obs[tid], done,
                    terminated=terminated, truncated=truncated, valid=valid,
                    interval_duration_s=interval_duration_s,
                )

    def update(self) -> Dict[str, Optional[float]]:
        return {tid: agent.update() for tid, agent in self.agents.items()}

    def episode_stats(self) -> Dict[str, Dict]:
        return {tid: agent.episode_stats() for tid, agent in self.agents.items()}

    def save(self, directory: str, episode: int):
        for agent in self.agents.values():
            agent.save(directory, episode)

    def save_research_ensemble(self, directory: str, episode: int,
                               run_manifest_sha256: str) -> CheckpointManifest:
        """Write a complete provenance-bearing ensemble without overwrites.

        Historical ``save``/``load_latest`` remain available solely for legacy
        resume behavior. Future research runs use this schema and immutable
        explicitly selected episodes.
        """
        if len(run_manifest_sha256) != 64:
            raise ValueError("run manifest SHA-256 is required")
        root = Path(directory); root.mkdir(parents=True, exist_ok=True)
        targets = {tid: root / f"{tid}_ep{episode:04d}.pt" for tid in self.agents}
        ensemble_path = root / f"ensemble_ep{episode:04d}.json"
        collisions = [str(path) for path in (*targets.values(), ensemble_path) if path.exists()]
        if collisions:
            raise FileExistsError("refusing to overwrite research checkpoints: " + ", ".join(collisions))
        cfg_hashes = {tid: hashlib.sha256(json.dumps(
            agent.cfg, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest() for tid, agent in self.agents.items()}
        published: list[Path] = []
        try:
            with tempfile.TemporaryDirectory(prefix=f".ensemble-ep{episode:04d}-", dir=root) as staging:
                staging_root = Path(staging)
                staged = {}
                for tid, agent in self.agents.items():
                    tmp = staging_root / targets[tid].name
                    numpy_state = np.random.get_state()
                    payload = {
                        "checkpoint_schema": "sumoflow-dqn-checkpoint-v2",
                        "online": agent.online_net.state_dict(), "target": agent.target_net.state_dict(),
                        "optimizer": agent.optimiser.state_dict(), "episode": episode,
                        "epsilon": agent.epsilon, "steps": agent.total_steps,
                        "updates": agent.total_updates, "tls_id": tid,
                        "architecture_config_sha256": cfg_hashes[tid],
                        "run_manifest_sha256": run_manifest_sha256,
                        "python_rng_state": random.getstate(),
                        "numpy_rng_state": {
                            "bit_generator": numpy_state[0],
                            "state": numpy_state[1].tolist(), "position": numpy_state[2],
                            "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4],
                        },
                        "torch_rng_state": torch.get_rng_state(),
                    }
                    torch.save(payload, tmp); staged[tid] = tmp
                checkpoint_rows = [{"tls_id": tid, "episode": episode,
                    "filename": targets[tid].name, "sha256": _sha256(staged[tid]),
                    "architecture_config_sha256": cfg_hashes[tid]}
                    for tid in sorted(staged)]
                ensemble_payload = {
                    "schema_version": "sumoflow-dqn-ensemble-v2", "episode": episode,
                    "run_manifest_sha256": run_manifest_sha256,
                    "tls_ids": sorted(self.agents), "checkpoints": checkpoint_rows,
                }
                ensemble_raw = (json.dumps(ensemble_payload, indent=2, sort_keys=True) + "\n").encode()
                staged_manifest = staging_root / ensemble_path.name
                staged_manifest.write_bytes(ensemble_raw)
                for tid in sorted(staged):
                    staged[tid].replace(targets[tid]); published.append(targets[tid])
                staged_manifest.replace(ensemble_path); published.append(ensemble_path)
        except Exception:
            for path in published:
                path.unlink(missing_ok=True)
            raise
        return CheckpointManifest(episode, tuple(
            CheckpointRecord(tid, episode, str(path.resolve()), _sha256(path))
            for tid, path in targets.items()
        ), schema_version="sumoflow-dqn-ensemble-v2",
            ensemble_manifest_path=str(ensemble_path.resolve()),
            ensemble_manifest_sha256=_sha256(ensemble_path),
            run_manifest_sha256=run_manifest_sha256)

    def load_latest(self, directory: str):
        """Best-effort per-agent loading retained for explicit training resume."""
        ckpt_dir = Path(directory)
        if not ckpt_dir.exists():
            logger.warning("Checkpoint dir '%s' not found; starting fresh.", directory)
            return
        for tid, agent in self.agents.items():
            checkpoints = sorted(ckpt_dir.glob(f"{tid}_ep*.pt"))
            if checkpoints:
                agent.load(str(checkpoints[-1]))
            else:
                logger.info("No checkpoint for %s; starting fresh.", tid)

    def load_for_research_evaluation(self, directory: str, episode: int) -> CheckpointManifest:
        """Require and atomically load a complete immutable v2 ensemble."""
        if not isinstance(episode, int) or isinstance(episode, bool) or episode < 0:
            raise CheckpointValidationError("checkpoint episode must be a non-negative integer")
        root = Path(directory)
        ensemble_path = root / f"ensemble_ep{episode:04d}.json"
        if not ensemble_path.is_file():
            raise CheckpointValidationError(f"research ensemble manifest not found: {ensemble_path}")
        try:
            manifest = json.loads(ensemble_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointValidationError(f"cannot read ensemble manifest: {exc}") from exc
        expected_ids = sorted(self.agents)
        if (manifest.get("schema_version") != "sumoflow-dqn-ensemble-v2"
                or manifest.get("episode") != episode
                or manifest.get("tls_ids") != expected_ids):
            raise CheckpointValidationError("research ensemble manifest is incompatible")
        run_hash = manifest.get("run_manifest_sha256")
        if not isinstance(run_hash, str) or len(run_hash) != 64:
            raise CheckpointValidationError("research ensemble lacks a valid run manifest hash")
        rows = manifest.get("checkpoints")
        if (not isinstance(rows, list) or len(rows) != len(expected_ids)
                or {row.get("tls_id") for row in rows} != set(expected_ids)):
            raise CheckpointValidationError("research ensemble checkpoint membership is incomplete")

        prepared: Dict[str, Dict[str, Any]] = {}
        records = []
        for row in rows:
            tls_id = row["tls_id"]; agent = self.agents[tls_id]
            expected_name = f"{tls_id}_ep{episode:04d}.pt"
            if row.get("filename") != expected_name or row.get("episode") != episode:
                raise CheckpointValidationError(f"invalid checkpoint identity for TLS {tls_id}")
            path = root / expected_name
            if not path.is_file() or _sha256(path) != row.get("sha256"):
                raise CheckpointValidationError(f"checkpoint hash mismatch for TLS {tls_id}")
            try:
                checkpoint = torch.load(path, map_location=agent.device, weights_only=True)
            except Exception as exc:
                raise CheckpointValidationError(f"cannot read research checkpoint {path}: {exc}") from exc
            cfg_hash = hashlib.sha256(json.dumps(
                agent.cfg, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            required = {"online", "target", "optimizer", "episode", "tls_id",
                        "checkpoint_schema", "architecture_config_sha256",
                        "run_manifest_sha256", "python_rng_state", "numpy_rng_state",
                        "torch_rng_state"}
            if (required - set(checkpoint)
                    or checkpoint.get("checkpoint_schema") != "sumoflow-dqn-checkpoint-v2"
                    or checkpoint.get("tls_id") != tls_id
                    or checkpoint.get("episode") != episode
                    or checkpoint.get("run_manifest_sha256") != run_hash
                    or checkpoint.get("architecture_config_sha256") != cfg_hash
                    or row.get("architecture_config_sha256") != cfg_hash):
                raise CheckpointValidationError(f"research checkpoint provenance is incompatible for TLS {tls_id}")
            try:
                online_probe = copy.deepcopy(agent.online_net)
                target_probe = copy.deepcopy(agent.target_net)
                online_probe.load_state_dict(checkpoint["online"], strict=True)
                target_probe.load_state_dict(checkpoint["target"], strict=True)
            except Exception as exc:
                raise CheckpointValidationError(
                    f"research checkpoint architecture is incompatible for TLS {tls_id}: {exc}"
                ) from exc
            prepared[tls_id] = checkpoint
            records.append(CheckpointRecord(tls_id, episode, str(path.resolve()), row["sha256"]))

        # No live model mutation occurs before the complete manifest and every
        # constituent file, hash, provenance field, and tensor shape validates.
        for tls_id, checkpoint in prepared.items():
            agent = self.agents[tls_id]
            agent.online_net.load_state_dict(checkpoint["online"], strict=True)
            agent.target_net.load_state_dict(checkpoint["target"], strict=True)
            agent.epsilon = checkpoint.get("epsilon", agent.epsilon_end)
            agent.total_steps = checkpoint.get("steps", 0)
        return CheckpointManifest(
            episode, tuple(records), "sumoflow-dqn-ensemble-v2",
            str(ensemble_path.resolve()), _sha256(ensemble_path), run_hash,
        )

    def load_legacy_for_evaluation(self, directory: str, episode: int) -> CheckpointManifest:
        """Explicit historical strict loader; never valid for new research runs."""
        return self.load_for_evaluation(directory, episode)

    def load_for_evaluation(self, directory: str, episode: int) -> CheckpointManifest:
        """Atomically load a complete historical ensemble.

        This compatibility path predates v2 ensemble manifests. New research
        evaluation must call :meth:`load_for_research_evaluation`.
        """
        if not isinstance(episode, int) or isinstance(episode, bool) or episode < 0:
            raise CheckpointValidationError("checkpoint episode must be a non-negative integer")
        checkpoint_dir = Path(directory)
        if not checkpoint_dir.is_dir():
            raise CheckpointValidationError(f"checkpoint directory not found: {checkpoint_dir}")

        prepared: Dict[str, Dict[str, Any]] = {}
        records: list[CheckpointRecord] = []
        for tls_id, agent in self.agents.items():
            path = checkpoint_dir / f"{tls_id}_ep{episode:04d}.pt"
            if not path.is_file():
                raise CheckpointValidationError(
                    f"missing checkpoint for TLS {tls_id}, episode {episode}: {path}"
                )
            try:
                try:
                    checkpoint = torch.load(path, map_location=agent.device, weights_only=True)
                except TypeError:  # compatibility with older supported PyTorch
                    checkpoint = torch.load(path, map_location=agent.device)
            except Exception as exc:
                raise CheckpointValidationError(f"cannot read checkpoint {path}: {exc}") from exc
            if not isinstance(checkpoint, dict):
                raise CheckpointValidationError(f"checkpoint is not a mapping: {path}")
            missing_keys = {"online", "target", "episode"} - set(checkpoint)
            if missing_keys:
                raise CheckpointValidationError(
                    f"checkpoint {path} is missing keys: {sorted(missing_keys)}"
                )
            if checkpoint["episode"] != episode:
                raise CheckpointValidationError(
                    f"checkpoint {path} records episode {checkpoint['episode']!r}, expected {episode}"
                )
            if "checkpoint_schema" in checkpoint:
                raise CheckpointValidationError(
                    "schema-declared checkpoints require the research ensemble loader"
                )
            try:
                online_probe = copy.deepcopy(agent.online_net)
                target_probe = copy.deepcopy(agent.target_net)
                online_probe.load_state_dict(checkpoint["online"], strict=True)
                target_probe.load_state_dict(checkpoint["target"], strict=True)
            except Exception as exc:
                raise CheckpointValidationError(
                    f"checkpoint {path} is incompatible with TLS {tls_id}: {exc}"
                ) from exc
            prepared[tls_id] = checkpoint
            records.append(
                CheckpointRecord(
                    tls_id=tls_id,
                    episode=episode,
                    path=str(path.resolve()),
                    sha256=_sha256(path),
                )
            )

        # Mutation begins only after every checkpoint has passed validation.
        for tls_id, checkpoint in prepared.items():
            agent = self.agents[tls_id]
            agent.online_net.load_state_dict(checkpoint["online"], strict=True)
            agent.target_net.load_state_dict(checkpoint["target"], strict=True)
            agent.epsilon = checkpoint.get("epsilon", agent.epsilon_end)
            agent.total_steps = checkpoint.get("steps", 0)
        return CheckpointManifest(episode=episode, checkpoints=tuple(records))
