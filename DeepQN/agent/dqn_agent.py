"""
dqn/agent/dqn_agent.py:
DQN Agent for one TLS junction.

Features:
- Double DQN (DDQN): online net selects action, target net evaluates it.
  Reduces Q-value overestimation.
- Soft or hard target-network update.
- Epsilon-greedy exploration with linear or exponential decay.
- Works with both UniformReplayBuffer and PrioritisedReplayBuffer.
- Each of the 7 TLS junctions gets its own DQNAgent instance that shares
  the same DuelingQNetwork *architecture* but has independent weights.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, Optional

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
        done:       bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay.push(state, action, reward, next_state, done)
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
        d  = torch.FloatTensor(batch.dones).to(self.device)
        w  = torch.FloatTensor(batch.weights).to(self.device)

        # Current Q-values
        q_values = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target: online selects, target evaluates
        with torch.no_grad():
            next_actions = self.online_net(s2).argmax(dim=1, keepdim=True)
            next_q = self.target_net(s2).gather(1, next_actions).squeeze(1)
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
    Container for all 7 per-junction DQN agents.
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
        done:     bool,
    ):
        for tid, agent in self.agents.items():
            if tid in obs and tid in actions:
                agent.push(
                    obs[tid], actions[tid], rewards[tid], next_obs[tid], done
                )

    def update(self) -> Dict[str, Optional[float]]:
        return {tid: agent.update() for tid, agent in self.agents.items()}

    def episode_stats(self) -> Dict[str, Dict]:
        return {tid: agent.episode_stats() for tid, agent in self.agents.items()}

    def save(self, directory: str, episode: int):
        for agent in self.agents.values():
            agent.save(directory, episode)

    def load_latest(self, directory: str):
        """Load the most recent checkpoint per agent from ``directory``."""
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