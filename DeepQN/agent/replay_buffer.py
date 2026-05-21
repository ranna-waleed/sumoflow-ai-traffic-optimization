"""
dqn/agent/replay_buffer.py
--------------------------
Experience replay buffers for the DQN agent.

Provides:
  - UniformReplayBuffer  : standard random-sample replay
  - PrioritisedReplayBuffer: PER (Schaul et al., 2016) — optional upgrade

Both expose the same interface:
    push(state, action, reward, next_state, done)
    sample(batch_size) -> Batch
    __len__()
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np


# Transition 

@dataclass
class Transition:
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


@dataclass
class Batch:
    states:      np.ndarray   # (B, state_dim)
    actions:     np.ndarray   # (B,)
    rewards:     np.ndarray   # (B,)
    next_states: np.ndarray   # (B, state_dim)
    dones:       np.ndarray   # (B,)
    weights:     np.ndarray   # (B,)   importance weights (ones for uniform)
    indices:     np.ndarray   # (B,)   for PER priority updates


#  Uniform replay buffer

class UniformReplayBuffer:
    """
    Fixed-capacity circular replay buffer.

    Parameters
    ----------
    capacity : int   — maximum number of stored transitions
    seed     : int   — random seed for reproducible sampling
    """

    def __init__(self, capacity: int = 50_000, seed: int = 42):
        self.capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)
        random.seed(seed)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self._buffer.append(
            Transition(
                state      = np.array(state,      dtype=np.float32),
                action     = int(action),
                reward     = float(reward),
                next_state = np.array(next_state, dtype=np.float32),
                done       = bool(done),
            )
        )

    def sample(self, batch_size: int) -> Batch:
        if len(self) < batch_size:
            raise ValueError(
                f"Buffer has {len(self)} transitions; requested {batch_size}."
            )
        transitions = random.sample(self._buffer, batch_size)
        return self._collate(transitions, np.ones(batch_size, dtype=np.float32))

    def __len__(self) -> int:
        return len(self._buffer)

    @staticmethod
    def _collate(transitions: List[Transition], weights: np.ndarray) -> Batch:
        return Batch(
            states      = np.stack([t.state      for t in transitions]),
            actions     = np.array([t.action     for t in transitions], dtype=np.int64),
            rewards     = np.array([t.reward     for t in transitions], dtype=np.float32),
            next_states = np.stack([t.next_state for t in transitions]),
            dones       = np.array([t.done       for t in transitions], dtype=np.float32),
            weights     = weights,
            indices     = np.zeros(len(transitions), dtype=np.int64),
        )


# Prioritised replay buffer

class PrioritisedReplayBuffer:
    """
    Proportional Prioritised Experience Replay (PER).

    Transitions with higher TD-error are sampled more frequently.
    Importance-sampling weights correct for the sampling bias.

    Parameters
    ----------
    capacity : int
    alpha    : float — priority exponent (0 = uniform, 1 = full priority)
    beta     : float — IS-weight exponent (0 = no correction, 1 = full)
    beta_inc : float — increment applied to beta after each sample call
    eps      : float — small constant added to priority to avoid zero
    """

    def __init__(
        self,
        capacity: int   = 50_000,
        alpha:    float = 0.6,
        beta:     float = 0.4,
        beta_inc: float = 0.001,
        eps:      float = 1e-6,
        seed:     int   = 42,
    ):
        self.capacity = capacity
        self.alpha    = alpha
        self.beta     = beta
        self.beta_inc = beta_inc
        self.eps      = eps
        np.random.seed(seed)

        self._buffer:     List[Optional[Transition]] = [None] * capacity
        self._priorities: np.ndarray                 = np.zeros(capacity, dtype=np.float32)
        self._ptr:        int                         = 0
        self._size:       int                         = 0

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        max_p = self._priorities[:self._size].max() if self._size > 0 else 1.0
        self._buffer[self._ptr] = Transition(
            state      = np.array(state,      dtype=np.float32),
            action     = int(action),
            reward     = float(reward),
            next_state = np.array(next_state, dtype=np.float32),
            done       = bool(done),
        )
        self._priorities[self._ptr] = max_p
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        if self._size < batch_size:
            raise ValueError(f"Buffer has {self._size}; requested {batch_size}.")

        probs = self._priorities[:self._size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self._size, batch_size, replace=False, p=probs)
        transitions = [self._buffer[i] for i in indices]

        # IS weights
        weights = (self._size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_inc)

        return Batch(
            states      = np.stack([t.state      for t in transitions]),
            actions     = np.array([t.action     for t in transitions], dtype=np.int64),
            rewards     = np.array([t.reward     for t in transitions], dtype=np.float32),
            next_states = np.stack([t.next_state for t in transitions]),
            dones       = np.array([t.done       for t in transitions], dtype=np.float32),
            weights     = weights.astype(np.float32),
            indices     = indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Call after each training step with the per-sample TD errors."""
        priorities = np.abs(td_errors) + self.eps
        for idx, p in zip(indices, priorities):
            self._priorities[idx] = float(p)

    def __len__(self) -> int:
        return self._size


# Factory 

def build_replay_buffer(cfg: dict) -> UniformReplayBuffer | PrioritisedReplayBuffer:
    """Build replay buffer from the ``dqn`` section of dqn_config.yaml."""
    capacity = cfg.get("replay_buffer_size", 50_000)
    seed     = cfg.get("seed", 42)
    use_per  = cfg.get("prioritised_replay", False)

    if use_per:
        return PrioritisedReplayBuffer(capacity=capacity, seed=seed)
    return UniformReplayBuffer(capacity=capacity, seed=seed)