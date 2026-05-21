"""
dqn/agent/network.py
--------------------
Dueling Deep Q-Network architecture for traffic signal control.

Dueling advantage:
  Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]

This separates state-value estimation from action-advantage estimation,
which is beneficial here because many states (e.g. light traffic) have
similar values regardless of the chosen phase.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN Q-network.

    Parameters
    ----------
    state_dim   : int  — observation vector size (default 37)
    action_dim  : int  — number of discrete actions (default 2: keep/switch)
    hidden_dims : list — widths of shared MLP hidden layers
    """

    def __init__(
        self,
        state_dim:   int       = 37,
        action_dim:  int       = 2,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim  = state_dim
        self.action_dim = action_dim

        #  Shared feature extractor 
        layers: List[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.feature_net = nn.Sequential(*layers)

        #  Value stream 
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        #  Advantage stream 
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # Weight init
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : Tensor shape (batch, state_dim) or (state_dim,)

        Returns
        -------
        q_values : Tensor shape (batch, action_dim)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        feats     = self.feature_net(state)
        value     = self.value_stream(feats)           # (B, 1)
        advantage = self.advantage_stream(feats)       # (B, action_dim)

        # Dueling combination: subtract mean advantage to enforce identifiability
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

    def get_action(self, state: torch.Tensor) -> int:
        """Greedy action for a single state (no gradient)."""
        with torch.no_grad():
            q = self.forward(state)
            return int(q.argmax(dim=-1).item())


# Convenience factory 

def build_network(cfg: dict, device: torch.device) -> DuelingQNetwork:
    """Build a DuelingQNetwork from the ``dqn`` section of dqn_config.yaml."""
    net = DuelingQNetwork(
        state_dim   = cfg.get("state_dim",   37),
        action_dim  = cfg.get("action_dim",  2),
        hidden_dims = cfg.get("hidden_dims", [256, 256]),
    )
    return net.to(device)