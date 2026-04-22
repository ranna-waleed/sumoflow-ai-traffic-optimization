# dqn/agent.py
# Deep Q-Network agent for traffic signal control

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


#  Q-Network 
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x):
        return self.net(x)


#  DQN Agent 
class DQNAgent:
    def __init__(
        self,
        state_size:   int   = 6,
        action_size:  int   = 4,
        hidden:       int   = 128,
        lr:           float = 1e-3,
        gamma:        float = 0.95,
        epsilon:      float = 1.0,
        epsilon_min:  float = 0.01,
        epsilon_decay:float = 0.995,
        batch_size:   int   = 64,
        memory_size:  int   = 10_000,
        target_update:int   = 10,    # update target net every N episodes
    ):
        self.state_size    = state_size
        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.episode       = 0

        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(state_size, action_size, hidden).to(self.device)
        self.target_net = QNetwork(state_size, action_size, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        print(f"[DQN] Agent initialized | device={self.device} | ε={self.epsilon:.3f}")

    def act(self, state) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax().item()

    def act_with_q(self, state):
        """
        Same as act() but also returns all Q-values.
        Used for logging and explainability —
        shows WHY the agent chose this action.
        Returns: (action: int, q_values: list[float])
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0)
        q_list = q_values.cpu().numpy().tolist()

        # During inference epsilon=0 so always exploit
        # During training still use epsilon-greedy
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = int(np.argmax(q_list))

        return action, q_list

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train on a random minibatch from memory."""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (Bellman equation)
        with torch.no_grad():
            next_q  = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target(self):
        """Copy policy net weights to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path=None):
        path = path or os.path.join(MODELS_DIR, "dqn_best.pth")
        torch.save({
            "policy_net":   self.policy_net.state_dict(),
            "target_net":   self.target_net.state_dict(),
            "epsilon":      self.epsilon,
            "episode":      self.episode,
        }, path)
        print(f"[DQN] Saved → {path}")

    def load(self, path=None):
        path = path or os.path.join(MODELS_DIR, "dqn_best.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"DQN weights not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.episode = checkpoint.get("episode", 0)
        self.policy_net.eval()
        print(f"[DQN] Loaded from {path} | episode={self.episode} | ε={self.epsilon:.3f}")

    def is_trained(self) -> bool:
        return os.path.exists(os.path.join(MODELS_DIR, "dqn_best.pth"))