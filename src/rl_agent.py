

import argparse
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class FPLSeasonEnv:
    """
    Lightweight simulation environment for training the RL agent offline
    using the BiLSTM forecaster's predicted points as the reward model.
    A simplified action space is used: each step the agent chooses how many
    transfers to make (0-2) from a ranked shortlist, and which of its 2
    highest-EV players to captain.
    """

    def __init__(self, player_pool: np.ndarray, n_gameweeks: int = 38, squad_size: int = 15,
                 budget: float = 100.0, free_transfers: int = 1):
        self.player_pool = player_pool  # shape (n_players, n_gameweeks) of forecast EV
        self.n_gameweeks = n_gameweeks
        self.squad_size = squad_size
        self.budget = budget
        self.free_transfers = free_transfers
        self.reset()

    def reset(self):
        self.gw = 0
        self.squad = list(np.random.choice(len(self.player_pool), self.squad_size, replace=False))
        self.bank = self.budget
        self.ft = self.free_transfers
        return self._get_state()

    def _get_state(self):
        horizon = min(3, self.n_gameweeks - self.gw)
        squad_ev = self.player_pool[self.squad, self.gw:self.gw + horizon].mean(axis=1)
        return np.concatenate([squad_ev, [self.bank, self.ft, self.gw / self.n_gameweeks]])

    def step(self, action: int):
        """action in {0: hold, 1: make 1 transfer to best available, 2: take a hit for 2 transfers}"""
        reward = 0.0
        if action in (1, 2):
            n_transfers = 1 if action == 1 else 2
            hit = max(0, n_transfers - self.ft) * 4  # -4 pts per transfer beyond free ones
            worst_idx = int(np.argmin(self.player_pool[self.squad, self.gw]))
            candidates = [i for i in range(len(self.player_pool)) if i not in self.squad]
            best_candidate = max(candidates, key=lambda i: self.player_pool[i, self.gw:self.gw + 3].mean())
            self.squad[worst_idx] = best_candidate
            self.ft = max(0, self.ft - n_transfers) if self.ft >= n_transfers else 0
            reward -= hit
        else:
            self.ft = min(2, self.ft + 1)

        captain_idx = int(np.argmax(self.player_pool[self.squad, self.gw]))
        gw_points = self.player_pool[self.squad, self.gw].sum() + self.player_pool[self.squad[captain_idx], self.gw]
        reward += gw_points

        self.gw += 1
        done = self.gw >= self.n_gameweeks
        next_state = self._get_state() if not done else np.zeros_like(self._get_state())
        return next_state, reward, done


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int = 3, gamma: float = 0.95, lr: float = 1e-3):
        self.q_net = QNetwork(state_dim, n_actions)
        self.target_net = QNetwork(state_dim, n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.n_actions = n_actions
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.q_net(torch.tensor(state, dtype=torch.float32))
            return int(torch.argmax(q).item())

    def remember(self, *transition):
        self.memory.append(Transition(*transition))

    def replay(self, batch_size: int = 64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch])
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(0.05, self.epsilon * 0.995)


def train_agent(n_players: int = 200, n_gameweeks: int = 38, episodes: int = 2000):
    rng = np.random.default_rng(42)
    player_pool = np.clip(rng.normal(4, 2.5, size=(n_players, n_gameweeks)), 0, None)
    env = FPLSeasonEnv(player_pool, n_gameweeks=n_gameweeks)
    state_dim = len(env.reset())
    agent = DQNAgent(state_dim)

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
        if ep % 50 == 0:
            agent.target_net.load_state_dict(agent.q_net.state_dict())
            print(f"Episode {ep}/{episodes} | season_points={total_reward:.0f} | epsilon={agent.epsilon:.3f}")

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--episodes", type=int, default=2000)
    args = parser.parse_args()
    if args.train:
        train_agent(episodes=args.episodes)
