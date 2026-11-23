"""MLP actor-critic shared by all roles (PLAN.md section 4)."""

import numpy as np
import torch
import torch.nn as nn

from .env import N_ACTIONS
from .obs import OBS_DIM

MASKED_LOGIT = -1e9  # finite, so masked entries never produce NaNs


def _init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=256, n_actions=N_ACTIONS):
        super().__init__()
        self.trunk = nn.Sequential(
            _init(nn.Linear(obs_dim, hidden), np.sqrt(2)), nn.ReLU(),
            _init(nn.Linear(hidden, hidden), np.sqrt(2)), nn.ReLU(),
        )
        self.pi = _init(nn.Linear(hidden, n_actions), 0.01)
        self.v = _init(nn.Linear(hidden, 1), 1.0)

    def forward(self, obs, mask):
        h = self.trunk(obs)
        logits = self.pi(h).masked_fill(~mask, MASKED_LOGIT)
        return logits, self.v(h).squeeze(-1)
