"""Game runner and win-rate helpers."""

import math
from collections import defaultdict

import numpy as np
import torch

from .bots import Bot
from .env import AvalonEnv, EVIL, GOOD
from .obs import encode

SETUPS = ("A", "B")  # A: PPO plays good vs bot evil; B: PPO plays evil vs bot good


def rates_by_seed(rows):
    """Group evaluation counts by independent training run, including older CSV files."""
    rates = defaultdict(dict)
    for row in rows:
        condition = row["condition"]
        seed = "-" if condition == "bots" else int(row["seed"])
        group = rates[condition, row["setup"]]
        if seed in group:
            raise ValueError(f"Duplicate {condition} seed {seed} in setup {row['setup']}")
        group[seed] = int(row["wins"]) / int(row["n"])
    return rates


class PolicyAgent:
    """Wraps a trained network so it can play a seat. Samples from the policy."""

    def __init__(self, model, condition, rng):
        self.model, self.condition, self.rng = model, condition, rng

    @torch.no_grad()
    def act(self, env):
        seat = env.current_player()
        obs = torch.from_numpy(encode(env, seat, self.condition))[None]
        mask = torch.from_numpy(env.legal_mask())[None]
        logits, _ = self.model(obs, mask)
        p = torch.softmax(logits[0], -1).numpy().astype(np.float64)
        return int(self.rng.choice(len(p), p=p / p.sum()))


def setup_win_rate(model, condition, setup, tier, n_games, seed=0):
    """Win rate of the PPO side in Setup A or B against `tier` bots."""
    rng = np.random.default_rng(seed)
    agent, bot = PolicyAgent(model, condition, rng), Bot(tier, rng)
    ppo_side = GOOD if setup == "A" else EVIL
    env = AvalonEnv(seed)
    wins = 0
    model.eval()
    for _ in range(n_games):
        env.reset()
        agents = [agent if env.side(s) == ppo_side else bot for s in range(5)]
        winner, _ = play_game(env, agents)
        wins += winner == ppo_side
    model.train()
    return wins, n_games


def play_game(env, agents):
    """Play one game to completion. `agents[seat].act(env)` returns an action."""
    while not env.done:
        env.step(agents[env.current_player()].act(env))
    return env.winner, env.win_reason


def wilson_ci(wins, n, z=1.96):
    """95% Wilson interval for a win rate."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def good_win_rate(make_agents, n_games, seed=0):
    """make_agents(env) -> list of 5 agents (env roles already dealt)."""
    env = AvalonEnv(seed)
    wins, reasons = 0, {}
    for _ in range(n_games):
        env.reset()
        winner, reason = play_game(env, make_agents(env))
        wins += winner == GOOD
        reasons[reason] = reasons.get(reason, 0) + 1
    return wins / n_games, wilson_ci(wins, n_games), reasons
