"""Self-play rollouts with an opponent pool, GAE, and the PPO update."""

from collections import defaultdict

import numpy as np
import torch
from torch.distributions import Categorical

from .env import N_PLAYERS, GOOD, AvalonEnv
from .obs import encode

CURRENT = "current"


class Collector:
    """Runs n_envs games in parallel. Each game is either current-vs-current
    self-play, or current vs. a pool snapshot (current plays one random side).
    Only decisions made by the current network are kept for training."""

    def __init__(self, model, pool, condition, n_envs, seed, p_selfplay=0.5, gamma=1.0, lam=0.95):
        if n_envs < 1:
            raise ValueError("n_envs must be positive")
        self.model, self.pool, self.condition = model, pool, condition
        self.rng = np.random.default_rng(seed)
        self.envs = [AvalonEnv(seed * 10_000 + i) for i in range(n_envs)]
        self.p_selfplay, self.gamma, self.lam = p_selfplay, gamma, lam
        self.ctrl = [None] * n_envs
        self.traj = [None] * n_envs
        self.stats = defaultdict(int)

    def _reset(self, i):
        env = self.envs[i].reset()
        self.traj[i] = [[] for _ in range(N_PLAYERS)]
        snap = None
        if len(self.pool) and self.rng.random() >= self.p_selfplay:
            snap = self.pool.sample(self.rng)
        if snap is None:
            self.ctrl[i] = [CURRENT] * N_PLAYERS
        else:
            side = int(self.rng.integers(2))
            self.ctrl[i] = [CURRENT if env.side(s) == side else snap for s in range(N_PLAYERS)]

    def _finish(self, i, out):
        env = self.envs[i]
        selfplay = all(c is CURRENT for c in self.ctrl[i])
        if selfplay:
            self.stats["selfplay_games"] += 1
            self.stats["selfplay_good_wins"] += env.winner == GOOD
        else:
            cur_seat = self.ctrl[i].index(CURRENT)
            self.stats["pool_games"] += 1
            self.stats["pool_cur_wins"] += env.winner == env.side(cur_seat)
        for seat in range(N_PLAYERS):
            t = self.traj[i][seat]
            if not t:
                continue
            values = np.array([x[4] for x in t], dtype=np.float32)
            rewards = np.zeros(len(t), dtype=np.float32)
            rewards[-1] = env.reward(seat)
            adv = np.zeros(len(t), dtype=np.float32)
            last = 0.0
            for k in reversed(range(len(t))):
                next_v = values[k + 1] if k + 1 < len(t) else 0.0
                delta = rewards[k] + self.gamma * next_v - values[k]
                last = delta + self.gamma * self.lam * last
                adv[k] = last
            for k, (o, m, a, lp, v) in enumerate(t):
                out["obs"].append(o)
                out["mask"].append(m)
                out["act"].append(a)
                out["logp"].append(lp)
                out["adv"].append(adv[k])
                out["ret"].append(adv[k] + values[k])
            t.clear()

    @torch.no_grad()
    def collect(self, n_steps):
        """Collect at least n_steps current-policy decisions, finishing every started game."""
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        out = defaultdict(list)
        steps = 0
        active = list(range(len(self.envs)))
        for i in active:
            self._reset(i)
        while active:
            groups = defaultdict(list)
            for i in active:
                env = self.envs[i]
                seat = env.current_player()
                groups[id(self.ctrl[i][seat])].append(
                    (i, seat, encode(env, seat, self.condition), env.legal_mask()))
            actions = {}
            for items in groups.values():
                i0, s0 = items[0][0], items[0][1]
                ctrl = self.ctrl[i0][s0]
                net = self.model if ctrl is CURRENT else ctrl
                obs = torch.from_numpy(np.stack([x[2] for x in items]))
                mask = torch.from_numpy(np.stack([x[3] for x in items]))
                logits, values = net(obs, mask)
                dist = Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)
                if ctrl is CURRENT:
                    steps += len(items)
                for j, (i, seat, o, m) in enumerate(items):
                    actions[i] = int(a[j])
                    if ctrl is CURRENT:
                        self.traj[i][seat].append((o, m, int(a[j]), float(logp[j]), float(values[j])))
            next_active = []
            for i in active:
                env = self.envs[i]
                env.step(actions[i])
                if env.done:
                    self._finish(i, out)
                    if steps >= n_steps:
                        continue
                    self._reset(i)
                next_active.append(i)
            active = next_active
        batch = {
            "obs": torch.from_numpy(np.stack(out["obs"])),
            "mask": torch.from_numpy(np.stack(out["mask"])),
            "act": torch.tensor(out["act"]),
            "logp": torch.tensor(out["logp"]),
            "adv": torch.tensor(out["adv"]),
            "ret": torch.tensor(out["ret"]),
        }
        return batch

    def pop_stats(self):
        s, self.stats = dict(self.stats), defaultdict(int)
        return s


def ppo_update(model, opt, batch, epochs=4, minibatch=1024, clip=0.2, ent_coef=0.01,
               vf_coef=0.5, max_grad_norm=0.5):
    n = len(batch["act"])
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    logs = defaultdict(list)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch):
            idx = perm[start:start + minibatch]
            logits, v = model(batch["obs"][idx], batch["mask"][idx])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(batch["act"][idx])
            ratio = torch.exp(logp - batch["logp"][idx])
            a = adv[idx]
            pg_loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            v_loss = 0.5 * (v - batch["ret"][idx]).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = pg_loss + vf_coef * v_loss - ent_coef * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            with torch.no_grad():
                logs["pg_loss"].append(pg_loss.item())
                logs["v_loss"].append(v_loss.item())
                logs["entropy"].append(entropy.item())
                logs["approx_kl"].append((batch["logp"][idx] - logp).mean().item())
                logs["clipfrac"].append(((ratio - 1).abs() > clip).float().mean().item())
    return {k: float(np.mean(v)) for k, v in logs.items()}
