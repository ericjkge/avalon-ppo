"""Train one self-play Double DQN run. Every seat plays the current Q-network
with epsilon-greedy exploration; each seat's decisions form its own transitions.

Usage: python -m scripts.train_dqn --seed N --tag _dqn [--steps 3000000]
Writes the same files as scripts.train. The saved network's policy head holds Q-values.
"""

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from avalon.belief import CONDITIONS
from avalon.env import GOOD, N_ACTIONS, N_PLAYERS, AvalonEnv
from avalon.evaluate import setup_win_rate
from avalon.model import ActorCritic
from avalon.obs import OBS_DIM, encode


class Replay:
    def __init__(self, size):
        self.size, self.n, self.i = size, 0, 0
        self.obs = np.zeros((size, OBS_DIM), dtype=np.float16)
        self.obs2 = np.zeros((size, OBS_DIM), dtype=np.float16)
        self.mask2 = np.ones((size, N_ACTIONS), dtype=bool)
        self.act = np.zeros(size, dtype=np.int64)
        self.rew = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)

    def add(self, obs, act, rew, obs2=None, mask2=None):
        i = self.i
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.done[i] = obs2 is None
        self.obs2[i] = 0 if obs2 is None else obs2
        self.mask2[i] = True if mask2 is None else mask2
        self.i = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, rng, k):
        j = rng.integers(self.n, size=k)
        t = torch.from_numpy
        return (t(self.obs[j].astype(np.float32)), t(self.act[j]), t(self.rew[j]),
                t(self.obs2[j].astype(np.float32)), t(self.mask2[j]), t(self.done[j]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS, default="deductive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--no-talk", dest="talk", action="store_false")
    ap.add_argument("--quest-bonus", type=float, default=0.2)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--collect", type=int, default=4096, help="decisions between learning phases")
    ap.add_argument("--grad-steps", type=int, default=128, help="gradient steps per learning phase")
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--target-every", type=int, default=1000, help="gradient steps between target copies")
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-frac", type=float, default=0.25, help="share of the budget spent annealing epsilon")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--eval-tier", default="hard")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--tag", default="_dqn")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run = Path(args.out) / f"{args.condition}_s{args.seed}{args.tag}"
    run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        ap.error(f"{run} already contains a run. Choose a new --out or --tag.")
    (run / "config.json").write_text(json.dumps(dict(vars(args), algo="dqn"), indent=2))

    q = ActorCritic()
    target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    buf = Replay(args.buffer)
    envs = [AvalonEnv(args.seed * 10_000 + i, talk=args.talk) for i in range(args.n_envs)]
    pending = [[None] * N_PLAYERS for _ in envs]  # per seat: (obs, action, quests won so far)
    all_legal = torch.ones(args.minibatch, N_ACTIONS, dtype=torch.bool)

    fields = ["update", "steps", "minutes", "selfplay_good_wr", "epsilon", "td_loss", "q_mean", "evalA", "evalB"]
    f = open(run / "log.csv", "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()

    t0, steps, update, grad_steps = time.time(), 0, 0, 0
    while steps < args.steps:
        eps = max(args.eps_end, 1 - (1 - args.eps_end) * steps / (args.eps_frac * args.steps))
        games = good_wins = 0
        goal = steps + args.collect
        while steps < goal:
            seats = [e.current_player() for e in envs]
            obs = np.stack([encode(e, s, args.condition) for e, s in zip(envs, seats)])
            masks = np.stack([e.legal_mask() for e in envs])
            with torch.no_grad():
                greedy = q(torch.from_numpy(obs), torch.from_numpy(masks))[0].argmax(1).numpy()
            for i, (env, seat) in enumerate(zip(envs, seats)):
                a = int(rng.choice(np.flatnonzero(masks[i]))) if rng.random() < eps else int(greedy[i])
                prev = pending[i][seat]
                if prev is not None:
                    buf.add(prev[0], prev[1], args.quest_bonus * (env.quests_won(seat) - prev[2]), obs[i], masks[i])
                pending[i][seat] = (obs[i], a, env.quests_won(seat))
                env.step(a)
                if env.done:
                    for s, prev in enumerate(pending[i]):
                        if prev is not None:
                            buf.add(prev[0], prev[1],
                                    args.quest_bonus * (env.quests_won(s) - prev[2]) + env.reward(s))
                    pending[i] = [None] * N_PLAYERS
                    games += 1
                    good_wins += env.winner == GOOD
                    env.reset()
            steps += len(envs)

        losses, qs = [], []
        if buf.n >= args.warmup:
            for _ in range(args.grad_steps):
                o, a, r, o2, m2, d = buf.sample(rng, args.minibatch)
                with torch.no_grad():  # Double DQN: the online net picks, the target net scores
                    a2 = q(o2, m2)[0].argmax(1, keepdim=True)
                    y = r + (1 - d) * target(o2, m2)[0].gather(1, a2).squeeze(1)
                qsa = q(o, all_legal)[0].gather(1, a[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(qsa, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                grad_steps += 1
                if grad_steps % args.target_every == 0:
                    target.load_state_dict(q.state_dict())
                losses.append(loss.item())
                qs.append(qsa.mean().item())

        update += 1
        row = dict(update=update, steps=steps, minutes=round((time.time() - t0) / 60, 2), epsilon=eps,
                   selfplay_good_wr=good_wins / max(games, 1),
                   td_loss=float(np.mean(losses)) if losses else "", q_mean=float(np.mean(qs)) if qs else "")
        if update % args.eval_every == 0 or steps >= args.steps:
            for setup in ("A", "B"):
                wins, n = setup_win_rate(q, args.condition, setup, args.eval_tier, args.eval_games,
                                         seed=10_000 + update, talk=args.talk, greedy=True)
                row[f"eval{setup}"] = wins / n
            torch.save(q.state_dict(), run / f"ckpt_{update:05d}.pt")
            print(f"[{run.name}] upd {update} steps {steps} {row['minutes']}m eps {eps:.2f} "
                  f"loss {row['td_loss']} sp_good {row['selfplay_good_wr']:.2f} "
                  f"A {row['evalA']:.2f} B {row['evalB']:.2f}", flush=True)
        w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        f.flush()

    f.close()
    pending_file = run / "model.pt.tmp"
    torch.save(q.state_dict(), pending_file)
    pending_file.replace(run / "model.pt")
    print(f"[{run.name}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
