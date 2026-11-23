"""Final evaluation: every trained run vs. bots in Setups A and B, plus the bot Reference.

Usage: python -m scripts.eval [--runs results/runs] [--games 2000] [--tier hard]
Writes results/csv/eval_<tier>.csv and prints a summary table.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from avalon.bots import Bot
from avalon.env import GOOD
from avalon.evaluate import good_win_rate, setup_win_rate, wilson_ci
from avalon.model import ActorCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--tier", default="hard")
    ap.add_argument("--out", default="results/csv")
    args = ap.parse_args()
    torch.set_num_threads(2)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    rows = []
    # Reference: bots in every seat. Report each side's win rate.
    rng = np.random.default_rng(123)
    bot = Bot(args.tier, rng)
    wr, _, _ = good_win_rate(lambda env: [bot] * 5, args.games, seed=123)
    gw = round(wr * args.games)
    for setup, wins in (("A", gw), ("B", args.games - gw)):
        lo, hi = wilson_ci(wins, args.games)
        rows.append(dict(condition="bots", seed="-", setup=setup, wins=wins, n=args.games,
                         rate=wins / args.games, lo=lo, hi=hi))

    for run in sorted(Path(args.runs).iterdir()):
        if not (run / "model.pt").exists():
            continue
        cfg = json.loads((run / "config.json").read_text())
        model = ActorCritic()
        model.load_state_dict(torch.load(run / "model.pt"))
        for setup in ("A", "B"):
            wins, n = setup_win_rate(model, cfg["condition"], setup, args.tier, args.games, seed=777)
            lo, hi = wilson_ci(wins, n)
            rows.append(dict(condition=cfg["condition"], seed=cfg["seed"], setup=setup, wins=wins,
                             n=n, rate=wins / n, lo=lo, hi=hi))
            print(f"{run.name:>16} setup {setup}: {wins / n:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)

    path = Path(args.out) / f"eval_{args.tier}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # Summary: pool games across seeds per condition
    agg = defaultdict(lambda: [0, 0, []])
    for r in rows:
        a = agg[(r["condition"], r["setup"])]
        a[0] += r["wins"]
        a[1] += r["n"]
        a[2].append(r["rate"])
    print(f"\nPPO-side win rate vs {args.tier} bots (A = PPO good, B = PPO evil; 'bots' = bot reference)")
    print(f"{'condition':>10} {'setup':>5} {'pooled':>8} {'95% CI':>16} {'per-seed':>24}")
    for (c, s), (wins, n, rates) in agg.items():
        lo, hi = wilson_ci(wins, n)
        print(f"{c:>10} {s:>5} {wins / n:8.3f} [{lo:.3f}, {hi:.3f}] {' '.join(f'{x:.3f}' for x in rates):>24}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
