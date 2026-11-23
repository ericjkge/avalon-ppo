"""Evaluate matched training runs vs. bots in Setups A and B, plus the bot reference.

Usage: python -m scripts.eval [--runs results/runs] [--games 2000] [--tier hard]
Writes results/csv/eval_<tier>.csv and prints a summary table.
"""

import argparse
import csv
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch

from avalon.belief import CONDITIONS
from avalon.bots import Bot, TIERS
from avalon.evaluate import good_win_rate, rates_by_seed, setup_win_rate, wilson_ci
from avalon.model import ActorCritic


def select_runs(root, conditions, seeds, steps, tag=""):
    """Choose one named, finished checkpoint per experimental cell before any games run."""
    if len(set(seeds)) != len(seeds) or len(set(conditions)) != len(conditions):
        raise ValueError("Each condition and seed must appear only once")
    training_keys = ("n_envs", "batch", "minibatch", "epochs", "lr", "ent", "clip",
                     "pool_every", "pool_size", "p_selfplay")
    selected = []
    reference = None
    for condition, seed in product(conditions, seeds):
        run = root / f"{condition}_s{seed}{tag}"
        for name in ("config.json", "log.csv", "model.pt"):
            if not (run / name).is_file():
                raise ValueError(f"Missing {run / name}")
        cfg = json.loads((run / "config.json").read_text())
        expected = dict(condition=condition, seed=seed, steps=steps, tag=tag)
        for key, value in expected.items():
            if cfg.get(key) != value:
                raise ValueError(f"{run.name}: expected {key}={value!r}, found {cfg.get(key)!r}")
        with (run / "log.csv").open(newline="") as f:
            last_steps = 0
            for row in csv.DictReader(f):
                last_steps = int(row["steps"])
        if last_steps < steps:
            raise ValueError(f"{run.name}: training log has not reached {steps} decisions")
        settings = {k: cfg[k] for k in training_keys}
        settings["rollout_method"] = cfg.get("rollout_method", "legacy")
        if reference is None:
            reference = settings
        for k, v in reference.items():
            if settings[k] != v:
                raise ValueError(f"{run.name}: {k} differs from {selected[0][0].name}")
        selected.append((run, cfg))
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=40_000_000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--tier", choices=TIERS, default="hard")
    ap.add_argument("--out", default="results/csv")
    args = ap.parse_args()
    if min(args.games, args.steps) < 1:
        ap.error("--games and --steps must be positive")
    try:
        runs = select_runs(args.runs, args.conditions, args.seeds, args.steps, args.tag)
    except (KeyError, ValueError) as exc:
        ap.error(str(exc))
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
        rows.append(dict(condition="bots", seed="-", run="", setup=setup, wins=wins, n=args.games,
                         rate=wins / args.games, lo=lo, hi=hi))

    for run, cfg in runs:
        model = ActorCritic()
        model.load_state_dict(torch.load(run / "model.pt"))
        for setup in ("A", "B"):
            wins, n = setup_win_rate(model, cfg["condition"], setup, args.tier, args.games, seed=777)
            lo, hi = wilson_ci(wins, n)
            rows.append(dict(condition=cfg["condition"], seed=cfg["seed"], run=run.name, setup=setup, wins=wins,
                             n=n, rate=wins / n, lo=lo, hi=hi))
            print(f"{run.name:>16} setup {setup}: {wins / n:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)

    path = Path(args.out) / f"eval_{args.tier}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    by_seed = rates_by_seed(rows)
    for setup in ("A", "B"):
        print(f"\nSetup {setup}, bot reference {by_seed['bots', setup]['-']:.3f}")
        for c in args.conditions:
            rates = by_seed[c, setup]
            print(f"{c:>10} mean {np.mean(list(rates.values())):.3f}  " +
                  "  ".join(f"s{s} {rates[s]:.3f}" for s in args.seeds))
        for left, right in combinations(args.conditions, 2):
            diffs = [by_seed[right, setup][s] - by_seed[left, setup][s] for s in args.seeds]
            print(f"{right} minus {left}, mean {np.mean(diffs):+.3f}  " +
                  "  ".join(f"s{s} {d:+.3f}" for s, d in zip(args.seeds, diffs)))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
