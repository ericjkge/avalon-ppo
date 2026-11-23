"""Train one self-play PPO run.

Usage: python -m scripts.train --condition {none,deductive,oracle} --seed N [--steps 3000000]
Writes results/runs/<condition>_s<seed>/{config.json, log.csv, model.pt, ckpt_*.pt}
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from avalon.belief import CONDITIONS
from avalon.evaluate import setup_win_rate
from avalon.model import ActorCritic
from avalon.pool import OpponentPool
from avalon.ppo import Collector, ppo_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--pool-every", type=int, default=10)
    ap.add_argument("--pool-size", type=int, default=20)
    ap.add_argument("--p-selfplay", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--eval-tier", default="hard")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    run = Path(args.out) / f"{args.condition}_s{args.seed}{args.tag}"
    run.mkdir(parents=True, exist_ok=True)
    if any(run.iterdir()):
        ap.error(f"{run} already contains a run. Choose a new --out or --tag.")
    try:
        config = (run / "config.json").open("x")
    except FileExistsError:
        ap.error(f"Another trainer claimed {run}. Choose a new --out or --tag.")
    with config:
        json.dump(dict(vars(args), rollout_method="complete_games"), config, indent=2)

    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    pool = OpponentPool(args.pool_every, args.pool_size)
    collector = Collector(model, pool, args.condition, args.n_envs, args.seed, args.p_selfplay)

    fields = ["update", "steps", "minutes", "selfplay_good_wr", "pool_cur_wr", "pg_loss", "v_loss",
              "entropy", "approx_kl", "clipfrac", "evalA", "evalB"]
    f = open(run / "log.csv", "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()

    t0, steps, update = time.time(), 0, 0
    while steps < args.steps:
        batch = collector.collect(min(args.batch, args.steps - steps))
        steps += len(batch["act"])
        metrics = ppo_update(model, opt, batch, args.epochs, args.minibatch, args.clip, args.ent)
        update += 1
        pool.maybe_add(update, model)
        s = collector.pop_stats()
        row = dict(update=update, steps=steps, minutes=round((time.time() - t0) / 60, 2), **metrics)
        row["selfplay_good_wr"] = s.get("selfplay_good_wins", 0) / max(s.get("selfplay_games", 0), 1)
        row["pool_cur_wr"] = s.get("pool_cur_wins", 0) / max(s.get("pool_games", 0), 1)
        if update % args.eval_every == 0 or steps >= args.steps:
            for setup in ("A", "B"):
                wins, n = setup_win_rate(model, args.condition, setup, args.eval_tier,
                                         args.eval_games, seed=10_000 + update)
                row[f"eval{setup}"] = wins / n
            torch.save(model.state_dict(), run / f"ckpt_{update:05d}.pt")
            print(f"[{run.name}] upd {update} steps {steps} {row['minutes']}m "
                  f"ent {row['entropy']:.3f} kl {row['approx_kl']:.4f} sp_good {row['selfplay_good_wr']:.2f} "
                  f"A {row['evalA']:.2f} B {row['evalB']:.2f}", flush=True)
        w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        f.flush()

    f.close()
    # A reader can use model.pt as a completion marker, including during a run.
    final = run / "model.pt"
    pending = run / "model.pt.tmp"
    torch.save(model.state_dict(), pending)
    pending.replace(final)
    print(f"[{run.name}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
