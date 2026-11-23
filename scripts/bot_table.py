"""Bot-vs-bot calibration table: good win rate for every (good tier, evil tier).

Usage: python -m scripts.bot_table [--games 2000] [--p-hide ...] [--p-trust ...] [--tau ...]
                                   [--q-evil ...] [--q-merlin ...] [--q-fail ...]
"""

import argparse

import numpy as np

from avalon.bots import Bot, TIERS
from avalon.evaluate import good_win_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--p-hide", type=float, default=0.3)
    ap.add_argument("--p-trust", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--q-evil", type=float, default=0.55)
    ap.add_argument("--q-merlin", type=float, default=0.55)
    ap.add_argument("--q-fail", type=float, default=0.8)
    ap.add_argument("--p-blend", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    kw = dict(p_hide=args.p_hide, p_trust=args.p_trust, tau=args.tau,
              q_evil=args.q_evil, q_merlin=args.q_merlin, q_fail=args.q_fail, p_blend=args.p_blend)

    print(f"Good win rate, {args.games} games per cell (rows: good tier, cols: evil tier)")
    print(f"{'':>8}" + "".join(f"{t:>14}" for t in TIERS))
    diag = {}
    for g in TIERS:
        row = f"{g:>8}"
        for e in TIERS:
            good, evil = Bot(g, rng, **kw), Bot(e, rng, **kw)

            def make_agents(env, good=good, evil=evil):
                return [evil if s in env.evil else good for s in range(5)]

            wr, (lo, hi), reasons = good_win_rate(make_agents, args.games, args.seed)
            row += f"{wr:>7.2f} ±{(hi - lo) / 2:.2f} "
            if g == e:
                diag[g] = reasons
        print(row)
    print("\nHow games end (same tier on both sides):")
    for t, reasons in diag.items():
        total = sum(reasons.values())
        print(f"  {t:>6}: " + ", ".join(f"{k} {v / total:.0%}" for k, v in sorted(reasons.items())))


if __name__ == "__main__":
    main()
