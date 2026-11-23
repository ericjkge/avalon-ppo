"""Figures for the slides.

Usage: python -m scripts.plot [--tier hard]
  results/figures/winrate_<tier>.png   bar chart, PPO-side win rate per condition (A and B)
  results/figures/curves.png           eval win rate during training, per condition
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from avalon.evaluate import wilson_ci  # noqa: E402

COND = ["none", "deductive", "oracle"]
LABEL = {"none": "No belief", "deductive": "Deductive", "oracle": "Oracle"}
COLOR = {"none": "#8a8f98", "deductive": "#3b6fd8", "oracle": "#e0883a"}


def bars(tier, out):
    rows = list(csv.DictReader(open(f"results/csv/eval_{tier}.csv")))
    agg = defaultdict(lambda: [0, 0, []])
    for r in rows:
        a = agg[(r["condition"], r["setup"])]
        a[0] += int(r["wins"])
        a[1] += int(r["n"])
        a[2].append(float(r["rate"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    titles = {"A": "Setup A: PPO plays good", "B": "Setup B: PPO plays evil"}
    for ax, setup in zip(axes, ("A", "B")):
        for j, c in enumerate(COND):
            if (c, setup) not in agg:
                continue
            wins, n, rates = agg[(c, setup)]
            lo, hi = wilson_ci(wins, n)
            p = wins / n
            ax.bar(j, p, color=COLOR[c], width=0.6)
            ax.errorbar(j, p, yerr=[[p - lo], [hi - p]], color="black", capsize=4, lw=1)
            ax.scatter([j] * len(rates), rates, color="black", s=12, zorder=3, alpha=0.6)
            ax.text(j, hi + 0.015, f"{p:.2f}", ha="center", fontsize=10)
        ref = agg.get(("bots", setup))
        if ref:
            ax.axhline(ref[0] / ref[1], color="black", ls="--", lw=1)
            ax.text(-0.45, ref[0] / ref[1] + 0.015, f"{tier} bots: {ref[0] / ref[1]:.2f}", fontsize=8, ha="left")
        ax.set_xticks(range(3), [LABEL[c] for c in COND])
        ax.set_title(titles[setup])
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"PPO-side win rate vs {tier} bots")
    fig.tight_layout()
    fig.savefig(out / f"winrate_{tier}.png", dpi=200)
    print("wrote", out / f"winrate_{tier}.png")


def curves(out, runs="results/runs"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for run in sorted(Path(runs).iterdir()):
        if not (run / "log.csv").exists():
            continue
        c = json.loads((run / "config.json").read_text())["condition"]
        rows = [r for r in csv.DictReader(open(run / "log.csv")) if r["evalA"]]
        x = np.array([int(r["steps"]) for r in rows]) / 1e6
        for ax, key in zip(axes, ("evalA", "evalB")):
            ax.plot(x, [float(r[key]) for r in rows], color=COLOR[c], alpha=0.8, lw=1.2,
                    label=LABEL[c] if run.name.endswith("_s0") else None)
    for ax, t in zip(axes, ("Setup A: PPO plays good", "Setup B: PPO plays evil")):
        ax.set_title(t)
        ax.set_xlabel("training decisions (millions)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("win rate vs bots (eval during training)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=200)
    print("wrote", out / "curves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="hard")
    args = ap.parse_args()
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    bars(args.tier, out)
    curves(out)


if __name__ == "__main__":
    main()
