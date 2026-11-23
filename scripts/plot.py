"""Figures for the slides.

Usage: python -m scripts.plot [--tier hard]
  results/figures/winrate_<tier>.png   bar chart, PPO-side win rate per condition (A and B)
  results/figures/curves.png           eval win rate during training, per condition
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from avalon.evaluate import rates_by_seed  # noqa: E402

COND = ["none", "deductive", "oracle"]
LABEL = {"none": "No belief", "deductive": "Deductive", "oracle": "Oracle"}
COLOR = {"none": "#8a8f98", "deductive": "#3b6fd8", "oracle": "#e0883a"}


def bars(tier, out, rows):
    agg = rates_by_seed(rows)
    conditions = [c for c in COND if (c, "A") in agg]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    titles = {"A": "Setup A: PPO plays good", "B": "Setup B: PPO plays evil"}
    for ax, setup in zip(axes, ("A", "B")):
        for j, c in enumerate(conditions):
            per_seed = agg[c, setup]
            rates = [per_seed[s] for s in sorted(per_seed)]
            mean = np.mean(rates)
            ax.bar(j, mean, color=COLOR[c], width=0.6, label="Seed mean" if j == 0 else None)
            offsets = np.linspace(-0.12, 0.12, len(rates)) if len(rates) > 1 else np.zeros(1)
            ax.scatter(j + offsets, rates, color="black", s=16, zorder=3, clip_on=False,
                       label="Training seed" if j == 0 else None)
        ref = agg.get(("bots", setup))
        if ref:
            ax.axhline(ref["-"], color="black", ls="--", lw=1, label=f"{tier.title()} bots")
        ax.set_xticks(range(len(conditions)), [LABEL[c] for c in conditions])
        ax.set_title(titles[setup])
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"PPO-side win rate vs {tier} bots")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out / f"winrate_{tier}.png", dpi=200)
    plt.close(fig)
    print("wrote", out / f"winrate_{tier}.png")


def curves(out, runs, evaluated):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    labelled = set()
    for cell in evaluated:
        c = cell["condition"]
        if c == "bots" or cell["setup"] != "A":
            continue
        run = runs / (cell.get("run") or f"{c}_s{cell['seed']}")
        with (run / "log.csv").open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["evalA"]]
        x = np.array([int(r["steps"]) for r in rows]) / 1e6
        for ax, key in zip(axes, ("evalA", "evalB")):
            ax.plot(x, [float(r[key]) for r in rows], color=COLOR[c], alpha=0.8, lw=1.2,
                    label=LABEL[c] if c not in labelled else None)
        labelled.add(c)
    for ax, t in zip(axes, ("Setup A: PPO plays good", "Setup B: PPO plays evil")):
        ax.set_title(t)
        ax.set_xlabel("training decisions (millions)")
        ax.locator_params(axis="x", nbins=5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("win rate vs bots (eval during training)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=200)
    plt.close(fig)
    print("wrote", out / "curves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="hard")
    ap.add_argument("--runs", type=Path, default=Path("results/runs"))
    ap.add_argument("--csv-dir", type=Path, default=Path("results/csv"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.csv_dir / f"eval_{args.tier}.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    bars(args.tier, args.out, rows)
    curves(args.out, args.runs, rows)


if __name__ == "__main__":
    main()
