# Overnight log

Decisions and results, in order. All runs are local; nothing is committed.

## Built
- `avalon/model.py` (MLP actor-critic, as in PLAN.md section 4)
- `avalon/pool.py` (save a copy every 10 updates, keep the last 20)
- `avalon/ppo.py` (64 parallel games, per-seat GAE, PPO update)
- `avalon/evaluate.py` (`PolicyAgent`, `setup_win_rate`)
- `scripts/train.py`, `scripts/eval.py`, `scripts/plot.py`
- Training speed: about 1M decisions per minute per run, much faster than the 10–40 min per 3M I estimated.

## Sanity run (deductive, seed 0, 10M decisions, PLAN.md default settings)
Win rate during training vs. Hard bots: Setup A rose from 0.03 to 0.27; Setup B stayed around 0.40–0.51. Still rising at 10M.

Final model, 1000 games per cell:
| Bot tier | A (PPO good) | B (PPO evil) |
|---|---|---|
| Easy | 0.33 | 0.93 |
| Medium | 0.30 | 0.46 |
| Hard | 0.31 | 0.48 |

Bot reference with Hard bots in every seat: good 0.39, evil 0.61. PPO is learning but is still weaker than the Hard bots on both sides at 10M.
**Sanity check passed:** PPO as good wins 31–33% against every evil tier; Easy good bots win 1% against Easy evil.

## Settings sweep (condition = none, so tuning can't favor the treatment)
4 variants × 10M decisions, run in parallel: base / entropy 0.003 / lr 1e-3 / batch 32768 (minibatch 4096).

Results: final model vs. Hard bots, 1000 games each.
| Variant | A | B | Mean |
|---|---|---|---|
| **base** (PLAN.md settings) | 0.235 | 0.494 | **0.364** |
| batch 32768 | 0.225 | 0.440 | 0.333 |
| lr 1e-3 | 0.220 | 0.440 | 0.330 |
| entropy 0.003 | 0.160 | 0.405 | 0.282 |

**Decision:** keep PLAN.md's settings. Raise the budget from 3M to **40M decisions per run**: win rate was still rising at 10M, and each run takes under an hour.
(Side note: at 10M, none-base A = 0.235 vs. the deductive sanity run's A = 0.31. One seed each, so it's only a hint.)

## Main runs
- Seed 0 for all three conditions launched first (3 in parallel), then seeds 1–2.
- Seed 0 done (about 50–55 min each, 3 in parallel). Win rate vs. Hard bots at the last in-training eval (300 games, noisy):
  | Condition | A (good) | B (evil) |
  |---|---|---|
  | none | 0.29–0.34 | 0.46–0.49 |
  | deductive | 0.46–0.52 | 0.47–0.48 |
  | oracle | 0.49–0.58 | 0.44–0.46 |
- Seeds 1–2: 6 runs in parallel with `--threads 1` (only changes speed, not results).
- Seeds 1–2 done (62–67 min each, 6 in parallel). All 9 runs finished with no errors.

## Final results (`scripts/eval.py`; 2000 games per run per setup vs. Hard bots)
PPO-side win rate. A = PPO plays all good seats; B = PPO plays both evil seats.

| Condition | A pooled [95% CI] | A per seed (s0 s1 s2) | B pooled [95% CI] | B per seed |
|---|---|---|---|---|
| Hard bots (reference) | 0.417 [0.396, 0.439] | — | 0.583 [0.561, 0.604] | — |
| None | **0.281** [0.270, 0.293] | 0.299 0.243 0.302 | 0.448 [0.436, 0.461] | 0.454 0.414 0.477 |
| Deductive | **0.422** [0.410, 0.435] | 0.469 0.422 0.374 | 0.491 [0.479, 0.504] | 0.470 0.510 0.492 |
| Oracle | **0.533** [0.521, 0.546] | 0.574 0.476 0.550 | 0.453 [0.440, 0.465] | 0.432 0.484 0.442 |

Files: `results/csv/eval_hard.csv`, `results/figures/winrate_hard.png`, `results/figures/curves.png`.

### Takeaways
1. **Headline: deductive belief closes about 56% of the None → Oracle gap in Setup A:** (0.422 − 0.281) / (0.533 − 0.281). The per-seed ranges don't overlap: none 0.24–0.30, deductive 0.37–0.47, oracle 0.48–0.57.
2. **Deductive good roughly matches the Hard good bots** (0.42 vs. 0.42). No-belief PPO is clearly worse than the bots; Oracle is better.
3. **Learning curves:** Oracle plateaus by about 10M decisions. None plateaus around 0.30. Deductive tracks None until about 25M, then pulls ahead and is **still rising at 40M**, so its final number likely understates it. The deductive input seems to make a better policy learnable, but more slowly.
4. **Setup B (PPO evil) barely learns:** flat at about 0.45 from the start in every condition, below the Hard evil bots (0.58). Deductive evil is slightly higher (0.49). The belief input never changes what evil sees, so the gap can only come from training against different good teammates in self-play. With 3 seeds, treat it as weak evidence.

### Caveats
- The pooled CIs treat all 6000 games as independent and ignore seed-to-seed variance. Cite the per-seed ranges alongside them.
- The bot reference here is 0.417; the calibration table gave 0.39. That's sampling noise (different seeds, about ±0.02).
- Settings were tuned briefly on the None condition only (see sweep); no per-condition tuning.
- One step budget (40M); Deductive hadn't plateaued.

### Possible next steps (not done)
- Train Deductive (and None, for fairness) to 80M to see if the gap widens.
- Find out why evil training stalls: maybe evil's reward is noisier, or the pool is mostly weak early copies.
