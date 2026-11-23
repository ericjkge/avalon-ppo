# Avalon PPO

Self-play PPO agents for 5-player Avalon (no chat), built for a CS 1840 mini project.

**Question:** how much does explicit belief modeling help self-play PPO in a hidden-role game? We vary only the belief input that Servants see (none / deductive / oracle) and measure win rate against rule-based Hard bots. See [PLAN.md](PLAN.md) for the design and [LOG.md](LOG.md) for every decision and result.

## Results
3 seeds per condition, 40M training decisions each, 2000 evaluation games per seed per setup.

| Condition | PPO plays good (Setup A) | PPO plays evil (Setup B) |
|---|---|---|
| No belief | 0.28 (seeds 0.24–0.30) | 0.45 |
| Deductive | 0.42 (seeds 0.37–0.47) | 0.49 |
| Oracle | 0.53 (seeds 0.48–0.57) | 0.45 |
| Hard bots in every seat (reference) | 0.42 | 0.58 |

## How to run
```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate

pytest                                                    # 34 tests
python -m scripts.bot_table                               # bot-vs-bot calibration table
python -m scripts.train --condition deductive --seed 0    # one run (~1 hour for 40M decisions)
python -m scripts.eval --games 2000 --tier hard           # evaluate every run in results/runs
python -m scripts.plot                                    # figures in results/figures
```
To reproduce the results, run `train.py` with `--steps 40000000` for each condition (`none`, `deductive`, `oracle`) and seed (0–2). The default is 3M.

## Repo layout
```
avalon/
  env.py        game rules, phases, action mask
  obs.py        observation vector (~490 floats)
  belief.py     belief inputs (none / deductive / oracle) + Bayesian belief for bots
  bots.py       Easy / Medium / Hard rule-based bots
  model.py      MLP actor-critic
  ppo.py        parallel self-play rollouts, GAE, PPO update
  pool.py       opponent pool of past network copies
  evaluate.py   game runner, Setup A/B win rates, confidence intervals
scripts/        train, eval, plot, bot_table
tests/          env, belief and bot tests
results/        runs, csv, figures (not in git)
```
