# Avalon PPO

Self-play PPO agents for 5-player Avalon (no chat), built for a CS 1840 mini project.

**Question:** how much does explicit belief modeling help self-play PPO in a hidden-role game? We vary only the belief input that Servants see (none / deductive / oracle) and measure win rate against rule-based Hard bots. See [PLAN.md](PLAN.md) for the design and [LOG.md](LOG.md) for every decision and result.

## Initial results
These figures come from commit `ebe920c`, before the rollout and Hard-bot corrections. That experiment used 3 seeds per condition, 40M training decisions each and 2000 evaluation games per seed per setup.

| Condition | PPO plays good (Setup A) | PPO plays evil (Setup B) |
|---|---|---|
| No belief | 0.28 (seeds 0.24–0.30) | 0.45 |
| Deductive | 0.42 (seeds 0.37–0.47) | 0.49 |
| Oracle | 0.53 (seeds 0.48–0.57) | 0.45 |
| Hard bots in every seat (reference) | 0.42 | 0.58 |

With the corrected Hard bot and the evaluation reference seed (123), good won 845 of 2,000 bot-only games (42.25%, 95% Wilson interval 40.10–44.43%). Updated PPO results require new training runs.

## How to run
```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate

pytest
python -m scripts.bot_table                               # bot-vs-bot calibration table
for condition in none deductive oracle; do
  for seed in 0 1 2; do
    python -m scripts.train --condition "$condition" --seed "$seed" --steps 40000000 || exit 1
  done
done
python -m scripts.eval --games 2000 --tier hard
python -m scripts.plot                                    # figures in results/figures
```
Each training run needs an empty output directory. Use a fresh `--out` directory or a new `--tag` to keep earlier runs. PPO finishes ongoing games before updating, so the final logged decision count can exceed the requested budget.

Final evaluation selects the nine named runs above and checks their training configuration and logged budgets. To evaluate a smaller experiment, pass matching `--conditions`, `--seeds`, `--steps` and, for tagged runs, `--tag`. The plot reads the run names from the evaluation CSV. Pass `--runs` to evaluation and plotting if you used a different training directory.

The CSV retains the Wilson interval for each fixed policy. The figure shows each training seed and the equally weighted mean. The evaluation summary also reports win-rate differences between conditions for the same training seed.

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
