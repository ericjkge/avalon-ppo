#!/bin/sh
# Runs every training job, most important first, PARALLEL at a time (default 5).
# Each job needs an empty run directory, so finished jobs are skipped on a rerun.
# Usage: sh scripts/run_all.sh   (logs in results/logs, runs in results/runs)
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
mkdir -p results/logs
STEPS=${STEPS:-40000000}
DQN_STEPS=${DQN_STEPS:-10000000}
{
  for s in 0 1 2; do echo "ppo_s$s $PY -m scripts.train --seed $s --steps $STEPS --threads 1 --tag _ppo"; done
  echo "dqn_s0 $PY -m scripts.train_dqn --seed 0 --steps $DQN_STEPS --threads 1 --tag _dqn"
  echo "reinforce_s0 $PY -m scripts.train --algo reinforce --seed 0 --steps $STEPS --threads 1 --tag _reinforce"
  for s in 0 1 2; do echo "sparse_s$s $PY -m scripts.train --quest-bonus 0 --seed $s --steps $STEPS --threads 1 --tag _sparse"; done
  for s in 0 1 2; do echo "notalk_s$s $PY -m scripts.train --no-talk --seed $s --steps $STEPS --threads 1 --tag _notalk"; done
  for s in 1 2; do
    echo "dqn_s$s $PY -m scripts.train_dqn --seed $s --steps $DQN_STEPS --threads 1 --tag _dqn"
    echo "reinforce_s$s $PY -m scripts.train --algo reinforce --seed $s --steps $STEPS --threads 1 --tag _reinforce"
  done
} | while read -r name cmd; do
  [ -f "results/runs/deductive_s${name##*_s}_${name%_s*}/model.pt" ] || printf '%s\n' "$cmd > results/logs/$name.log 2>&1"
done | xargs -P "${PARALLEL:-5}" -I{} sh -c '{}'
echo "all jobs finished $(date)" >> results/logs/queue.log
