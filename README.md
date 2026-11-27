# Avalon self-play: which RL methods learn to deceive?

A CS 1840 mini project. We train agents to play 5-player Avalon by self-play with three methods from the course (PPO, REINFORCE, DQN) and compare the strategies they end up with.

**Question:** Good Avalon play is balanced: players mix their actions so their role cannot be read. Which self-play RL methods learn balanced play, and what do the others learn instead?

**Independent variable:** the training method (plus two single-change variants of PPO).
**Dependent variables:** how readable each agent's hidden role is, how exploitable the agent is, and how it fares against the other agents.

> The earlier version of this project asked whether a belief input helps PPO. That experiment is recorded in [PLAN.md](PLAN.md) and [LOG.md](LOG.md), which describe the old rules and are not updated for this one.

## Why this question

Two facts about how Avalon is played well motivate the design.

1. **Good strategies are mixed.** A Merlin who rejects every team containing an evil player wins quests but is identified and assassinated. An evil player who only approves teams with evil on them is identified by their votes. Each role has to randomize between using its knowledge and concealing it.
2. **Servants have to be candidate Merlins.** The Assassin picks one of three good players. Good does best when the three are indistinguishable. That takes Merlin acting less informed *and* Servants acting more informed than they are.

These hold for any player, human or learned, so they give us a yardstick that does not depend on the training method. For every agent we ask the same two things:

- **Is it balanced?** Where the game calls for mixing, does the agent randomize, or does it always take the same action?
- **Is its good team indistinguishable?** Can an observer tell Merlin from the Servants by how they vote, propose and talk?

Win rate alone cannot answer either. An agent can win often against the opponents it trained with while being easy to read, and a better opponent would then beat it. That is why the comparison below rests on behavior and exploitability as well as on wins.

The methods are worth comparing on this yardstick because they are built differently. PPO and REINFORCE output a probability for every action. DQN acts greedily on its Q-values, so it takes the same action every time it is in the same situation. Whether that difference shows up in how balanced the learned players are is one of the things the experiment measures.

## The game

Five seats, dealt at random.

| Role | Side | Knows |
|---|---|---|
| Merlin | Good | Both evil seats |
| Servant (×2) | Good | Only their own role |
| Assassin, Minion | Evil | Each other |

Each round:

1. **Propose.** The leader picks a team. Team sizes by quest are 2, 3, 2, 3, 3.
2. **Talk.** Every seat speaks once, in a random order. A seat may pass, say "sus *j*" (I think player *j* is evil) or say "res *j*" (I think player *j* is good). Statements are public at once, so later speakers have heard earlier ones.
3. **Vote.** All five vote at the same time. Three or more approvals pass the team. Leadership then moves to the next seat.
4. **If rejected,** the next leader proposes. The fifth rejection on one quest fails that quest.
5. **If approved,** evil players on the team each choose Success or Fail. Good players' cards are always Success. One Fail fails the quest. Only the number of Fail cards is made public.

Three failed quests win the game for evil. After three successful quests the Assassin names one good player; evil wins if that player is Merlin, and good wins otherwise.

Differences from tabletop Avalon: talk is limited to the fixed vocabulary above, there are no extra roles (Percival, Morgana, Mordred), and five rejections fail a quest instead of ending the game.

### Reward

Every seat gets +1 if its side wins and −1 if it loses. In addition, each passed quest pays 0.2 to every good seat and each failed quest pays 0.2 to every evil seat. No discounting.

### Actions

38 discrete actions. Only the ones legal in the current phase can be chosen.

| Index | Action |
|---|---|
| 0–9 | Propose one of the 10 two-player teams |
| 10–19 | Propose one of the 10 three-player teams |
| 20, 21 | Approve, Reject |
| 22, 23 | Success, Fail |
| 24–26 | Assassinate the 1st, 2nd or 3rd good seat |
| 27 | Pass (say nothing) |
| 28–32 | Sus seat 0–4 (not yourself) |
| 33–37 | Res seat 0–4 (not yourself) |

### What a player observes

A vector of 1,239 numbers:

- its own role and seat, and the evil seats if it is Merlin or evil
- the phase, quest number, leader, rejection count and proposed team
- each finished quest's outcome and Fail-card count
- for every past proposal: the leader, the team, each seat's statement, each seat's vote, whether it passed, and the Fail-card count
- for Servants, a "deductive belief": each player's chance of being evil given only the Fail counts so far (held fixed in every run)

Hidden from a player: other players' roles, votes still being cast, and who played a Fail card. The whole history is in the observation, so the network needs no memory.

## Training methods

Every method uses the same game, the same observation, and the same network: two hidden layers of 256 units, shared by all five seats, which learns each role from the role input. All training is self-play. No rule-based bot is ever a training opponent.

| Run tag | Method | How it trains | Seeds | Budget |
|---|---|---|---|---|
| `_ppo` | **PPO** (reference) | Half the games are the current network in every seat; half are the current network on one side against one of its 20 most recent saved copies. GAE (λ = 0.95), clip 0.2, 4 epochs, entropy bonus 0.01. | 3 | 40M decisions |
| `_reinforce` | **REINFORCE with a baseline** | Same games as PPO. One pass over each batch with the plain policy-gradient loss, no clipping, Monte Carlo returns minus the learned value. | 3 | 40M |
| `_dqn` | **Double DQN** | The current Q-network plays every seat with ε-greedy exploration (ε from 1.0 to 0.05 over the first quarter). Replay buffer of 200k transitions, target network, greedy at evaluation. | 3 | 10M |
| `_sparse` | PPO, no quest bonus | Reference PPO with only the ±1 game-end reward. | 3 | 40M |
| `_notalk` | PPO, no talk | Reference PPO with the talk step removed from the game. | 3 | 40M |

The first three rows compare methods. The last two each change one thing about the reference, to ask whether the quest bonus and the talk round change the strategy that PPO learns.

DQN has a smaller budget because it is about twice as slow per decision. It reuses each stored decision about 16 times, so its budget is not directly comparable to the others'.

## Evaluation

There is no fixed "correct" opponent in a hidden-role game, and winning against the agents you trained with can mean you only learned to counter them. So we compare agents three ways, none of which depends on a hand-written opponent.

> **Status:** the game, the three trainers and the run queue are built and tested. The evaluation scripts below are not written yet.

**1. Cross-play.** Every agent's good team plays every agent's evil team, including other seeds of the same method. This gives a table of good-side win rates. An agent that wins only against its own training partners shows a strong diagonal and weak rows elsewhere. The row and column averages rank the methods as good and as evil.

**2. Exploitability.** Freeze an agent, train a fresh PPO opponent against it for a fixed budget, and record how often the opponent wins. A lower number means the agent is harder to exploit. This is the direct test of the "it only beat what it trained with" worry.

**3. Behavior metrics**, from logged games:

| Metric | What it shows | Tied to |
|---|---|---|
| Assassin accuracy, against the 1/3 chance level | How well good hides Merlin | Insights 1 and 2 |
| Accuracy of a simple classifier that picks Merlin from the three good players using their votes, proposals and statements | How readable Merlin is, independent of the agent's own Assassin | Insight 2 |
| Merlin's and Servants' rejection rates on teams with an evil player | Whether Merlin hides, and whether Servants cover for Merlin | Insights 1 and 2 |
| Policy entropy at those vote decisions | Whether the agent mixes at all | Insight 1 |
| Evil's Fail rate by quest number | Whether evil passes early quests to gain trust | Insight 1 |
| Evil's approval rate on teams without evil | Whether evil blends its votes | Insight 1 |
| Share of sus statements that name an evil player, by the speaker's role | Whether talk is truthful, and whether Merlin leaks through it | Talk |
| Change in win rate when statements are scrambled at test time | Whether anyone listens to talk | Talk |
| How games end (three failed quests, Merlin assassinated, Assassin missed) | Where each side's wins come from | All |

### What we expect

Written before the results, so they can be checked.

- **PPO** learns a mixed strategy: Merlin rejects fewer bad teams than full knowledge allows, and Assassin accuracy sits above 1/3 but well below 1.
- **REINFORCE** learns the same kind of strategy more slowly, because it uses each batch once and its gradient estimates are noisier. At the same budget it should be closer to random play.
- **DQN** is the most readable and the most exploitable, since it cannot mix. It may also be unstable, because its replay buffer holds games against older versions of its opponents.
- **Without the quest bonus,** evil passes more early quests and good protects Merlin more, because nothing pays for quest results on their own.
- **Talk** carries little information among trained agents: an evil player can say whatever a good player would. If Merlin's accusations are more accurate than a Servant's, talk gives the Assassin an extra clue.

### Sanity check only

The repo has rule-based Easy, Medium and Hard bots ([avalon/bots.py](avalon/bots.py)) from the earlier experiment. Training logs each run's win rate against the Hard bots as a quick sign that the run is improving. These numbers are not part of the comparison.

## How to run

```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest

sh scripts/run_all.sh        # all 15 training runs, 5 at a time
```

Single runs:

```bash
.venv/bin/python -m scripts.train --seed 0 --steps 40000000 --tag _ppo
.venv/bin/python -m scripts.train --algo reinforce --seed 0 --steps 40000000 --tag _reinforce
.venv/bin/python -m scripts.train --quest-bonus 0 --seed 0 --steps 40000000 --tag _sparse
.venv/bin/python -m scripts.train --no-talk --seed 0 --steps 40000000 --tag _notalk
.venv/bin/python -m scripts.train_dqn --seed 0 --steps 10000000 --tag _dqn
```

Each run writes `config.json`, `log.csv`, periodic checkpoints and a final `model.pt` to `results/runs/deductive_s<seed><tag>/`. A run refuses to start if its folder is not empty, so delete a half-finished folder before rerunning. `scripts/run_all.sh` skips runs that already have a `model.pt`.

On an 8-core laptop with five runs in parallel, a 40M-decision run takes roughly six hours.

## Repo layout

```
avalon/
  env.py        game rules, phases, legal-action mask
  obs.py        observation vector
  belief.py     deductive belief input; Bayesian belief used by the Hard bots
  bots.py       Easy / Medium / Hard rule-based bots (sanity check only)
  model.py      the shared network
  ppo.py        parallel self-play games, GAE, PPO and REINFORCE updates
  pool.py       saved copies of the network used as opponents
  evaluate.py   game runner and win-rate helpers
scripts/
  train.py      PPO and REINFORCE
  train_dqn.py  Double DQN
  run_all.sh    the full queue of runs
  eval.py, plot.py, bot_table.py   from the earlier experiment
tests/          rules, talk round, bots, rollouts
results/        runs and logs (not in git)
```
