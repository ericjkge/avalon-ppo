# Avalon PPO: Plan

## Question
How much does explicit belief modeling help self-play PPO in a hidden-role game?
- **Independent variable:** the belief input (None / Deductive / Oracle)
- **Dependent variable:** win rate by side vs. bots

## 1. Environment
- 5 players: Merlin, 2 Servants (good); Assassin, Minion (evil). No chat.
- Quest sizes 2, 3, 2, 3, 3. One Fail card fails a quest. Leader rotates.
- 5 rejected proposals in a row → evil wins.
- Good wins 3 quests → Assassin guesses Merlin. A correct guess → evil wins.
- Votes and quest cards are simultaneous: each seat acts in turn, choices hidden until all have acted.
- Reward: ±1 to every seat at game end.

**Actions (27, masked by phase)**
| Idx | Phase | Mask |
|---|---|---|
| 0–19 | Propose (10 teams of size 2, 10 of size 3) | wrong size |
| 20–21 | Vote: Approve / Reject | — |
| 22–23 | Quest: Success / Fail | Fail for good players |
| 24–26 | Assassinate (non-evil players, relative index) | — |

## 2. Observation (one flat vector, the same in all conditions)
- Own role (one-hot, 4), own seat (one-hot, 5)
- Known evil (5 bits): true evil players for Merlin and evil; zeros for Servants
- Current state: quest #, phase, leader, rejection count, proposed team, past quest results
- History: 25 proposal slots × (leader, team, 5 votes, approved, # fails), zero-padded
- **Belief slot (5 floats): the only thing that differs between conditions**

## 3. Conditions (the belief slot only changes for Servants; others always get the truth)
| Condition | Servant belief slot |
|---|---|
| None | zeros |
| Deductive | P(evil) per player, from consistent evil pairs |
| Oracle | true evil players (0/1) |

**Deductive:** list the 6 evil pairs that exclude self → remove pairs where a quest had more Fails than the pair's members on that team → weight the rest equally → player i's value = share of remaining pairs that contain i. Successful quests rule nothing out.

## 4. Model
```
            Observation (~490 floats)
 ┌──────────────────────────────────────────────┐
 │ role (5) │ seat (5) │ known evil (5)         │
 │ current state (~45)                          │
 │ history (25 slots × 17 = 425)                │
 │ belief (5)  ← None / Deductive / Oracle      │
 └──────────────────────────────────────────────┘
                       │
              Linear 490→256, ReLU
              Linear 256→256, ReLU     shared trunk
                       │
           ┌───────────┴───────────┐
     Linear 256→27            Linear 256→1
     + mask (illegal → −∞)    V(s)
     softmax → π(a|s)
```
- About 200K parameters. One network for all roles; it's identical in every condition.
- The value head sees the same observation as the policy.

## 5. Training
- PPO: clip 0.2, γ = 1, λ = 0.95, lr 3e-4 (Adam), entropy 0.01, 4 epochs, advantage normalization.
- **Opponent pool:** save a copy every 10 updates, keep the last 20. Each game: 50% current vs. current, 50% current vs. a random saved copy (current takes good or evil 50/50). Train only on the current network's seats. Pure self-play until the first copy is saved.
- Fixed step budget (40M decisions; see LOG.md) for every condition; 3 seeds each → 9 runs.

## 6. Bots
| | Easy | Medium | Hard |
|---|---|---|---|
| **Good propose** | random team incl. self | self + lowest suspicion* | self + lowest Bayesian P(evil)** |
| **Good vote** | always approve | approve if team suspicion is the minimum possible for its size; always approve the 5th | reject if any member has P(evil) ≥ 0.99 or team's expected evil count > min possible + `tau`; always approve the 5th |
| **Merlin** | acts like a Servant | proposes clean teams, rejects any team containing evil | proposes like a Hard Servant; rejects teams with evil but approves them with prob `p_hide` |
| **Evil propose** | random team incl. self | self + random good players | self + good players with the lowest suspicion |
| **Evil vote** | approve if team has evil | approve if team has evil | approve if team has evil, else approve with prob `p_blend`; reject the 5th otherwise |
| **Evil quest** | always Fail | Fail, but only one evil player fails per quest | like Medium, but plays Success on quest 1 with prob `p_trust` |
| **Assassin** | random | guesses the good player who rejected the most teams containing evil | Medium's score plus a bonus for proposing teams without evil |

*Suspicion = failed quests a player was on, + 0.5 per approve vote on or proposal of a failed team.
**Bayesian belief over the 12 assignments a Servant considers (evil pair × Merlin). Updated on every resolved vote, proposal and quest with a simple behavior model: evil approves teams with evil w.p. `q_evil`; Merlin approves and proposes clean teams w.p. `q_merlin`; Servants vote 50/50; a quest with evil fails w.p. `q_fail`. Assignments inconsistent with Fail counts are eliminated. Same idea as DeepRole's belief update, with a hand-set behavior model instead of a learned strategy.

**Calibrated defaults:** `p_hide`=0.3, `p_trust`=0.5, `tau`=0.25, `q_evil`=`q_merlin`=0.55, `q_fail`=0.8, `p_blend`=0.4.

**Calibration target:** good wins 25–45% with Hard bots in every seat (DeepRole: 34% in all-DeepRole games, 40% human-only).

Good win rate with the corrected proposal likelihood, 2000 games per cell using `scripts.bot_table --seed 0`. Rows are good tiers and columns are evil tiers.
| | Easy | Medium | Hard |
|---|---|---|---|
| Easy | 0.01 | 0.01 | 0.04 |
| Medium | 0.15 | 0.01 | 0.02 |
| Hard | 0.42 | 0.36 | **0.42** |

Known limits: Medium good is near the floor; Hard evil isn't stronger than Medium evil against Hard good. Only Hard bots are used for evaluation.

## 7. Evaluation
≥1000 games per cell, 95% confidence intervals, against **Hard** bots.
| Setup | PPO | Bots |
|---|---|---|
| A (main result) | all good seats | all evil seats |
| B | all evil seats | all good seats |
| Reference | — | all seats (good wins 42.25% at evaluation seed 123) |

**Caveat:** Hard good bots use their own Bayesian belief, which affects how to read Setup B.

## 8. TODO
- [x] Environment + action mask + observation
- [x] Deductive belief + unit tests on hand-built histories
- [x] Easy / Medium / Hard bots
- [x] Evaluation code (A / B / Reference, CIs)
- [x] Sanity check: bot-vs-bot tables, calibration
- [x] PPO + opponent pool
- [ ] Repeat the 9 full training runs with complete-game PPO rollouts (initial runs in LOG.md)
- [ ] Evaluate those checkpoints against corrected Hard bots and report matched-seed differences
