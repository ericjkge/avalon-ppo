---
marp: true
theme: default
size: "16:9"
paginate: true
title: Does explicit deduction help PPO play Avalon?
---

# Does explicit deduction help PPO play Avalon?

You’re a Loyal Servant. You and player 3 go on a quest.
You play Success. **One Fail card comes back.**

**Who played it?**

<!--
0:45 · Pause for an answer. Player 3 must have played Fail because there were only two players on the quest. The game never revealed the author of the card, yet you know who it was.

Our question is whether a reinforcement learning agent plays better when we calculate deductions like this for it. We can test that by giving different versions of the agent the exact same game history.
-->

---

## What can each player see?

| Role | Seats | Private knowledge |
|---|---:|---|
| Merlin | 1 | Knows both evil players |
| Loyal Servant | 2 | Knows their own role |
| Assassin and Minion | 2 | Know each other |

**Propose a team → everyone votes → approved team goes on a quest**

Everyone sees proposals, votes and quest results. There’s no chat.
After three successful quests, good wins only if the Assassin guesses Merlin incorrectly.

<!--
1:00 · Merlin and the two Servants play for good. The Assassin and Minion play for evil. A rejected team passes leadership to the next seat. Five consecutive rejections or three failed quests give evil the win. A single Fail card fails a quest.

Merlin faces a trade-off. Consistently rejecting teams with evil players can protect the quests, but those public votes may also give the Assassin a clue about Merlin’s identity.
-->

---

## One failed quest, three ways to represent it

You are seat 0. The team was **{0, 3}** and one Fail came back.
Every condition receives that history.

| Servant input | Additional information |
|---|---|
| **None** | No belief estimate |
| **Deductive** | Each seat’s probability of being evil is **[0, ⅓, ⅓, 1, ⅓]** |
| **Oracle** | The actual two evil seats, even before the first quest |

Deduction summarizes public clues. Oracle supplies hidden information.

<!--
1:30 · Read the five entries in seat order, from 0 through 4. You know you are good and player 3 is evil. The remaining possible evil pairs are {1, 3}, {2, 3}, and {3, 4}. Weighting those pairs equally gives each possible partner a one-third share.

More generally, we eliminate any pair that could not have produced the observed number of Fail cards. A successful quest rules out no pair because evil can play Success. Only the Servants’ information changes across conditions.
-->

---

## How self-play PPO learns to choose an action

**Observation → shared network → legal action → game outcome**

- One actor–critic network plays all five seats in each run. Its inputs include the acting player’s role, private knowledge and public history.
- A mask filters 27 possible actions to those legal for that player and phase.
- Each seat earns **+1** if its side wins and **−1** if it loses.

<!--
1:10 · Think back to player 3. On a later turn, should the Servant approve a team that includes player 3? There is no immediate reward for a good vote. The critic estimates the seat’s expected return, and PPO uses the game outcome to update the probabilities of earlier actions. Its clipped objective limits how strongly an update rewards a change in action probability.

Training mixes games against the current policy with games against saved copies. Each belief condition uses the same architecture and training settings.
-->

---

## How we measure team strength

| Matchup | Question |
|---|---|
| **A · PPO good vs. Hard evil bots** | Does helping the Servants improve good’s win rate? |
| **B · PPO evil vs. Hard good bots** | Does the training condition also affect the learned evil strategy? |
| **Hard bots in all five seats** | How often does each side win with the reference strategy? |

We trained **3 seeds per condition** for **40M decisions per run** and evaluated each seed in **2,000 games per matchup**.

<!--
1:05 · Setup A is the main comparison. Both sides have different jobs, and good can still lose at assassination after succeeding on three quests. That’s why we measure the all-bot reference instead of assuming a 50% baseline.

The experimental variable is the Servants’ belief input. The outcome is a whole side’s win rate. All three good players share the outcome, so this design cannot isolate how much an individual Merlin or Servant contributed.
-->

---

## Deduction raised good’s win rate by 14.1 points in the initial runs

| Initial experiment | Good-side win rate | Evil-side win rate |
|---|---:|---:|
| None | 28.1% | 44.8% |
| Deductive | **42.2%** | 49.1% |
| Oracle | 53.3% | 45.3% |
| Hard bots in every seat | 41.7% | 58.3% |

These runs used the original PPO collector and Hard bots.

<!--
1:50 · Read down the good-side column first. Deductive gained 14.1 percentage points over None, closing about 56% of the 25.2-point gap between None and Oracle. Across the three seeds, the good-side rates were 29.9%, 24.3%, 30.2% for None, 46.9%, 42.2%, 37.4% for Deductive, and 57.4%, 47.6%, 55.0% for Oracle.

Read the evil-side column separately. The spread between conditions is smaller. Every evil player already knows both evil seats, so any difference here arises through training. These percentages describe the initial implementation. The reference row comes from that same experiment.

Source: https://github.com/ericjkge/avalon-ppo/blob/ebe920ca82968c71a680a697c69fffa5584f598f/LOG.md#L51-L70
-->

---

## Does the deduction advantage hold after the fixes?

The initial experiment suggests that good plays better when the Servants receive the deduction explicitly.

PPO now finishes active games before updating the policy. With the corrected proposal model, Hard good bots won **845 of 2,000** bot-only games (**42.25%**).

**How large is the Deductive–None gap when all three conditions train and play under the corrected implementation?**

<!--
1:10 · Return to the opening quest. Player 3’s identity was already inferable. The initial runs suggest that making this inference explicit helped the good side use it.

We subsequently found that some trajectories crossed a policy update and that the Hard bot’s model gave inconsistent weights to team proposals. Both are corrected. The 42.25% figure checks the revised bot baseline. Estimating the revised PPO effect requires full matched-seed training and evaluation.

Calibration and rerun instructions: README.md in this worktree.
-->

---

## Q&A · How does this relate to DeepRole?

**DeepRole** integrates deductive reasoning with counterfactual regret minimization and uses deep value networks trained through self-play.

**Our experiment** holds PPO fixed and changes what a Servant receives. It asks how much a simple deduction from public quest history helps a learned policy.

[Serrino et al., *Finding Friend and Foe in Multi-Agent Games*, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/912d2b1c7b2826caf99687388d2e8f7c-Abstract.html)

<!--
Optional backup slide. The opponent in our evaluation is a rule-based Hard bot. A claim about relative playing strength against DeepRole would require a separate matchup.
-->
