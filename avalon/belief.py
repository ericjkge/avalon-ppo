"""Belief inputs: the only part of the observation that differs between conditions.

Only Servants are affected. Merlin and both evil players know the evil team,
so they always get the truth.
"""

from itertools import combinations

import numpy as np

from .env import N_PLAYERS, SERVANT

CONDITIONS = ("none", "deductive", "oracle")


def quest_records(env):
    """Public (team, fails) for every quest that was played."""
    return [(p.team, p.fails) for p in env.history if p.fails is not None]


def deductive_from(seat, records):
    """P(evil) per player for a good player who only knows they are good.

    Keeps every evil pair (excluding `seat`) consistent with the quest results:
    a quest with k Fail cards had at least k evil players on it. Successful
    quests rule nothing out, since evil can play Success. Remaining pairs are
    weighted equally.
    """
    others = [i for i in range(N_PLAYERS) if i != seat]
    pairs = [
        pair for pair in combinations(others, 2)
        if all(len(set(pair) & set(team)) >= fails for team, fails in records)
    ]
    b = np.zeros(N_PLAYERS, dtype=np.float32)
    for pair in pairs:
        b[list(pair)] += 1.0
    return b / len(pairs)  # the true pair is always consistent, so len >= 1


def _hypotheses(seat):
    """(evil mask, Merlin mask) for every assignment a good non-Merlin seat considers."""
    others = [i for i in range(N_PLAYERS) if i != seat]
    evil, merlin = [], []
    for pair in combinations(others, 2):
        for m in others:
            if m not in pair:
                e = np.zeros(N_PLAYERS)
                e[list(pair)] = 1
                mm = np.zeros(N_PLAYERS)
                mm[m] = 1
                evil.append(e)
                merlin.append(mm)
    return np.array(evil), np.array(merlin)  # 12 hypotheses


def bayesian_from(seat, history, q_evil=0.8, q_merlin=0.8, q_fail=0.8):
    """P(evil) per player for a Servant who also reads votes and proposals.

    Behavior model (one number per role, a deliberately simple stand-in for
    DeepRole's "likelihood under my own strategy"):
      - evil votes for teams with evil w.p. q_evil, for clean teams w.p. 1 - q_evil
      - Merlin votes for clean teams w.p. q_merlin, and proposes clean teams w.p. q_merlin
      - Servants vote 50/50 (uninformative)
      - a quest with evil on it gets at least one Fail w.p. q_fail
    Votes on the 5th attempt are ignored (everyone has a reason to approve).
    Returns (P(evil), P(Merlin)), each of length 5.
    """
    E, M = _hypotheses(seat)
    S = 1 - E - M
    S[:, seat] = 0
    logw = np.zeros(len(E))
    for p in history:
        if not p.votes:
            continue  # vote still pending: nothing public yet
        team = list(p.team)
        n_evil = E[:, team].sum(1)
        clean = n_evil == 0

        if p.leader != seat:  # Merlin proposes clean teams
            pm = np.where(clean, q_merlin, 1 - q_merlin)
            logw += np.log(M[:, p.leader] * pm + (1 - M[:, p.leader]))

        if p.attempt < 4:
            for i, approve in enumerate(p.votes):
                if i == seat:
                    continue
                pe = np.where(clean, 1 - q_evil, q_evil)
                pm = np.where(clean, q_merlin, 1 - q_merlin)
                if not approve:
                    pe, pm = 1 - pe, 1 - pm
                logw += np.log(E[:, i] * pe + M[:, i] * pm + S[:, i] * 0.5)

        if p.fails is not None:
            lik = np.where(clean, 1.0, q_fail if p.fails > 0 else 1 - q_fail)
            lik = np.where(n_evil < p.fails, 0.0, lik)
            with np.errstate(divide="ignore"):
                logw += np.log(lik)

    w = np.exp(logw - logw.max())
    w /= w.sum()
    return (w @ E).astype(np.float32), (w @ M).astype(np.float32)


def truth(env):
    b = np.zeros(N_PLAYERS, dtype=np.float32)
    b[list(env.evil)] = 1.0
    return b


def belief_input(env, seat, condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition == "oracle" or env.roles[seat] != SERVANT:
        return truth(env)
    if condition == "none":
        return np.zeros(N_PLAYERS, dtype=np.float32)
    return deductive_from(seat, quest_records(env))
