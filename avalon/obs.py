"""Flat observation vector, identical across conditions except the belief slot."""

import numpy as np

from .belief import belief_input
from .env import A_PASS, A_RES, A_SUS, MAX_ATTEMPTS, N_PHASES, N_PLAYERS, N_QUESTS, N_ROLES, SERVANT

N_SLOTS = N_QUESTS * MAX_ATTEMPTS  # one history slot per possible proposal
VOTE_DIM = N_PLAYERS * 3 + 2  # leader, team, votes, approved, fails / 2
TALK_DIM = N_PLAYERS + N_PLAYERS * N_PLAYERS  # has spoken; speaker x target, -1 sus / +1 res
SLOT_DIM = VOTE_DIM + TALK_DIM

SECTIONS = (
    ("role", N_ROLES),
    ("seat", N_PLAYERS),
    ("known_evil", N_PLAYERS),
    ("phase", N_PHASES),
    ("quest", N_QUESTS),
    ("leader", N_PLAYERS),
    ("rejections", MAX_ATTEMPTS),
    ("proposed_team", N_PLAYERS),
    ("quest_outcome", N_QUESTS * 3),  # succeeded / failed / not played
    ("quest_fails", N_QUESTS),  # Fail cards / 2
    ("history", N_SLOTS * SLOT_DIM),
    ("belief", N_PLAYERS),
)
OFFSETS = {}
_o = 0
for _name, _dim in SECTIONS:
    OFFSETS[_name] = slice(_o, _o + _dim)
    _o += _dim
OBS_DIM = _o


def known_evil(env, seat):
    """What the seat privately knows: Merlin and evil see the evil team."""
    v = np.zeros(N_PLAYERS, dtype=np.float32)
    if env.roles[seat] != SERVANT:
        v[list(env.evil)] = 1.0
    return v


def encode(env, seat, condition):
    x = np.zeros(OBS_DIM, dtype=np.float32)

    def put(name, idx, val=1.0):
        x[OFFSETS[name].start + idx] = val

    put("role", env.roles[seat])
    put("seat", seat)
    x[OFFSETS["known_evil"]] = known_evil(env, seat)
    if not env.done:
        put("phase", env.phase)
        put("quest", env.quest)
        put("leader", env.leader)
        put("rejections", env.attempt)
    if env.team is not None:
        for i in env.team:
            put("proposed_team", i)
    for q in range(N_QUESTS):
        if q < len(env.quest_fails):
            put("quest_outcome", q * 3 + int(env.quest_failed[q]))
            put("quest_fails", q, env.quest_fails[q] / 2)
        else:
            put("quest_outcome", q * 3 + 2)

    h0 = OFFSETS["history"].start
    for p in env.history:
        base = h0 + (p.quest * MAX_ATTEMPTS + p.attempt) * SLOT_DIM
        x[base + p.leader] = 1.0
        for i in p.team:
            x[base + N_PLAYERS + i] = 1.0
        for i, v in enumerate(p.votes):  # empty while the vote is pending
            x[base + 2 * N_PLAYERS + i] = float(v)
        x[base + 3 * N_PLAYERS] = float(p.approved)
        if p.fails is not None:
            x[base + 3 * N_PLAYERS + 1] = p.fails / 2
        for s, a in p.talk.items():
            x[base + VOTE_DIM + s] = 1.0
            if a != A_PASS:
                sign, target = (-1.0, a - A_SUS) if a < A_RES else (1.0, a - A_RES)
                x[base + VOTE_DIM + N_PLAYERS * (1 + s) + target] = sign

    x[OFFSETS["belief"]] = belief_input(env, seat, condition)
    return x
