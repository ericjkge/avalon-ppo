"""Rule-based Easy / Medium / Hard bots (see PLAN.md section 6).

Bots read the env directly but only use information their role is entitled to:
the public history, their own role, and (for Merlin and evil) the evil team.
"""

import numpy as np

from .belief import bayesian_from
from .env import (
    A_APPROVE, A_ASSASSINATE, A_FAIL, A_REJECT, A_SUCCESS, ASSASSINATE,
    MAX_ATTEMPTS, MERLIN, N_PLAYERS, PROPOSE, QUEST, TEAM_INDEX, TEAMS, VOTE,
    is_evil_role,
)

EASY, MEDIUM, HARD = "easy", "medium", "hard"
TIERS = (EASY, MEDIUM, HARD)


def suspicion(env):
    """Public: number of failed quests each player was on."""
    s = np.zeros(N_PLAYERS)
    for p in env.history:
        if p.fails:
            s[list(p.team)] += 1
    return s


def vote_suspicion(env, w=0.5):
    """Suspicion plus w for each approve vote on, or proposal of, a team that failed."""
    s = suspicion(env)
    for p in env.history:
        if p.fails:
            s += w * np.array(p.votes, dtype=float)
            s[p.leader] += w
    return s


def min_team_score(score, k):
    return min(score[list(t)].sum() for t in TEAMS if len(t) == k)


class Bot:
    def __init__(self, tier, rng=None, p_hide=0.3, p_trust=0.5, tau=0.25,
                 q_evil=0.55, q_merlin=0.55, q_fail=0.8, p_blend=0.4):
        assert tier in TIERS, tier
        self.tier = tier
        self.rng = rng if rng is not None else np.random.default_rng()
        self.p_hide = p_hide  # Hard Merlin: prob of approving a team with evil
        self.p_trust = p_trust  # Hard evil: prob of playing Success on quest 1
        self.tau = tau  # Hard Servant: slack on expected evil count when voting
        self.p_blend = p_blend  # Hard evil: prob of approving a clean team (not the 5th)
        # Hard Servant's model of how other roles behave (see belief.bayesian_from)
        self.model = dict(q_evil=q_evil, q_merlin=q_merlin, q_fail=q_fail)

    def act(self, env):
        seat = env.current_player()
        if env.phase == PROPOSE:
            return self._propose(env, seat)
        if env.phase == VOTE:
            return self._vote(env, seat)
        if env.phase == QUEST:
            return self._quest(env, seat)
        if env.phase == ASSASSINATE:
            return self._assassinate(env)
        raise RuntimeError("game is over")

    # ---- helpers -------------------------------------------------------
    def _lowest(self, candidates, score, n):
        """The n candidates with the lowest score, ties broken randomly."""
        jitter = self.rng.random(len(candidates))
        order = sorted(range(len(candidates)), key=lambda j: (score[candidates[j]], jitter[j]))
        return [candidates[j] for j in order[:n]]

    def _belief(self, env, seat):
        return bayesian_from(seat, env.history, **self.model)[0]

    # ---- phases --------------------------------------------------------
    def _propose(self, env, seat):
        k = env.team_size()
        others = [i for i in range(N_PLAYERS) if i != seat]
        evil_seat = is_evil_role(env.roles[seat])
        good_others = [i for i in env.non_evil if i != seat]

        if self.tier == EASY:
            pick = list(self.rng.choice(others, k - 1, replace=False))
        elif evil_seat:
            if self.tier == MEDIUM:
                pick = list(self.rng.choice(good_others, k - 1, replace=False))
            else:  # Hard: the good players who look most trustworthy
                pick = self._lowest(good_others, suspicion(env), k - 1)
        elif env.roles[seat] == MERLIN and self.tier == MEDIUM:
            pick = list(self.rng.choice(good_others, k - 1, replace=False))
        elif self.tier == MEDIUM:
            pick = self._lowest(others, vote_suspicion(env), k - 1)
        else:  # Hard Servant, and Hard Merlin hiding as a Servant
            score = self._belief(env, seat) + 0.01 * vote_suspicion(env)
            pick = self._lowest(others, score, k - 1)
        return TEAM_INDEX[tuple(sorted([seat] + [int(i) for i in pick]))]

    def _vote(self, env, seat):
        team = list(env.team)
        last = env.attempt == MAX_ATTEMPTS - 1
        has_evil = any(i in env.evil for i in team)  # only used by roles that know

        if is_evil_role(env.roles[seat]):
            if has_evil or (last and self.tier != HARD):
                return A_APPROVE
            if self.tier == HARD and not last and self.rng.random() < self.p_blend:
                return A_APPROVE
            return A_REJECT
        if self.tier == EASY or last:
            return A_APPROVE
        if env.roles[seat] == MERLIN:
            if not has_evil or (self.tier == HARD and self.rng.random() < self.p_hide):
                return A_APPROVE
            return A_REJECT
        if self.tier == MEDIUM:
            s = vote_suspicion(env)
            s[seat] = 0  # I know I'm good
            ok = s[team].sum() <= min_team_score(s, len(team))
        else:
            b = self._belief(env, seat)
            ok = b[team].max() < 0.99 and b[team].sum() <= min_team_score(b, len(team)) + self.tau
        return A_APPROVE if ok else A_REJECT

    def _quest(self, env, seat):
        if self.tier == EASY:
            return A_FAIL
        failer = min(i for i in env.team if i in env.evil)  # evil coordinate: one Fail
        if seat != failer:
            return A_SUCCESS
        if self.tier == HARD and env.quest == 0 and self.rng.random() < self.p_trust:
            return A_SUCCESS
        return A_FAIL

    def _assassinate(self, env):
        cands = list(env.non_evil)
        if self.tier == EASY:
            return A_ASSASSINATE + int(self.rng.integers(len(cands)))
        score = np.zeros(N_PLAYERS)
        for p in env.history:
            p_has_evil = any(i in env.evil for i in p.team)
            if p_has_evil and p.votes:
                for c in cands:
                    score[c] += not p.votes[c]  # rejected a team with evil
            if self.tier == HARD and not p_has_evil:
                score[p.leader] += 1  # proposed a clean team
        best = self._lowest(cands, -score, 1)[0]
        return A_ASSASSINATE + cands.index(best)
