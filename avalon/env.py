"""5-player Avalon (no chat) as a turn-based multi-agent environment.

One seat acts at a time. Votes and quest cards are collected seat by seat but
only written to the public history once everyone has acted, so they behave as
simultaneous. Good players' quest cards are forced to Success and are filled in
automatically, so the env only asks for real decisions.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

N_PLAYERS = 5
MERLIN, SERVANT, ASSASSIN, MINION = range(4)
N_ROLES = 4
ROLE_NAMES = ("Merlin", "Servant", "Assassin", "Minion")
ROLE_DECK = (MERLIN, SERVANT, SERVANT, ASSASSIN, MINION)
QUEST_SIZES = (2, 3, 2, 3, 3)
N_QUESTS = len(QUEST_SIZES)
MAX_ATTEMPTS = 5

PROPOSE, VOTE, QUEST, ASSASSINATE, DONE = range(5)
N_PHASES = 4  # DONE is never observed

GOOD, EVIL = 0, 1

# Actions: 0-19 propose a team, then vote, quest card, assassination target.
TEAMS = tuple(c for k in (2, 3) for c in combinations(range(N_PLAYERS), k))
TEAM_INDEX = {t: i for i, t in enumerate(TEAMS)}
A_APPROVE, A_REJECT, A_SUCCESS, A_FAIL, A_ASSASSINATE = 20, 21, 22, 23, 24
N_ACTIONS = 27


def is_evil_role(role):
    return role in (ASSASSIN, MINION)


@dataclass
class Proposal:
    quest: int
    attempt: int
    leader: int
    team: tuple
    votes: tuple = ()  # approve (True) / reject per seat; empty until resolved
    approved: bool = False
    fails: int | None = None  # number of Fail cards; None if the quest wasn't played


class AvalonEnv:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, roles=None, leader=None):
        if roles is None:
            roles = self.rng.permutation(ROLE_DECK)
        self.roles = tuple(int(r) for r in roles)
        assert sorted(self.roles) == sorted(ROLE_DECK), self.roles
        self.evil = tuple(i for i, r in enumerate(self.roles) if is_evil_role(r))
        self.non_evil = tuple(i for i in range(N_PLAYERS) if i not in self.evil)
        self.merlin = self.roles.index(MERLIN)
        self.assassin = self.roles.index(ASSASSIN)
        self.leader = int(self.rng.integers(N_PLAYERS)) if leader is None else leader
        self.quest = 0
        self.attempt = 0
        self.quest_fails = []  # Fail-card count per completed quest
        self.history = []  # list[Proposal]
        self.phase = PROPOSE
        self.winner = None
        self.win_reason = None
        self._queue = [self.leader]
        self._votes = {}
        self._cards = {}
        return self

    # ---- queries -------------------------------------------------------
    @property
    def done(self):
        return self.phase == DONE

    @property
    def team(self):
        """The team currently being voted on or questing, else None."""
        return self.history[-1].team if self.phase in (VOTE, QUEST) else None

    def team_size(self):
        return QUEST_SIZES[self.quest]

    def current_player(self):
        return self._queue[0] if self._queue else None

    def side(self, seat):
        return EVIL if seat in self.evil else GOOD

    def legal_mask(self):
        m = np.zeros(N_ACTIONS, dtype=bool)
        if self.phase == PROPOSE:
            lo = 0 if self.team_size() == 2 else 10
            m[lo:lo + 10] = True
        elif self.phase == VOTE:
            m[[A_APPROVE, A_REJECT]] = True
        elif self.phase == QUEST:
            m[[A_SUCCESS, A_FAIL]] = True  # only evil seats are ever asked
        elif self.phase == ASSASSINATE:
            m[A_ASSASSINATE:A_ASSASSINATE + 3] = True
        return m

    def reward(self, seat):
        assert self.done
        return 1.0 if self.side(seat) == self.winner else -1.0

    # ---- transitions ---------------------------------------------------
    def step(self, action):
        action = int(action)
        if self.done:
            raise RuntimeError("game is over")
        if not self.legal_mask()[action]:
            raise ValueError(f"illegal action {action} in phase {self.phase}")
        seat = self._queue.pop(0)

        if self.phase == PROPOSE:
            self.history.append(Proposal(self.quest, self.attempt, seat, TEAMS[action]))
            self.phase = VOTE
            self._votes = {}
            self._queue = list(range(N_PLAYERS))
        elif self.phase == VOTE:
            self._votes[seat] = action == A_APPROVE
            if not self._queue:
                self._resolve_vote()
        elif self.phase == QUEST:
            self._cards[seat] = action == A_FAIL
            if not self._queue:
                self._resolve_quest()
        elif self.phase == ASSASSINATE:
            target = self.non_evil[action - A_ASSASSINATE]
            if target == self.merlin:
                self._end(EVIL, "assassinated_merlin")
            else:
                self._end(GOOD, "assassin_missed")

    def _resolve_vote(self):
        p = self.history[-1]
        p.votes = tuple(self._votes[i] for i in range(N_PLAYERS))
        p.approved = sum(p.votes) > N_PLAYERS // 2
        self.leader = (self.leader + 1) % N_PLAYERS
        if p.approved:
            self.phase = QUEST
            self._cards = {}
            self._queue = [i for i in p.team if i in self.evil]
            if not self._queue:
                self._resolve_quest()
        else:
            self.attempt += 1
            if self.attempt == MAX_ATTEMPTS:
                self._end(EVIL, "five_rejections")
            else:
                self.phase = PROPOSE
                self._queue = [self.leader]

    def _resolve_quest(self):
        p = self.history[-1]
        p.fails = sum(self._cards.values())
        self.quest_fails.append(p.fails)
        n_failed = sum(f > 0 for f in self.quest_fails)
        n_succeeded = len(self.quest_fails) - n_failed
        if n_failed == 3:
            self._end(EVIL, "three_fails")
        elif n_succeeded == 3:
            self.phase = ASSASSINATE
            self._queue = [self.assassin]
        else:
            self.quest += 1
            self.attempt = 0
            self.phase = PROPOSE
            self._queue = [self.leader]

    def _end(self, winner, reason):
        self.phase = DONE
        self.winner = winner
        self.win_reason = reason
        self._queue = []
