import numpy as np
import pytest

from avalon.env import (
    A_APPROVE, A_ASSASSINATE, A_FAIL, A_REJECT, A_SUCCESS, ASSASSIN, ASSASSINATE,
    AvalonEnv, EVIL, GOOD, MERLIN, MINION, PROPOSE, QUEST, SERVANT, TEAM_INDEX,
    TEAMS, VOTE,
)
from avalon.obs import OBS_DIM, encode

# Seats: 0 Merlin, 1 Servant, 2 Servant, 3 Assassin, 4 Minion
ROLES = (MERLIN, SERVANT, SERVANT, ASSASSIN, MINION)


def new_env(leader=0):
    return AvalonEnv(0).reset(roles=ROLES, leader=leader)


def vote_all(env, approve):
    for _ in range(5):
        env.step(A_APPROVE if approve else A_REJECT)


def play_quest(env, team, fail):
    """Propose `team`, approve it, and have evil members play Fail or Success."""
    env.step(TEAM_INDEX[team])
    vote_all(env, True)
    while env.phase == QUEST:
        env.step(A_FAIL if fail else A_SUCCESS)


def test_teams():
    assert len(TEAMS) == 20
    assert [len(t) for t in TEAMS] == [2] * 10 + [3] * 10


def test_reset_deals_roles():
    env = AvalonEnv(1)
    for _ in range(50):
        env.reset()
        assert sorted(env.roles) == sorted(ROLES)
        assert len(env.evil) == 2


def test_propose_mask_matches_quest_size():
    env = new_env()
    m = env.legal_mask()
    assert m[:10].all() and not m[10:].any()
    with pytest.raises(ValueError):
        env.step(TEAM_INDEX[(0, 1, 2)])


def test_majority_needed():
    env = new_env()
    env.step(TEAM_INDEX[(0, 1)])
    for a in (A_APPROVE, A_APPROVE, A_REJECT, A_REJECT, A_REJECT):
        env.step(a)
    assert env.phase == PROPOSE and env.attempt == 1 and env.leader == 1
    env.step(TEAM_INDEX[(0, 1)])
    for a in (A_APPROVE, A_APPROVE, A_APPROVE, A_REJECT, A_REJECT):
        env.step(a)
    # Clean team: good cards are automatic, so the quest resolves immediately
    assert env.quest_fails == [0] and env.quest == 1 and env.attempt == 0


def test_five_rejections_evil_wins():
    env = new_env()
    for _ in range(5):
        env.step(TEAM_INDEX[(0, 1)])
        vote_all(env, False)
    assert env.done and env.winner == EVIL and env.win_reason == "five_rejections"
    assert env.reward(3) == 1 and env.reward(0) == -1


def test_only_evil_asked_on_quest():
    env = new_env()
    env.step(TEAM_INDEX[(1, 3)])
    vote_all(env, True)
    assert env.phase == QUEST and env.current_player() == 3


def test_three_fails_evil_wins():
    env = new_env()
    play_quest(env, (0, 3), fail=True)
    play_quest(env, (0, 3, 4), fail=True)
    assert env.quest_fails == [1, 2]
    play_quest(env, (1, 4), fail=True)
    assert env.done and env.winner == EVIL and env.win_reason == "three_fails"


@pytest.mark.parametrize("target,winner", [(0, EVIL), (1, GOOD)])
def test_assassination(target, winner):
    env = new_env()
    play_quest(env, (0, 1), fail=False)
    play_quest(env, (0, 1, 2), fail=False)
    play_quest(env, (1, 2), fail=False)
    assert env.phase == ASSASSINATE and env.current_player() == 3
    assert env.non_evil == (0, 1, 2)
    env.step(A_ASSASSINATE + target)
    assert env.done and env.winner == winner


def test_pending_votes_not_in_obs():
    env = new_env()
    env.step(TEAM_INDEX[(0, 1)])
    before = encode(env, 4, "none")
    env.step(A_REJECT)
    env.step(A_APPROVE)
    after = encode(env, 4, "none")
    assert before.shape == (OBS_DIM,)
    np.testing.assert_array_equal(before, after)


def test_random_games_finish():
    env = AvalonEnv(2)
    rng = np.random.default_rng(3)
    for _ in range(300):
        env.reset()
        for _ in range(200):
            if env.done:
                break
            env.step(rng.choice(np.flatnonzero(env.legal_mask())))
            if not env.done:
                assert env.phase in (PROPOSE, VOTE, QUEST, ASSASSINATE)
                encode(env, env.current_player(), "deductive")
        assert env.done
