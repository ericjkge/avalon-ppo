import numpy as np
import pytest

from avalon.env import (
    A_APPROVE, A_ASSASSINATE, A_FAIL, A_PASS, A_REJECT, A_RES, A_SUCCESS, A_SUS, ASSASSIN,
    ASSASSINATE, AvalonEnv, EVIL, GOOD, MERLIN, MINION, PROPOSE, QUEST, SERVANT, TALK,
    TEAM_INDEX, TEAMS, VOTE,
)
from avalon.obs import OBS_DIM, encode

# Seats: 0 Merlin, 1 Servant, 2 Servant, 3 Assassin, 4 Minion
ROLES = (MERLIN, SERVANT, SERVANT, ASSASSIN, MINION)


def new_env(leader=0, talk=False):
    return AvalonEnv(0, talk=talk).reset(roles=ROLES, leader=leader)


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


@pytest.mark.parametrize("action", [-7, -27, 27, 38, 20.0, 20.9, "20", True, np.bool_(True)])
def test_invalid_action_cannot_consume_or_change_a_vote(action):
    env = new_env()
    env.step(np.int64(TEAM_INDEX[(0, 1)]))
    with pytest.raises(ValueError):
        env.step(action)
    assert env.current_player() == 0
    vote_all(env, True)
    assert env.history[-1].votes == (True,) * 5


def test_quest_cards_are_hidden_and_the_published_count_is_anonymous():
    games = [new_env(), new_env()]
    for env, first in zip(games, (A_FAIL, A_SUCCESS)):
        env.step(TEAM_INDEX[(3, 4)])
        vote_all(env, True)
        env.step(first)
        assert env.current_player() == 4
    for mode in ("none", "deductive", "oracle"):
        np.testing.assert_array_equal(encode(games[0], 4, mode), encode(games[1], 4, mode))
    for env, second in zip(games, (A_SUCCESS, A_FAIL)):
        env.step(second)
        assert env.history[-1].fails == 1
    assert games[0].history == games[1].history


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


def test_five_rejections_fail_the_quest_and_play_continues():
    env = new_env()
    for _ in range(5):
        env.step(TEAM_INDEX[(0, 1)])
        vote_all(env, False)
    assert not env.done and env.phase == PROPOSE
    assert env.quest == 1 and env.attempt == 0 and env.leader == 0
    assert env.quest_failed == [True] and env.quest_fails == [0]
    assert env.quests_won(3) == 1 and env.quests_won(0) == 0
    assert all(p.fails is None for p in env.history)
    for _ in range(10):
        env.step(TEAM_INDEX[(0, 1, 2)] if env.team_size() == 3 else TEAM_INDEX[(0, 1)])
        vote_all(env, False)
    assert env.done and env.winner == EVIL and env.win_reason == "three_fails"
    assert env.reward(3) == 1 and env.reward(0) == -1


def test_talk_round_is_optional_public_and_every_seat_speaks_once():
    env = new_env(talk=True)
    env.step(TEAM_INDEX[(0, 1)])
    assert env.phase == TALK and env.team == (0, 1)
    order = list(env._queue)
    assert sorted(order) == [0, 1, 2, 3, 4]
    first, second = order[0], order[1]
    m = env.legal_mask()
    assert m[A_PASS] and not m[:A_PASS].any()
    assert not m[A_SUS + first] and not m[A_RES + first] and m.sum() == 9
    with pytest.raises(ValueError):
        env.step(A_SUS + first)
    before = encode(env, second, "none")
    env.step(A_SUS + second)
    assert env.current_player() == second
    assert (encode(env, second, "none") != before).sum() == 2  # spoke bit and the accusation
    env.step(A_PASS)
    for _ in range(3):
        env.step(A_RES + first if env.current_player() != first else A_PASS)
    assert env.phase == VOTE and env.history[-1].talk[first] == A_SUS + second
    assert len(env.history[-1].talk) == 5


def test_talk_order_varies():
    env = AvalonEnv(5)
    orders = set()
    for _ in range(30):
        env.reset()
        env.step(int(np.flatnonzero(env.legal_mask())[0]))
        orders.add(tuple(env._queue))
    assert len(orders) > 10


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
                assert env.phase in (PROPOSE, TALK, VOTE, QUEST, ASSASSINATE)
                encode(env, env.current_player(), "deductive")
        assert env.done
