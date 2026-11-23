import numpy as np
import pytest

from avalon.bots import Bot, HARD, MEDIUM, TIERS
from avalon.env import (
    A_APPROVE, A_FAIL, A_REJECT, A_SUCCESS, ASSASSIN, AvalonEnv, MERLIN, MINION,
    QUEST, SERVANT, TEAM_INDEX,
)
from avalon.evaluate import play_game

ROLES = (MERLIN, SERVANT, SERVANT, ASSASSIN, MINION)


@pytest.mark.parametrize("tier", TIERS)
def test_games_finish(tier):
    rng = np.random.default_rng(0)
    env = AvalonEnv(0)
    for _ in range(300):
        env.reset()
        play_game(env, [Bot(tier, rng)] * 5)  # env raises on illegal actions
        assert env.done


def test_mixed_tiers_finish():
    rng = np.random.default_rng(1)
    env = AvalonEnv(1)
    for _ in range(300):
        env.reset()
        play_game(env, [Bot(TIERS[rng.integers(3)], rng) for _ in range(5)])


@pytest.mark.parametrize("tier", (MEDIUM, HARD))
def test_good_proposer_includes_self(tier):
    env = AvalonEnv(0).reset(roles=ROLES, leader=1)
    action = Bot(tier).act(env)
    team = next(t for t, i in TEAM_INDEX.items() if i == action)
    assert 1 in team


def test_medium_merlin_rejects_evil_team():
    env = AvalonEnv(0).reset(roles=ROLES, leader=1)
    env.step(TEAM_INDEX[(1, 3)])
    assert env.current_player() == 0
    assert Bot(MEDIUM).act(env) == A_REJECT


def test_evil_approves_own_team():
    env = AvalonEnv(0).reset(roles=ROLES, leader=1)
    env.step(TEAM_INDEX[(1, 3)])
    for _ in range(4):
        env.step(A_APPROVE)
    assert env.current_player() == 4
    assert Bot(MEDIUM).act(env) == A_APPROVE


def test_only_one_evil_fails():
    env = AvalonEnv(0).reset(roles=ROLES, leader=0)
    env.quest = 1  # a size-3 quest
    env.step(TEAM_INDEX[(0, 3, 4)])
    for _ in range(5):
        env.step(A_APPROVE)
    assert env.phase == QUEST
    assert Bot(MEDIUM).act(env) == A_FAIL  # seat 3 is the designated failer
    env.step(A_FAIL)
    assert Bot(MEDIUM).act(env) == A_SUCCESS  # seat 4 stays hidden
