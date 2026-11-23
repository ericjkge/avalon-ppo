import numpy as np
import pytest

from avalon.belief import belief_input, deductive_from
from avalon.env import ASSASSIN, AvalonEnv, MERLIN, MINION, SERVANT

ROLES = (MERLIN, SERVANT, SERVANT, ASSASSIN, MINION)


def test_no_info_is_uniform():
    np.testing.assert_allclose(deductive_from(0, []), [0, 0.5, 0.5, 0.5, 0.5])


def test_success_rules_nothing_out():
    np.testing.assert_allclose(deductive_from(0, [((1, 2), 0)]), [0, 0.5, 0.5, 0.5, 0.5])


def test_fail_with_self_pins_partner():
    # Seat 0 went on (0, 1) and it failed -> 1 is evil; the other evil is 2, 3 or 4
    np.testing.assert_allclose(deductive_from(0, [((0, 1), 1)]), [0, 1, 1 / 3, 1 / 3, 1 / 3])


def test_two_fails_pins_pair_within_team():
    b = deductive_from(0, [((1, 2, 3), 2)])
    np.testing.assert_allclose(b, [0, 2 / 3, 2 / 3, 2 / 3, 0])


def test_two_quests_combine():
    # (1,2) failed and (2,3) failed -> pairs must hit both: (1,3), (2,x) for x in 1,3,4
    b = deductive_from(0, [((1, 2), 1), ((2, 3), 1)])
    # consistent pairs: (1,2),(1,3),(2,3),(2,4)
    np.testing.assert_allclose(b, [0, 2 / 4, 3 / 4, 2 / 4, 1 / 4])


@pytest.mark.parametrize("seat,condition,expected", [
    (1, "none", [0, 0, 0, 0, 0]),
    (1, "deductive", [0.5, 0, 0.5, 0.5, 0.5]),  # Servant can't tell Merlin apart
    (1, "oracle", [0, 0, 0, 1, 1]),
    (0, "none", [0, 0, 0, 1, 1]),  # Merlin always gets the truth
    (3, "deductive", [0, 0, 0, 1, 1]),  # so does evil
])
def test_conditions(seat, condition, expected):
    env = AvalonEnv(0).reset(roles=ROLES, leader=0)
    np.testing.assert_allclose(belief_input(env, seat, condition), expected)


# ---- Bayesian belief (used by Hard good bots) ------------------------------
from avalon.belief import bayesian_from
from avalon.env import Proposal


def test_bayesian_prior():
    pe, pm = bayesian_from(0, [])
    np.testing.assert_allclose(pe, [0, 0.5, 0.5, 0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(pm, [0, 0.25, 0.25, 0.25, 0.25], atol=1e-6)


def test_bayesian_respects_deduction():
    # (0, 1) failed -> 1 is evil no matter how votes are modelled
    p = Proposal(0, 0, 0, (0, 1), votes=(True,) * 5, approved=True, fails=1)
    pe, _ = bayesian_from(0, [p])
    assert pe[1] > 0.999


def test_bayesian_reads_votes():
    # Seat 3 approves a team containing 4 that later fails; seat 2 rejects it.
    # The failing team implicates 4, so approving it should look evil.
    p = Proposal(0, 0, 1, (1, 4), votes=(True, True, False, True, True), approved=True, fails=1)
    pe, _ = bayesian_from(0, [p])
    assert pe[3] > pe[2]


def test_bayesian_pending_vote_ignored():
    p = Proposal(0, 0, 1, (1, 4))
    np.testing.assert_allclose(bayesian_from(0, [p])[0], bayesian_from(0, [])[0])
