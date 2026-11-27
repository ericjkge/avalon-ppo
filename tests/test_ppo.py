import pytest
import torch

from avalon.env import A_APPROVE
from avalon.model import ActorCritic
from avalon.pool import OpponentPool
from avalon.ppo import Collector


@pytest.mark.parametrize("use_snapshot", [False, True])
def test_rollouts_finish_under_one_policy_and_exclude_snapshot_seats(use_snapshot):
    torch.manual_seed(0)
    model = ActorCritic(hidden=16)
    pool = OpponentPool(every=1)
    with torch.no_grad():
        model.v.weight.zero_()
        model.v.bias.fill_(-10)
    if use_snapshot:
        pool.maybe_add(1, model)
    collector = Collector(model, pool, "none", 4, seed=0, p_selfplay=0, lam=1, quest_bonus=0)
    for version in (1, 2):
        with torch.no_grad():
            model.pi.bias[A_APPROVE] = version
            model.v.bias.fill_(version)
            batch = collector.collect(100)
            logits, values = model(batch["obs"], batch["mask"])
            expected = torch.distributions.Categorical(logits=logits).log_prob(batch["act"])
        assert len(expected) >= 100
        torch.testing.assert_close(batch["logp"], expected)
        torch.testing.assert_close(batch["ret"] - batch["adv"], values)
        torch.testing.assert_close(batch["ret"].abs(), torch.ones_like(values))
        assert all(env.done for env in collector.envs)
        stats = collector.pop_stats()
        assert stats["pool_games" if use_snapshot else "selfplay_games"] >= 4
