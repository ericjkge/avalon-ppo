import json
import sys

import numpy as np
import pytest
from matplotlib.figure import Figure

from avalon.evaluate import rates_by_seed
from scripts import train
from scripts.eval import select_runs
from scripts.plot import bars


def make_run(root, condition="none", seed=0, tag="", **settings):
    run = root / f"{condition}_s{seed}{tag}"
    run.mkdir()
    cfg = dict(condition=condition, seed=seed, tag=tag, steps=40, n_envs=4, batch=10,
               minibatch=5, epochs=4, lr=.0003, ent=.01, clip=.2, pool_every=10,
               pool_size=20, p_selfplay=.5)
    cfg.update(settings)
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "log.csv").write_text("update,steps\n1,41\n")
    (run / "model.pt").write_bytes(b"checkpoint")
    return run


def test_selection_uses_expected_cells_and_ignores_pilots(tmp_path):
    pilot = make_run(tmp_path, tag="_pilot", steps=10)
    conditions = ("none", "deductive", "oracle")
    seeds = (0, 1, 2)
    expected = [make_run(tmp_path, c, s, threads=s+1) for c in conditions for s in seeds]
    selected = select_runs(tmp_path, conditions, seeds, steps=40)
    assert [run for run, cfg in selected] == expected
    assert select_runs(tmp_path, ["none"], [0], 10, "_pilot")[0][0] == pilot
    with pytest.raises(ValueError, match="only once"):
        select_runs(tmp_path, ["none"], [0, 0], 40)


@pytest.mark.parametrize("problem,message", [
    ("wrong_budget", "expected steps"), ("incomplete", "has not reached"),
    ("missing_checkpoint", "model.pt"), ("wrong_setting", "lr differs"),
    ("mixed_collectors", "rollout_method differs"), ("missing_cell", "deductive_s0"),
])
def test_final_evaluation_rejects_mismatched_runs(tmp_path, problem, message):
    make_run(tmp_path)
    if problem != "missing_cell":
        change = {"wrong_budget": {"steps": 10}, "wrong_setting": {"lr": .01},
                  "mixed_collectors": {"rollout_method": "complete_games"}}.get(problem, {})
        run = make_run(tmp_path, "deductive", **change)
        if problem == "incomplete":
            (run / "log.csv").write_text("update,steps\n1,10\n")
        if problem == "missing_checkpoint":
            (run / "model.pt").unlink()
    with pytest.raises(ValueError, match=message):
        select_runs(tmp_path, ["none", "deductive"], [0], steps=40)


def test_training_rerun_preserves_all_existing_artifacts(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    monkeypatch.setattr(sys, "argv", ["train", "--condition", "none", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before


def test_seed_figure_weights_each_policy_equally_and_shows_actual_rates(tmp_path, monkeypatch):
    # Different game counts distinguish a mean across seeds (0.5) from pooling (0.1).
    rows = [dict(condition="none", seed=seed, setup=setup, wins=wins, n=n)
            for setup in ("A", "B") for seed, wins, n in ((0, 0, 9), (1, 1, 1))]
    with pytest.raises(ValueError, match="Duplicate"):
        rates_by_seed(rows + [rows[0]])
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **kw: figures.append(fig))
    bars("hard", tmp_path, rows)
    for ax in figures[0].axes:
        assert ax.patches[0].get_height() == pytest.approx(.5)
        np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 1], [0, 1])
