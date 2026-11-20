"""learning-speed-extension/1 (current_semantics/extension.py): the event, the replay check
and the censored paired comparison."""
from __future__ import annotations

import inspect

from ontology_rgat.current_semantics import extension as ext
from ontology_rgat.current_semantics import pilot


def _rows(difficulties):
    return [{"update": i, "difficulty": d, "success": 0.5, "wall_s": float(i)}
            for i, d in enumerate(difficulties)]


def test_event_is_the_first_update_trained_at_full_difficulty():
    assert ext.first_full(_rows([0.0, 0.5, 0.9, 1.0, 1.0])) == 3
    assert ext.first_full(_rows([0.0, 0.3, 0.3])) is None


def test_a_replay_may_differ_only_in_wall_time():
    a, b = _rows([0.0])[0], _rows([0.0])[0]
    b["wall_s"] = 999.0
    assert ext.same_row(a, b)
    b["success"] = 0.5000001
    assert not ext.same_row(a, b)


def test_censoring_orders_the_pair_and_two_censored_models_tie():
    seen = lambda u: {"update": u, "censored": False}
    cut = {"update": ext.CAP, "censored": True}
    proposed = {1: seen(150), 2: cut, 3: seen(200), 4: cut, 5: seen(100)}
    other = {1: cut, 2: seen(400), 3: seen(180), 4: cut, 5: seen(120)}
    out = ext.compare(proposed, other)
    assert (out["proposed_first"], out["other_first"], out["ties"]) == (2, 2, 1)
    bounds = {r["seed"]: r["difference_is_bound"] for r in out["per_seed"]}
    assert bounds == {1: True, 2: True, 3: False, 4: True, 5: False}


def test_the_extension_cap_is_the_unminiaturised_budget():
    assert ext.CAP == 600


def test_stop_at_full_is_off_by_default():
    parameter = inspect.signature(pilot.train_seed).parameters["stop_at_full"]
    assert parameter.default is False
