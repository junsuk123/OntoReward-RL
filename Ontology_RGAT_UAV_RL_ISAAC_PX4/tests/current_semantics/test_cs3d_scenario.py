"""spatial-current-scenario/1 (task v5): the omnidirectional composition scenario."""
from __future__ import annotations

import collections
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from ontology_rgat.current_semantics.backend import local_plant_config, make_local_backend
from ontology_rgat.current_semantics.contracts import (DEFAULT_CONFIG_PATH, contract_manifest,
                                                       load_config)
from ontology_rgat.current_semantics.scenario import ScenarioLocalBackend, draw, sector

CONFIG = load_config()
SPEC = CONFIG.task.scenario
SEEDS = range(995001, 995061)


def test_held_out_set_is_a_balanced_non_cyclic_permutation():
    assert SPEC.bins == 4 and len(SPEC.held_out) == 4 and len(SPEC.allowed) == 12
    assert sorted(a for a, _ in SPEC.held_out) == sorted(b for _, b in SPEC.held_out) == [0, 1, 2, 3]
    relative = collections.Counter((b - a) % 4 for a, b in SPEC.allowed)
    assert set(relative) == {0, 1, 2, 3}          # every relative class is trained
    trained_a = collections.Counter(a for a, _ in SPEC.allowed)
    assert set(trained_a.values()) == {3}         # uniform bearing marginal in training


@pytest.mark.parametrize("split", ["in_distribution", "composition"])
def test_draws_stay_in_their_split_and_are_seeded(split):
    combos = {(d.bearing_bin, d.heading_bin) for d in (draw(SPEC, s, split) for s in SEEDS)}
    allowed = set(SPEC.allowed) if split == "in_distribution" else set(SPEC.held_out)
    assert combos <= allowed and len(combos) == len(allowed)
    assert draw(SPEC, 995001, split) == draw(SPEC, 995001, split)
    for s in SEEDS:
        d = draw(SPEC, s, split)
        assert sector(d.bearing_rad, 4) == d.bearing_bin and sector(d.heading_rad, 4) == d.heading_bin
        assert SPEC.offset_radius_m[0] <= d.offset_radius_m <= SPEC.offset_radius_m[1]
    with pytest.raises(ValueError):
        draw(SPEC, 1, "train")


def test_the_plant_flies_the_drawn_geometry_and_primes_a_fresh_frame():
    for seed in list(SEEDS)[:20]:
        flight = ScenarioLocalBackend(local_plant_config(CONFIG), scenario=SPEC,
                                      split="composition")
        m, truth = flight.reset(seed)
        d = flight.episode_scenario
        r = np.asarray(truth.relative_position)
        assert sector(math.atan2(-r[1], -r[0]), 4) == d.bearing_bin
        assert math.hypot(*r[:2]) == pytest.approx(d.offset_radius_m)
        assert sector(math.atan2(flight.pad_velocity[1], flight.pad_velocity[0]), 4) == d.heading_bin
        if m.optical_position is not None:   # the primed frame is the NEW geometry
            assert np.linalg.norm(np.asarray(m.optical_position) - r) < 0.15
    # Easy rungs shrink the offset as the /12 curriculum does.
    easy = ScenarioLocalBackend(local_plant_config(CONFIG), difficulty=0.0, scenario=SPEC,
                                split="in_distribution")
    _, truth = easy.reset(995001)
    assert math.hypot(*np.asarray(truth.relative_position)[:2]) < 0.15 * SPEC.offset_radius_m[1] + 1e-9


def test_the_backend_reports_the_draw_and_none_keeps_the_reference_plant():
    backend = make_local_backend(CONFIG, scenario="in_distribution")
    plain = make_local_backend(CONFIG)
    try:
        _, info = backend.reset(995001)
        _, info_plain = plain.reset(995001)
        assert info["scenario"]["split"] == "in_distribution" and "scenario" not in info_plain
    finally:
        backend.close(); plain.close()


def _with(tmp_path, edit):
    data = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    edit(data["task"]["scenario"])
    out = tmp_path / "config.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


@pytest.mark.parametrize("edit, match", [
    (lambda s: s.update(held_out=[[0, 1], [1, 2], [2, 3], [3, 0]]), "cyclic shift"),
    (lambda s: s.update(held_out=[[0, 0], [1, 0], [2, 3], [3, 1]]), "exactly once"),
    (lambda s: s.update(offset_radius_m=[0.9, 0.3]), "low < high"),
    (lambda s: s.update(version="spatial-current-scenario/0"), "version"),
])
def test_config_refuses_an_unbalanced_or_degenerate_hold_out(tmp_path, edit, match):
    with pytest.raises(ValueError, match=match):
        load_config(_with(tmp_path, edit))


def test_the_scenario_is_part_of_the_task_hash():
    manifest = contract_manifest(CONFIG)
    assert manifest["task"]["contract"]["version"] == "spatial-current-task/5"
    assert manifest["task"]["contract"]["scenario"]["held_out"] == [[0, 0], [1, 2], [2, 3], [3, 1]]
