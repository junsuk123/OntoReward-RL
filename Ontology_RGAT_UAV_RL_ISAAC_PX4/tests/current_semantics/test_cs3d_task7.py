"""Task v7 / G4 pilot 10 (spatial_view_contact_v3.yaml, 2026-11-02, status doc 18).

v3 is v2 with one change, scenario /3: the pad starts 1.0-1.5 m away at an initial
speed of 1.0-1.5 m/s, and the held-out set composes a trained motion with a trained
image axis.  Nothing in v1 or v2 may move.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pytest
import yaml

from ontology_rgat.current_semantics.backend import make_local_backend
from ontology_rgat.current_semantics.contracts import (DEFAULT_CONFIG_PATH, TASK_CONFIG_PATHS,
                                                       contract_manifest, load_config)
from ontology_rgat.current_semantics.experiment import TASK_RUN_ROOTS, run_root_for
from ontology_rgat.current_semantics.scenario import (CompositionScenario, CouplingScenario,
                                                      draw, isaac_scenario, pad_speed)

V1 = load_config(DEFAULT_CONFIG_PATH)
V2 = load_config(TASK_CONFIG_PATHS["spatial-current-task/6"])
V3_PATH = TASK_CONFIG_PATHS["spatial-current-task/7"]
V3 = load_config(V3_PATH)


def test_v3_changes_only_the_task_and_the_pilot_and_v1_v2_keep_their_classes():
    one, three = contract_manifest(V1), contract_manifest(V3)
    assert {k for k in one if k.endswith("_hash") and one[k] != three[k]} == {"task_hash", "g4_hash"}
    assert contract_manifest(V1)["task_hash"].startswith("4c903c99")
    assert type(V1.task.scenario) is CompositionScenario
    assert type(V2.task.scenario) is CompositionScenario
    assert isinstance(V3.task.scenario, CouplingScenario)
    assert (V3.task.version, V3.g4.protocol) == ("spatial-current-task/7", "g4-pilot/10")
    assert V3.g4.training_randomization == V2.g4.training_randomization


def test_v3_holds_out_narrow_fleeing_and_wide_approaching_with_balanced_marginals():
    spec = V3.task.scenario
    assert set(spec.held_out) == {(1, 1), (3, 3), (0, 2), (2, 0)}
    assert set(collections.Counter(a for a, _ in spec.allowed).values()) == {3}
    assert set(collections.Counter(b for _, b in spec.allowed).values()) == {3}
    # Fleeing trains on the wide axis only; the narrow axis trains with the other classes.
    fleeing = {a for a, b in spec.allowed if (b - a) % 4 == 0}
    assert fleeing == {0, 2}
    assert {(b - a) % 4 for a, b in spec.allowed if a in (1, 3)} == {1, 2, 3}
    assert {(d.bearing_bin, d.heading_bin) for d in
            (draw(spec, s, "composition") for s in range(999801, 999861))} == set(spec.held_out)


def test_v3_draws_a_fast_pad_from_the_edge_and_eases_it_in_the_curriculum():
    spec = V3.task.scenario
    speeds = [pad_speed(spec, s, 0.5, 1.0) for s in range(999801, 999861)]
    assert min(speeds) >= 1.0 and max(speeds) <= 1.5
    assert pad_speed(spec, 999801, 0.5, 0.0) == 0.5
    assert pad_speed(V2.task.scenario, 999801, 0.5, 1.0) is None
    backend = make_local_backend(V3, scenario="in_distribution")
    try:
        for seed in (999801, 999802, 999803):
            backend.reset(seed)
            flight = backend.flight
            assert 1.0 <= flight.v0 <= 1.5
            np.testing.assert_allclose(np.linalg.norm(flight.pad_velocity[:2]), flight.v0)
    finally:
        backend.close()


def test_isaac_refuses_scenario_3_until_it_carries_the_pad_speed():
    with pytest.raises(ValueError, match="not carried by the Isaac scenario"):
        isaac_scenario(V3.task.scenario, 999801, "in_distribution")


def _variant(tmp_path, edit):
    data = yaml.safe_load(Path(V3_PATH).read_text(encoding="utf-8"))
    edit(data)
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("edit, message", [
    (lambda d: d["task"]["scenario"].pop("pad_speed_m_s"), "missing keys"),
    (lambda d: d["task"]["scenario"].update(pad_speed_m_s=[1.5, 1.0]), "0 < low <= high"),
    (lambda d: d["task"]["scenario"].update(held_out=[[1, 1], [3, 3], [0, 2], [2, 2]]),
     "exactly once"),
    (lambda d: d["g4_pilot"].update(protocol="g4-pilot/9"), "checks learnability"),
])
def test_v3_refuses_a_malformed_scenario(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        load_config(_variant(tmp_path, edit))


def test_task7_owns_its_run_root():
    root = TASK_RUN_ROOTS["spatial-current-task/7"]
    assert run_root_for(V3, None) == root and root.name.endswith("_task7")
    with pytest.raises(ValueError, match="holds spatial-current-task/6"):
        run_root_for(V3, TASK_RUN_ROOTS["spatial-current-task/6"])
