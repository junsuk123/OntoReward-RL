"""Task v6 / G4 pilot 8 (spatial_view_contact_v2.yaml, 2026-11-01).

v2 is v1 with three changes: scenario /2 holds out one relative-geometry class
(B = A, the pad fleeing), the pilot protocol is g4-pilot/8, and the handover
own-state error is deleted.  Adding it must not move one v1 hash: development/1,
its Isaac evaluation and the handover-fix flight are registered under them.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pytest
import yaml

from ontology_rgat.current_semantics import development as dev
from ontology_rgat.current_semantics.backend import make_local_backend
from ontology_rgat.current_semantics.contracts import (DEFAULT_CONFIG_PATH, TASK_CONFIG_PATHS,
                                                       contract_manifest, load_config)
from ontology_rgat.current_semantics.experiment import TASK_RUN_ROOTS, run_root_for
from ontology_rgat.current_semantics.scenario import draw

V1 = load_config(DEFAULT_CONFIG_PATH)
V2_PATH = TASK_CONFIG_PATHS["spatial-current-task/6"]
V2 = load_config(V2_PATH)


def test_v1_hashes_did_not_move():
    manifest = contract_manifest(V1)
    assert manifest["task_hash"].startswith("4c903c99")
    assert manifest["g4_hash"].startswith("4d6c0362")
    assert manifest["algorithm_hash"].startswith("70ebb7e0")
    plan = dev.load_development(dev.DEFAULT_DEVELOPMENT_PATH, V1)
    assert plan.protocol == "development/1"
    assert dev.plan_hash(plan, V1).startswith("d6be7db0")


def test_v2_changes_only_the_task_and_the_pilot():
    one, two = contract_manifest(V1), contract_manifest(V2)
    differ = {k for k in one if k.endswith("_hash") and one[k] != two[k]}
    assert differ == {"task_hash", "g4_hash"}
    assert (V2.task.version, V2.g4.protocol) == ("spatial-current-task/6", "g4-pilot/9")
    assert V2.task.reward == V1.task.reward and V2.local_plant == V1.local_plant
    assert V2.g4.training_randomization.vision_blackout == \
        V1.g4.training_randomization.vision_blackout
    assert V2.g4.training_randomization.handover_own_state_error is None


def test_v2_holds_out_the_whole_approaching_class_with_uniform_marginals():
    # Pilot 8 held out fleeing (class 0) and failed the declared hardness rule; pilot 9,
    # the declared fallback, holds out approaching (class 2).
    spec = V2.task.scenario
    assert spec.version == "spatial-current-scenario/2"
    assert {(b - a) % 4 for a, b in spec.held_out} == {2}
    assert {(b - a) % 4 for a, b in spec.allowed} == {0, 1, 3}
    assert set(collections.Counter(a for a, _ in spec.allowed).values()) == {3}
    assert set(collections.Counter(b for _, b in spec.allowed).values()) == {3}
    for seed in range(999201, 999261):
        d = draw(spec, seed, "composition")
        assert (d.heading_bin - d.bearing_bin) % 4 == 2
        d = draw(spec, seed, "in_distribution")
        assert (d.heading_bin - d.bearing_bin) % 4 != 2


def _variant(tmp_path, edit, source=V2_PATH):
    data = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    edit(data)
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("edit, message", [
    # pilot 8 deleted the error: naming it again is refused, not ignored
    (lambda d: d["g4_pilot"]["training_randomization"].update(handover_own_state_error={
        "horizontal_max_m_s": 1.5, "vertical_max_m_s": 0.4, "tau_min_s": 0.15,
        "tau_max_s": 0.5}), "unknown keys"),
    # scenario /2 must hold out one whole relative class
    (lambda d: d["task"]["scenario"].update(held_out=[[0, 0], [1, 2], [2, 3], [3, 1]]),
     "one cyclic shift"),
    # each pilot protocol checks its own task
    (lambda d: d["g4_pilot"].update(protocol="g4-pilot/7"), "missing keys|checks learnability"),
])
def test_v2_refuses_a_mixed_contract(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        load_config(_variant(tmp_path, edit))


def test_v1_still_refuses_a_cyclic_shift_and_a_missing_error(tmp_path):
    with pytest.raises(ValueError, match="must not be a cyclic shift"):
        load_config(_variant(tmp_path, lambda d: d["task"]["scenario"].update(
            held_out=[[0, 0], [1, 1], [2, 2], [3, 3]]), DEFAULT_CONFIG_PATH))
    with pytest.raises(ValueError, match="missing keys"):
        load_config(_variant(tmp_path, lambda d: d["g4_pilot"]["training_randomization"].pop(
            "handover_own_state_error"), DEFAULT_CONFIG_PATH))


def test_each_task_writes_only_its_own_run_root(tmp_path):
    v1_root, v2_root = (TASK_RUN_ROOTS["spatial-current-task/5"],
                        TASK_RUN_ROOTS["spatial-current-task/6"])
    assert run_root_for(V1, None) == v1_root and run_root_for(V2, None) == v2_root
    assert v2_root.name.endswith("_task6") and v2_root.parent == v1_root.parent
    with pytest.raises(ValueError, match="holds spatial-current-task/5"):
        run_root_for(V2, v1_root)
    with pytest.raises(ValueError, match="holds spatial-current-task/6"):
        run_root_for(V1, v2_root)
    assert run_root_for(V2, tmp_path) == tmp_path


def test_v2_training_episodes_carry_no_own_state_error():
    randomization = V2.g4.training_randomization
    backend = make_local_backend(V2, randomization=randomization)
    try:
        _, info = backend.reset(990001)
        assert backend.error is None
        record = info["training_randomization"]
        assert record["own_state_error_m_s"] is None and record["tau_s"] is None
        before = backend.observer.last_measurement
        for _ in range(3):
            backend.step(np.zeros(3))
        seen = backend.observer.last_measurement
        assert seen is not before
    finally:
        backend.close()


def test_development_protocols_belong_to_one_task():
    assert dev.default_plan_path(V1) == dev.DEFAULT_DEVELOPMENT_PATH
    assert dev.default_plan_path(V2).name == "development_v2.yaml"
    with pytest.raises(ValueError, match="not a development protocol"):
        dev.load_development(dev.DEFAULT_DEVELOPMENT_PATH, V2)
