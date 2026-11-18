"""isaac-development/2: the /1 cases re-flown under their own artifacts (2026-11-13).

/1 is unchanged (a passed gate, every model's best.pt).  /2 pairs case by case with
/1, never touches /1's files, accepts a complete development procedure whatever its
gate, and excludes (and lists) models without best.pt instead of refusing.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ontology_rgat.current_semantics import development as dev
from ontology_rgat.current_semantics import isaac_development as idev
from ontology_rgat.current_semantics.contracts import DEFAULT_CONFIG_PATH, load_config


@pytest.fixture(scope="module")
def loaded():
    config = load_config()
    return config, dev.load_development(dev.DEFAULT_DEVELOPMENT_PATH, config)


def test_v2_flies_exactly_the_v1_cases_under_its_own_artifacts(loaded):
    config, dev_plan = loaded
    v1 = idev.load_plan(idev.DEFAULT_PATH, config, dev_plan)
    v2 = idev.load_plan(idev.REFLIGHT_PATH, config, dev_plan)
    assert v1.protocol == "isaac-development/1" and v2.protocol == "isaac-development/2"
    assert v2.conditions == v1.conditions
    s1, s2 = idev.spec_of(v1), idev.spec_of(v2)
    assert s1.requires_gate and not s2.requires_gate
    assert {s1.registration, s1.result, s1.out_dir}.isdisjoint(
        {s2.registration, s2.result, s2.out_dir})
    assert (s1.registration, s1.result, s1.out_dir) == (
        idev.REGISTRATION, idev.RESULT, idev.OUT_DIR)          # /1 names unchanged


def test_v2_refuses_other_cases(loaded, tmp_path):
    config, dev_plan = loaded
    data = yaml.safe_load(idev.REFLIGHT_PATH.read_text(encoding="utf-8"))
    data["conditions"] = {"composition_id": data["conditions"]["composition_id"]}
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the isaac-development/1"):
        idev.load_plan(path, config, dev_plan)


def _root_with_checkpoints(root: Path, dev_plan, missing: set[str]) -> None:
    for key in idev.model_keys(dev_plan):
        if key not in missing:
            path = idev.checkpoint_path(root, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(key.encode())


@pytest.mark.parametrize("plan_path, ok", [(idev.DEFAULT_PATH, False),
                                           (idev.REFLIGHT_PATH, True)])
def test_missing_checkpoints_refuse_v1_and_are_excluded_by_v2(loaded, tmp_path, monkeypatch,
                                                              plan_path, ok):
    config, dev_plan = loaded
    missing = {"10/semantic_mlp_ppo", "13/semantic_mlp_ppo"}
    _root_with_checkpoints(tmp_path, dev_plan, missing)
    seen = {}

    def passed(config, dev_plan, root, *, requires_gate=True):
        seen["requires_gate"] = requires_gate
        return None, "development-plan-hash"

    monkeypatch.setattr(idev, "_development_passed", passed)
    status, payload = idev.register(DEFAULT_CONFIG_PATH, plan_path,
                                    dev.DEFAULT_DEVELOPMENT_PATH, tmp_path)
    assert seen["requires_gate"] is (not ok)
    if not ok:
        assert status == 3 and "no best.pt" in payload["reason"]
        return
    assert status == 0 and payload["protocol"] == "isaac-development/2"
    assert set(payload["excluded_models"]) == missing
    assert len(payload["checkpoints"]) == 18 and payload["episodes"] == 18 * 30
    (tmp_path / idev.spec_of(idev.load_plan(plan_path, config, dev_plan)).registration
     ).write_text(__import__("json").dumps(payload), encoding="utf-8")
    plan = idev.load_plan(plan_path, config, dev_plan)
    keys = idev.registered_keys(plan, dev_plan, tmp_path)
    assert len(keys) == 18 and not (set(keys) & missing)
    assert idev.registered_keys(idev.load_plan(idev.DEFAULT_PATH, config, dev_plan),
                                dev_plan, tmp_path) == idev.model_keys(dev_plan)
