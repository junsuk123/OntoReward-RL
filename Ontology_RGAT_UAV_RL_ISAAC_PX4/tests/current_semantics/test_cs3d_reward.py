"""spatial-current-reward/4: the common dense reward of task v4 (reward.py)."""
from __future__ import annotations

from dataclasses import replace
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from ontology_rgat.current_semantics import contracts
from ontology_rgat.current_semantics.contracts import DEFAULT_CONFIG_PATH, load_config
from ontology_rgat.current_semantics.reward import (CommonRewardFunction, desired_velocity,
                                                    lost_view_weight, potential,
                                                    terminal_reward)
from ontology_rgat.landing.terminal import REFERENCE_TERMINAL_REWARDS

CONFIG = load_config()
REWARD = CONFIG.task.reward
FIELD = REWARD.field
LOST = FIELD.lost_view
CH = CONFIG.local_plant.contact_height_m
PACKAGE = Path(contracts.__file__).resolve().parent


def truth(r, v, roll_pitch=(0.0, 0.0)):
    return SimpleNamespace(relative_position=np.asarray(r, float),
                           relative_velocity=np.asarray(v, float),
                           roll_pitch=np.asarray(roll_pitch, float))


def test_terminal_values_are_the_evaluators_table():
    assert terminal_reward("RUNNING") == 0.0
    assert terminal_reward("SUCCESS") == REFERENCE_TERMINAL_REWARDS["SUCCESS"] == 25.0
    assert terminal_reward("TASK_TIMEOUT") == REFERENCE_TERMINAL_REWARDS["TASK_TIMEOUT"]
    for status in ("UNSAFE_CONTACT", "MISSED_PAD_CONTACT", "SAFETY_ENVELOPE_VIOLATION"):
        assert terminal_reward(status) == -40.0
    assert terminal_reward("SUCCESS") > terminal_reward("TASK_TIMEOUT") > -40.0


def test_the_field_ends_in_a_contact_the_verdict_accepts():
    at_pad = desired_velocity(np.zeros(2), 0.0, FIELD)
    np.testing.assert_allclose(at_pad, [0.0, 0.0, -FIELD.touchdown_sink_m_s])
    assert FIELD.touchdown_sink_m_s < CONFIG.contact.z_speed_m_s
    # Descent is asked for only inside the alignment radius, which lies inside the pad.
    outside = desired_velocity(np.array([FIELD.alignment_radius_m + 1e-6, 0.0]), 0.0, FIELD)
    assert outside[2] == 0.0
    assert FIELD.alignment_radius_m < CONFIG.contact.pad_radius_m
    high = desired_velocity(np.zeros(2), 10.0, FIELD)
    assert high[2] == pytest.approx(-FIELD.descent_speed_max_m_s)
    far = desired_velocity(np.array([30.0, -40.0]), 2.0, FIELD)
    assert np.linalg.norm(far[:2]) == pytest.approx(FIELD.lateral_speed_max_m_s)
    assert np.dot(far[:2], [30.0, -40.0]) < 0.0 and far[2] == 0.0
    near = desired_velocity(np.array([0.2, 0.0]), 1.0, FIELD)
    assert near[0] == pytest.approx(-FIELD.lateral_gain_per_s * 0.2)


def test_potential_peaks_on_the_field_at_the_touchdown_point():
    best = potential([0.0, 0.0, CH], [0.0, 0.0, -FIELD.touchdown_sink_m_s], REWARD, CH)
    assert best == 0.0
    rng = np.random.default_rng(0)
    for _ in range(200):
        r, v = rng.uniform(-5, 5, 3), rng.uniform(-2, 2, 3)
        assert potential(r, v, REWARD, CH) < 0.0
    # Further away and off the field both cost, at the registered weights.
    assert potential([1.0, 0.0, CH], -FIELD.lateral_gain_per_s * np.array([1.0, 0, 0]), REWARD,
                     CH) == pytest.approx(-REWARD.distance_weight_per_m)
    assert potential([0.0, 0.0, CH], [0.0, 0.0, -FIELD.touchdown_sink_m_s - 1.0], REWARD,
                     CH) == pytest.approx(-REWARD.velocity_weight_per_m_s)
    with pytest.raises(ValueError):
        potential([0.0, math.nan, 1.0], np.zeros(3), REWARD, CH)


def test_shaping_telescopes_and_terminal_potential_is_kept():
    rng = np.random.default_rng(1)
    states = [truth(rng.uniform(-3, 3, 3), rng.uniform(-1, 1, 3)) for _ in range(30)]
    ages = rng.uniform(0.0, 2.0, len(states))
    function = CommonRewardFunction(REWARD, CH)
    start = function.reset(states[0], vision_age_s=ages[0])
    total_shaping = total_cost = 0.0
    for index, state in enumerate(states[1:], start=1):
        status = "MISSED_PAD_CONTACT" if index == len(states) - 1 else "RUNNING"
        reward, parts = function.step(state, status=status, dt=0.1,
                                      action_normalized=np.full(3, 0.5),
                                      vision_age_s=ages[index])
        assert reward == pytest.approx(parts["shaping"] + parts["cost"] + parts["terminal"]
                                       + parts["tilt"])
        total_shaping += parts["shaping"]; total_cost += parts["cost"]
    end = potential(states[-1].relative_position, states[-1].relative_velocity, REWARD, CH,
                    ages[-1])
    assert total_shaping == pytest.approx(end - start)  # Phi(s_T) is not zeroed
    assert parts["terminal"] == -40.0
    assert total_cost == pytest.approx(-(REWARD.time_cost_per_s + 0.25
                                         * REWARD.control_cost_per_s) * 0.1 * 29)


def test_a_constant_state_pays_only_the_time_cost():
    function = CommonRewardFunction(REWARD, CH)
    state = truth([2.0, 1.0, 3.0], [0.0, 0.0, 0.0])
    function.reset(state, vision_age_s=0.1)
    reward, parts = function.step(state, status="RUNNING", dt=0.1, action_normalized=np.zeros(3),
                                  vision_age_s=0.1)
    assert parts["shaping"] == 0.0 and reward == pytest.approx(-REWARD.time_cost_per_s * 0.1)
    # /4: the action's change pays nothing beyond its control cost (no noise tax).
    _, jump = function.step(state, status="RUNNING", dt=0.1, action_normalized=np.ones(3),
                            vision_age_s=0.1)
    assert jump["cost"] == pytest.approx(-(REWARD.time_cost_per_s + REWARD.control_cost_per_s)
                                         * 0.1)
    assert set(jump) == {"potential", "shaping", "cost", "tilt", "terminal"}
    with pytest.raises(RuntimeError):
        CommonRewardFunction(REWARD, CH).step(state, status="RUNNING", dt=0.1,
                                              action_normalized=np.zeros(3), vision_age_s=0.1)


def _with(path, tmp_path, edit):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    edit(data["task"]["reward"])
    out = tmp_path / "config.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


@pytest.mark.parametrize("edit, match", [
    (lambda r: r["field"].update(touchdown_sink_m_s=0.30), "touchdown vertical speed"),
    (lambda r: r.update(command_change_cost=3.0), "unknown keys"),
    (lambda r: r["field"]["lost_view"].update(full_after_s=0.5), "below full_after_s"),
    (lambda r: r["field"]["lost_view"].update(search_height_m=0.4), "below search_height_m"),
    (lambda r: r["field"]["lost_view"].update(terminal_radius_m=0.4), "alignment radius"),
    (lambda r: r["field"]["lost_view"].update(ramp_m=0.0), "positive"),
    (lambda r: r["field"].update(alignment_radius_m=0.5), "inside the pad"),
    (lambda r: r.update(version="spatial-current-reward/3"), "version"),
    (lambda r: r.update(extra=1.0), "unknown keys"),
    (lambda r: r.update(time_cost_per_s=0.0), "positive"),
    (lambda r: r.update(tilt_cost_onset_deg=21.0), "below the 21 deg envelope"),
])
def test_config_refuses_a_reward_the_verdict_cannot_accept(tmp_path, edit, match):
    with pytest.raises(ValueError, match=match):
        load_config(_with(DEFAULT_CONFIG_PATH, tmp_path, edit))


def test_reward_is_part_of_the_task_hash():
    manifest = contracts.contract_manifest(CONFIG)
    assert manifest["task"]["contract"]["reward"]["version"] == "spatial-current-reward/4"
    assert manifest["task"]["contract"]["version"] == "spatial-current-task/5"
    assert manifest["task"]["contract"]["reward"]["field"]["lost_view"]["after_s"] == 0.5
    changed = replace(CONFIG, task=replace(CONFIG.task, reward=replace(
        REWARD, time_cost_per_s=REWARD.time_cost_per_s + 0.1)))
    assert contracts.contract_manifest(changed)["task_hash"] != manifest["task_hash"]
    assert contracts.contract_manifest(changed)["algorithm_hash"] == manifest["algorithm_hash"]


def test_no_policy_path_module_imports_the_reward():
    # Truth enters only the task: the reward is imported by the backend alone.
    for name in ("models.py", "features.py", "graph.py", "ppo.py", "observation.py",
                 "geometry.py"):
        tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module}
        assert not any(module.endswith("reward") for module in imported), name


def test_tilt_cost_follows_its_registered_shape():
    function = CommonRewardFunction(REWARD, CH)
    level = truth([1.0, 0.0, 2.0], [0.0, 0.0, 0.0])
    function.reset(level, vision_age_s=0.1)
    onset = REWARD.tilt_cost_onset_deg
    for degrees, excess in ((onset - 1.0, 0.0), ((onset + 21.0) / 2, 0.5), (21.0, 1.0)):
        tilted = truth([1.0, 0.0, 2.0], [0.0, 0.0, 0.0], (math.radians(degrees), 0.0))
        _, parts = function.step(tilted, status="RUNNING", dt=0.1, action_normalized=[-0.8, 0, 0],
                                 vision_age_s=0.1)
        assert parts["tilt"] == pytest.approx(-REWARD.tilt_cost_per_s * excess ** 2 * 0.1)


def test_a_stale_view_turns_the_sink_into_a_search_climb():
    high, centred = 1.5, np.array([0.1, 0.0])
    fresh = desired_velocity(centred, high, FIELD, 0.1)
    assert fresh[2] < 0.0                                          # aligned: sink
    np.testing.assert_allclose(desired_velocity(centred, high, FIELD, LOST.after_s), fresh)
    stale = desired_velocity(centred, high, FIELD, LOST.full_after_s)
    assert stale[2] == pytest.approx(LOST.climb_speed_m_s)          # climb, full speed
    np.testing.assert_allclose(stale[:2], fresh[:2])                # lateral untouched
    half = desired_velocity(centred, high, FIELD, (LOST.after_s + LOST.full_after_s) / 2)
    assert half[2] == pytest.approx((fresh[2] + stale[2]) / 2)      # blended in age
    # No accepted frame at all counts as an infinite age.
    np.testing.assert_allclose(desired_velocity(centred, high, FIELD, None), stale)
    np.testing.assert_allclose(desired_velocity(centred, high, FIELD, math.inf), stale)
    # The climb stops at the search height and never asks to descend above it.
    near_top = LOST.search_height_m - 0.1
    assert desired_velocity(centred, near_top, FIELD, 5.0)[2] == pytest.approx(
        LOST.climb_gain_per_s * 0.1)
    assert desired_velocity(centred, LOST.search_height_m + 3.0, FIELD, 5.0)[2] == 0.0
    with pytest.raises(ValueError):
        desired_velocity(centred, high, FIELD, -0.1)


def test_the_blind_last_stretch_keeps_sinking():
    # Below the terminal height and near the centre the camera cannot see the pad on
    # any landing (pilot6_design_checks: blind from <= 0.43 m, <= 0.56 m off centre).
    low = LOST.terminal_height_m - 0.1
    for offset in (0.0, 0.3, LOST.terminal_radius_m):
        e = np.array([offset, 0.0])
        np.testing.assert_allclose(desired_velocity(e, low, FIELD, 5.0),
                                   desired_velocity(e, low, FIELD, 0.0))
        assert lost_view_weight(5.0, low, offset, LOST) == 0.0
    # Leaving either bound ramps the climb in over ramp_m.
    assert lost_view_weight(5.0, LOST.terminal_height_m + LOST.ramp_m / 2, 0.0, LOST) \
        == pytest.approx(0.5)
    assert lost_view_weight(5.0, low, LOST.terminal_radius_m + LOST.ramp_m, LOST) == 1.0
    # Low but far off the pad: climb to search for it.
    assert desired_velocity(np.array([2.0, 0.0]), 0.2, FIELD, 5.0)[2] > 0.0


def test_losing_the_view_costs_potential_and_climbing_recovers_it():
    r, sinking = [0.1, 0.0, 1.5 + CH], [0.0, 0.0, -0.5]
    climbing = [0.0, 0.0, LOST.climb_speed_m_s]
    seen = potential(r, sinking, REWARD, CH, 0.1)
    lost = potential(r, sinking, REWARD, CH, 2.0)
    assert lost < seen
    assert potential(r, climbing, REWARD, CH, 2.0) > lost
