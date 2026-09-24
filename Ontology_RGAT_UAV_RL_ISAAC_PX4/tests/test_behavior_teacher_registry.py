"""A config may only name a behaviour teacher the warm start can actually fly.

2026-09-23 the primary experiment moved to ``pn_guidance_v1`` as both its
control condition and its teacher. The arm passed validation at start-up --
``_baseline_arms`` checks controller ids -- and the demonstration stage could
bind the law, but the teacher gate still listed only the PD and the retired
servo by hand, so the run booted Isaac, the four PX4 instances and the
gateways, fine-tuned the keypoint encoder, and only then died on
``unknown behavior teacher``. Everything the run had paid for was torn down.

The gate is cheap and the run is not, so it belongs here rather than at minute
ten of a machine-hour.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from ontology_rgat.benchmarks.experiment import load_experiment    # noqa: E402
from run_three_pipeline import (ANALYTIC_CONTROLLERS,              # noqa: E402
                                BEHAVIOR_TEACHERS,
                                PN_GUIDANCE_CONTROLLER,
                                PRIVILEGED_VELOCITY_TEACHER,
                                behavior_cloning_settings,
                                demonstration_fingerprint)

CFG = SimpleNamespace(sim=SimpleNamespace(max_steps=300))
EXPERIMENTS = sorted((ROOT / "config/experiments").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda path: path.stem)
def test_every_shipped_config_names_a_teacher_the_warm_start_can_fly(path):
    settings = behavior_cloning_settings(load_experiment(path))
    if not bool(settings.get("enabled", False)):
        return
    teacher = str(settings.get("teacher", PRIVILEGED_VELOCITY_TEACHER))
    assert teacher in BEHAVIOR_TEACHERS, (
        f"{path.name} clones from {teacher!r}, which the demonstration stage "
        f"rejects; it flies one of {sorted(BEHAVIOR_TEACHERS)}")


def test_a_teacher_the_stage_flies_is_also_an_arm_the_comparison_can_run():
    """The two registries name the same laws.

    A teacher is a control law; the control condition is the same law flown
    without a policy behind it. Letting the sets drift is how a run clones from
    a law its own baseline arm cannot fly, or declares an arm no warm start can
    produce demonstrations from.
    """
    assert set(BEHAVIOR_TEACHERS) == set(ANALYTIC_CONTROLLERS)


def test_each_teacher_signs_its_own_attempt_rows():
    """``method`` identifies the law that flew, not the stage it flew for.

    The attempts CSV is the only record of which law produced a stored
    demonstration set. Two laws sharing a method column makes a set collected
    by PN guidance read back as the privileged PD's, which is the exact claim
    the paper rests the warm start on.
    """
    methods = [entry["method"] for entry in BEHAVIOR_TEACHERS.values()]
    assert len(set(methods)) == len(methods), methods
    for teacher, entry in BEHAVIOR_TEACHERS.items():
        assert entry["information"].strip(), teacher


def _fingerprint(settings):
    config = load_experiment(
        ROOT / "config/experiments/planar_three_arm_comparison.yaml")
    return demonstration_fingerprint(config, settings, cfg=CFG, system={})


def test_pn_gains_decide_what_a_stored_demonstration_set_means():
    """A set flown on different gains is a different set.

    Stored teacher flights are reused across runs by fingerprint. The PN gains
    change what the law flies -- not how long it is given to fly it -- so a
    fingerprint blind to them would hand a PPO warm start demonstrations from a
    law the config no longer describes.
    """
    config = load_experiment(
        ROOT / "config/experiments/planar_three_arm_comparison.yaml")
    settings = behavior_cloning_settings(config)
    assert str(settings["teacher"]) == PN_GUIDANCE_CONTROLLER
    baseline = _fingerprint(settings)
    for key, value in (("pn_navigation_gain", 4.0),
                       ("pn_approach_speed_m_s", 0.9),
                       ("pn_closing_gain", 2.4),
                       ("pn_climb_reference_m_s", 0.8),
                       ("pn_reference_scale", 0.40),
                       ("pn_flare_range_m", 2.0),
                       ("pn_flare_descent_m_s", 0.5)):
        changed = dict(settings)
        changed[key] = value
        assert _fingerprint(changed) != baseline, key


def test_another_law_s_gains_do_not_re_fly_a_stored_set():
    """The PD's stored flights survive a PN gain being tuned, and back.

    Re-fingerprinting on every knob any teacher owns is what re-flew the
    four-flight teacher set three times on 2026-09-21. A gain the active law
    never reads cannot change what its demonstrations mean.
    """
    config = load_experiment(
        ROOT / "config/experiments/two_pipeline_comparison.yaml")
    settings = behavior_cloning_settings(config)
    assert str(settings["teacher"]) == PRIVILEGED_VELOCITY_TEACHER
    baseline = demonstration_fingerprint(config, settings, cfg=CFG, system={})
    changed = dict(settings)
    changed["pn_navigation_gain"] = 4.0
    changed["pn_closing_gain"] = 2.4
    assert demonstration_fingerprint(
        config, changed, cfg=CFG, system={}) == baseline
