"""The platform-motion ladder must be climbable inside the training budget.

Evaluation always flies the full platform motion (``c = 1``). A ladder the
budget cannot finish trains the policy on a slower deck than the one it is
scored on, and nothing in a run reports that the curriculum stalled: the
mismatch is silent and is attributed to the policy instead.
"""
import pytest

from ontology_rgat.benchmarks.experiment import load_experiment
from ontology_rgat.curriculum import (PlatformMotionCurriculum,
                                      assert_curriculum_reaches_nominal,
                                      episodes_to_reach_nominal)

from conftest import ROOT

PLANAR = ROOT / "config/experiments/planar_three_arm_comparison.yaml"


def _curriculum_and_budget(path):
    config = load_experiment(path)
    curriculum = dict(config["curriculum"])
    return curriculum, int(config["training"]["episodes_full"])


def test_gated_ladder_costs_its_minimum_window_per_level():
    assert episodes_to_reach_nominal(
        80, 20, performance_gated=True, episodes_per_update=512) == 1580
    assert episodes_to_reach_nominal(
        1, 20, performance_gated=True, episodes_per_update=512) == 0
    # Linear advancement pays the update interval instead.
    assert episodes_to_reach_nominal(
        3, 20, performance_gated=False, episodes_per_update=512) == 1024


def test_the_shipped_eighty_level_ladder_is_refused_on_this_budget():
    """The defect this guard exists for, stated as the failure it was."""
    with pytest.raises(ValueError, match="cannot reach nominal difficulty"):
        assert_curriculum_reaches_nominal(
            80, 20, performance_gated=True, episodes_per_update=512,
            training_episodes=1000)


def test_planar_experiment_can_climb_its_own_ladder():
    curriculum, budget = _curriculum_and_budget(PLANAR)
    assert_curriculum_reaches_nominal(
        curriculum["levels"], curriculum["minimum_episodes_at_level"],
        performance_gated=curriculum["performance_gated"],
        episodes_per_update=curriculum["update_every_episodes"],
        training_episodes=budget)
    required = episodes_to_reach_nominal(
        curriculum["levels"], curriculum["minimum_episodes_at_level"],
        performance_gated=curriculum["performance_gated"],
        episodes_per_update=curriculum["update_every_episodes"])
    # Reaching nominal on the final episode is not training at nominal: keep
    # real headroom so the policy flies the evaluated deck before selection.
    assert budget - required >= 100


def test_a_climbed_ladder_ends_at_full_platform_motion():
    curriculum, _budget = _curriculum_and_budget(PLANAR)
    ladder = PlatformMotionCurriculum(
        levels=curriculum["levels"],
        episodes_per_update=curriculum["update_every_episodes"],
        performance_gated=curriculum["performance_gated"],
        assessment_window=curriculum["assessment_window"],
        minimum_episodes_at_level=curriculum["minimum_episodes_at_level"],
        success_rate_threshold=curriculum["success_rate_threshold"])
    assert ladder.c == 0.0
    passing = {"paper_success": 1.0, "geometric_fov_loss_fraction": 0.0,
               "relative_position_rmse_m": 0.0}
    episodes = 0
    while ladder.c < 1.0 and episodes < 10_000:
        ladder.observe(passing)
        episodes += 1
    assert ladder.c == pytest.approx(1.0)
    assert episodes <= 1000, "a flawless policy must finish inside the budget"
