"""P1: the paper's environment, reward and interfaces, checked against the text.

The reward oracle in this file is written directly from Table III, Sec. III-C
and Sec. III-D-4 and deliberately shares no code with
``ontology_rgat.reward_modes.shin2026``.  A transcription error in the
implementation would otherwise agree with itself.

Nothing here asserts a performance number.  Table IV and Table V are loaded as
reproduction targets and the tests only check that they are recorded correctly
and are not being presented as results of this repository.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from config_loader import load_config as load_system  # noqa: E402

from ontology_rgat.benchmarks.experiment import load_experiment  # noqa: E402
from ontology_rgat.benchmarks.paper_reference import (  # noqa: E402
    BACKEND_DEVIATIONS, PAPER_REFERENCE_PATH, assert_declared_deviations,
    assert_paper_constants, deviation, load_paper_reference,
    paper_constant_mismatches, paper_table_iv, paper_table_v, unreported_items)
from ontology_rgat.benchmarks.randomization import (  # noqa: E402
    sample_domain_randomization, sample_initial_condition)
from ontology_rgat.reward_modes import ShinRewardConfig  # noqa: E402
from ontology_rgat.reward_modes.shin2026 import ShinReward  # noqa: E402


SYSTEM = ROOT / "config/shin2026-system.yaml"
EXPERIMENT = ROOT / "config/experiments/shin2026_baseline.yaml"


# ----------------------------------------------- independent reward oracle

def paper_shaping_reward(previous_state, next_state, action, vertical_velocity):
    """Table III, transcribed independently of the production implementation.

    ``previous_state``/``next_state`` are [dx, dy, dz, dvx, dvy, dvz] in the
    drone body frame, ``action`` is [vx, vy, vz, wz] and ``vertical_velocity``
    is the drone's own body-frame v_z.
    """
    def clip(value, low, high):
        return min(max(value, low), high)

    d_xy_prev = float(np.hypot(previous_state[0], previous_state[1]))
    d_xy_next = float(np.hypot(next_state[0], next_state[1]))
    dz_prev, dz_next = float(previous_state[2]), float(next_state[2])

    lateral_progress = clip(d_xy_prev - d_xy_next, -1.0, 1.0)
    vertical_progress = (clip(abs(dz_prev) - abs(dz_next), -1.0, 1.0)
                         / max(d_xy_next, 1.0))
    vertical_speed_penalty = -clip(float(vertical_velocity) + 0.5, 0.0, float("inf"))
    undershoot_penalty = -(dz_next if dz_next > 0.0 else 0.0)
    yaw_rate_penalty = -abs(float(action[3]))

    return (1.0 * lateral_progress
            + 1.0 * vertical_progress
            + 0.5 * vertical_speed_penalty
            + 1.0 * undershoot_penalty
            + 2.0 * yaw_rate_penalty)


def paper_active_perception_reward(next_estimation_loss,
                                   alpha=0.1, beta=1.0, tau=0.01):
    """Sec. III-C: r_active_t = -alpha [beta (L_est_{t+1} - tau)]_0^1."""
    inner = beta * (float(next_estimation_loss) - tau)
    return -alpha * min(max(inner, 0.0), 1.0)


def paper_auxiliary_estimation_loss(truth, estimate):
    """Eq. (1): mean squared error over the six relative-state components."""
    truth = np.asarray(truth, dtype=float).reshape(6)
    estimate = np.asarray(estimate, dtype=float).reshape(6)
    return float(np.sum((truth - estimate) ** 2) / 6.0)


def paper_step_reward(previous_state, next_state, action, vertical_velocity,
                      next_estimation_loss, *, success=False, failure=False):
    """Sec. III-D-4: terminal outcomes replace the step reward."""
    if success:
        return 10.0
    if failure:
        return -10.0
    return (paper_shaping_reward(previous_state, next_state, action,
                                 vertical_velocity)
            + paper_active_perception_reward(next_estimation_loss))


# ---------------------------------------------------------------- reward

def _cases(seed=0, count=64):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield (rng.uniform(-6, 6, 6), rng.uniform(-6, 6, 6),
               rng.uniform(-2, 2, 4), float(rng.uniform(-2.5, 2.5)),
               float(rng.uniform(0.0, 1.5)))


def test_the_implementation_matches_an_independent_table_iii_oracle():
    reward = ShinReward(ShinRewardConfig())
    for previous, following, action, vertical, loss in _cases():
        value, parts = reward(
            previous, following, action,
            drone_vertical_velocity=vertical, next_estimation_loss=loss)
        expected = paper_step_reward(previous, following, action, vertical, loss)
        assert value == pytest.approx(expected, abs=1e-9)
        # Term by term, so a compensating pair of errors cannot pass.
        assert parts["lateral_progress"] == pytest.approx(
            np.clip(np.hypot(*previous[:2]) - np.hypot(*following[:2]), -1, 1))
        assert parts["yaw_rate_penalty"] == pytest.approx(-2.0 * abs(action[3]))
        assert parts["active_perception"] == pytest.approx(
            paper_active_perception_reward(loss))


def test_active_perception_uses_the_paper_gains_and_saturates_as_printed():
    reference = load_paper_reference()["active_perception_reward"]
    assert (reference["alpha"], reference["beta"], reference["tau"]) == (0.1, 1.0, 0.01)
    config = ShinRewardConfig()
    assert (config.active_alpha, config.active_beta, config.active_tau) == (0.1, 1.0, 0.01)
    # Below tau the penalty is exactly zero, and the bracket saturates at 1.
    assert paper_active_perception_reward(0.0) == 0.0
    assert paper_active_perception_reward(0.01) == 0.0
    assert paper_active_perception_reward(0.51) == pytest.approx(-0.05)
    assert paper_active_perception_reward(50.0) == pytest.approx(-0.1)
    reward = ShinReward(ShinRewardConfig())
    previous, following = np.zeros(6), np.zeros(6)
    for loss in (0.0, 0.01, 0.26, 0.51, 50.0):
        _, parts = reward(previous, following, np.zeros(4),
                          drone_vertical_velocity=0.0, next_estimation_loss=loss)
        assert parts["active_perception"] == pytest.approx(
            paper_active_perception_reward(loss))


def test_the_auxiliary_loss_is_the_mean_over_six_components():
    truth = np.array([1.0, -2.0, 0.5, 0.25, -0.75, 1.5])
    estimate = np.zeros(6)
    assert paper_auxiliary_estimation_loss(truth, estimate) == pytest.approx(
        np.mean(truth ** 2))


def test_terminal_outcomes_replace_rather_than_augment_the_step_reward():
    reward = ShinReward(ShinRewardConfig())
    previous = np.array([2.0, 1.0, -3.0, 0.0, 0.0, 0.0])
    following = np.array([1.0, 0.5, -2.0, 0.0, 0.0, 0.0])
    action = np.array([0.4, -0.3, -0.5, 0.9])
    shaping = paper_shaping_reward(previous, following, action, -0.4)
    assert shaping != 0.0  # the case below would be vacuous otherwise

    value, parts = reward(previous, following, action,
                          drone_vertical_velocity=-0.4, next_estimation_loss=0.5,
                          physical_contact=True, terminal=True)
    assert value == 10.0
    assert parts["task"] == 10.0
    assert all(parts[name] == 0.0 for name in parts if name != "task")

    value, parts = reward(previous, following, action,
                          drone_vertical_velocity=-0.4, next_estimation_loss=0.5,
                          crash=True, terminal=True)
    assert value == -10.0
    value, parts = reward(previous, following, action,
                          drone_vertical_velocity=-0.4, next_estimation_loss=0.5,
                          excessive_drift=True, terminal=True)
    assert value == -10.0


# ----------------------------------------------- Table I / Table II sampling

def test_the_reset_sampler_reproduces_the_table_i_distribution_at_c_equals_one():
    initial = load_paper_reference()["initial_conditions"]
    draws = [sample_initial_condition(seed, curriculum=1.0)
             for seed in range(4000)]
    position = np.array([draw["relative_position_m"] for draw in draws])
    yaw = np.array([draw["platform_yaw_misalignment_rad"] for draw in draws])
    speed = np.array([draw["platform_speed_m_s"] for draw in draws])
    speed_step = np.array([draw["speed_step_m_s"] for draw in draws])
    yaw_step = np.array([draw["yaw_rate_step_rad_s"] for draw in draws])

    altitude = initial["altitude_offset_m"]
    assert position[:, 2].min() >= altitude["low"] - 1e-9
    assert position[:, 2].max() <= altitude["high"] + 1e-9
    lateral = initial["lateral_offset_m"]
    assert position[:, :2].min() >= lateral["low"] - 1e-9
    assert position[:, :2].max() <= lateral["high"] + 1e-9
    yaw_range = initial["platform_yaw_misalignment_deg"]
    assert np.degrees(yaw).min() >= yaw_range["low"] - 1e-6
    assert np.degrees(yaw).max() <= yaw_range["high"] + 1e-6

    platform = initial["initial_platform_speed_m_s"]
    assert speed.min() >= platform["low"] - 1e-9
    assert speed.max() <= platform["high"] + 1e-9
    # Table I draws speeds across the whole range, not only slow ones.
    assert speed.max() > 7.0
    assert np.all(np.asarray(
        [draw["platform_yaw_rate_rad_s"] for draw in draws]) == 0.0)

    step = initial["platform_speed_perturbation_m_s"]
    assert speed_step.min() >= step["low"] - 1e-9
    assert speed_step.max() <= step["high"] + 1e-9
    yaw_perturbation = initial["platform_yaw_rate_perturbation_deg_s"]
    assert np.degrees(yaw_step).min() >= yaw_perturbation["low"] - 1e-6
    assert np.degrees(yaw_step).max() <= yaw_perturbation["high"] + 1e-6


def test_the_curriculum_scales_platform_motion_and_c_zero_is_stationary():
    for seed in range(50):
        assert sample_initial_condition(seed, curriculum=0.0)[
            "platform_speed_m_s"] == 0.0
        assert sample_initial_condition(seed, curriculum=0.0)[
            "speed_step_m_s"] == 0.0
    half = [sample_initial_condition(s, curriculum=0.5)["platform_speed_m_s"]
            for s in range(400)]
    full = [sample_initial_condition(s, curriculum=1.0)["platform_speed_m_s"]
            for s in range(400)]
    assert np.allclose(np.asarray(half) * 2.0, full)


def test_the_domain_randomization_sampler_matches_table_ii():
    table = load_paper_reference()["domain_randomization"]
    samples = [sample_domain_randomization(seed) for seed in range(3000)]

    def bounds(values):
        array = np.asarray(values, dtype=float)
        return float(array.min()), float(array.max())

    scalar_fields = {
        "velocity_gain_xy": "velocity_gain_xy",
        "velocity_gain_z": "velocity_gain_z",
        "attitude_gain_roll_pitch": "attitude_gain_roll_pitch",
        "attitude_gain_yaw": "attitude_gain_yaw",
        "ground_texture_scale": "ground_texture_scale",
        "light_direction_deg": "light_direction_deg",
    }
    for attribute, key in scalar_fields.items():
        low, high = bounds([getattr(s, attribute) for s in samples])
        assert low >= table[key]["low"] - 1e-9, key
        assert high <= table[key]["high"] + 1e-9, key
        span = table[key]["high"] - table[key]["low"]
        assert high - low > 0.75 * span, f"{key} barely explores its range"

    low, high = bounds([s.brightness for s in samples])
    assert low >= table["ground_brightness_factor"]["low"] - 1e-9
    assert high <= table["ground_brightness_factor"]["high"] + 1e-9

    for attribute, key, scale in (
            ("external_force_n", "external_force_n", 1.0),
            ("external_torque_nm", "external_torque_nm", 1.0),
            ("initial_velocity_m_s", "initial_velocity_m_s", 1.0),
            ("rgb_scale", "rgb_scaling", 1.0)):
        low, high = bounds([getattr(s, attribute) for s in samples])
        assert low >= table[key]["low"] * scale - 1e-12, key
        assert high <= table[key]["high"] * scale + 1e-12, key

    rate = np.degrees([s.initial_angular_rate_rad_s for s in samples])
    assert rate.min() >= table["initial_angular_rate_deg_s"]["low"] - 1e-6
    assert rate.max() <= table["initial_angular_rate_deg_s"]["high"] + 1e-6

    texture = np.asarray([s.ground_texture_id for s in samples])
    assert texture.min() == table["ground_texture_id"]["low"]
    assert texture.max() == table["ground_texture_id"]["high"]


def test_the_external_torque_reading_is_declared_not_silently_chosen():
    """The printed '-4e3, 4e3 N.m' is unusable; the reading must be on record."""
    interpretations = {entry["item"]: entry
                       for entry in load_paper_reference()["interpretations"]}
    entry = interpretations["domain_randomization.external_torque_nm"]
    assert entry["printed"] == "-4e3, 4e3 [N.m]"
    assert entry["adopted"] == "-4e-3, 4e-3 N.m"
    assert entry["rationale"].strip()
    sample = sample_domain_randomization(11)
    assert np.all(np.abs(sample.external_torque_nm) <= 4e-3 + 1e-12)


# ------------------------------------------------------- resolved config

def test_the_resolved_configuration_matches_every_transcribed_paper_value():
    system = load_system(SYSTEM)
    experiment = load_experiment(EXPERIMENT)
    assert paper_constant_mismatches(system, experiment) == []
    assert_paper_constants(system, experiment)


def test_a_drifted_paper_constant_is_caught_rather_than_tolerated():
    system = deepcopy(load_system(SYSTEM))
    experiment = deepcopy(load_experiment(EXPERIMENT))
    system["vision"]["camera"]["horizontal_fov_deg"] = 75.0
    experiment["estimator"]["latent_dimension"] = 128
    experiment["reward"]["active_perception"]["tau"] = 0.05
    mismatches = paper_constant_mismatches(system, experiment)
    assert len(mismatches) == 3
    with pytest.raises(AssertionError, match="disagrees with the transcribed"):
        assert_paper_constants(system, experiment)


def test_the_network_interface_reserves_the_first_six_latent_components():
    from ontology_rgat.pipelines import PIPELINES
    network = load_paper_reference()["network"]
    for spec in PIPELINES.values():
        assert spec.reserved_latent_dimensions == network["relative_state_dim"]
        assert spec.actor_latent_slice == slice(
            network["actor_latent_slice_start"], None)


# ------------------------------------------------------ declared deviations

def test_every_transfer_deviation_from_the_paper_is_declared_with_evidence():
    system = load_system(SYSTEM)
    experiment = load_experiment(EXPERIMENT)
    keys = assert_declared_deviations(system, experiment)
    assert set(keys) == {"simulator", "platform_motion_model", "platform_speed",
                         "command_envelope", "battery_termination",
                         "entry_handover", "keypoint_encoder"}
    for entry in BACKEND_DEVIATIONS:
        assert entry.paper.strip() and entry.repository.strip()
        assert entry.reason.strip() and entry.evidence
        # A deviation that hit only one arm would confound the comparison.
        assert entry.identical_across_arms
    # No deviation may be registered against the paper's own backend.
    assert not [e for e in BACKEND_DEVIATIONS if e.backend == "paper_reproduction"]


def test_the_speed_deviation_reports_the_real_gap_rather_than_a_rounded_one():
    system = load_system(SYSTEM)
    reference = load_paper_reference()["initial_conditions"]
    paper_high = reference["initial_platform_speed_m_s"]["high"]
    configured = list(system["pad"]["speed_range_m_s"])
    cap = float(system["pad"]["vehicle_max_speed_m_s"])
    assert paper_high == 8.0
    # The gap is more than an order of magnitude; the registry says so.
    assert max(configured) < paper_high / 10.0
    assert cap < paper_high
    entry = deviation("platform_speed")
    assert "8" in entry.paper and "largest single fidelity gap" in entry.reason


def test_the_command_envelope_cannot_track_the_paper_platform_speed():
    """A recorded consequence, so no report can quietly claim Table V speeds."""
    experiment = load_experiment(EXPERIMENT)
    limit = list(experiment["control"]["max_velocity_m_s"])
    paper_speed = load_paper_reference()[
        "initial_conditions"]["initial_platform_speed_m_s"]["high"]
    assert max(limit[:2]) < paper_speed
    entry = deviation("command_envelope")
    assert "not stated" in entry.paper
    assert "cannot track an 8 m/s" in entry.reason


def test_the_entry_gate_is_never_stricter_than_the_paper_initial_velocity():
    """Table II randomizes body velocity to U(-1,1) m/s on every axis.

    A handover gate tighter than that would reject hover states the paper's own
    episodes start from, and on this rendered stage it sat inside the PX4 limit
    cycle and starved the run of episodes entirely.
    """
    system = load_system(SYSTEM)
    table = load_paper_reference()["domain_randomization"]["initial_velocity_m_s"]
    tolerance = float(system["benchmark"]["entry_speed_tolerance_m_s"])
    assert tolerance >= float(table["high"])
    assert float(system["benchmark"]["entry_view_margin"]) <= 1.0
    entry = deviation("entry_handover")
    assert "setup only" in entry.reason
    assert entry.identical_across_arms


def test_battery_termination_is_an_addition_to_the_paper_terminal_set():
    reference = load_paper_reference()
    assert reference["simulation"]["termination"] == [
        "touchdown", "exits the workspace", "horizon"]
    entry = deviation("battery_termination")
    assert entry.identical_across_arms
    # The paper's own terminal rewards remain exactly two values.
    terminal = reference["terminal_reward"]
    assert terminal["successful_landing"] == 10.0
    assert terminal["crash_or_excessive_drift"] == -10.0


# ------------------------------------------- reproduction targets, not results

def test_table_iv_is_recorded_as_a_target_and_never_as_a_repository_result():
    table = paper_table_iv()
    assert table["evaluation_episodes"] == 10000
    rows = {(row["approach"], row["variant"]): row for row in table["rows"]}
    proposed = rows[("Proposed", None)]
    assert (proposed["success_rate"], proposed["pos_rmse"],
            proposed["vel_rmse"]) == (97, 0.474, 0.589)
    assert rows[("Proposed", "w/o state estimation")]["success_rate"] == 73
    assert rows[("Hybrid", "EKF + RL")]["pos_rmse"] == 1.331
    # Rows the paper leaves blank stay blank rather than being imputed.
    assert rows[("RL [16]", None)]["pos_rmse"] is None
    assert table["table_values_are_rounded_to_integer_percent"]
    assert table["body_text_success_rates"]["proposed_wo_state_estimation"] == 73.99

    target = paper_table_v()
    assert target["evaluation_episodes_per_scenario"] == 1000
    assert target["success_rate_percent"]["8m/s"] == 98.8
    assert target["success_rate_percent"]["u_turn"] == 55.0

    # These numbers describe the paper. Nothing in the runtime package may
    # carry them as its own measured output.
    package = ROOT / "python/ontology_rgat"
    offenders = []
    for path in package.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("0.474", "98.8", "73.99", "69.89"):
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)}: {needle}")
    assert offenders == [], f"paper numbers embedded in runtime code: {offenders}"


def test_unreported_paper_items_are_enumerated_with_where_this_repo_sets_them():
    reference = load_paper_reference()
    entries = reference["unreported"]
    assert len(entries) >= 12
    for entry in entries:
        assert entry["item"].strip()
        assert entry["repo_location"].strip()
    items = "\n".join(unreported_items())
    # The four that would otherwise be easiest to present as reproduced.
    assert "PPO hyperparameters" in items
    assert "Action scaling and command limits" in items
    assert "PACMAN" in items
    assert "training seeds" in items


def test_the_paper_reference_file_is_the_single_transcription_source():
    assert PAPER_REFERENCE_PATH.is_file()
    reference = load_paper_reference()
    assert str(reference["paper"]["doi"]) == "10.1109/LRA.2026.3674011"
    assert reference["paper"]["pages"] == "5542-5549"
    experiment = load_experiment(EXPERIMENT)
    assert str(experiment["paper"]["doi"]) == str(reference["paper"]["doi"])
