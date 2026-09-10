"""Navigation uncertainty has to reach the ontology, the policy and the reward.

The whole point of the urban experiment is that the pad-relative pose is
sometimes untrustworthy and the vehicle has to act on knowing that. Every link
in that chain is easy to break silently -- a default that reads as open sky, an
observation channel dropped in a refactor, a node whose value stops being
populated -- and none of those break a test that only checks flight dynamics.
These pin the chain end to end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ontology_rgat.config import default_config  # noqa: E402
from ontology_rgat.semantic import (  # noqa: E402
    build_ontology_graph, compute_features, make_observation)


@pytest.fixture(scope="module")
def cfg():
    return default_config(mode="quick", target="sitl")


def _state(cfg, *, gnss_quality: float, marker_quality: float,
           gnss_valid: bool = True):
    """One control step, identical but for how much the navigation is trusted."""
    battery = {"reserve": 1.0, "state_of_charge": 1.0, "enabled": True,
               "hover_seconds_remaining": 60.0, "depleted": False,
               "energy_used_j": 0.0, "power_w": 180.0}
    meas = {"pos": np.array([0.6, -0.4, 3.0]), "vel": np.array([0.1, 0.0, -0.5]),
            "rpy": np.array([0.02, 0.01, 0.0]), "omega": np.zeros(3),
            "marker_quality": marker_quality,
            "marker_detected": marker_quality > 0.1,
            "pad_velocity": np.array([2.0, 0.5, 0.0]), "pad_speed": 2.06,
            "battery": battery}
    diag = {"mean_wind_i": np.zeros(3), "aero_force_i": np.zeros(3),
            "aero_force_mag": 0.0, "acceleration_i": np.zeros(3),
            "marker_quality": marker_quality, "battery": battery,
            "pad": {"valid": True, "velocity": np.array([2.0, 0.5, 0.0]),
                    "speed": 2.06},
            "pad_velocity_i": np.array([2.0, 0.5, 0.0]), "pad_speed": 2.06,
            "gnss": {"enabled": True, "valid": gnss_valid,
                     "quality": gnss_quality, "deck_quality": gnss_quality,
                     "sigma_xy_m": (1.0 - gnss_quality) * 20.0,
                     "deck_sigma_xy_m": (1.0 - gnss_quality) * 20.0,
                     "nlos_detected_fraction": (1.0 - gnss_quality) * 0.5,
                     "satellites_tracked": int(4 + 8 * gnss_quality),
                     "fix_type": 3, "cn0_mean_db": 45.0, "hdop": 1.0,
                     "vdop": 1.6, "residual_rms_m": 0.0}}
    sem = compute_features(diag, meas, None, cfg)
    return sem, make_observation(meas, sem, cfg)


def test_a_degrading_fix_lowers_the_semantic_confidence(cfg):
    open_sky, _ = _state(cfg, gnss_quality=1.0, marker_quality=0.0)
    canyon, _ = _state(cfg, gnss_quality=0.2, marker_quality=0.0)

    assert canyon.gnss_integrity < open_sky.gnss_integrity
    assert canyon.nav_confidence < open_sky.nav_confidence
    # And it reaches the node the goal is read off, not just its own channel.
    assert canyon.touchdown_safety < 0.5 * open_sky.touchdown_safety


def test_the_policy_can_see_the_uncertainty(cfg):
    """Three dedicated channels, and they have to actually move."""
    _, open_sky = _state(cfg, gnss_quality=1.0, marker_quality=0.0)
    _, canyon = _state(cfg, gnss_quality=0.2, marker_quality=0.0)

    moved = np.flatnonzero(np.abs(canyon - open_sky) > 1e-9)
    assert moved.size == 3
    # The last three channels are the receiver's account of itself.
    assert set(moved.tolist()) == {open_sky.size - 3, open_sky.size - 2,
                                   open_sky.size - 1}
    integrity, nlos, sigma = canyon[-3:]
    assert integrity == pytest.approx(0.2)
    assert nlos > 0.0 and sigma > 0.0


def test_markers_rescue_a_bad_fix(cfg):
    """The substitutability the ontology exists to learn.

    With the tags in frame the fix hardly matters; the moment they leave it the
    fix is all there is. If this ever becomes a product instead of a noisy-OR,
    a blind vehicle with a good fix and a sighted one with a bad fix stop being
    distinguishable, and the whole experiment loses its point.
    """
    blind, _ = _state(cfg, gnss_quality=0.2, marker_quality=0.0)
    sighted, _ = _state(cfg, gnss_quality=0.2, marker_quality=0.9)

    assert sighted.nav_confidence > 4.0 * blind.nav_confidence
    # The fix itself is untouched -- only what can be done despite it.
    assert sighted.gnss_integrity == pytest.approx(blind.gnss_integrity)


def test_the_reward_prices_flying_on_a_pose_nothing_vouches_for(cfg):
    """And prices it large enough to change behaviour, not just to exist."""
    weight = cfg.reward.manual.w_nav
    open_sky, _ = _state(cfg, gnss_quality=1.0, marker_quality=0.0)
    canyon, _ = _state(cfg, gnss_quality=0.2, marker_quality=0.0)

    penalty = weight * ((1.0 - canyon.nav_confidence)
                        - (1.0 - open_sky.nav_confidence))
    assert penalty > 0.0
    # Over one episode this has to be comparable with the terminal rewards, or
    # the policy is better off ignoring it and taking the crash.
    episode = penalty * cfg.sim.max_time
    assert episode > 0.25 * abs(cfg.reward.manual.success)


def test_the_ontology_graph_carries_the_uncertainty_nodes(cfg):
    """The R-GAT reads node values; a node that stops being populated is mute."""
    names = list(cfg.ontology.node_names)
    canyon, _ = _state(cfg, gnss_quality=0.2, marker_quality=0.3)
    graph = build_ontology_graph(canyon, cfg)
    values = canyon.node_values

    assert values[names.index("GnssIntegrity")] == pytest.approx(0.2)
    assert values[names.index("MarkerQuality")] == pytest.approx(0.3)
    # GnssIntegrity must still be wired to what it supports, or its value is
    # read and then goes nowhere.
    gnss_node = names.index("GnssIntegrity")
    assert int(np.sum(graph.src == gnss_node)) >= 3


def test_an_invalid_fix_is_not_quietly_open_sky(cfg):
    """A receiver that says it has no solution must not read as a good one."""
    lost, _ = _state(cfg, gnss_quality=0.8, marker_quality=0.0, gnss_valid=False)

    assert lost.gnss_integrity == pytest.approx(0.0)
    assert lost.nav_confidence == pytest.approx(0.0)
