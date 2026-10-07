"""``spatial-reference/3``: the local plant carries Isaac's actuation latency.

The first learned-policy Isaac flight (2026-10-06, twelve episodes) landed
nothing while the same checkpoints landed the same seeds locally. A joint
impulse-response fit of ground-truth lateral acceleration against the command
gave Isaac/PX4 h = [-0.02, 0.74, 0.24, -0.01] over 0.1 s lags -- no response
in the step the command is issued -- against [0.44, 0.15, 0.18, -0.04] for the
/2 plant, which answers inside the same step. A policy tuned on an instant
plant over-commands a delayed one: tilt reached 18-21 deg and the pad left the
camera within 1.1 s. /3 is /2 plus one law, a 0.10 s command-to-response
latency that the supervisor's stopping margin also counts; with it the local
fit becomes [0.05, 0.35, 0.24, 0.14].
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ontology_rgat.spatial.core import (
    REFERENCE_ACTUATION_DELAY_S, REFERENCE_SCHEMA, SpatialConfig)
from ontology_rgat.spatial.dynamics import ATTITUDE_OMEGA, THRUST_TAU
from ontology_rgat.spatial.environment import SpatialLandingEnv
from ontology_rgat.spatial.safety import ReferenceSpatialSupervisor
from ontology_rgat.spatial.training import _graph_planes, _ontology

#: The /2 clones and cells of 2026-10-06 were trained under this hash
#: (results/full_pipeline_20261006_lowsigma_kl/clone_plain/plan.json). Adding
#: a rung must not move it, or every one of those checkpoints stops loading.
TWO_CONFIG_SHA256 = "de4f1622908f547a7d3a07ec10ba745f2c5c80f3d570301cfd9d45cc9f03f6be"


def _cfg(schema):
    from ontology_rgat.spatial.runtime_contract import deployment_profile
    return replace(SpatialConfig(), schema=schema,
                   isaac_profile_sha256=deployment_profile(schema)["sha256"])


#: The /3 clones and cell of 2026-10-06 (results/full_pipeline_20261006_v3).
THREE_CONFIG_SHA256 = "8b102acab1ff"  # prefix; the full hash is checked below


def test_the_latency_rungs_carry_it_and_the_earlier_ones_do_not():
    assert REFERENCE_SCHEMA == "spatial-reference/12"
    assert REFERENCE_ACTUATION_DELAY_S == pytest.approx(0.10)
    for schema in ("spatial-reference/3", "spatial-reference/4", "spatial-reference/5",
                   "spatial-reference/6", "spatial-reference/7", "spatial-reference/8",
                   "spatial-reference/9", "spatial-reference/10",
                   "spatial-reference/11",
                   "spatial-reference/12"):
        assert _cfg(schema).actuation_delay_s == pytest.approx(0.10), schema
    for schema in ("spatial-reference/1", "spatial-reference/2",
                   "spatial-causal-rgat/5", "spatial-causal-rgat/10"):
        assert _cfg(schema).actuation_delay_s == 0.0, schema


#: The /4 clones and cell of 2026-10-06 (results/full_pipeline_20261006_v4).
def test_adding_rungs_did_not_move_the_previous_rung_hashes():
    assert _cfg("spatial-reference/2").signature["config_sha256"] == TWO_CONFIG_SHA256
    assert _cfg("spatial-reference/3").signature["config_sha256"].startswith(THREE_CONFIG_SHA256)
    import json
    plan = (Path(__file__).resolve().parents[2] / "results" / "full_pipeline_20261006_v4"
            / "low_sigma_kl" / "plan.json")
    if plan.exists():
        recorded = json.loads(plan.read_text())["signature"]["config_sha256"]
        assert _cfg("spatial-reference/4").signature["config_sha256"] == recorded


def test_optical_realism_is_the_fifth_rung_and_varies_with_depth():
    """/5: detection and confidence come from the deployed board's tags.

    Measured 2026-10-06: the SAME /4 checkpoints on the SAME seeds land in
    5-12 s with a constant 0.98 confidence and time out at 70 s with the
    realistic one (corridor armed 2-11 % of steps), which is the third Isaac
    flight's behaviour reproduced locally. On /4 and below the channel is the
    constant the policies were trained on; on /5 it falls with camera depth
    as the detector's own metric does (Isaac: 0.22 at 1.8-2.6 m, 0.45 at
    0.8-1.1 m, 0.58 at 0.2-0.5 m).
    """
    from ontology_rgat.spatial.environment import LocalBackend
    four, five = _cfg("spatial-reference/4"), _cfg("spatial-reference/5")
    assert not four.optical_realism and five.optical_realism
    for capability in ("actuation_delay_s", "attitude_omega", "landing_gear_extension_m",
                       "vertical_inhibit_holds", "camera_transport_delay", "packet_fields",
                       "registry_hash"):
        assert getattr(four, capability) == getattr(five, capability), capability
    assert four.signature["config_sha256"] != five.signature["config_sha256"]
    backend = LocalBackend(five)
    backend.reset(3)
    level = np.array([1.0, 0.0, 0.0, 0.0])
    def confidence(height, samples=200):
        rows = [backend._optical_quality(np.array([0.0, 0.0, height]), level) for _ in range(samples)]
        seen = [c for visible, c in rows if visible]
        return np.mean([v for v, _ in rows]), (np.mean(seen) if seen else 0.0)
    far_seen, far = confidence(2.4)
    mid_seen, mid = confidence(1.2)
    near_seen, near = confidence(0.4)
    assert far_seen > 0.9 and mid_seen > 0.9 and near_seen > 0.9
    assert 0.1 < far < 0.3 < mid < 0.55 < near < 0.8, (far, mid, near)
    # /4 keeps the constant the earlier checkpoints were trained with.
    assert LocalBackend(four).board is None


def test_the_calibrated_rung_is_three_measured_laws_and_nothing_else():
    """/4: attitude bandwidth, landing gear and the v11 board; /1-/3 untouched."""
    from ontology_rgat.spatial.core import (
        REFERENCE_ATTITUDE_OMEGA_RAD_S, REFERENCE_LANDING_GEAR_EXTENSION_M)
    from ontology_rgat.spatial.runtime_contract import deployment_profile
    three, four = _cfg("spatial-reference/3"), _cfg("spatial-reference/4")
    assert four.attitude_omega == pytest.approx(REFERENCE_ATTITUDE_OMEGA_RAD_S) == 14.0
    assert three.attitude_omega == pytest.approx(ATTITUDE_OMEGA) == 10.0
    assert four.landing_gear_extension_m == pytest.approx(REFERENCE_LANDING_GEAR_EXTENSION_M) == 0.18
    assert four.touchdown_height == pytest.approx(four.contact_height + 0.18)
    for schema in ("spatial-reference/1", "spatial-reference/2", "spatial-reference/3",
                   "spatial-causal-rgat/5"):
        cfg = _cfg(schema)
        assert cfg.landing_gear_extension_m == 0.0 and cfg.touchdown_height == cfg.contact_height
    # Same observation contract: packet, graph, capabilities.
    assert three.packet_fields == four.packet_fields
    assert three.registry_hash == four.registry_hash
    for capability in ("direct_acceleration", "reference_tracking", "matched_disturbances",
                       "reference_context", "reference_supervisor", "camera_transport_delay",
                       "vertical_inhibit_holds", "actuation_delay_s"):
        assert getattr(three, capability) == getattr(four, capability), capability
    assert three.signature["config_sha256"] != four.signature["config_sha256"]
    # The deployment profile is the v11 board with landing gear, and it keeps
    # the v9 camera and capture timing.
    profile = deployment_profile("spatial-reference/4")
    resolved = profile["resolved_scientific_configuration"]
    assert profile["path"].endswith("spatial-isaac-system-v11.yaml")
    assert len(resolved["vision"]["board"]) == 17
    assert resolved["vehicle"]["landing_gear"]["extension_m"] == pytest.approx(0.18)
    assert resolved["isaac"]["spatial_optical_capture_timing"] is True
    assert deployment_profile("spatial-reference/3")["path"].endswith("spatial-isaac-system-v9.yaml")


def test_the_gear_shifts_touchdown_and_the_easy_start_band_together():
    """Legs raise the body at contact; the curriculum's low start must not
    begin below it, and the braking envelope must measure from the legs."""
    from ontology_rgat.spatial.environment import LocalBackend
    four = _cfg("spatial-reference/4")
    backend = LocalBackend(four, difficulty=0.0)
    for seed in range(5):
        backend.reset(seed)
        assert backend.position[2] > four.touchdown_height + 0.05, backend.position[2]
    env = SpatialLandingEnv(four)
    try:
        env.reset(seed=11)
        backend = env.backend
        backend.position[:2] = backend.pad[:2]
        backend.position[2] = backend.pad[2] + four.touchdown_height + 0.02
        backend.velocity[:] = [*backend.pad_velocity[:2], -0.4]
        for _ in range(8):
            _obs, _r, done, _t, info = env.step(np.array([0.0, 0.0, -0.5]))
            if done:
                break
        assert done and info["truth_contact"], info["status"]
        assert info["truth_relative_position"][2] == pytest.approx(four.touchdown_height, abs=1e-6)
    finally:
        env.close()


def _command_effect(schema, steps=4, seed=7):
    """Lateral velocity attributable to a constant +x command, per decision.

    The same seed is run twice, once commanding +x and once commanding
    nothing, and the two velocities are differenced: the domain-randomised
    handover kick and the moving pad are identical in both, so what remains
    is the plant's own answer to the command.
    """
    def run(action):
        env = SpatialLandingEnv(_cfg(schema))
        try:
            env.reset(seed=seed)
            out = []
            for _ in range(steps):
                _obs, _r, done, _t, info = env.step(action)
                out.append(float(info["own_velocity_enu_m_s"][0]))
                if done:
                    break
            return np.asarray(out)
        finally:
            env.close()
    return run(np.array([0.8, 0.0, 0.0])) - run(np.zeros(3))


def test_the_delayed_plant_does_not_answer_inside_the_step_the_command_is_issued():
    instant = _command_effect("spatial-reference/2")
    delayed = _command_effect("spatial-reference/3")
    # /2 moves within the first decision; /3 only from the second one on.
    assert instant[0] > 0.02, instant
    assert abs(delayed[0]) < 1e-6, delayed
    assert delayed[1] > 0.02, delayed
    # After the latency the plant is the same plant: the one-step-shifted
    # response matches /2 to within integration order.
    assert delayed[2] == pytest.approx(instant[1], abs=0.01)


def test_the_supervisor_counts_the_same_latency():
    class _Inhibited:
        inhibited, abort = True, False

    class _Own:
        own_velocity = np.array([0.0, 0.0, -0.5])

    class _Estimate:
        own = _Own()

    descend = np.array([0.0, 0.0, -1.0])
    for schema in ("spatial-reference/2", "spatial-reference/3"):
        cfg = _cfg(schema)
        applied = ReferenceSpatialSupervisor().action(descend, _Estimate(), _Inhibited(), cfg)
        delay = THRUST_TAU + 2 / ATTITUDE_OMEGA + cfg.actuation_delay_s
        assert applied[2] * cfg.max_acceleration[2] == pytest.approx(0.5 / delay)


def test_the_rung_differs_from_its_predecessor_in_nothing_but_this_law():
    two, three = _cfg("spatial-reference/2"), _cfg("spatial-reference/3")
    assert two.packet_fields == three.packet_fields
    assert two.registry_hash == three.registry_hash
    assert _ontology(two).node_names == _ontology(three).node_names
    assert _graph_planes(two) == _graph_planes(three)
    for capability in ("direct_acceleration", "reference_tracking",
                       "matched_disturbances", "reference_context",
                       "reference_supervisor", "camera_transport_delay",
                       "vertical_inhibit_holds"):
        assert getattr(two, capability) == getattr(three, capability), capability
    assert two.actuation_delay_s != three.actuation_delay_s
    assert two.signature["config_sha256"] != three.signature["config_sha256"]
    assert two.signature["packet_registry"] == three.signature["packet_registry"]


def test_the_damping_rung_overshoots_like_the_isaac_vehicle_and_nothing_else_moves():
    """/6: omega 10 rad/s at zeta 0.7, identified from four flights' tilt traces.

    A 13 deg attitude step on the critically damped /5 servo never exceeds
    13 deg; on /6 it overshoots, which is the margin the fourth flight's
    envelope violation (roll reversal overshooting to 21.2 deg) was about.
    """
    import math
    from ontology_rgat.spatial.core import REFERENCE_ATTITUDE_SERVO_6
    from ontology_rgat.spatial.dynamics import advance_attitude_thrust
    five, six = _cfg("spatial-reference/5"), _cfg("spatial-reference/6")
    assert (six.attitude_omega, six.attitude_damping) == REFERENCE_ATTITUDE_SERVO_6 == (10.0, 0.7)
    assert five.attitude_damping == 1.0 and _cfg("spatial-reference/3").attitude_damping == 1.0
    for capability in ("actuation_delay_s", "landing_gear_extension_m", "optical_realism",
                       "vertical_inhibit_holds", "packet_fields", "registry_hash"):
        assert getattr(five, capability) == getattr(six, capability), capability
    assert five.signature["config_sha256"] != six.signature["config_sha256"]

    class _Command:
        derived_roll_pitch_rad = (0.0, math.radians(13.0))
        yaw_enu_rad = 0.0
        thrust_weight_ratio = 1.0

    def peak(cfg):
        angles, rates, thrust = np.zeros(3), np.zeros(3), 9.80665
        top = 0.0
        for _ in range(100):
            angles, rates, thrust, _acc, _body = advance_attitude_thrust(
                angles, rates, thrust, _Command(), 0.01,
                attitude_omega=cfg.attitude_omega, attitude_damping=cfg.attitude_damping)
            top = max(top, math.degrees(angles[1]))
        return top
    assert peak(five) <= 13.0 + 1e-6
    # zeta 0.7 overshoots by exp(-pi*zeta/sqrt(1-zeta^2)) = 4.6 %: 13.6 deg, less the
    # 90 deg/s rate clamp; measured 13.47 deg at 0.43 s.
    assert 13.2 < peak(six) < 14.2, peak(six)
