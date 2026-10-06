"""The vertical inhibit holds the rate; it does not ratchet altitude upward.

``spatial-reference/1`` and every ladder rung rewrite the vertical command only
when it is NEGATIVE: a descending vehicle is braked with up to full upward
authority and an ascending one passes through untouched. The
direct-acceleration plant has no restoring force on altitude, so that asymmetry
integrates a zero-mean policy upward. Measured on a fresh policy at the easiest
curriculum rung, a requested -0.065 m/s^2 came out as an applied +0.082, the
median climb was +19.71 m, and 27 of 40 episodes ended on the 20 m ceiling with
the vehicle nowhere near the pad.

``spatial-reference/2`` holds the vertical rate instead. Everything else --
packet, graph, ontology, reward, every capability -- is identical, so /1 stays
loadable and the two rungs remain directly comparable.

The swept oracle sits inside the terminal-descent corridor on 64 % of its steps
and has 2.8 % of its actions overridden, so a gain sweep cannot see any of
this; that is why a 96-cell sweep reported a 95.8 % ceiling on a contract
whose exploration distribution never descends.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ontology_rgat.spatial.core import (
    REFERENCE_SCHEMA, REFERENCE_SCHEMAS, SpatialConfig)
from ontology_rgat.spatial.dynamics import ATTITUDE_OMEGA, THRUST_TAU
from ontology_rgat.spatial.environment import SpatialLandingEnv
from ontology_rgat.spatial.safety import ReferenceSpatialSupervisor
from ontology_rgat.spatial.training import SpatialAgent, _graph_planes, _ontology

DELAY = THRUST_TAU + 2 / ATTITUDE_OMEGA


def _delay(schema):
    """The supervisor's model of the plant delay: thrust lag, the rung's own
    attitude bandwidth and, from /3 on, the actuation latency."""
    cfg = replace(SpatialConfig(), schema=schema)
    return THRUST_TAU + 2 / cfg.attitude_omega + cfg.actuation_delay_s


class _Inhibited:
    """The supervisor decision under test: inhibited, not aborting."""
    inhibited, abort = True, False


class _Own:
    def __init__(self, vz):
        self.own_velocity = np.array([0.0, 0.0, float(vz)])


class _Estimate:
    def __init__(self, vz):
        self.own = _Own(vz)


def _applied(schema, requested, vz):
    cfg = replace(SpatialConfig(), schema=schema)
    supervisor = ReferenceSpatialSupervisor()
    normalized = supervisor.action(
        np.asarray(requested, dtype=float), _Estimate(vz), _Inhibited(), cfg)
    return np.asarray(normalized) * np.asarray(cfg.max_acceleration)


def test_the_active_contract_is_a_holding_rung():
    assert REFERENCE_SCHEMA == "spatial-reference/8"
    assert REFERENCE_SCHEMAS == ("spatial-reference/1", "spatial-reference/2",
                                 "spatial-reference/3", "spatial-reference/4",
                                 "spatial-reference/5", "spatial-reference/6",
                                 "spatial-reference/7", "spatial-reference/8")
    for schema in REFERENCE_SCHEMAS[1:]:
        assert replace(SpatialConfig(), schema=schema).vertical_inhibit_holds, schema
    assert not replace(SpatialConfig(), schema="spatial-reference/1").vertical_inhibit_holds
    # Every frozen ladder rung keeps the one-sided brake it was trained on.
    for rung in range(3, 11):
        cfg = replace(SpatialConfig(), schema=f"spatial-causal-rgat/{rung}")
        assert not cfg.vertical_inhibit_holds


@pytest.mark.parametrize("vz", [0.0, 0.4, -0.4])
def test_neither_rung_lets_an_inhibited_vehicle_descend(vz):
    """The inhibit is not weakened. This is the safety property it exists for."""
    for schema in REFERENCE_SCHEMAS:
        assert _applied(schema, [0.0, 0.0, -1.0], vz)[2] >= 0.0 or vz > 0.0


def test_the_one_sided_brake_passes_a_climb_through_and_the_hold_does_not():
    """The defect, stated as a difference between the two rungs."""
    climb = [0.0, 0.0, 1.0]
    # /1: a positive command is not touched at all, from any vertical rate.
    assert _applied("spatial-reference/1", climb, 0.0)[2] == pytest.approx(2.0)
    assert _applied("spatial-reference/1", climb, 0.5)[2] == pytest.approx(2.0)
    # /2: the command is replaced by whatever drives the rate to zero, so a
    # vehicle already climbing is brought back rather than accelerated.
    for schema in REFERENCE_SCHEMAS[1:]:
        assert _applied(schema, climb, 0.0)[2] == pytest.approx(0.0)
        assert _applied(schema, climb, 0.5)[2] == pytest.approx(-0.5 / _delay(schema))


def test_the_hold_still_brakes_a_descent_like_the_one_sided_law():
    """What changes is the free climb, not the descent brake."""
    descend = [0.0, 0.0, -1.0]
    for schema in REFERENCE_SCHEMAS:
        assert _applied(schema, descend, -0.5)[2] == pytest.approx(0.5 / _delay(schema))


def test_the_two_rungs_differ_in_nothing_but_the_signature_and_this_law():
    one = replace(SpatialConfig(), schema="spatial-reference/1")
    two = replace(SpatialConfig(), schema="spatial-reference/2")
    assert one.packet_fields == two.packet_fields
    assert one.registry_hash == two.registry_hash
    assert _ontology(one).node_names == _ontology(two).node_names
    assert _graph_planes(one) == _graph_planes(two)
    for capability in ("direct_acceleration", "reference_tracking",
                       "matched_disturbances", "reference_context",
                       "reference_supervisor", "camera_transport_delay"):
        assert getattr(one, capability) == getattr(two, capability), capability
    # A checkpoint trained under one must not silently load under the other.
    assert one.signature["config_sha256"] != two.signature["config_sha256"]
    assert one.signature["packet_registry"] == two.signature["packet_registry"]


def test_a_sampled_policy_no_longer_climbs_out_of_the_task():
    """The behavioural claim, on the rung the policy actually trains on.

    Six seeds at the easiest rung. Under /1 the median climb was +19.71 m over
    40 seeds and most episodes ended on the 20 m ceiling; the bound here is
    loose on purpose -- it is a regression guard against the ratchet returning,
    not a performance assertion.
    """
    climbs, statuses = [], []
    for schema in REFERENCE_SCHEMAS:
        cfg = replace(SpatialConfig(), schema=schema)
        env = SpatialLandingEnv(cfg)
        try:
            per_schema = []
            for seed in (5001, 5002, 5003, 5004, 5005, 5006):
                agent = SpatialAgent("ppo_vector_canonical", cfg, 828, -1.1)
                observation, _ = env.reset(seed=seed, difficulty=0.0)
                heights = []
                for _ in range(int(cfg.horizon / cfg.dt) + 2):
                    observation, _r, done, _t, info = env.step(
                        agent.act(observation)[1])
                    heights.append(info["truth_relative_position"][2])
                    if done:
                        statuses.append((schema, info["status"]))
                        break
                per_schema.append(max(heights) - heights[0])
            climbs.append(float(np.median(per_schema)))
        finally:
            env.close()
    braking, *holding = climbs
    for schema, climb in zip(REFERENCE_SCHEMAS[1:], holding):
        assert climb < 1.0, f"{schema}: the vertical hold climbed {climb:.2f} m"
        assert climb < braking
        # Under /1 the climb runs into the 20 m causal envelope; the holding
        # rungs must not.
        held = [status for s, status in statuses if s == schema]
        assert held.count("SAFETY_ENVELOPE_VIOLATION") == 0, (schema, held)
