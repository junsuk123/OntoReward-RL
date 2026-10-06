"""Invariants that must hold across the 2D and 3D routes simultaneously.

The 3D route was created by copying the 2D one, so each route's own test file
could stay green while the two drifted apart: for a week they optimized
different terminal tables and different discount horizons, which makes a
dimension study meaningless. Per-route tests cannot catch that by construction
-- only a test that loads both can. New shared contract pieces belong here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ontology_rgat.landing.terminal import (
    REFERENCE_CURRICULUM_START_UNSAFE,
    REFERENCE_DISCOUNT_TAU_S,
    REFERENCE_TERMINAL_REWARDS,
    UNSAFE_REASONS,
    validate_curriculum_ramp,
    validate_terminal_ordering,
)
from ontology_rgat.spatial.core import SpatialConfig
from ontology_rgat.spatial.environment import Evaluator
from ontology_rgat.two_axis.config import REFERENCE_CONFIG_PATH, load_config

PYTHON_ROOT = Path(__file__).resolve().parents[1] / "python"


@pytest.fixture(scope="module")
def routes():
    return load_config(REFERENCE_CONFIG_PATH), SpatialConfig()


def test_both_routes_optimize_the_same_terminal_table(routes):
    reference, spatial = routes
    assert dict(reference.reward.terminal_bonus) == REFERENCE_TERMINAL_REWARDS
    assert Evaluator.BONUSES == REFERENCE_TERMINAL_REWARDS
    assert Evaluator.UNSAFE_STATUSES == frozenset(UNSAFE_REASONS)
    assert tuple(reference.reward.unsafe_reasons) == UNSAFE_REASONS
    assert spatial.registry_hash  # route still constructs under the shared table


def test_both_routes_share_the_discount_horizon(routes):
    reference, spatial = routes
    assert reference.timing.discount_time_constant_s == REFERENCE_DISCOUNT_TAU_S
    assert spatial.discount_tau == REFERENCE_DISCOUNT_TAU_S
    # A 3D mission that outlasts the 2D one would re-weight identical outcomes.
    assert spatial.horizon == reference.timing.mission_duration_limit_s


def test_both_routes_share_the_curriculum_ramp(routes):
    reference, spatial = routes
    start = REFERENCE_CURRICULUM_START_UNSAFE
    assert reference.curriculum.start_unsafe_contact_penalty == start
    assert spatial.curriculum.start_unsafe_contact_penalty == start
    validate_curriculum_ramp(start, REFERENCE_TERMINAL_REWARDS)
    validate_terminal_ordering(REFERENCE_TERMINAL_REWARDS)


def test_the_unsafe_penalty_is_never_identified_by_its_magnitude():
    """SAFE_ABORT once took the unsafe outcomes' own value of -40.

    Every site that recognized an unsafe outcome by matching that literal then
    scored a safe abort as a crash. Status sets, never magnitudes.
    """
    assert "SAFE_ABORT" not in Evaluator.UNSAFE_STATUSES
    assert Evaluator.NOMINAL_UNSAFE_PENALTY == max(
        REFERENCE_TERMINAL_REWARDS[name] for name in UNSAFE_REASONS)
    assert Evaluator.NOMINAL_UNSAFE_PENALTY < REFERENCE_TERMINAL_REWARDS["SAFE_ABORT"]


def test_only_one_module_defines_terminal_reward_magnitudes():
    """Structural guard: the drift happened because four copies existed.

    Any module that pairs a terminal reason with a numeric literal is defining
    a second table. Tests that only read the values cannot see a copy appear --
    they read whichever one their own route happens to import.
    """
    pattern = re.compile(
        r"""["']?(SUCCESS|SAFE_ABORT|TASK_TIMEOUT|UNSAFE_CONTACT|"""
        r"""UNAUTHORIZED_CONTACT|MISSED_PAD_CONTACT|SAFETY_ENVELOPE_VIOLATION)"""
        r"""["']?\s*[:=,]\s*-?\d+(\.\d+)?""")
    allowed = {Path("ontology_rgat/landing/terminal.py")}
    offenders = {}
    for path in sorted(PYTHON_ROOT.rglob("*.py")):
        relative = path.relative_to(PYTHON_ROOT)
        if relative in allowed or "_archive" in relative.parts:
            continue
        hits = sorted({match.group(1) for match in pattern.finditer(path.read_text())})
        if hits:
            offenders[str(relative)] = hits
    assert not offenders, (
        "terminal reward magnitudes defined outside landing/terminal.py: "
        f"{offenders}")


def test_one_axis_spec_reproduces_the_reference_registry():
    """The axis-generic spec must *derive* the 2D packet, not merely resemble it.

    If this holds, the 3D packet is a substitution rather than a design choice:
    the same per-axis fields emitted twice. If it ever fails, the generic spec
    has drifted from the ported MATLAB contract and any 3D packet built from it
    is measuring something else.
    """
    from ontology_rgat.landing import packet as spec
    from ontology_rgat.two_axis.contracts import load_reference_registry

    reference = load_reference_registry()
    expected = [(field["name"], field["source"], field["normalization"])
                for field in reference.fields]
    produced = [(name, item.source, spec.normalization(item))
                for name, item, _ in spec.field_specs(spec.PLANAR_AXES)]
    assert produced == expected


def test_the_spatial_packet_is_the_planar_one_with_a_second_axis():
    from ontology_rgat.landing import packet as spec

    reference = [item for item in spec.FIELD_SPECS if item.reference]
    per_axis = sum(item.per_axis for item in reference)
    shared = len(reference) - per_axis
    assert (per_axis, shared) == (12, 14)
    assert len(spec.field_names(spec.PLANAR_AXES)) == per_axis + shared == 26
    assert len(spec.field_names(spec.SPATIAL_AXES)) == 2 * per_axis + shared == 38
    # Every shared field keeps its name and position; only axis fields multiply.
    planar = spec.field_names(spec.PLANAR_AXES)
    spatial = spec.field_names(spec.SPATIAL_AXES)
    shared_names = [item.name for item in reference if not item.per_axis]
    assert [n for n in planar if n in shared_names] == shared_names
    assert [n for n in spatial if n in shared_names] == shared_names
    assert spec.registry(spec.PLANAR_AXES)["sha256"] != spec.registry(
        spec.SPATIAL_AXES)["sha256"]


def test_extras_are_opt_in_and_never_shift_the_reference_core():
    """An Isaac-matching extra must not be able to redefine the ported contract.

    Extras append; they never reorder or displace a reference field, so a
    checkpoint trained on the core still reads the same values at the same
    indices for every field it knows about.
    """
    from ontology_rgat.landing import packet as spec

    for axes in (spec.PLANAR_AXES, spec.SPATIAL_AXES):
        core = spec.field_names(axes)
        full = spec.field_names(axes, extras=True)
        assert full[:len(core)] == core
        extra = full[len(core):]
        assert set(extra).isdisjoint(core)
        assert spec.registry(axes)["sha256"] != spec.registry(
            axes, extras=True)["sha256"]
    # The active 3D contract: the derived core plus the declared 3D extras
    # (capture-vs-now bearing per axis, transport age, disturbance per axis
    # and vertically).
    assert len(spec.field_names(spec.SPATIAL_AXES, extras=True)) == 44


def test_normalize_and_decode_are_inverses_on_both_axis_counts():
    """One pair of functions, so a scale cannot differ by direction or dimension.

    Four hand-maintained copies of this pair existed before: forward in
    two_axis/contracts.py and spatial/core.py, inverse in ontology_v28.py and
    spatial/context.py, each repeating the scales.
    """
    import math

    import numpy as np

    from ontology_rgat.landing import observation, packet as spec

    rng = np.random.default_rng(20261005)
    for axes, fov in ((spec.PLANAR_AXES, math.radians(25)),
                      (spec.SPATIAL_AXES, {"x": math.radians(45),
                                           "y": math.radians(36.9)})):
        names = spec.field_names(axes)
        raw = {}
        for name, item, _ in spec.field_specs(axes):
            if item.scale is spec.BINARY:
                raw[name] = float(rng.integers(0, 2))
            elif item.scale == spec.PROBABILITY:
                raw[name] = float(rng.random())
            elif item.scale == spec.IDENTITY:
                raw[name] = float(rng.uniform(-1, 1))
            elif item.scale == spec.MISSION_LIMIT:
                raw[name] = float(rng.uniform(0, 70))
            else:
                raw[name] = float(rng.uniform(-3, 3))
        values = observation.normalize(raw, axes, half_fov=fov)
        assert values.shape == (len(names),)
        back = observation.decode(values, axes, half_fov=fov)
        for name, item, _ in spec.field_specs(axes):
            if item.scale == spec.MISSION_LIMIT:
                continue  # clipped by construction, not invertible
            assert back[name] == pytest.approx(raw[name], rel=1e-5, abs=1e-6), name


def test_a_missing_or_nonfinite_packet_field_is_refused():
    import math

    from ontology_rgat.landing import observation, packet as spec

    raw = {name: 0.0 for name in spec.field_names(spec.SPATIAL_AXES)}
    observation.normalize(raw, spec.SPATIAL_AXES, half_fov=math.radians(45))
    incomplete = dict(raw)
    incomplete.pop("predictedFovMargin_y")
    with pytest.raises(KeyError, match="predictedFovMargin_y"):
        observation.normalize(incomplete, spec.SPATIAL_AXES, half_fov=math.radians(45))
    nonfinite = dict(raw, h=float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        observation.normalize(nonfinite, spec.SPATIAL_AXES, half_fov=math.radians(45))


def _unified():
    from dataclasses import replace

    from ontology_rgat.spatial.core import REFERENCE_SCHEMA

    return replace(SpatialConfig(), schema=REFERENCE_SCHEMA)


def test_the_unified_contract_is_the_derived_one_not_a_ladder_rung():
    from ontology_rgat.landing import packet as spec

    cfg = _unified()
    assert cfg.axes == spec.SPATIAL_AXES
    assert tuple(cfg.packet_fields) == spec.field_names(cfg.axes, extras=True)
    assert cfg.registry_hash == spec.registry(cfg.axes, extras=True)["sha256"]
    # Every quantity the reference graph reads is present, which was not true
    # of the default spatial contract: predicted bearing, predicted margin and
    # acceleration uncertainty arrived only at ladder rungs 6 and 7.
    for required in ("predictedBearing_x", "predictedBearing_y",
                     "predictedFovMargin_x", "predictedFovMargin_y",
                     "accelerationStd"):
        assert required in cfg.packet_fields


def test_the_unified_contract_builds_one_reference_plane_per_axis():
    """The 3D graph must be the 2D graph per axis, with real channel meanings.

    The previous default filled the nine rows positionally: row 2's `validity`
    channel carried the remaining mission time. The typed relation kernels index
    channels, so that is not a 3D ontology, it is nine rows of unrelated numbers.
    """
    import numpy as np

    from ontology_rgat.landing.plane_graph import FEATURE_CHANNELS
    from ontology_rgat.spatial.environment import SpatialLandingEnv

    cfg = _unified()
    env = SpatialLandingEnv(cfg)
    try:
        observation, _ = env.reset(seed=3001)
        for _ in range(12):
            observation, *_ = env.step(np.zeros(3))
    finally:
        env.close()
    from ontology_rgat.landing.ontology import SPATIAL_EXTENSIONS, schema

    ontology = schema(SPATIAL_EXTENSIONS)
    graphs = np.asarray(observation.graph.X)
    assert graphs.shape == (len(cfg.axes), ontology.node_count,
                            len(FEATURE_CHANNELS))
    for plane in graphs:
        # `bias` is the constant channel and `typeId` the node index; a
        # positional fill overwrites both.
        assert np.allclose(plane[:, FEATURE_CHANNELS.index("bias")], 1.0)
        np.testing.assert_allclose(
            plane[:, FEATURE_CHANNELS.index("typeId")],
            np.arange(1, ontology.node_count + 1) / ontology.node_count,
            atol=1e-6)
        # Row 2 is DroneTranslation; its validity channel is a validity flag.
        assert plane[2, FEATURE_CHANNELS.index("validity")] == 1.0
    assert np.isfinite(graphs).all() and np.abs(graphs).max() <= 1.0


def test_both_dimensions_agree_on_the_graph_definition():
    """2D and 3D must build their planes from the same function object.

    Identity, not equality: two copies that happen to agree today is exactly
    the state this refactor removed, and it is invisible to a value assertion.
    """
    from ontology_rgat.landing import plane_graph
    from ontology_rgat.spatial import context
    from ontology_rgat.two_axis import ontology_v28

    assert ontology_v28.build_plane is plane_graph.build_plane
    assert context.build_plane is plane_graph.build_plane
    assert ontology_v28.descent_eligibility is plane_graph.descent_eligibility
    assert context.descent_eligibility is plane_graph.descent_eligibility
    # The channel order is the contract the typed relation kernels index.
    assert plane_graph.FEATURE_CHANNELS == ontology_v28.FEATURE_CHANNELS


def test_head_geometry_is_implied_by_the_axis_count():
    """The 2D head's four magic numbers are what one horizontal axis produces."""
    from ontology_rgat.landing.packet import PLANAR_AXES, SPATIAL_AXES, head_geometry

    planar = head_geometry(PLANAR_AXES)
    assert (planar.packet_dim, planar.action_dim, planar.descent_axis,
            planar.graph_planes) == (26, 2, 1, 1)
    spatial = head_geometry(SPATIAL_AXES, extras=True)
    assert (spatial.packet_dim, spatial.action_dim, spatial.descent_axis,
            spatial.graph_planes) == (44, 3, 2, 2)
    # The descent command is always the last action component.
    for axes in (PLANAR_AXES, SPATIAL_AXES):
        geometry = head_geometry(axes)
        assert geometry.descent_axis == geometry.action_dim - 1


def test_the_spatial_head_is_built_from_the_derived_geometry():
    from dataclasses import replace

    from ontology_rgat.landing.packet import head_geometry
    from ontology_rgat.spatial.core import REFERENCE_SCHEMA
    from ontology_rgat.spatial.training import SpatialAgent

    cfg = replace(SpatialConfig(), schema=REFERENCE_SCHEMA)
    geometry = head_geometry(cfg.axes, extras=True)
    agent = SpatialAgent("ppo_ontology_rgat", cfg, seed=3)
    assert agent.actor.graph_planes == geometry.graph_planes
    assert agent.actor.descent_axis == geometry.descent_axis
    assert agent.actor.log_std.shape == (geometry.action_dim,)
    assert len(cfg.packet_fields) == geometry.packet_dim


def _graph_blind_fields(fields, build, base):
    """Packet fields that provably never change the graph."""
    import numpy as np

    reference = np.asarray(build(base))
    blind = []
    for index, name in enumerate(fields):
        moved = False
        for delta in (0.17, -0.23, 0.41):
            probe = np.array(base, dtype=np.float32)
            probe[index] = np.clip(probe[index] + delta, -0.98, 0.98)
            if probe[index] == base[index]:
                continue
            if not np.array_equal(np.asarray(build(probe)), reference):
                moved = True
                break
        if not moved:
            blind.append(name)
    return blind


def test_3d_gives_the_baseline_no_information_the_graph_arms_cannot_see():
    """The confound that would have hidden the proposed method's advantage.

    ``ppo_semantic_flat`` and ``ppo_ontology_rgat`` read the graph alone
    (``self.raw(graphs.flatten(1))``); only ``ppo_vector_canonical`` reads the
    packet. So a packet field absent from the context rows is information the
    BASELINE holds exclusively. Before the ontology was extended, 3D had six
    such fields against the planar task's two, and the three extra ones --
    capture-vs-now bearing per axis and the transport age -- were exactly the
    quantities describing what 3D adds. The baseline would have been handed the
    new phenomena and the proposed arm denied them.

    What may remain blind is the planar task's own inherited set: the
    policy-memory fields, which upstream's nine rows do not read either.
    """
    import numpy as np
    from dataclasses import replace

    from ontology_rgat.spatial.context import axis_context_graphs
    from ontology_rgat.spatial.core import REFERENCE_SCHEMA
    from ontology_rgat.spatial.environment import SpatialLandingEnv

    cfg = replace(SpatialConfig(), schema=REFERENCE_SCHEMA)
    env = SpatialLandingEnv(cfg)
    try:
        observation, _ = env.reset(seed=3001)
        for _ in range(15):
            observation, *_ = env.step(np.zeros(3))
    finally:
        env.close()
    blind = _graph_blind_fields(
        list(cfg.packet_fields),
        lambda values: axis_context_graphs(values, cfg),
        np.asarray(observation.packet.values, dtype=np.float32))
    assert set(blind) <= {"previousAction_x", "previousAction_y",
                          "previousNormalizedActionZ"}, blind
    # The quantities describing what 3D adds must all reach the graph.
    for name in ("estimatedBearing_x", "estimatedBearing_y",
                 "opticalTransportAge", "disturbanceEstimate_x",
                 "disturbanceEstimate_y", "disturbanceEstimateZ"):
        assert name in cfg.packet_fields and name not in blind, name


def test_the_spatial_ontology_extends_the_planar_one_without_disturbing_it():
    from ontology_rgat.landing.ontology import (
        BASE_NODES, PLANAR_EXTENSIONS, SPATIAL_EXTENSIONS, schema)
    from ontology_rgat.two_axis.ontology_v28 import (
        GRAPH_EDGES, GRAPH_SCHEMA_HASH, NODE_NAMES, READOUT_GROUPS, SCHEMA_VERSION)
    from ontology_rgat.landing.ontology import schema_hash
    from ontology_rgat.landing.plane_graph import FEATURE_CHANNELS

    planar = schema(PLANAR_EXTENSIONS)
    assert planar.node_names == NODE_NAMES == BASE_NODES
    assert planar.edges == GRAPH_EDGES
    assert planar.readout_groups == READOUT_GROUPS
    # The ported contract's structure digest must not move.
    assert schema_hash(planar, feature_channels=FEATURE_CHANNELS,
                       version=SCHEMA_VERSION,
                       decoding="python-causal-v3-smooth-v28") == GRAPH_SCHEMA_HASH

    spatial = schema(SPATIAL_EXTENSIONS)
    assert spatial.node_names[:len(BASE_NODES)] == BASE_NODES
    assert spatial.extensions == ("DisturbanceEstimate", "MeasurementLatency")
    # Indices the actor depends on must not shift.
    assert spatial.node_names.index("DescentEligibility") == 7
    # An extension joins an existing readout group; there are still four.
    assert len(spatial.readout_groups) == len(READOUT_GROUPS) == 4
    assert set(spatial.declared_edges) > set(planar.declared_edges)


def test_every_declared_ontology_extension_has_a_context_row():
    """A node with no row would be nine zeros the attention still spends on."""
    from ontology_rgat.landing.ontology import EXTENSION_NAMES
    from ontology_rgat.landing.plane_graph import EXTENSION_ROWS

    assert set(EXTENSION_NAMES) == set(EXTENSION_ROWS)


def test_both_routes_price_sustained_body_rate_in_the_running_term():
    """Angular rate must cost something while it is being held, in BOTH routes.

    It already appears inside ``readiness``, but readiness is paid as a
    difference, so the sum telescopes to the endpoint and a vehicle that spins
    for the whole approach pays nothing for it. That is what let the 3D
    policies sit outside the terminal-descent corridor, whose ``settled``
    condition is a body-rate limit: measured on the run1 checkpoints, the rate
    condition fails on 61-100 % of in-band steps while the tilt condition
    passes on 69-100 %.

    The weight lives in landing/terminal.py so a dimension study keeps one
    objective; this checks both routes actually read it and that the term is
    in ``running``, not in a telescoping difference.
    """
    import math
    from dataclasses import replace

    import numpy as np

    from ontology_rgat.landing.terminal import REFERENCE_SPIN_WEIGHT
    from ontology_rgat.spatial.core import SpatialConfig
    from ontology_rgat.spatial.environment import Evaluator, Truth
    from ontology_rgat.two_axis.config import RewardConfig
    from ontology_rgat.two_axis.reward import compute_reward

    assert RewardConfig().spin_weight == REFERENCE_SPIN_WEIGHT
    # Shipped off: weight 1.0 was trained and lost (see landing/terminal.py).
    # The wiring still has to be symmetric for the day it is raised, so the
    # comparison below forces a nonzero weight instead of the default.
    assert REFERENCE_SPIN_WEIGHT == 0.0, "raising this needs a trained comparison"
    weight = 1.0

    # 2D: the same state, spinning at the touchdown limit versus still.
    common = dict(ex_true_m=0.2, h_true_m=1.0, measured_bearing_rad=0.05,
                  bearing_valid=True, normalized_policy_action=np.zeros(2),
                  fov_rad=1.2, dt_s=0.1, terminal_reason=None,
                  config=replace(RewardConfig(), spin_weight=weight),
                  previous_goal_cost=0.0,
                  touchdown_pitch_rate_rad_s=math.radians(10.0))
    still = compute_reward(pitch_rate_rad_s=0.0, **common)
    spinning = compute_reward(pitch_rate_rad_s=math.radians(10.0), **common)
    assert spinning.spin_cost > still.spin_cost == 0.0
    assert spinning.running < still.running, (spinning.running, still.running)

    # 3D: same comparison through the evaluator's own components.
    cfg = SpatialConfig()
    import ontology_rgat.spatial.environment as spatial_env

    def state(rate):
        return Truth(relative_position=np.array([0.2, 0.0, 1.0]),
                     relative_velocity=np.zeros(3), roll_pitch=np.zeros(2),
                     angular_rate=np.array([rate, 0.0, 0.0]), contact=False)
    kwargs = dict(elapsed=1.0, dt=cfg.dt, action=np.zeros(3),
                  safety=type("S", (), dict(inhibited=False, abort=False))(),
                  abort_elapsed=0.0, bearings=np.zeros(2), visible=True)
    original = spatial_env.REFERENCE_SPIN_WEIGHT
    spatial_env.REFERENCE_SPIN_WEIGHT = weight
    try:
        quiet = Evaluator(cfg)
        quiet.reset(state(0.0))
        _, _, quiet_parts = quiet.evaluate(state(0.0), **kwargs)
        fast = Evaluator(cfg)
        fast.reset(state(cfg.touchdown_rate))
        _, _, fast_parts = fast.evaluate(state(cfg.touchdown_rate), **kwargs)
    finally:
        spatial_env.REFERENCE_SPIN_WEIGHT = original
    assert fast_parts["spin"] > quiet_parts["spin"] == 0.0
    assert fast_parts["running"] < quiet_parts["running"]

    # Same body rate, in units of each route's own limit, costs the same.
    assert math.isclose(fast_parts["spin"], spinning.spin_cost, rel_tol=1e-9)
