"""The ontology situation graph as the policy's state.

The method's claim is narrow and checkable: both learned arms solve the same
MDP, and the only thing that differs is whether the actor and the critic see
``G_t``. Everything here exists to make one of those words executable.
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from ontology_rgat.benchmarks.experiment import load_experiment    # noqa: E402
from ontology_rgat.perception.semantic_observation import (        # noqa: E402
    SemanticObservation)
from ontology_rgat.pipelines import (                              # noqa: E402
    ABLATION_PIPELINES, ALL_PIPELINES, LEGACY_PIPELINES, PIPELINES,
    PipelineSpec, assert_primary_baseline_equivalence,
    graph_state_pipeline_ids, validate_pipeline_configuration)
from ontology_rgat.ppo.graph_state_encoder import (                # noqa: E402
    GRAPH_STATE_REPRESENTATIONS, GraphStateEncoder, graph_feature_tensor,
    graph_state_topology)
from ontology_rgat.ppo.recurrent import (                          # noqa: E402
    PipelineActorCritic, recurrent_ppo_loss)
from ontology_rgat.ppo.recurrent_train import _reward              # noqa: E402
from ontology_rgat.rgat.state_graph import (                       # noqa: E402
    STATE_FEATURE_ROWS, STATE_GRAPH_EDGES, STATE_GRAPH_INPUT_DIM,
    STATE_GRAPH_VERSION, STATE_GOAL_NODE, STATE_NODE_NAMES,
    STATE_RELATION_NAMES, STATE_RISK_NODES, StateGraphGeometry,
    StateGraphScales, build_state_graph, empty_state_graph,
    state_node_values, unreachable_state_input_nodes)

from conftest import default_experiment_config                     # noqa: E402


def _observation(centroid=(0.0, 0.0), *, scale=0.06, visible=1.0, loss=0.0,
                 vz_safety=0.6, motion=0.9, confidence=0.9, memory=1.0):
    return SemanticObservation(
        keypoint_confidence=float(confidence),
        visible_keypoint_fraction=float(visible),
        image_alignment=0.5, apparent_target_scale=min(1.0, float(scale) / 0.75),
        image_plane_motion_safety=float(motion), scale_rate_safety=0.9,
        visibility_memory=float(memory), reacquisition_trend=0.5,
        vertical_motion_safety=float(vz_safety), attitude_stability=0.9,
        battery_risk=0.1, visual_loss_risk=float(loss),
        centroid_xy=tuple(float(v) for v in centroid), raw_scale=float(scale),
        visual_loss_duration_s=0.0 if loss <= 0.0 else 0.5)


# ------------------------------------------------------------- the schema

def test_the_schema_is_the_minimum_that_leaves_a_graph_to_reason_over():
    """Three properties, each of which the reduced study measured as load-bearing.

    Lose the two-stage structure and there is no message to pass; merge the
    relation types and "relational" attention is just attention; leave an
    input node off every path to the goal and the schema silently drops a
    feature. A schema that fails any of them is not a smaller ontology, it is
    a different method.
    """
    graph = empty_state_graph()
    assert len(STATE_NODE_NAMES) == 9
    assert len(STATE_RELATION_NAMES) == 4
    assert set(STATE_RELATION_NAMES) == {
        "degrades", "supports", "contributes", "self"}
    assert unreachable_state_input_nodes() == ()

    # Two stages: no risk node reaches the goal without passing through an
    # intermediate, EXCEPT the two the study kept as declared shortcuts.
    successors = {name: set() for name in STATE_NODE_NAMES}
    for source, target, _ in STATE_GRAPH_EDGES:
        successors[source].add(target)
    direct = {name for name in STATE_RISK_NODES
              if STATE_GOAL_NODE in successors[name]}
    assert direct == {"MeasurementAge", "RelativeRange"}
    assert "TouchdownSafety" in STATE_NODE_NAMES
    assert successors["TouchdownSafety"] == {STATE_GOAL_NODE}

    # Self-loops for every node, and the declared edges on top.
    self_relation = STATE_RELATION_NAMES.index("self")
    assert int((np.asarray(graph.rel) == self_relation).sum()) == len(STATE_NODE_NAMES)
    declared = tuple(
        (graph.node_names[s], graph.node_names[d], graph.relation_names[r])
        for s, d, r in zip(graph.src, graph.dst, graph.rel)
        if graph.relation_names[r] != "self")
    assert declared == STATE_GRAPH_EDGES


def test_the_goal_node_is_always_zero_so_the_answer_cannot_leak_into_the_state():
    for observation in (_observation(), _observation((0.8, 0.0), scale=0.4),
                        _observation(visible=0.0, loss=1.0)):
        graph = build_state_graph(observation)
        assert graph.node_names[graph.goal_node] == STATE_GOAL_NODE
        assert graph.X[0, graph.goal_node] == 0.0


def test_the_graph_takes_no_argument_through_which_truth_could_enter():
    """The boundary is the signature and the code, not a convention.

    Names are read out of the module's syntax tree rather than out of its
    text, so a prose mention of "the critic" in a docstring is not mistaken
    for the critic being read.
    """
    import ast

    parameters = tuple(inspect.signature(build_state_graph).parameters)
    assert parameters == ("observation", "geometry", "scales", "previous",
                          "committed", "dt")
    module = sys.modules[build_state_graph.__module__]
    tree = ast.parse(inspect.getsource(module))
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, (ast.Name, ast.arg)):
            used.add(node.id if isinstance(node, ast.Name) else node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            used.add(node.value)
    for token in ("true_relative_state", "platform_position", "pad_position",
                  "deck_velocity", "simulator_truth", "ground_truth",
                  "geometric_pad_center_in_fov", "critic", "privileged"):
        assert token not in used, token
    with pytest.raises(AttributeError):
        build_state_graph(SimpleNamespace(simulator_truth=np.zeros(6)))


def test_soft_saturation_keeps_resolution_at_both_ends_of_the_range():
    """A clipped normalisation cannot cover two orders of magnitude.

    The reduced study measured this: with ``min(1, x/scale)`` the range node
    saturated on three quarters of its samples, altitude disappeared from the
    state, and the policy climbed to the ceiling and never landed. Widening
    the constant collapsed the other end instead. The fix is a saturation that
    is never flat.
    """
    scales = StateGraphScales()
    index = STATE_NODE_NAMES.index("RelativeRange")
    # raw_scale goes as 1 / range, so these are far, mid and very close.
    far = state_node_values(_observation(scale=0.006), scales=scales)[index]
    mid = state_node_values(_observation(scale=0.06), scales=scales)[index]
    near = state_node_values(_observation(scale=0.6), scales=scales)[index]
    assert 0.0 < near < mid < far < 1.0
    # Strictly monotone and never saturated across the whole flight envelope,
    # so the gradient never vanishes. The sweep stays inside ``range_bounds_m``
    # -- outside it the value is deliberately clamped, which is a bound on the
    # range estimate rather than a saturation of the node.
    low, high = scales.range_bounds_m
    samples = [state_node_values(_observation(scale=s), scales=scales)[index]
               for s in np.geomspace(scales.reference_scale / (high * 0.98),
                                     scales.reference_scale / (low * 1.02), 40)]
    assert all(a > b for a, b in zip(samples, samples[1:]))
    assert max(samples) < 1.0 and min(samples) > 0.0


def test_the_direction_channel_separates_states_the_node_values_cannot():
    """Node values are magnitudes of risk; a policy needs the sign too.

    Without it a pad ahead and a pad behind produce an identical state, and no
    controller can be built on that. The channel adds no new physical quantity
    -- it is the sign of the quantity the node already carries.
    """
    ahead = build_state_graph(_observation((0.6, 0.0)))
    behind = build_state_graph(_observation((-0.6, 0.0)))
    # Same magnitudes...
    alignment = STATE_NODE_NAMES.index("AlignmentError")
    assert ahead.X[0, alignment] != pytest.approx(0.0)
    # ...opposite directions, on the row that carries them.
    assert ahead.X[4, alignment] == pytest.approx(-behind.X[4, alignment])
    assert ahead.X[4, alignment] != 0.0
    # And the whole feature matrix differs, which is the point.
    assert not np.allclose(ahead.X, behind.X)


def test_the_visibility_node_carries_the_branch_the_guidance_law_switches_on():
    """Tracking and searching command opposite vertical actions.

    The reduced study measured the reference law's mean vertical command below
    1.5 m as +0.02 while tracking and +2.0 while searching -- nearly the climb
    limit. A state that cannot tell the two apart cannot reproduce either, and
    the policy it trains sits in a limit cycle. This is the same information
    the baseline actor already has; it is not privileged.
    """
    visibility = STATE_NODE_NAMES.index("PadVisibility")
    tracking = build_state_graph(_observation())
    searching = build_state_graph(_observation(visible=0.0, loss=0.8))
    committed = build_state_graph(_observation(scale=0.5), committed=True)
    assert tracking.X[4, visibility] == pytest.approx(+1.0)
    assert searching.X[4, visibility] == pytest.approx(-1.0)
    assert committed.X[4, visibility] == pytest.approx(0.0)


def test_the_feature_matrix_is_deterministic_and_shaped_as_declared():
    observation = _observation((0.3, -0.1), scale=0.08)
    first = build_state_graph(observation)
    second = build_state_graph(observation)
    np.testing.assert_array_equal(first.X, second.X)
    assert first.X.shape == (STATE_GRAPH_INPUT_DIM, len(STATE_NODE_NAMES))
    assert STATE_GRAPH_INPUT_DIM == STATE_FEATURE_ROWS + len(STATE_NODE_NAMES)
    np.testing.assert_allclose(first.X[1], 1.0 - first.X[0], atol=1e-6)
    np.testing.assert_allclose(first.X[2, :len(STATE_RISK_NODES)], 1.0)
    np.testing.assert_allclose(first.X[2, len(STATE_RISK_NODES):], 0.0)
    np.testing.assert_allclose(first.X[3], 1.0)
    np.testing.assert_array_equal(
        first.X[STATE_FEATURE_ROWS:], np.eye(len(STATE_NODE_NAMES)))
    assert np.all((first.X[0] >= 0.0) & (first.X[0] <= 1.0))
    assert np.all(np.abs(first.X[4]) <= 1.0)


def test_the_camera_geometry_is_a_profile_constant_not_an_observation():
    """A different camera changes the bearing, and nothing else."""
    wide = StateGraphGeometry(nadir_column=-0.577, tan_half_horizontal=2.0)
    narrow = StateGraphGeometry(nadir_column=-0.577, tan_half_horizontal=0.5)
    alignment = STATE_NODE_NAMES.index("AlignmentError")
    observation = _observation((0.2, 0.0))
    assert (state_node_values(observation, geometry=wide)[alignment]
            > state_node_values(observation, geometry=narrow)[alignment])


# ------------------------------------------------------------- the encoder

def test_the_readout_reads_every_node_and_not_a_chosen_one():
    """A state has no reason to discard eight of nine node embeddings.

    The reward readout takes the goal node of a one-unit layer; that is the
    retired method. Here every node has to reach ``g_t``, which is checked by
    perturbing each node in turn.
    """
    encoder = GraphStateEncoder(hidden_dim=16, graph_dim=16, seed=3).eval()
    base = graph_feature_tensor(empty_state_graph())
    with torch.no_grad():
        reference = encoder(base)
    for node in range(len(STATE_NODE_NAMES)):
        perturbed = base.clone()
        perturbed[0, node, 0] += 0.7
        with torch.no_grad():
            moved = encoder(perturbed)
        assert not torch.allclose(reference, moved, atol=1e-7), (
            f"{STATE_NODE_NAMES[node]} does not reach the readout")


def test_the_encoder_gradient_matches_central_differences():
    torch.manual_seed(5)
    encoder = GraphStateEncoder(hidden_dim=8, graph_dim=8, seed=5).double()
    X = torch.as_tensor(
        build_state_graph(_observation((0.25, -0.1))).X.T.astype(np.float64)
    )[None]
    parameter = encoder.layer1.kernel.weight
    encoder(X).sum().backward()
    analytic = parameter.grad.clone()
    epsilon = 1e-6
    for index in ((0, 0, 0, 0), (2, 0, 3, 1), (1, 0, 7, 2)):
        with torch.no_grad():
            parameter[index] += epsilon
            high = float(encoder(X).sum())
            parameter[index] -= 2 * epsilon
            low = float(encoder(X).sum())
            parameter[index] += epsilon
        assert analytic[index].item() == pytest.approx(
            (high - low) / (2 * epsilon), abs=1e-5)


@pytest.mark.parametrize("representation", GRAPH_STATE_REPRESENTATIONS)
def test_every_representation_runs_through_the_same_encoder(representation):
    """The ablations are settings, not forks of the implementation."""
    topology = graph_state_topology(representation)
    encoder = GraphStateEncoder(representation=representation,
                                hidden_dim=8, graph_dim=8)
    assert encoder.topology.n_nodes == len(STATE_NODE_NAMES)
    output = encoder(graph_feature_tensor(empty_state_graph()))
    assert output.shape == (1, 8)
    assert torch.isfinite(output).all()
    if representation == "ontology_rgat":
        assert topology.n_relations == len(STATE_RELATION_NAMES)
        assert topology.n_edges == len(STATE_GRAPH_EDGES) + len(STATE_NODE_NAMES)
    elif representation == "gat":
        # Same edges, relation types merged: this ablates "relational".
        assert topology.n_relations == 1
        assert topology.n_edges == len(STATE_GRAPH_EDGES) + len(STATE_NODE_NAMES)
    else:
        # Self-loops only: this ablates message passing, not the node set.
        assert topology.n_edges == len(STATE_NODE_NAMES)


def test_the_relational_encoder_has_strictly_more_to_say_than_the_merged_one():
    relational = GraphStateEncoder(representation="ontology_rgat",
                                   hidden_dim=8, graph_dim=8)
    merged = GraphStateEncoder(representation="gat", hidden_dim=8, graph_dim=8)
    pooled = GraphStateEncoder(representation="node_pool",
                               hidden_dim=8, graph_dim=8)
    count = lambda model: sum(p.numel() for p in model.parameters())  # noqa: E731
    assert count(relational) > count(merged)
    assert count(merged) >= count(pooled)


def test_the_encoder_is_trained_rather_than_frozen():
    """There is no offline stage, no artifact and no checksum here.

    A configuration that tried to freeze it, or to load it from a file, would
    be the retired reward-side method wearing the new name, so both are
    refused rather than ignored.
    """
    encoder = GraphStateEncoder(hidden_dim=8, graph_dim=8)
    assert all(parameter.requires_grad for parameter in encoder.parameters())
    described = encoder.describe()
    assert described["frozen"] is False
    assert described["trained_by"] == "ppo"
    assert described["graph_version"] == STATE_GRAPH_VERSION

    config = load_experiment(default_experiment_config())
    for bad in ({"freeze_during_ppo": True},
                {"pretrained_artifact": "rgat/state_encoder.pt"},
                {"representation": "transformer"}):
        broken = {**config, "graph_state": {**config["graph_state"], **bad}}
        with pytest.raises(ValueError):
            validate_pipeline_configuration(broken)


# --------------------------------------------------------------- the arms

def test_the_single_experimental_factor_is_the_state_representation():
    assert_primary_baseline_equivalence()
    baseline, proposed = (PIPELINES[name] for name in
                          ("shin_se_fixed", "shin_se_onto_rgat_state"))
    differing = [field.name for field in PipelineSpec.__dataclass_fields__.values()
                 if field.name != "name"
                 and getattr(baseline, field.name) != getattr(proposed, field.name)]
    assert sorted(differing) == [
        "graph_state_enabled", "graph_state_representation",
        "ontology_enabled", "ontology_input_mode"]


def test_a_graph_state_arm_cannot_also_touch_the_reward():
    for name in graph_state_pipeline_ids():
        spec = ALL_PIPELINES[name]
        assert spec.reward_mode in {"shin_table_active", "shin_table_no_active"}
        assert not spec.fov_risk_reward_enabled
        assert not spec.use_direct_rgat_potential
        assert not spec.use_adaptive_reward_weights
    with pytest.raises(ValueError):
        PipelineSpec(
            name="illegal", state_estimation_enabled=True,
            auxiliary_estimation_loss_enabled=True, active_perception_enabled=True,
            reward_mode="shin_table_active", ontology_enabled=True,
            ontology_input_mode="state_situation_graph",
            use_direct_rgat_potential=True,
            graph_state_enabled=True, graph_state_representation="ontology_rgat")


def test_a_run_cannot_mix_the_state_method_with_the_retired_reward_method():
    """Two factors is not a controlled comparison."""
    config = load_experiment(default_experiment_config())
    mixed = {**config,
             "pipelines": ["shin_se_onto_rgat_state", "shin_se_onto_rgat_recovery"],
             "pipeline_contract": {
                 **config["pipeline_contract"],
                 "shin_se_onto_rgat_recovery": {
                     "state_estimation": True, "auxiliary_estimation_loss": True,
                     "active_perception": True, "reward_mode": "shin_table_active",
                     "ontology_enabled": True,
                     "ontology_input_mode": "fov_semantic_observation",
                     "use_direct_rgat_potential": False}},
             "fov_risk": {"lambda_fov": 0.1, "prediction_horizon_seconds": 1.0,
                          "visibility_criterion": "geometric_pad_center_in_fov"}}
    with pytest.raises(ValueError, match="two factors"):
        validate_pipeline_configuration(mixed)


def test_the_ablations_keep_their_own_ids_and_are_not_the_default():
    assert set(ABLATION_PIPELINES) == {
        "shin_se_onto_gat_state", "shin_se_node_pool_state"}
    assert not set(ABLATION_PIPELINES) & set(PIPELINES)
    assert "shin_se_onto_rgat_recovery" in LEGACY_PIPELINES


# ----------------------------------------------------- the policy and PPO

def _model(name, **overrides):
    settings = dict(image_embedding=16, lstm_hidden=12, latent_dim=16,
                    actor_hidden=8, critic_hidden=8, graph_hidden_dim=8,
                    graph_dim=8, pipeline=name)
    settings.update(overrides)
    return PipelineActorCritic(**settings)


def _inputs(model, steps=2):
    images = torch.rand(1, steps, 1, 32, 32)
    proprio = torch.zeros(1, steps, 7)
    proprio[..., 3] = 1.0
    truth = torch.zeros(1, steps, 6)
    graph = None
    if model.graph_state_enabled:
        graph = graph_feature_tensor(
            [build_state_graph(_observation((0.1 * index, 0.0)))
             for index in range(steps)])[None]
    return images, proprio, truth, graph


def test_the_actor_and_the_critic_each_hold_their_own_encoder():
    """They already have separate learning rates and Adam state.

    A shared encoder would have to merge two differently scaled gradients into
    one parameter. They see the same ``G_t`` and have the same architecture.
    """
    model = _model("shin_se_onto_rgat_state")
    assert model.policy_graph_encoder is not None
    assert model.value_graph_encoder is not None
    assert model.policy_graph_encoder is not model.value_graph_encoder
    # Same architecture, different initialisation.
    assert (model.policy_graph_encoder.describe()
            == model.value_graph_encoder.describe())
    assert not torch.equal(model.policy_graph_encoder.readout.weight,
                           model.value_graph_encoder.readout.weight)


def test_an_arm_is_handed_exactly_the_representation_it_declares():
    """Neither a missing graph nor an unexpected one may pass silently."""
    proposed = _model("shin_se_onto_rgat_state")
    images, proprio, truth, graph = _inputs(proposed)
    with pytest.raises(ValueError, match="requires the situation graph"):
        proposed(images, proprio, true_relative_state=truth)
    baseline = _model("shin_se_fixed")
    with pytest.raises(ValueError, match="must not be handed one"):
        baseline(images, proprio, true_relative_state=truth,
                 graph_features=graph)


def test_the_graph_changes_the_action_and_the_value():
    model = _model("shin_se_onto_rgat_state").eval()
    images, proprio, truth, graph = _inputs(model)
    with torch.no_grad():
        first = model(images, proprio, true_relative_state=truth,
                      graph_features=graph)
        second = model(images, proprio, true_relative_state=truth,
                       graph_features=graph * 0.0)
    assert not torch.allclose(first.action_mean, second.action_mean)
    assert not torch.allclose(first.value, second.value)
    assert first.actor_graph_embedding.shape == (1, 2, model.graph_dim)


def test_ppo_trains_the_encoder_end_to_end():
    """The gradient reaches the relational kernels, and the update moves them.

    Freezing the encoder, or precomputing ``g_t`` and detaching it, would be a
    different method: the graph would be a fixed feature rather than a learned
    representation.
    """
    torch.manual_seed(1)
    model = _model("shin_se_onto_rgat_state")
    images, proprio, truth, graph = _inputs(model)
    pre_squash = torch.randn(1, 2, 3) * 0.4
    batch = {
        "images": images, "proprioception": proprio,
        "true_relative_state": truth, "graph_features": graph,
        "pre_squash_action": pre_squash, "action": torch.tanh(pre_squash),
        "old_log_prob": torch.zeros(1, 2), "advantage": torch.randn(1, 2),
        "return": torch.randn(1, 2),
        "truth_valid": torch.ones(1, 2, dtype=torch.bool),
    }
    before = model.policy_graph_encoder.layer1.kernel.weight.detach().clone()
    loss, metrics = recurrent_ppo_loss(model, batch)
    loss.backward()
    assert metrics["graph_state_enabled"] == 1.0
    for encoder in (model.policy_graph_encoder, model.value_graph_encoder):
        total = sum(float(p.grad.abs().sum()) for p in encoder.parameters()
                    if p.grad is not None)
        assert total > 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    optimizer.step()
    assert not torch.equal(
        before, model.policy_graph_encoder.layer1.kernel.weight.detach())


def test_the_two_learned_arms_receive_an_identical_reward():
    """The claim the whole design rests on, checked on the dispatch."""
    previous = SimpleNamespace(
        critic=SimpleNamespace(true_relative_state=np.array([0.5, -0.2, -1.0,
                                                             0.1, 0.0, -0.1])),
        actor=SimpleNamespace(body_velocity=np.array([0.0, 0.0, -0.1])),
        state={})
    following = SimpleNamespace(
        critic=SimpleNamespace(true_relative_state=np.array([0.45, -0.18, -0.9,
                                                             0.08, 0.0, -0.08])),
        actor=SimpleNamespace(body_velocity=np.array([0.0, 0.0, -0.08])),
        state={}, command=np.array([0.1, 0.0, -0.1, 0.0, 0.02]),
        normalized_command=np.array([0.1, -0.1, 0.02]),
        physical_contact=False, strict_success=False, crash=False,
        excessive_drift=False, battery_depleted=False, terminal=False)
    estimate, next_estimate = np.zeros(6), np.ones(6) * 0.05

    baseline, base_parts, base_loss = _reward(
        "shin_se_fixed", previous, following, estimate, next_estimate, None)
    for name in ("shin_se_onto_rgat_state", "shin_se_onto_gat_state",
                 "shin_se_node_pool_state"):
        value, parts, loss = _reward(
            name, previous, following, estimate, next_estimate, None)
        assert value == pytest.approx(baseline), name
        assert parts == base_parts, name
        assert loss == pytest.approx(base_loss), name
    # No ontology key is even present, on either side.
    assert not {key for key in base_parts
                if "ontology" in key or "fov_unavailability" in key}


def test_the_update_replays_the_graph_the_rollout_actually_acted_on():
    """A stored transition without its ``G_t`` is not usable, and says so.

    The graph is a constructed input, not a differentiable function of the
    encoder, so the update has to replay the matrices the rollout stored. A
    silently substituted empty graph would score the policy on a state it
    never saw.
    """
    from ontology_rgat.ppo.recurrent_train import _chunk_graph_features

    model = _model("shin_se_onto_rgat_state")
    graph = build_state_graph(_observation((0.2, 0.0)))
    chunk = [{"state_graph_X": graph.X}, {"state_graph_X": graph.X}]
    replayed = _chunk_graph_features(model, chunk, torch.device("cpu"))
    assert replayed.shape == (1, 2, len(STATE_NODE_NAMES), STATE_GRAPH_INPUT_DIM)
    np.testing.assert_allclose(replayed[0, 0].numpy(), graph.X.T, atol=1e-6)
    with pytest.raises(ValueError, match="G_t on every stored transition"):
        _chunk_graph_features(model, [{"state_graph_X": None}],
                              torch.device("cpu"))
    # A baseline arm replays nothing, because it has nothing to replay.
    assert _chunk_graph_features(_model("shin_se_fixed"), chunk,
                                 torch.device("cpu")) is None


def test_a_warm_start_without_situation_graphs_is_refused_not_guessed():
    """Recomputing ``G_t`` after the fact would need a recurrence a file has not."""
    from ontology_rgat.ppo.behavior_cloning import (STATE_GRAPH_FIELD,
                                                    behavior_clone)

    model = _model("shin_se_onto_rgat_state")
    count = 8
    dataset = {
        "embedding": torch.randn(count, 16),
        "proprioception": torch.randn(count, 7) * 0.1,
        "action": torch.zeros(count, 3),
        "truth": torch.zeros(count, 6),
        "episode_id": torch.zeros(count, dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="situation graphs"):
        behavior_clone(model, dataset, epochs=1)
    graph = build_state_graph(_observation())
    dataset[STATE_GRAPH_FIELD] = torch.as_tensor(
        np.stack([graph.X.T.astype(np.float32)] * count))
    metrics = behavior_clone(model, dataset, epochs=1)
    assert math.isfinite(metrics["action_loss_after"])
