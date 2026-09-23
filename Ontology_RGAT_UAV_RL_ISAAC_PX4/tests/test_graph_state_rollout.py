"""One rollout, end to end, against a fake flight stack.

``collect_episode`` is the most intricate plumbing in the run: it builds the
situation graph from the frozen encoder's own output, hands it to the actor
*before* the rest of the network sees the frame, stores it on the transition,
publishes it to the operator views, and the PPO update replays it. A shape or
keyword error anywhere along that path surfaces hours into a flight otherwise.

The environment here is a fake, so nothing about the physics is being tested.
What is being tested is that the two arms go through the same code, that the
graph reaches and leaves the policy intact, and that their rewards agree.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaac_sim"))

from ontology_rgat.benchmarks.live_env import LiveStep                # noqa: E402
from ontology_rgat.benchmarks.shin2026 import (ActorObservation,      # noqa: E402
                                               CriticObservation)
from ontology_rgat.controllers import (PLANAR_ACTION_DIM,             # noqa: E402
                                       PlanarLongitudinalController)
from ontology_rgat.ppo.recurrent import PipelineActorCritic           # noqa: E402
from ontology_rgat.ppo.recurrent_train import (collect_episode,       # noqa: E402
                                               update_episode)
from ontology_rgat.rgat.state_graph import (STATE_GRAPH_INPUT_DIM,    # noqa: E402
                                            STATE_NODE_NAMES)


HORIZON = 6


def _config():
    return SimpleNamespace(
        sim=SimpleNamespace(dt=0.1, max_steps=HORIZON, world_xy_limit=60.0,
                            ground_z=0.08, crash_tilt=np.deg2rad(75.0)),
        criteria=SimpleNamespace(xy=0.35, vz=0.55, tilt=np.deg2rad(10.0),
                                 rate=np.deg2rad(45.0), rel_speed_xy=0.45),
        external=SimpleNamespace(
            landing_camera={"resolution": (512, 320), "horizontal_fov_deg": 90.0,
                            "pitch_down_deg": 60.0},
            reset_recoveries=0),
    )


class _FakeEnvironment:
    """Enough of ``LiveShinEnvironment`` for one deterministic episode.

    The vehicle closes on the deck a little each step and the pad drifts across
    the image, so the situation graph actually changes and a test that depended
    on it being constant would fail.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.controller = PlanarLongitudinalController(
            max_velocity=(1.6, 0.9), max_acceleration=(1.2, 0.8), dt=cfg.sim.dt,
            curriculum_min_action_scale=1.0)
        self.adapter = SimpleNamespace(controller=self.controller)
        self.bridge = SimpleNamespace(last_reset_ack={
            "detail": {"entry_yaw_enu_rad": 0.35,
                       "domain_randomization": {"brightness": 0.8}}})
        self.pad_motion_scale = 1.0
        self.steps = 0
        self.finished = False
        self.commands = []

    # -- observations -----------------------------------------------------
    def _step(self, command=None, terminal=False):
        rng = np.random.default_rng(1000 + self.steps)
        # A frame with a bright blob that walks across it, so the frozen
        # encoder's keypoints -- and therefore the graph -- move.
        # The encoder's certified frame size; ``grayscale_image_tensor``
        # refuses anything else, and using a smaller one here would test a
        # path the run never takes.
        image = np.zeros((320, 512), dtype=np.uint8)
        column = 60 + 40 * self.steps
        image[140:200, column:column + 70] = 220
        image += rng.integers(0, 12, image.shape, dtype=np.uint8)
        actor = ActorObservation(
            image=image,
            body_velocity=np.array([0.3, 0.0, -0.15 - 0.01 * self.steps]),
            attitude_quaternion=np.array([1.0, 0.0, 0.0, 0.0]))
        relative = np.array([1.2 - 0.15 * self.steps, 0.0,
                             -3.0 + 0.2 * self.steps, 0.1, 0.0, -0.15])
        critic = CriticObservation(actor=actor, true_relative_state=relative)
        planar = (np.zeros(5) if command is None
                  else command.as_planar_array())
        normalized = (np.zeros(PLANAR_ACTION_DIM) if command is None
                      else command.normalized_action)
        return LiveStep(
            actor=actor, critic=critic,
            state={"battery": {"enabled": False}, "extra": {},
                   "landed": False, "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                   "angular_velocity": [0.0, 0.0, 0.0],
                   "truth": {"valid": True, "position": relative[:3].tolist()}},
            command=planar, normalized_command=normalized,
            physical_contact=bool(terminal), unsafe_pad_contact=False,
            crash=False, excessive_drift=False, battery_depleted=False,
            terminal=bool(terminal), timeout=False,
            strict_success=bool(terminal),
            # Alternate so the FOV bookkeeping has both cases to count.
            geometric_pad_center_in_fov=bool(self.steps % 3),
            landing_metrics={
                "lateral_error": 0.2, "vertical_velocity": -0.2,
                "relative_horizontal_speed": 0.1, "tilt": 0.05,
                "roll": 0.01, "pitch": 0.02, "angular_rate": 0.1,
                "kinematic_sample": "current", "px4_attitude_failure": 0.0})

    # -- environment API --------------------------------------------------
    def reset(self, seed, curriculum=1.0, scenario="segmented_cruise_slow"):
        self.steps = 0
        self.controller.reset()
        self.controller.set_curriculum(float(curriculum))
        return self._step()

    def step(self, normalized_action):
        action = np.asarray(normalized_action, dtype=float).reshape(-1)
        assert action.shape == (PLANAR_ACTION_DIM,), action.shape
        command = self.controller.command(action)
        self.commands.append(command)
        self.steps += 1
        return self._step(command, terminal=self.steps >= HORIZON)

    def finish_episode(self):
        self.finished = True


def _model(name):
    torch.manual_seed(4)
    return PipelineActorCritic(
        image_embedding=16, lstm_hidden=12, latent_dim=16, actor_hidden=8,
        critic_hidden=8, graph_hidden_dim=8, graph_dim=8, pipeline=name).eval()


def _rollout(name):
    environment = _FakeEnvironment(_config())
    rows, metric = collect_episode(
        environment, _model(name), name, seed=7, deterministic=True,
        scenario="segmented_cruise_slow", phase="training")
    return environment, rows, metric


@pytest.mark.parametrize("name", ["shin_se_fixed", "shin_se_onto_rgat_state"])
def test_a_rollout_records_the_situation_graph_for_either_arm(name):
    """Both arms compute and store ``G_t``; only one of them consumes it.

    Publishing it for both is what lets the operator read the two sides of the
    comparison on identical channels instead of two different ontologies.
    """
    environment, rows, metric = _rollout(name)
    assert environment.finished
    assert len(rows) == HORIZON
    assert metric["steps"] == HORIZON
    for row in rows:
        graph = np.asarray(row["state_graph_X"])
        assert graph.shape == (STATE_GRAPH_INPUT_DIM, len(STATE_NODE_NAMES))
        assert np.isfinite(graph).all()
        assert np.all((graph[0] >= 0.0) & (graph[0] <= 1.0))
        # The goal node never carries a value.
        assert graph[0, -1] == 0.0
    # The graph is state-dependent, so it has to actually move over an episode
    # in which the pad moves.
    assert not np.allclose(rows[0]["state_graph_X"], rows[-1]["state_graph_X"])


@pytest.mark.parametrize("name", ["shin_se_fixed", "shin_se_onto_rgat_state"])
def test_a_rollout_stays_inside_the_planar_envelope(name):
    environment, rows, _ = _rollout(name)
    for row in rows:
        action = np.asarray(row["action"])
        command = np.asarray(row["command"])
        assert action.shape == (PLANAR_ACTION_DIM,)
        assert np.all(np.abs(action) <= 1.0)
        assert command.shape == (5,)
        # The two constrained degrees of freedom, on every recorded step.
        assert command[1] == 0.0
        assert command[3] == 0.0
    # And the tilt the controller actually held stays inside its own limit.
    limit = environment.controller.max_longitudinal_tilt
    assert all(abs(float(c.longitudinal_tilt_rad)) <= limit + 1e-9
               for c in environment.commands)


def test_the_two_arms_fly_the_same_episode_and_earn_the_same_reward_structure():
    """Same environment, same seed: the reward components must be the same kind.

    The two policies act differently, so the reward VALUES differ -- they are
    in different states. What must not differ is which components exist, which
    is what "the ontology does not touch the reward" means operationally.
    """
    _, baseline_rows, baseline_metric = _rollout("shin_se_fixed")
    _, proposed_rows, proposed_metric = _rollout("shin_se_onto_rgat_state")
    assert (set(baseline_rows[0]["reward_parts"])
            == set(proposed_rows[0]["reward_parts"]))
    for parts in (baseline_rows[0]["reward_parts"],
                  proposed_rows[0]["reward_parts"]):
        assert "ontology_fov_reward" not in parts
        assert "predicted_fov_unavailability" not in parts
        assert "attitude_penalty" in parts
    # The metrics both arms are compared on are produced for both.
    for metric in (baseline_metric, proposed_metric):
        for key in ("paper_success", "strict_success",
                    "geometric_fov_retention_ratio",
                    "relative_position_rmse_m", "episode_return"):
            assert key in metric, key
    assert baseline_metric["ontology_enabled"] is False
    assert proposed_metric["ontology_enabled"] is True


def test_a_zero_output_policy_earns_an_identical_reward_sequence_on_both_arms():
    """The strongest form of "the reward is untouched".

    With the actor's last layer zeroed both models emit the same action in the
    same state, so the two rollouts are the same trajectory. Any difference in
    the reward sequence would then have to come from the reward itself.
    """
    sequences = {}
    for name in ("shin_se_fixed", "shin_se_onto_rgat_state"):
        model = _model(name)
        with torch.no_grad():
            model.actor[-1].weight.zero_()
            model.actor[-1].bias.zero_()
            model.log_std.fill_(-8.0)
        rows, _ = collect_episode(
            _FakeEnvironment(_config()), model, name, seed=7,
            deterministic=True, scenario="segmented_cruise_slow")
        sequences[name] = [float(row["reward"]) for row in rows]
    np.testing.assert_allclose(sequences["shin_se_fixed"],
                               sequences["shin_se_onto_rgat_state"], atol=1e-12)


def test_the_recorded_rollout_can_be_replayed_by_the_ppo_update():
    """The rows a flight produces are exactly what the update consumes."""
    name = "shin_se_onto_rgat_state"
    environment, rows, _ = _rollout(name)
    model = _model(name).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.policy_graph_encoder.layer1.kernel.weight.detach().clone()
    metrics = update_episode(model, optimizer, rows, epochs=1,
                             sequence_length=3, target_kl=0.0)
    assert all(np.isfinite(value) for value in metrics.values())
    assert not torch.equal(
        before, model.policy_graph_encoder.layer1.kernel.weight.detach())


def test_the_operator_views_receive_the_graph_and_the_planar_command():
    """What the dashboard and the RViz HUD are handed, checked at the call."""
    published = []

    class _Monitor:
        def reset_started(self, **_kwargs):
            pass

        def reset_episode(self, **_kwargs):
            pass

        def step(self, **kwargs):
            published.append(kwargs)

    collect_episode(
        _FakeEnvironment(_config()), _model("shin_se_onto_rgat_state"),
        "shin_se_onto_rgat_state", seed=7, deterministic=True,
        scenario="segmented_cruise_slow", monitor=_Monitor())

    assert published
    sample = published[-1]
    values = np.asarray(sample["state_graph_values"], dtype=float)
    assert values.shape == (len(STATE_NODE_NAMES),)
    embedding = np.asarray(sample["graph_embedding"], dtype=float)
    assert embedding.shape == (8,)
    command = np.asarray(sample["planar_command"], dtype=float)
    assert command.shape == (5,) and command[1] == 0.0 and command[3] == 0.0
    assert np.asarray(sample["normalized_action"]).shape == (PLANAR_ACTION_DIM,)
    # The 3-D graph view shows the graph this arm's method actually uses.
    assert tuple(sample["semantic_graph"].node_names) == STATE_NODE_NAMES


def test_a_baseline_rollout_publishes_the_graph_but_no_embedding():
    """There is no encoder on the baseline, so there is no ``g_t`` to report."""
    published = []

    class _Monitor:
        def reset_started(self, **_kwargs):
            pass

        def reset_episode(self, **_kwargs):
            pass

        def step(self, **kwargs):
            published.append(kwargs)

    collect_episode(
        _FakeEnvironment(_config()), _model("shin_se_fixed"), "shin_se_fixed",
        seed=7, deterministic=True, scenario="segmented_cruise_slow",
        monitor=_Monitor())
    sample = published[-1]
    assert sample["graph_embedding"] is None
    assert np.asarray(sample["state_graph_values"]).shape == (
        len(STATE_NODE_NAMES),)
    assert tuple(sample["semantic_graph"].node_names) == STATE_NODE_NAMES
