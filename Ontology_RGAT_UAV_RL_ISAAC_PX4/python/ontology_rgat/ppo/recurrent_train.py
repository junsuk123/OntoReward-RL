"""Live recurrent PPO training/evaluation for the controlled benchmark."""
from __future__ import annotations

import os
import copy
import csv
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..bridge import BridgeError, EntryResetError, GatewayTimeout, PX4Failsafe
from ..curriculum import PlatformMotionCurriculum
from ..perception import (SEMANTIC_FEATURE_NAMES, grayscale_image_tensor,
                          semantic_graph, semantic_observation)
from ..pipelines import (available_pipeline_ids, get_pipeline,
                         primary_pipeline_ids)
from ..rgat.adaptive_model import adaptive_reward_graph
from ..reward_modes import (AdaptiveRewardConfig, AdaptiveWeightReward,
                            FixedBaselineRewardWeights,
                            OntoRewardPBRS, ShinReward, ShinRewardConfig,
                            NoSERewardContext, OntologyRewardContext,
                            ShinSERewardContext, TerminalFlags,
                            active_perception_reward, sparse_terminal_reward)
from ..reward_modes.adaptive_weight import shin_reward_components
from ..reward_modes.adaptive_weight import RewardComponentNormalizer
from .behavior_cloning import behavior_clone
from .recurrent import PipelineActorCritic, recurrent_ppo_loss


def _battery_sample(state):
    battery = state.get("battery") if isinstance(state.get("battery"), dict) else {}
    return battery if battery.get("enabled", False) else {}


def _battery_reserve(state) -> float:
    battery = _battery_sample(state)
    return float(np.clip(battery.get("reserve", 1.0), 0.0, 1.0))


def _tensor_observation(model, observation):
    image = grayscale_image_tensor(observation.image, device=model.device)
    proprio = torch.as_tensor(observation.proprioception[None],
                              dtype=torch.float32, device=model.device)
    return image, proprio


def _terminal_flags(following) -> TerminalFlags:
    raw_contact = bool(following.physical_contact)
    safe_contact = bool(getattr(following, "strict_success", raw_contact))
    return TerminalFlags(
        # The reward's historical ``physical_contact`` argument means a valid
        # landing outcome. Raw deck contact is insufficient: an off-centre or
        # tilted strike is a terminal failure.
        physical_contact=safe_contact,
        crash=bool(following.crash or (raw_contact and not safe_contact)),
        excessive_drift=following.excessive_drift,
        battery_depleted=following.battery_depleted,
        terminal=following.terminal)


def _transition_result_vertical_velocity(previous, following) -> float:
    """실제 전이 결과 속도. 최소 legacy test double만 이전 값을 fallback한다."""
    actor = getattr(following, "actor", None)
    if actor is None:  # 이전 API의 단위-test fixture 호환성
        actor = previous.actor
    return float(actor.body_velocity[2])


def _reward(method, previous, following, estimate, next_estimate, potential,
            *, gamma=0.99, shaping_lambda=1.0,
            current_semantic_graph=None, next_semantic_graph=None,
            current_adaptive_graph=None, next_adaptive_graph=None,
            reward_normalizer=None,
            estimation_loss_fn=None):
    """Dispatch reward through the selected pipeline's narrow data contract."""
    terminal = _terminal_flags(following)
    try:
        spec = get_pipeline(method)
    except ValueError:
        spec = None
    next_loss = None
    needs_estimation_loss = bool(
        spec is not None and spec.state_estimation_enabled)
    # Legacy modes retain their old behavior. The new no_se and onto_no_se
    # branches never calculate an estimation loss at all.
    if needs_estimation_loss or method in {
            "shin2026", "manual_no_active", "ontoreward",
            "ontoreward_plus_active", "sparse"}:
        if estimate is not None and next_estimate is not None:
            if estimation_loss_fn is None:
                next_loss = float(np.mean(
                    (next_estimate - following.critic.true_relative_state) ** 2))
            else:
                next_loss = float(estimation_loss_fn(
                    next_estimate, following.critic.true_relative_state))
        elif needs_estimation_loss:
            raise ValueError("shin_se reward requires its supervised estimate")
    if method == "sparse":
        value = sparse_terminal_reward(**terminal.as_kwargs())
        return value, {"task": value}, next_loss
    if (spec is not None and spec.name in {"shin_se_fixed", "no_se_fixed"}
            and reward_normalizer is not None):
        if current_adaptive_graph is None:
            raise ValueError("normalized fixed reward requires current adaptive graph")
        value, parts = AdaptiveWeightReward(
            FixedBaselineRewardWeights(reward_normalizer),
            config=AdaptiveRewardConfig(
                active_enabled=spec.active_perception_enabled),
            normalizer=reward_normalizer)(
                current_adaptive_graph,
                previous.critic.true_relative_state,
                following.critic.true_relative_state,
                following.command,
                    next_uav_vertical_velocity=_transition_result_vertical_velocity(
                        previous, following),
                next_estimation_loss=next_loss,
                **terminal.as_kwargs())
        return value, parts, next_loss
    if spec is not None and spec.reward_mode in {
            "shin_table_active", "shin_table_no_active"}:
        if spec.active_perception_enabled:
            context = ShinSERewardContext(
                current_training_relative_state=previous.critic.true_relative_state,
                next_training_relative_state=following.critic.true_relative_state,
                next_estimation_loss=float(next_loss), action=following.command,
                uav_vertical_velocity=_transition_result_vertical_velocity(
                    previous, following),
                terminal=terminal)
        else:
            context = NoSERewardContext(
                current_training_relative_state=previous.critic.true_relative_state,
                next_training_relative_state=following.critic.true_relative_state,
                action=following.command,
                uav_vertical_velocity=_transition_result_vertical_velocity(
                    previous, following),
                terminal=terminal)
        value, parts = ShinReward(ShinRewardConfig(
            active_enabled=spec.active_perception_enabled))(
                context.current_training_relative_state,
                context.next_training_relative_state, context.action,
                drone_vertical_velocity=context.uav_vertical_velocity,
                next_estimation_loss=(context.next_estimation_loss
                                      if isinstance(context, ShinSERewardContext)
                                      else None),
                **context.terminal.as_kwargs())
        return value, parts, next_loss
    if spec is not None and spec.reward_mode == "adaptive_weight":
        if current_adaptive_graph is None or next_adaptive_graph is None:
            raise ValueError("adaptive reward requires current and next semantic ontology graphs")
        if potential is None or not hasattr(potential, "normalizer"):
            raise ValueError("adaptive reward requires a frozen weight artifact")
        training_config = dict(getattr(potential, "metadata", {}).get(
            "training_config") or {})
        reward = AdaptiveWeightReward(
            potential,
            config=AdaptiveRewardConfig(
                active_enabled=spec.active_perception_enabled,
                gamma=float(gamma),
                semantic_potential_scale=float(training_config.get(
                    "semantic_potential_shaping_lambda", .75))),
            normalizer=potential.normalizer)
        value, parts = reward(
            current_adaptive_graph,
            previous.critic.true_relative_state,
            following.critic.true_relative_state,
            following.command,
            next_graph=next_adaptive_graph,
            next_uav_vertical_velocity=_transition_result_vertical_velocity(
                previous, following),
            next_estimation_loss=next_loss,
            **terminal.as_kwargs())
        return value, parts, next_loss
    if method in {"shin2026", "manual_no_active"}:
        # Table III uses the un-tilded physical relative geometry. Only the
        # active-perception term is coupled to the tilded estimator output.
        # Feeding estimates into both made a blind estimator manufacture its
        # own progress reward and removed the simulator's useful early signal.
        value, parts = ShinReward(ShinRewardConfig(
            active_enabled=method == "shin2026"))(
                previous.critic.true_relative_state,
                following.critic.true_relative_state, following.command,
                drone_vertical_velocity=_transition_result_vertical_velocity(
                    previous, following),
                next_estimation_loss=next_loss, **terminal.as_kwargs())
        return value, parts, next_loss
    if spec is not None and spec.reward_mode == "semantic_pbrs":
        if current_semantic_graph is None or next_semantic_graph is None:
            raise ValueError("onto_no_se reward requires semantic observation graphs")
        if potential is None:
            raise ValueError("onto_no_se requires a frozen direct R-GAT potential")
        context = OntologyRewardContext(
            graph=current_semantic_graph, next_graph=next_semantic_graph,
            terminal=terminal)
        pbrs = OntoRewardPBRS(
            potential, gamma=gamma, ppo_gamma=gamma,
            shaping_lambda=shaping_lambda, design_id=potential.design_id,
            frozen=True)
        value, parts = pbrs(
            context.graph, context.next_graph, **context.terminal.as_kwargs())
        return value, parts, None
    if potential is None:
        raise ValueError(f"{method} requires a frozen controlled R-GAT potential")
    pbrs = OntoRewardPBRS(
        potential, gamma=gamma, ppo_gamma=gamma,
        shaping_lambda=shaping_lambda, design_id=potential.design_id, frozen=True)
    value, parts = pbrs(
        {"estimated_relative_state": estimate,
         "battery_reserve": _battery_reserve(previous.state)},
        {"estimated_relative_state": next_estimate,
         "battery_reserve": _battery_reserve(following.state)},
        **terminal.as_kwargs())
    if method == "ontoreward_plus_active" and not following.terminal:
        active = active_perception_reward(next_loss, ShinRewardConfig())
        value += active
        parts["active_perception"] = active
    return value, parts, next_loss


def _semantic_from_output(output, proprioception, state, *, previous, dt):
    observation = semantic_observation(
        output.keypoints[0, -1].detach().cpu().numpy(),
        output.heatmaps[0, -1].detach().cpu().numpy(),
        np.asarray(proprioception, dtype=np.float32).reshape(-1),
        keypoint_visibility=output.keypoint_visibility[0, -1].detach().cpu().numpy(),
        battery_reserve=_battery_reserve(state), previous=previous, dt=dt)
    return observation, semantic_graph(observation)


def _landing_phase(observation) -> str:
    if observation.visual_loss_risk > 0.0 or observation.visible_keypoint_fraction < 0.5:
        return "recovery"
    if observation.apparent_target_scale < 0.25:
        return "approach"
    if observation.apparent_target_scale < 0.70:
        return "descent"
    return "touchdown"


def visual_recovery_metrics(rows, *, initial_in_fov: bool, success: bool,
                            dt: float) -> dict[str, float]:
    """Measure observability loss, action-level recovery and its outcome."""
    visibility = [bool(initial_in_fov)] + [bool(row["in_fov"]) for row in rows]
    losses = 0
    reacquisitions = 0
    loss_start = None
    completed_durations = []
    loss_transitions = []
    reacquisition_transitions = []
    for transition, (current, following) in enumerate(
            zip(visibility[:-1], visibility[1:])):
        if current and not following:
            losses += 1
            loss_start = transition
            loss_transitions.append(transition)
        elif not current and following:
            reacquisitions += 1
            reacquisition_transitions.append(transition)
            if loss_start is not None:
                completed_durations.append(max(1, transition - loss_start) * float(dt))
                loss_start = None

    feature_index = {name: index for index, name in enumerate(SEMANTIC_FEATURE_NAMES)}
    lost_commands = []
    low_visibility_commands = []
    for index, row in enumerate(rows):
        command = np.asarray(row.get("command", np.zeros(4)), dtype=float)
        if not visibility[index]:
            lost_commands.append(command)
        semantic = np.asarray(row.get("semantic_features", ()), dtype=float)
        if semantic.size == len(SEMANTIC_FEATURE_NAMES):
            low = (semantic[feature_index["visible_keypoint_fraction"]] < 0.5
                   or semantic[feature_index["keypoint_confidence"]] < 0.01)
            if low:
                low_visibility_commands.append(command)

    def fraction(commands, predicate):
        if not commands:
            return 0.0
        return float(np.mean([predicate(command) for command in commands]))

    output = {
        "visual_loss_events": float(losses),
        "visual_reacquisition_events": float(reacquisitions),
        "visual_reacquisition_rate": float(reacquisitions / max(losses, 1)),
        "mean_visual_reacquisition_time_s": (
            float(np.mean(completed_durations)) if completed_durations else 0.0),
        "recovery_climb_fraction": fraction(
            lost_commands, lambda command: command[2] > 0.05),
        "unsafe_descent_low_visibility_fraction": fraction(
            low_visibility_commands, lambda command: command[2] < -0.05),
        "recovery_landing_opportunity": float(losses > 0 and reacquisitions > 0),
        "successful_recovery_landing": float(
            losses > 0 and reacquisitions > 0 and bool(success)),
    }
    potential_loss = []
    potential_reacquisition = []
    for index in loss_transitions:
        parts = rows[index].get("reward_parts", {})
        if "phi" in parts and "phi_next" in parts:
            potential_loss.append(float(parts["phi_next"]) - float(parts["phi"]))
    for index in reacquisition_transitions:
        parts = rows[index].get("reward_parts", {})
        if "phi" in parts and "phi_next" in parts:
            potential_reacquisition.append(
                float(parts["phi_next"]) - float(parts["phi"]))
    if potential_loss or potential_reacquisition:
        output.update({
            "potential_delta_on_visual_loss_mean": (
                float(np.mean(potential_loss)) if potential_loss else 0.0),
            "potential_delta_on_reacquisition_mean": (
                float(np.mean(potential_reacquisition))
                if potential_reacquisition else 0.0),
        })
    return output


def collect_episode(env, model: PipelineActorCritic, method: str, seed: int,
                    *, curriculum=1.0, potential=None, deterministic=False,
                    gamma=0.99, shaping_lambda=1.0,
                    scenario="training_random_walk", monitor=None,
                    phase="evaluation", action_transform=None,
                    reward_normalizer=None):
    """Collect one real episode while keeping actor/reward contracts separate."""
    model_spec = model.pipeline_spec
    try:
        requested_spec = get_pipeline(method)
    except ValueError:
        requested_spec = None
    if (method in available_pipeline_ids() and requested_spec is not None
            and requested_spec.name != model_spec.name):
        raise ValueError(
            f"model pipeline {model_spec.name} cannot run reward pipeline {method}")
    step = env.reset(seed, curriculum, scenario=scenario)
    reset_detail = ((getattr(env.bridge, "last_reset_ack", {}) or {}).get(
        "detail") or {})
    domain_randomization = reset_detail.get("domain_randomization") or {}
    initial_battery = dict(_battery_sample(step.state))
    initial_in_fov = bool(step.pad_in_fov)
    action_scale = float(env.adapter.controller.action_scale)
    if monitor is not None:
        monitor.reset_episode(
            method=method, phase=phase, seed=seed, scenario=scenario,
            curriculum=curriculum, action_scale=action_scale,
            motion_scale=float(getattr(env, "pad_motion_scale", curriculum)))
    hidden = model.initial_state(1)
    rows = []
    visual_loss_run = 0
    longest_visual_loss = 0
    action_rng = np.random.default_rng(int(seed) + 9187)
    action_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 2718)
    with torch.no_grad():
        image, proprio = _tensor_observation(model, step.actor)
        truth = torch.as_tensor(step.critic.true_relative_state[None],
                                dtype=torch.float32, device=model.device)
        output = model(image, proprio, true_relative_state=truth, hidden=hidden,
                       episode_start=torch.tensor([True], device=model.device))
        semantic, graph = _semantic_from_output(
            output, step.actor.proprioception, step.state,
            previous=None, dt=float(env.cfg.sim.dt))
        adaptive_graph = adaptive_reward_graph(semantic)
        while True:
            mean = output.action_mean[:, -1]
            std = output.action_std[:, -1]
            noise = torch.randn(mean.shape, generator=action_generator).to(mean.device)
            pre_squash = mean if deterministic else mean + std * noise
            action = torch.tanh(pre_squash)
            if action_transform is not None:
                transformed = np.asarray(action_transform(
                    len(rows), action.cpu().numpy()[0], semantic, action_rng),
                    dtype=np.float32).reshape(-1)
                if transformed.shape != (4,) or not np.isfinite(transformed).all():
                    raise ValueError("estimator-free behavior transform returned invalid action")
                transformed = np.clip(transformed, -0.999999, 0.999999)
                action = torch.as_tensor(
                    transformed[None], dtype=mean.dtype, device=mean.device)
                pre_squash = torch.atanh(action)
            log_prob = model.log_prob(pre_squash, action, mean, std)
            following = env.step(action.cpu().numpy()[0])
            next_image, next_proprio = _tensor_observation(model, following.actor)
            next_truth = torch.as_tensor(following.critic.true_relative_state[None],
                                         dtype=torch.float32, device=model.device)
            next_output = model(next_image, next_proprio,
                                true_relative_state=next_truth,
                                hidden=output.hidden)
            next_semantic, next_graph = _semantic_from_output(
                next_output, following.actor.proprioception, following.state,
                previous=semantic, dt=float(env.cfg.sim.dt))
            next_adaptive_graph = adaptive_reward_graph(next_semantic)
            estimate = (None if output.relative_state is None else
                        output.relative_state[0, -1].cpu().numpy())
            next_estimate = (None if next_output.relative_state is None else
                             next_output.relative_state[0, -1].cpu().numpy())
            reward, parts, estimation_loss = _reward(
                method, step, following, estimate, next_estimate, potential,
                gamma=gamma, shaping_lambda=shaping_lambda,
                current_semantic_graph=graph, next_semantic_graph=next_graph,
                current_adaptive_graph=adaptive_graph,
                next_adaptive_graph=next_adaptive_graph,
                reward_normalizer=reward_normalizer,
                estimation_loss_fn=(
                    None if model.relative_state_head is None else
                    model.relative_state_head.numpy_loss))
            visual_loss_run = visual_loss_run + 1 if not following.pad_in_fov else 0
            longest_visual_loss = max(longest_visual_loss, visual_loss_run)
            row = {
                "image": np.asarray(step.actor.image, dtype=np.uint8),
                "proprioception": step.actor.proprioception.copy(),
                "truth": step.critic.true_relative_state.copy(),
                "next_truth": following.critic.true_relative_state.copy(),
                "pre_squash": pre_squash.cpu().numpy()[0],
                "action": action.cpu().numpy()[0],
                "command": following.command.copy(),
                "log_prob": float(log_prob.item()),
                "value": float(output.value.item()), "reward": float(reward),
                "done": float(following.terminal), "in_fov": following.pad_in_fov,
                "battery_reserve": _battery_reserve(step.state),
                "battery_energy_j": float(_battery_sample(step.state).get(
                    "remaining_j", 0.0)),
                "reward_parts": parts,
                "semantic_graph_X": graph.X.copy(),
                "adaptive_graph_X": adaptive_graph.X.copy(),
                "rho_raw": shin_reward_components(
                    step.critic.true_relative_state,
                    following.critic.true_relative_state,
                    following.command,
                    next_uav_vertical_velocity=float(
                        following.actor.body_velocity[2])),
                "next_uav_vertical_velocity": float(
                    following.actor.body_velocity[2]),
                "semantic_features": semantic.feature_vector.copy(),
                "next_semantic_features": next_semantic.feature_vector.copy(),
                "phase": str(phase), "scenario": str(scenario),
                "landing_phase": _landing_phase(semantic),
                "hidden_h": hidden[0].cpu().numpy(),
                "hidden_c": hidden[1].cpu().numpy(),
            }
            if estimate is not None:
                row["estimate"] = estimate
                row["estimation_loss"] = float(estimation_loss)
            rows.append(row)
            if monitor is not None:
                status = ("success" if following.strict_success
                          else "unsafe_pad_contact" if following.unsafe_pad_contact
                          else "battery_depleted" if following.battery_depleted
                          else "failure" if following.terminal else "running")
                monitor.step(
                    index=len(rows), dt=env.cfg.sim.dt, method=method,
                    reward=reward, reward_parts=parts, estimate=next_estimate,
                    truth=following.critic.true_relative_state,
                    in_fov=following.pad_in_fov,
                    estimation_loss=estimation_loss, state=following.state,
                    pipeline_spec=model_spec,
                    semantic_features=next_semantic.feature_vector,
                    semantic_graph=next_graph,
                    scenario=scenario, status=status)
            hidden = output.hidden
            step, output = following, next_output
            semantic, graph = next_semantic, next_graph
            adaptive_graph = next_adaptive_graph
            if following.terminal:
                break
    env.finish_episode()
    truth_final = step.critic.true_relative_state
    state = step.state
    final_battery = _battery_sample(state)
    landing = step.landing_metrics
    lateral_error = float(landing["lateral_error"])
    vertical_velocity = float(landing["vertical_velocity"])
    vertical_speed = abs(vertical_velocity)
    relative_horizontal_speed = float(landing["relative_horizontal_speed"])
    tilt = float(landing["tilt"])
    angular_rate = float(landing["angular_rate"])
    metric = {
        "seed": int(seed), "episode_return": float(sum(row["reward"] for row in rows)),
        # Retain the established report column name, but define it as the
        # complete safe-landing gate instead of raw contact.
        "paper_success": float(step.strict_success),
        "strict_success": float(step.strict_success),
        "pad_contact": float(step.physical_contact),
        "unsafe_pad_contact": float(step.unsafe_pad_contact),
        "crash_failure": float(step.crash),
        "collision_rate": float(step.crash),
        "excessive_drift_rate": float(step.excessive_drift),
        "failure": float(not step.strict_success),
        "touchdown_lateral_error": lateral_error,
        "touchdown_vertical_velocity": vertical_velocity,
        "touchdown_relative_horizontal_velocity": relative_horizontal_speed,
        "touchdown_tilt": tilt,
        "touchdown_roll": float(landing["roll"]),
        "touchdown_pitch": float(landing["pitch"]),
        "touchdown_angular_rate": angular_rate,
        "touchdown_kinematic_sample": str(landing["kinematic_sample"]),
        "landing_gate_contact": float(step.physical_contact),
        "landing_gate_position": float(
            lateral_error <= float(env.cfg.criteria.xy)),
        "landing_gate_vertical_speed": float(
            vertical_speed <= float(env.cfg.criteria.vz)),
        "landing_gate_relative_horizontal_speed": float(
            relative_horizontal_speed <= float(env.cfg.criteria.rel_speed_xy)),
        "landing_gate_attitude": float(
            tilt <= float(env.cfg.criteria.tilt)),
        "landing_gate_angular_rate": float(
            angular_rate <= float(env.cfg.criteria.rate)),
        "fov_loss_fraction": float(np.mean([not row["in_fov"] for row in rows])),
        # Reward-independent physical tracking error, available to every arm
        # and therefore safe to use for a common performance curriculum.
        "relative_position_rmse_m": float(np.sqrt(np.mean(np.asarray([
            row["truth"][:3] for row in rows], dtype=float) ** 2))),
        "longest_visual_loss_s": float(longest_visual_loss * env.cfg.sim.dt),
        "action_envelope_scale": action_scale,
        "touchdown_time_s": float(len(rows) * env.cfg.sim.dt),
        "battery_reserve_initial": float(initial_battery.get("reserve", 1.0)),
        "battery_reserve_final": _battery_reserve(state),
        "battery_energy_initial_j": float(initial_battery.get("remaining_j", 0.0)),
        "battery_energy_final_j": float(final_battery.get("remaining_j", 0.0)),
        "battery_energy_used_j": float(final_battery.get("energy_used_j", 0.0)),
        "battery_depleted": float(step.battery_depleted),
        "steps": len(rows), "status": ("success" if step.strict_success else
                                         "unsafe_pad_contact" if step.unsafe_pad_contact else
                                         "battery_depleted" if step.battery_depleted else
                                         "collision" if step.crash else
                                         "excessive_drift" if step.excessive_drift else
                                         "timeout" if step.timeout else "failure"),
        "pipeline": (model_spec.name if method in available_pipeline_ids()
                     else str(method)),
        "state_estimation_enabled": bool(model_spec.state_estimation_enabled),
        "active_perception_enabled": bool(
            model_spec.active_perception_enabled
            if method in available_pipeline_ids()
            else method in {"shin2026", "ontoreward_plus_active"}),
        "ontology_enabled": bool(
            model_spec.ontology_enabled
            if method in available_pipeline_ids()
            else method.startswith("ontoreward")),
        "reward_source": (model_spec.reward_mode
                          if method in available_pipeline_ids() else str(method)),
        "rgat_design_id": (getattr(potential, "design_id", "")
                           if (model_spec.ontology_enabled
                               or method.startswith("ontoreward")) else ""),
        "domain_randomization_enabled": float(bool(domain_randomization)),
        "domain_external_force_n": float(np.linalg.norm(
            domain_randomization.get("external_force_n", (0.0, 0.0, 0.0)))),
        "domain_external_torque_nm": float(np.linalg.norm(
            domain_randomization.get("external_torque_nm", (0.0, 0.0, 0.0)))),
        "domain_ground_texture_id": int(
            domain_randomization.get("ground_texture_id", 0)),
        "domain_brightness": float(domain_randomization.get("brightness", 1.0)),
        "selected_checkpoint_episode": int(getattr(
            model, "_selected_checkpoint_episode", 0)),
        "selected_checkpoint_score": float(getattr(
            model, "_selected_checkpoint_score", 0.0)),
    }
    metric.update(visual_recovery_metrics(
        rows, initial_in_fov=initial_in_fov,
        success=bool(step.strict_success), dt=float(env.cfg.sim.dt)))
    potential_loss_delta = []
    potential_reacquisition_delta = []
    visibility = [initial_in_fov] + [bool(row["in_fov"]) for row in rows]
    for index, row in enumerate(rows):
        parts = row.get("reward_parts", {})
        if "phi" not in parts or "phi_next" not in parts:
            continue
        delta = float(parts["phi_next"]) - float(parts["phi"])
        if visibility[index] and not visibility[index + 1]:
            potential_loss_delta.append(delta)
        elif not visibility[index] and visibility[index + 1]:
            potential_reacquisition_delta.append(delta)
    if potential_loss_delta or potential_reacquisition_delta:
        metric.update({
            "potential_delta_on_visual_loss_mean": (
                float(np.mean(potential_loss_delta))
                if potential_loss_delta else 0.0),
            "potential_delta_on_reacquisition_mean": (
                float(np.mean(potential_reacquisition_delta))
                if potential_reacquisition_delta else 0.0),
        })
    metric.update(visual_recovery_metrics(
        rows, initial_in_fov=initial_in_fov,
        success=bool(step.strict_success), dt=float(env.cfg.sim.dt)))
    if model_spec.state_estimation_enabled:
        position_error = np.asarray([
            row["estimate"][:3] - row["truth"][:3] for row in rows])
        velocity_error = np.asarray([
            row["estimate"][3:] - row["truth"][3:] for row in rows])
        estimation_losses = np.asarray([
            row["estimation_loss"] for row in rows], dtype=float)
        active_values = np.asarray([
            float(row["reward_parts"].get("active_perception", 0.0))
            for row in rows], dtype=float)
        active_limit = ShinRewardConfig().active_alpha
        lost_errors = [row["estimation_loss"]
                       for row in rows if not row["in_fov"]]
        metric.update({
            "position_rmse": float(np.sqrt(np.mean(position_error ** 2))),
            "velocity_rmse": float(np.sqrt(np.mean(velocity_error ** 2))),
            "normalized_estimation_loss_mean": float(np.mean(estimation_losses)),
            "active_reward_saturation_fraction": float(np.mean(
                active_values <= (-active_limit + 1e-8))),
            "active_reward_standard_deviation": float(np.std(active_values)),
            "visual_loss_estimation_error": (
                float(np.mean(lost_errors)) if lost_errors else 0.0),
        })
    if rows and "weight_1" in rows[0].get("reward_parts", {}):
        for index in range(1, 6):
            metric[f"adaptive_weight_{index}_mean"] = float(np.mean([
                row["reward_parts"].get(f"weight_{index}", 0.0) for row in rows]))
            metric[f"adaptive_component_{index}_cumulative"] = float(np.sum([
                row["reward_parts"].get(f"weighted_{index}", 0.0) for row in rows]))
        metric["adaptive_rgat_inference_latency_ms_mean"] = float(np.mean([
            row["reward_parts"].get("rgat_inference_latency_ms", 0.0)
            for row in rows]))
        metric["adaptive_rgat_parameter_count"] = int(
            getattr(potential, "parameter_count", 0))
    return rows, metric


def collect_episode_resilient(env, model: PipelineActorCritic, method: str,
                              seed: int, **kwargs):
    """Retry one seed after a recoverable SITL infrastructure interruption.

    The failed partial trajectory is discarded, so infrastructure downtime is
    neither labelled as a task failure nor used in a gradient update. The live
    environment first uses its bounded local reset attempts; an exhausted
    :class:`EntryResetError` gets the same fresh-stack retry here so a single
    bad PX4 boot cannot terminate a checkpointed multi-arm experiment.
    """
    external = getattr(env.cfg, "external", {})
    if hasattr(external, "get"):
        recoveries = int(external.get(
            "episode_recoveries", external.get("reset_recoveries", 0)))
    else:
        recoveries = int(getattr(
            external, "episode_recoveries",
            getattr(external, "reset_recoveries", 0)))
    attempt = 0
    while attempt <= recoveries:
        try:
            return collect_episode(env, model, method, seed, **kwargs)
        except BridgeError as exc:
            recoverable = (isinstance(exc, (EntryResetError, GatewayTimeout))
                           or (isinstance(exc, PX4Failsafe) and exc.recoverable)
                           or "simulator has stalled" in str(exc).lower())
            recover = getattr(env, "recover_infrastructure", None)
            if not recoverable or attempt >= recoveries or not callable(recover):
                raise
            next_attempt = attempt + 1
            print(
                f"WARNING: [{method}] episode infrastructure failed ({exc}). Discarding "
                f"the partial trajectory, restarting the owned stack, and "
                f"retrying seed {int(seed)} ({next_attempt} of {recoveries}).")
            restarted = recover()
            # A peer may already have rebuilt the one shared Isaac world while
            # this worker was in a UDP transaction.  Its interrupted trajectory
            # is still discarded, but that collateral disconnect must not use
            # up this method's own fault budget.
            if restarted is not False:
                attempt = next_attempt
    raise AssertionError("unreachable infrastructure-recovery state")


def _gae(rows, gamma, gae_lambda):
    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float32)
    values = np.asarray([row["value"] for row in rows], dtype=np.float32)
    dones = np.asarray([row["done"] for row in rows], dtype=np.float32)
    advantage = np.zeros_like(rewards)
    running = 0.0
    for index in range(len(rows) - 1, -1, -1):
        next_value = 0.0 if index == len(rows) - 1 else values[index + 1]
        delta = rewards[index] + gamma * (1.0 - dones[index]) * next_value - values[index]
        running = delta + gamma * gae_lambda * (1.0 - dones[index]) * running
        advantage[index] = running
    return advantage, advantage + values


def update_estimator_episode(model, optimizer, rows, *, epochs=2,
                             grad_clip=5.0, sequence_length=32):
    """Supervise vision/LSTM on actual hover frames before changing the actor."""
    if not model.pipeline_spec.auxiliary_estimation_loss_enabled:
        raise ValueError("estimator-only update is illegal for an estimator-free pipeline")
    device = model.device
    losses = []
    estimator_parameters = [parameter for parameter in [
        *model.encoder.parameters(), *model.temporal_backbone.parameters(),
        *model.relative_state_head.parameters()] if parameter.requires_grad]
    for _ in range(int(epochs)):
        for start in range(0, len(rows), int(sequence_length)):
            chunk = rows[start:min(start + int(sequence_length), len(rows))]
            images = np.stack([row["image"] for row in chunk])[:, None]
            initial_hidden = (
                torch.as_tensor(chunk[0]["hidden_h"], dtype=torch.float32, device=device),
                torch.as_tensor(chunk[0]["hidden_c"], dtype=torch.float32, device=device),
            )
            output = model(
                torch.as_tensor(images[None], dtype=torch.float32, device=device) / 255.0,
                torch.as_tensor(
                    np.stack([row["proprioception"] for row in chunk])[None],
                    dtype=torch.float32, device=device),
                hidden=initial_hidden,
                episode_start=torch.zeros(
                    (1, len(chunk)), dtype=torch.bool, device=device))
            truth = torch.as_tensor(
                np.stack([row["truth"] for row in chunk])[None],
                dtype=torch.float32, device=device)
            loss = model.relative_state_head.loss(output.relative_state, truth)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(estimator_parameters, float(grad_clip))
            optimizer.step()
            losses.append(float(loss.detach()))
    entropy = float(torch.log(
        model.log_std.detach().exp()
        * math.sqrt(2.0 * math.pi * math.e)).sum())
    auxiliary = float(np.mean(losses))
    return {
        "loss": auxiliary, "ppo_loss": 0.0, "value_loss": 0.0,
        "entropy": entropy, "kl_divergence": 0.0,
        "auxiliary_estimation_loss": auxiliary,
        "ppo_early_stop": 0.0, "ppo_epochs_completed": 0.0,
        "effective_learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def update_episode(model, optimizer, rows, *, gamma=.99, gae_lambda=.95,
                   epochs=5, clip=.2, value_coef=.5, entropy_coef=.003,
                   auxiliary_coef=1.0, grad_clip=5.0, sequence_length=32,
                   target_kl=.03, minimum_learning_rate=5e-6,
                   rollback_on_excessive_kl=True,
                   log_std_bounds=(-3.0, -0.8)):
    advantage, returns = _gae(rows, gamma, gae_lambda)
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    device = model.device
    metrics = []
    early_stop = False
    rollback_count = 0
    epochs_completed = 0
    target_kl = float(target_kl)
    log_std_low, log_std_high = (float(value) for value in log_std_bounds)
    if log_std_low > log_std_high:
        raise ValueError("log_std_bounds must be ordered")
    for _ in range(int(epochs)):
        epoch_model = (copy.deepcopy(model.state_dict())
                       if rollback_on_excessive_kl else None)
        epoch_optimizer = (copy.deepcopy(optimizer.state_dict())
                           if rollback_on_excessive_kl else None)
        rollback_epoch = False
        for start in range(0, len(rows), int(sequence_length)):
            stop = min(start + int(sequence_length), len(rows))
            chunk = rows[start:stop]
            images = np.stack([row["image"] for row in chunk])[:, None]
            initial_hidden = (
                torch.as_tensor(chunk[0]["hidden_h"], dtype=torch.float32, device=device),
                torch.as_tensor(chunk[0]["hidden_c"], dtype=torch.float32, device=device),
            )
            batch = {
                "images": torch.as_tensor(images[None], dtype=torch.float32,
                                          device=device) / 255.0,
                "proprioception": torch.as_tensor(
                    np.stack([row["proprioception"] for row in chunk])[None],
                    dtype=torch.float32, device=device),
                "true_relative_state": torch.as_tensor(
                    np.stack([row["truth"] for row in chunk])[None],
                    dtype=torch.float32, device=device),
                "episode_start": torch.zeros((1, len(chunk)), dtype=torch.bool,
                                             device=device),
                "initial_hidden": initial_hidden,
                "pre_squash_action": torch.as_tensor(
                    np.stack([row["pre_squash"] for row in chunk])[None],
                    dtype=torch.float32, device=device),
                "action": torch.as_tensor(
                    np.stack([row["action"] for row in chunk])[None],
                    dtype=torch.float32, device=device),
                "old_log_prob": torch.as_tensor(
                    [[row["log_prob"] for row in chunk]],
                    dtype=torch.float32, device=device),
                "advantage": torch.as_tensor(advantage[None, start:stop],
                                             dtype=torch.float32, device=device),
                "return": torch.as_tensor(returns[None, start:stop],
                                          dtype=torch.float32, device=device),
                "truth_valid": torch.ones((1, len(chunk)), dtype=torch.bool,
                                          device=device),
            }
            loss, values = recurrent_ppo_loss(
                model, batch, clip=clip, value_coef=value_coef,
                entropy_coef=entropy_coef, auxiliary_coef=auxiliary_coef)
            if target_kl > 0.0 and float(values["kl_divergence"]) > target_kl:
                metrics.append({key: float(value) for key, value in values.items()})
                early_stop = True
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            with torch.no_grad():
                model.log_std.clamp_(log_std_low, log_std_high)
            # Measure the update rather than the pre-update batch. This catches
            # a representation shift caused by the auxiliary estimator before
            # another recurrent chunk can amplify it.
            with torch.no_grad():
                _, values = recurrent_ppo_loss(
                    model, batch, clip=clip, value_coef=value_coef,
                    entropy_coef=entropy_coef, auxiliary_coef=auxiliary_coef)
            metrics.append({key: float(value) for key, value in values.items()})
            if target_kl > 0.0 and float(values["kl_divergence"]) > target_kl:
                early_stop = True
                rollback_epoch = bool(rollback_on_excessive_kl)
                current_lr = float(optimizer.param_groups[0]["lr"])
                if rollback_epoch:
                    model.load_state_dict(epoch_model)
                    optimizer.load_state_dict(epoch_optimizer)
                    rollback_count += 1
                reduced_lr = max(float(minimum_learning_rate), 0.5 * current_lr)
                for group in optimizer.param_groups:
                    group["lr"] = reduced_lr
                break
        if early_stop:
            break
        epochs_completed += 1
    summary = {
        key: float(np.mean([row[key] for row in metrics]))
        for key in metrics[0]
    }
    summary.update({
        "ppo_early_stop": float(early_stop),
        "ppo_kl_rollback_count": float(rollback_count),
        "ppo_epochs_completed": float(epochs_completed),
        "effective_learning_rate": float(optimizer.param_groups[0]["lr"]),
    })
    return summary


def no_landing_abort_episode(ppo, planned_policy_episodes=None) -> int:
    """Return the policy episode at which persistent zero success is fatal.

    Sparse touchdown success can legitimately lag dense perception/control
    improvements in a short live run. Keep the original grace as the earliest
    possible stop, but give a bounded fraction of the requested PPO budget to
    obtain a first landing. The cap still prevents a large publication run
    from wasting thousands of live episodes with a broken policy.
    """
    window = max(1, int(ppo.get("health_window_episodes", 20)))
    grace = max(window, int(ppo.get("health_grace_episodes", 40)))
    if planned_policy_episodes is None:
        return grace
    planned = max(0, int(planned_policy_episodes))
    fraction = float(ppo.get("health_no_landing_budget_fraction", 0.75))
    fraction = float(np.clip(fraction, 0.0, 1.0))
    maximum = max(grace, int(ppo.get(
        "health_no_landing_max_grace_episodes", 120)))
    budget_grace = int(math.ceil(planned * fraction))
    return max(grace, min(maximum, budget_grace))


def training_health_issue(history, ppo, *, warmup_episodes=0,
                          planned_policy_episodes=None) -> str | None:
    """Return why an unattended run is not learning, after a fair window."""
    window = max(1, int(ppo.get("health_window_episodes", 20)))
    policy_rows = [row for row in history
                   if int(float(row.get("episode", 0))) > int(warmup_episodes)]
    grace = max(window, int(ppo.get("health_grace_episodes", 40)))
    no_landing_grace = no_landing_abort_episode(
        ppo, planned_policy_episodes=planned_policy_episodes)
    if len(policy_rows) < window:
        return None
    recent = policy_rows[-window:]
    successes = sum(float(row.get("paper_success", 0.0)) for row in recent)
    fov_loss = float(np.mean([
        float(row.get("fov_loss_fraction", 1.0)) for row in recent]))
    limit = float(ppo.get("health_max_fov_loss_fraction", 0.80))
    issues = []
    battery_fraction = float(np.mean([
        float(row.get("battery_depleted", 0.0)) for row in recent]))
    battery_limit = float(ppo.get(
        "health_max_battery_depletion_fraction", 0.60))
    if battery_fraction > battery_limit:
        issues.append(
            f"battery depletion is {battery_fraction:.1%} (limit {battery_limit:.1%})")
    if len(policy_rows) < grace:
        return "; ".join(issues) or None
    if successes == 0.0 and len(policy_rows) >= no_landing_grace:
        issues.append(f"no landing in the last {window} policy episodes")
    if fov_loss > limit:
        issues.append(f"mean FOV loss is {fov_loss:.1%} (limit {limit:.1%})")
    loss_events = sum(float(row.get("visual_loss_events", 0.0)) for row in recent)
    reacquisitions = sum(float(row.get(
        "visual_reacquisition_events", 0.0)) for row in recent)
    minimum_events = int(ppo.get("health_min_visual_loss_events", 5))
    if loss_events >= minimum_events:
        reacquisition_rate = reacquisitions / max(loss_events, 1.0)
        minimum_reacquisition = float(ppo.get(
            "health_min_reacquisition_rate", 0.25))
        if reacquisition_rate < minimum_reacquisition:
            issues.append(
                f"visual reacquisition is {reacquisition_rate:.1%} "
                f"(minimum {minimum_reacquisition:.1%})")
        unsafe_descent = float(np.mean([float(row.get(
            "unsafe_descent_low_visibility_fraction", 0.0))
            for row in recent]))
        unsafe_limit = float(ppo.get(
            "health_max_unsafe_descent_low_visibility_fraction", 0.50))
        if unsafe_descent > unsafe_limit:
            issues.append(
                f"unsafe low-visibility descent is {unsafe_descent:.1%} "
                f"(limit {unsafe_limit:.1%})")

    def stalled_high(field, limit_key, default):
        values = [float(row[field]) for row in policy_rows
                  if row.get(field) not in (None, "")]
        if len(values) < 2 * window:
            return None
        previous, current = values[-2 * window:-window], values[-window:]
        before, after = float(np.mean(previous)), float(np.mean(current))
        limit_value = float(ppo.get(limit_key, default))
        improvement = float(ppo.get("health_min_relative_improvement", .05))
        if after > limit_value and after >= before * (1.0 - improvement):
            return before, after, limit_value
        return None

    rmse = stalled_high("position_rmse", "health_max_position_rmse_m", 3.0)
    if rmse is not None:
        issues.append(
            f"position RMSE stalled at {rmse[1]:.2f} m "
            f"(previous {rmse[0]:.2f}, limit {rmse[2]:.2f})")
    saturation = stalled_high(
        "active_reward_saturation_fraction",
        "health_max_active_reward_saturation_fraction", .80)
    if saturation is not None:
        issues.append(
            f"active reward saturation stalled at {saturation[1]:.1%} "
            f"(previous {saturation[0]:.1%}, limit {saturation[2]:.1%})")
    return "; ".join(issues) or None


def save_recurrent_checkpoint(path, model, optimizer, *, method, episode,
                              config_hash, curriculum, potential=None,
                              selection_score=None, selection_metric=None,
                              model_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_spec = getattr(model, "pipeline_spec", None)
    payload = {
        "format": "three-pipeline-recurrent-v3-scaled-estimator", "method": method,
        "pipeline_spec": (model_spec.to_manifest() if model_spec is not None else None),
        "episode": int(episode), "config_hash": config_hash,
        "model": (model.state_dict() if model_state is None else model_state),
        "optimizer": optimizer.state_dict(),
        "curriculum": curriculum.state_dict(),
        "reward_design_id": getattr(potential, "design_id", None),
        "reward_design_sha256": getattr(potential, "sha256", None),
        "selection_score": (None if selection_score is None else
                            float(selection_score)),
        "selection_metric": selection_metric,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def deployment_checkpoint_score(metric) -> float:
    """Reward-independent safety score for choosing a deployable PPO snapshot."""
    success = float(metric.get("paper_success", 0.0))
    contact = float(metric.get("pad_contact", 0.0))
    unsafe = float(metric.get("unsafe_pad_contact", 0.0))
    crash = float(metric.get("crash_failure", 0.0))
    lateral = min(max(float(metric.get("touchdown_lateral_error", 5.0)), 0.0), 5.0)
    fov = np.clip(float(metric.get("fov_loss_fraction", 1.0)), 0.0, 1.0)
    blind_descent = np.clip(float(metric.get(
        "unsafe_descent_low_visibility_fraction", 0.0)), 0.0, 1.0)
    relative_speed = min(max(float(metric.get(
        "touchdown_relative_horizontal_velocity", 2.0)), 0.0), 2.0)
    # Safe landing dominates. Unsafe contact can never beat a contact-free near
    # miss merely through dense reward, and visibility resolves similar flights.
    return float(100.0 * success + 10.0 * contact - 45.0 * unsafe
                 - 35.0 * crash - 7.0 * lateral - 10.0 * fov
                 - 8.0 * blind_descent - 3.0 * relative_speed)


def _migrate_legacy_shin_state_dict(state_dict):
    """Rename the pre-refactor estimator container without changing weights."""
    replacements = {
        "estimator.lstm.": "temporal_backbone.lstm.",
        "estimator.latent_head.": "temporal_backbone.latent_head.",
    }
    migrated = {}
    for name, value in state_dict.items():
        target = name
        for old, new in replacements.items():
            if name.startswith(old):
                target = new + name[len(old):]
                break
        migrated[target] = value
    return migrated


def _call_with_optional_lock(lock, function, *args, **kwargs):
    if lock is None:
        return function(*args, **kwargs)
    with lock:
        return function(*args, **kwargs)


def train_live(env_factory: Callable, model, method, seeds, output_dir,
               *, config_hash, potential=None, ppo=None, curriculum_config=None,
               monitor=None, restart_incompatible=False,
               demonstration_dataset=None, demonstration_anchor=None,
               optimizer_lock=None):
    ppo = ppo or {}
    demonstration_anchor = dict(demonstration_anchor or {})
    anchor_enabled = bool(
        demonstration_dataset is not None
        and demonstration_anchor.get("enabled", False))
    curriculum = PlatformMotionCurriculum(**(curriculum_config or {}))
    if hasattr(potential, "assert_frozen"):
        potential.assert_frozen()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo.get("learning_rate", 2e-4)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{method}.pt"
    best_checkpoint_path = output_dir / f"{method}.best.pt"
    history_path = output_dir / f"{method}_training.csv"
    reward_trace_path = output_dir / f"{method}_reward_steps.jsonl"
    logged_reward_episodes = set()
    if reward_trace_path.is_file():
        for line in reward_trace_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                if bool(record.get("episode_complete", False)):
                    logged_reward_episodes.add(int(record["episode"]))
            except (ValueError, KeyError, json.JSONDecodeError):
                # 마지막 줄이 중단 중 잘렸다면 안전하게 무시한다. 완성된
                # checkpoint/history가 권위이며 다음 commit 때 다시 쓴다.
                continue
    history = []
    completed = 0
    best_score = -float("inf")
    best_episode = 0
    model_spec = getattr(model, "pipeline_spec", None)
    if model_spec is None:
        try:
            model_spec = get_pipeline(method)
        except ValueError:
            model_spec = get_pipeline("shin_se")
    warmup_episodes = (max(0, int(ppo.get("perception_warmup_episodes", 0)))
                       if model_spec.state_estimation_enabled else 0)
    reward_normalizer = None
    if ppo.get("reward_component_scales") is not None:
        reward_normalizer = RewardComponentNormalizer(
            scales=tuple(float(value) for value in ppo["reward_component_scales"]),
            exact_paper_raw=bool(ppo.get("exact_paper_raw_reward", False)),
            source="shared_runtime_configuration")
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
        incompatibility = None
        legacy_test_model = not hasattr(model, "pipeline_spec")
        saved_format = saved.get("format")
        legacy_shin_checkpoint = (
            saved_format == "shin2026-recurrent-v1"
            and (legacy_test_model or (
                model_spec.state_estimation_enabled
                and method not in primary_pipeline_ids())))
        if (saved_format != "three-pipeline-recurrent-v3-scaled-estimator"
                and not legacy_shin_checkpoint):
            incompatibility = "unsupported checkpoint format"
        elif saved.get("method") != method or saved.get("config_hash") != config_hash:
            incompatibility = "checkpoint method/config mismatch"
        elif (saved_format == "three-pipeline-recurrent-v3-scaled-estimator"
              and not legacy_test_model
              and saved.get("pipeline_spec") != model_spec.to_manifest()):
            incompatibility = "checkpoint pipeline information-boundary mismatch"
        expected_design = getattr(potential, "sha256", None)
        # The Shin arm never consumes the ontology potential. Older baseline
        # checkpoints may nevertheless carry the run-level artifact hash, so
        # only reward-shaped arms are coupled to a particular design artifact.
        if (incompatibility is None
                and (model_spec.ontology_enabled or method.startswith("ontoreward")) and
                saved.get("reward_design_sha256") != expected_design):
            incompatibility = "checkpoint reward-design mismatch"
        if incompatibility is not None:
            if not restart_incompatible:
                raise ValueError(f"{incompatibility}: {checkpoint_path}")
            old_hash = str(saved.get("config_hash", "unknown"))[:12]
            archive = checkpoint_path.with_name(
                f"{checkpoint_path.stem}.incompatible-{old_hash}{checkpoint_path.suffix}")
            sequence = 1
            while archive.exists():
                archive = checkpoint_path.with_name(
                    f"{checkpoint_path.stem}.incompatible-{old_hash}-{sequence}"
                    f"{checkpoint_path.suffix}")
                sequence += 1
            os.replace(checkpoint_path, archive)
            if history_path.is_file():
                history_archive = archive.with_name(
                    f"{archive.stem}_training{history_path.suffix}")
                os.replace(history_path, history_archive)
            if best_checkpoint_path.is_file():
                best_archive = archive.with_name(
                    f"{archive.stem}.best{best_checkpoint_path.suffix}")
                os.replace(best_checkpoint_path, best_archive)
            print(f"Archived incompatible {method} checkpoint as {archive.name}; "
                  "starting with the current control configuration.")
        else:
            state_dict = saved["model"]
            if legacy_shin_checkpoint and not legacy_test_model:
                state_dict = _migrate_legacy_shin_state_dict(state_dict)
                print(f"Migrating legacy Shin checkpoint module names: {checkpoint_path}")
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(saved["optimizer"])
            completed = int(saved["episode"])
            saved_curriculum = dict(saved["curriculum"])
            interval_changed = int(saved_curriculum["episodes_per_update"]) != int(
                curriculum.episodes_per_update)
            if (interval_changed and ppo.get(
                    "allow_curriculum_interval_migration", False)):
                for name in ("levels", "initial_level"):
                    if int(saved_curriculum[name]) != int(getattr(curriculum, name)):
                        raise ValueError(f"curriculum checkpoint mismatch for {name}")
                curriculum.update(completed)
                print(
                    "Rescaled curriculum checkpoint interval from "
                    f"{saved_curriculum['episodes_per_update']} to "
                    f"{curriculum.episodes_per_update} episodes; continuing at "
                    f"level {curriculum.level} (c={curriculum.c:.3f}).")
            else:
                curriculum.load_state_dict(saved_curriculum)
            if history_path.is_file():
                with history_path.open(newline="", encoding="utf-8") as stream:
                    history = list(csv.DictReader(stream))
            print(f"Resuming {method} at episode {completed + 1} from {checkpoint_path}")
            if best_checkpoint_path.is_file():
                best_saved = torch.load(
                    best_checkpoint_path, map_location=model.device,
                    weights_only=False)
                if (best_saved.get("format") == saved_format
                        and best_saved.get("method") == method
                        and best_saved.get("config_hash") == config_hash
                        and best_saved.get("reward_design_sha256") == expected_design):
                    best_score = float(best_saved.get("selection_score", -float("inf")))
                    best_episode = int(best_saved.get("episode", 0))
                else:
                    print(f"Ignoring incompatible best-policy checkpoint: "
                          f"{best_checkpoint_path}")
    seed_list = list(seeds)
    planned_policy_episodes = max(0, len(seed_list) - warmup_episodes)
    no_landing_grace = no_landing_abort_episode(
        ppo, planned_policy_episodes=planned_policy_episodes)
    if completed > len(seed_list):
        raise ValueError("checkpoint has more episodes than this run requests")
    if monitor is not None:
        monitor.restore_training(method, history)

    def persist_history():
        if not history:
            return
        fields = sorted({key for row in history for key in row})
        temporary = history_path.with_suffix(history_path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        os.replace(temporary, history_path)

    if completed == len(seed_list):
        print(f"{method} training already complete ({completed} episodes).")
        if best_checkpoint_path.is_file():
            selected = torch.load(best_checkpoint_path, map_location=model.device,
                                  weights_only=False)
            model.load_state_dict(selected["model"])
            model._selected_checkpoint_episode = int(selected["episode"])
            model._selected_checkpoint_score = float(selected["selection_score"])
            print(f"Selected {method} best deployment checkpoint from episode "
                  f"{model._selected_checkpoint_episode}.")
        return history

    with env_factory() as env:
        for episode, seed in enumerate(seed_list[completed:], start=completed + 1):
            perception_warmup = episode <= warmup_episodes
            ppo_episode = max(0, episode - warmup_episodes)
            c = (curriculum.update(0) if perception_warmup
                 else curriculum.update(ppo_episode - 1))
            # The rollout metric describes this pre-update policy. Keep its
            # exact weights so deployment selection never attributes a good
            # flight to the subsequent PPO update.
            rollout_model_state = copy.deepcopy(model.state_dict())
            rows, metric = collect_episode_resilient(
                env, model, method, seed, curriculum=c, potential=potential,
                gamma=float(ppo.get("gamma", .99)),
                shaping_lambda=float(ppo.get("shaping_lambda", 1.0)),
                reward_normalizer=reward_normalizer,
                monitor=monitor, phase=("perception warm-up" if perception_warmup
                                        else "training"),
                # Position-backed velocity setpoints bound this exploration.
                # Sampling during the short Shin-only estimator warm-up avoids
                # collecting copies of an almost perfectly static trajectory.
                deterministic=bool(
                    perception_warmup and not ppo.get(
                        "perception_warmup_safe_exploration", True)))
            if perception_warmup:
                loss = _call_with_optional_lock(
                    optimizer_lock, update_estimator_episode,
                    model, optimizer, rows,
                    epochs=int(ppo.get("perception_warmup_epochs", 2)),
                    grad_clip=float(ppo.get("grad_clip", 5.0)),
                    sequence_length=int(ppo.get("sequence_length", 32)))
            else:
                loss = _call_with_optional_lock(
                    optimizer_lock, update_episode,
                    model, optimizer, rows, gamma=float(ppo.get("gamma", .99)),
                    gae_lambda=float(ppo.get("gae_lambda", .95)),
                    epochs=int(ppo.get("epochs", 5)), clip=float(ppo.get("clip", .2)),
                    value_coef=float(ppo.get("value_coef", .5)),
                    entropy_coef=float(ppo.get("entropy_coef", .003)),
                    auxiliary_coef=float(ppo.get(
                        "auxiliary_estimation_coefficient", 1.0)),
                    grad_clip=float(ppo.get("grad_clip", 5.0)),
                    sequence_length=int(ppo.get("sequence_length", 32)),
                    target_kl=float(ppo.get("target_kl", .03)),
                    minimum_learning_rate=float(ppo.get(
                        "minimum_learning_rate", 5e-6)),
                    rollback_on_excessive_kl=bool(ppo.get(
                        "rollback_on_excessive_kl", True)),
                    log_std_bounds=tuple(ppo.get(
                        "log_std_bounds", (-3.0, -0.8))))
                # Short live runs can forget a small set of successful
                # demonstrations before PPO observes its first sparse terminal
                # success. A decaying auxiliary BC pass provides demonstration
                # replay to every arm equally; value and reward learning remain
                # on-policy and the learned exploration variance is untouched.
                anchor_until = max(0, int(demonstration_anchor.get(
                    "until_policy_episode", 0)))
                anchor_interval = max(1, int(demonstration_anchor.get(
                    "interval_episodes", 1)))
                if (anchor_enabled and ppo_episode <= anchor_until
                        and (ppo_episode - 1) % anchor_interval == 0):
                    start_lr = float(demonstration_anchor.get(
                        "learning_rate", 5e-5))
                    end_lr = float(demonstration_anchor.get(
                        "minimum_learning_rate", start_lr * .2))
                    progress = ((ppo_episode - 1) / max(1, anchor_until - 1))
                    anchor_lr = start_lr + (end_lr - start_lr) * progress
                    anchor_metric = _call_with_optional_lock(
                        optimizer_lock, behavior_clone,
                        model, demonstration_dataset,
                        epochs=max(1, int(demonstration_anchor.get("epochs", 1))),
                        learning_rate=anchor_lr,
                        sequence_length=max(1, int(demonstration_anchor.get(
                            "sequence_length", 48))),
                        auxiliary_coefficient=float(demonstration_anchor.get(
                            "auxiliary_coefficient", .10)),
                        post_log_std=None)
                    loss.update({
                        "imitation_anchor_applied": 1.0,
                        "imitation_anchor_learning_rate": anchor_lr,
                        "imitation_anchor_action_loss_before": anchor_metric[
                            "action_loss_before"],
                        "imitation_anchor_action_loss_after": anchor_metric[
                            "action_loss_after"],
                    })
                else:
                    loss["imitation_anchor_applied"] = 0.0
            if hasattr(potential, "assert_frozen"):
                # Reward design is not a PPO module/optimizer parameter. This
                # hash+mode assertion catches accidental mutation immediately.
                potential.assert_frozen()
            metric.update(loss)
            prior_steps = sum(
                int(float(row.get("steps", 0))) for row in history
                if row.get("optimization_phase") == "ppo")
            metric.update({"method": method, "scenario": "training_random_walk",
                           "episode": episode, "curriculum_level": curriculum.level,
                           "curriculum": float(c),
                           "optimization_phase": ("perception_warmup"
                                                  if perception_warmup else "ppo"),
                           "training_sample_efficiency": ppo_episode,
                           "ppo_episode": ppo_episode,
                           "ppo_environment_steps": (prior_steps
                               + (0 if perception_warmup else int(metric["steps"])))})
            log_every = max(1, int(ppo.get("reward_log_every_steps", 1)))
            if episode not in logged_reward_episodes:
                with reward_trace_path.open("a", encoding="utf-8") as stream:
                    for step_index, transition in enumerate(rows):
                        if step_index % log_every and step_index != len(rows) - 1:
                            continue
                        record = {
                            "episode": episode, "time_index": step_index,
                            "episode_complete": step_index == len(rows) - 1,
                            "method": method,
                            "phase": ("perception_warmup" if perception_warmup else "ppo"),
                            "landing_phase": transition.get("landing_phase", "unknown"),
                            "scenario": "training_random_walk",
                            "success": int(metric["paper_success"]),
                            "failure": int(metric["failure"]),
                            "failure_type": metric["status"],
                            "disturbance_level": float(
                                metric.get("domain_external_force_n", 0.0)),
                            "reward": float(transition["reward"]),
                            **{key: (float(value) if isinstance(
                                value, (int, float, np.integer, np.floating)) else value)
                               for key, value in transition["reward_parts"].items()},
                        }
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                logged_reward_episodes.add(episode)
            advanced = (False if perception_warmup else curriculum.observe(metric))
            metric["curriculum_advanced"] = float(advanced)
            metric["next_curriculum_level"] = curriculum.level
            history.append(metric)
            if monitor is not None:
                monitor.training_update(method, metric)
            save_recurrent_checkpoint(
                checkpoint_path, model, optimizer, method=method,
                episode=episode, config_hash=config_hash, curriculum=curriculum,
                potential=potential)
            selection_score = deployment_checkpoint_score(metric)
            if selection_score > best_score:
                best_score = selection_score
                best_episode = episode
                save_recurrent_checkpoint(
                    best_checkpoint_path, model, optimizer, method=method,
                    episode=episode, config_hash=config_hash,
                    curriculum=curriculum, potential=potential,
                    selection_score=selection_score,
                    selection_metric={
                        key: metric.get(key) for key in (
                            "paper_success", "pad_contact", "unsafe_pad_contact",
                            "crash_failure", "touchdown_lateral_error",
                            "fov_loss_fraction",
                            "unsafe_descent_low_visibility_fraction")},
                    model_state=rollout_model_state)
            persist_history()
            issue = training_health_issue(
                history, ppo, warmup_episodes=warmup_episodes,
                planned_policy_episodes=planned_policy_episodes)
            if issue is not None:
                raise RuntimeError(f"training health gate stopped {method}: {issue}")
            health_grace = max(
                int(ppo.get("health_window_episodes", 20)),
                int(ppo.get("health_grace_episodes", 40)))
            recent_policy = [row for row in history
                             if int(float(row.get("episode", 0))) > warmup_episodes]
            recent_window = recent_policy[-max(
                1, int(ppo.get("health_window_episodes", 20))):]
            if (ppo_episode >= health_grace
                    and ppo_episode < no_landing_grace
                    and not any(float(row.get("paper_success", 0.0))
                                for row in recent_window)
                    and (ppo_episode == health_grace
                         or episode == completed + 1)):
                print(
                    f"WARNING: {method} has no landing in the last "
                    f"{len(recent_window)} policy episodes; dense health metrics "
                    f"remain inside their limits, so training continues until "
                    f"policy episode {no_landing_grace} before zero success is fatal.")
            print(f"{method} episode {episode}/{len(seed_list)} "
                  f"return={metric['episode_return']:+.3f} "
                  f"success={int(metric['paper_success'])} c={c:.3f}")
    if best_checkpoint_path.is_file():
        selected = torch.load(best_checkpoint_path, map_location=model.device,
                              weights_only=False)
        model.load_state_dict(selected["model"])
        model._selected_checkpoint_episode = int(selected["episode"])
        model._selected_checkpoint_score = float(selected["selection_score"])
        print(f"Selected {method} best deployment checkpoint from episode "
              f"{model._selected_checkpoint_episode} (score={best_score:.2f}).")
    return history
