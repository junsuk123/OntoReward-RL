"""Live recurrent PPO training/evaluation for the controlled benchmark."""
from __future__ import annotations

import os
import csv
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..curriculum import PlatformMotionCurriculum
from ..mathx import quat_to_euler_zyx
from ..perception import grayscale_image_tensor
from ..reward_modes import (OntoRewardPBRS, ShinReward, ShinRewardConfig,
                            active_perception_reward, sparse_terminal_reward)
from .recurrent import ShinRecurrentActorCritic, recurrent_ppo_loss


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


def _reward(method, previous, following, estimate, next_estimate, potential,
            *, gamma=0.99, shaping_lambda=1.0):
    terminal = dict(
        physical_contact=following.physical_contact, crash=following.crash,
        excessive_drift=following.excessive_drift,
        battery_depleted=following.battery_depleted,
        terminal=following.terminal)
    next_loss = float(np.mean((next_estimate - following.critic.true_relative_state) ** 2))
    if method == "sparse":
        value = sparse_terminal_reward(**terminal)
        return value, {"task": value}, next_loss
    if method in {"shin2026", "manual_no_active"}:
        value, parts = ShinReward(ShinRewardConfig(
            active_enabled=method == "shin2026"))(
                estimate, next_estimate, following.command,
                drone_vertical_velocity=previous.actor.body_velocity[2],
                next_estimation_loss=next_loss, **terminal)
        return value, parts, next_loss
    if potential is None:
        raise ValueError(f"{method} requires a frozen controlled R-GAT potential")
    pbrs = OntoRewardPBRS(
        potential, gamma=gamma, ppo_gamma=gamma,
        shaping_lambda=shaping_lambda, design_id=potential.design_id, frozen=True)
    value, parts = pbrs(
        {"estimated_relative_state": estimate,
         "battery_reserve": _battery_reserve(previous.state)},
        {"estimated_relative_state": next_estimate,
         "battery_reserve": _battery_reserve(following.state)}, **terminal)
    if method == "ontoreward_plus_active" and not following.terminal:
        active = active_perception_reward(next_loss, ShinRewardConfig())
        value += active
        parts["active_perception"] = active
    return value, parts, next_loss


def collect_episode(env, model: ShinRecurrentActorCritic, method: str, seed: int,
                    *, curriculum=1.0, potential=None, deterministic=False,
                    gamma=0.99, shaping_lambda=1.0,
                    scenario="training_random_walk", monitor=None,
                    phase="evaluation"):
    """Collect one true simulator episode without crossing the actor boundary."""
    step = env.reset(seed, curriculum, scenario=scenario)
    initial_battery = dict(_battery_sample(step.state))
    action_scale = float(env.adapter.controller.action_scale)
    if monitor is not None:
        monitor.reset_episode(
            method=method, phase=phase, seed=seed, scenario=scenario,
            curriculum=curriculum, action_scale=action_scale)
    hidden = model.initial_state(1)
    rows = []
    visual_loss_run = 0
    longest_visual_loss = 0
    with torch.no_grad():
        image, proprio = _tensor_observation(model, step.actor)
        truth = torch.as_tensor(step.critic.true_relative_state[None],
                                dtype=torch.float32, device=model.device)
        output = model(image, proprio, true_relative_state=truth, hidden=hidden,
                       episode_start=torch.tensor([True], device=model.device))
        while True:
            mean = output.action_mean[:, -1]
            std = output.action_std[:, -1]
            pre_squash = mean if deterministic else mean + std * torch.randn_like(mean)
            action = torch.tanh(pre_squash)
            log_prob = model.log_prob(pre_squash, action, mean, std)
            following = env.step(action.cpu().numpy()[0])
            next_image, next_proprio = _tensor_observation(model, following.actor)
            next_truth = torch.as_tensor(following.critic.true_relative_state[None],
                                         dtype=torch.float32, device=model.device)
            next_output = model(next_image, next_proprio,
                                true_relative_state=next_truth,
                                hidden=output.hidden)
            estimate = output.relative_state[0, -1].cpu().numpy()
            next_estimate = next_output.relative_state[0, -1].cpu().numpy()
            reward, parts, estimation_loss = _reward(
                method, step, following, estimate, next_estimate, potential,
                gamma=gamma, shaping_lambda=shaping_lambda)
            visual_loss_run = visual_loss_run + 1 if not following.pad_in_fov else 0
            longest_visual_loss = max(longest_visual_loss, visual_loss_run)
            rows.append({
                "image": np.asarray(step.actor.image, dtype=np.uint8),
                "proprioception": step.actor.proprioception.copy(),
                "truth": step.critic.true_relative_state.copy(),
                "pre_squash": pre_squash.cpu().numpy()[0],
                "action": action.cpu().numpy()[0],
                "log_prob": float(log_prob.item()),
                "value": float(output.value.item()), "reward": float(reward),
                "done": float(following.terminal), "estimate": estimate,
                "estimation_loss": estimation_loss, "in_fov": following.pad_in_fov,
                "battery_reserve": _battery_reserve(step.state),
                "battery_energy_j": float(_battery_sample(step.state).get(
                    "remaining_j", 0.0)),
                "reward_parts": parts,
                "hidden_h": hidden[0].cpu().numpy(),
                "hidden_c": hidden[1].cpu().numpy(),
            })
            if monitor is not None:
                status = ("success" if following.physical_contact
                          else "battery_depleted" if following.battery_depleted
                          else "failure" if following.terminal else "running")
                monitor.step(
                    index=len(rows), dt=env.cfg.sim.dt, method=method,
                    reward=reward, reward_parts=parts, estimate=next_estimate,
                    truth=following.critic.true_relative_state,
                    in_fov=following.pad_in_fov,
                    estimation_loss=estimation_loss, state=following.state,
                    scenario=scenario, status=status)
            hidden = output.hidden
            step, output = following, next_output
            if following.terminal:
                break
    env.finish_episode()
    position_error = np.asarray([row["estimate"][:3] - row["truth"][:3] for row in rows])
    velocity_error = np.asarray([row["estimate"][3:] - row["truth"][3:] for row in rows])
    lost_errors = [row["estimation_loss"] for row in rows if not row["in_fov"]]
    truth_final = step.critic.true_relative_state
    state = step.state
    final_battery = _battery_sample(state)
    rpy = quat_to_euler_zyx(np.asarray(state["quaternion_wxyz"], dtype=float))
    metric = {
        "seed": int(seed), "episode_return": float(sum(row["reward"] for row in rows)),
        "paper_success": float(step.physical_contact),
        "strict_success": float(step.strict_success),
        "position_rmse": float(np.sqrt(np.mean(position_error ** 2))),
        "velocity_rmse": float(np.sqrt(np.mean(velocity_error ** 2))),
        "touchdown_lateral_error": float(np.linalg.norm(truth_final[:2])),
        "touchdown_vertical_velocity": float(step.actor.body_velocity[2]),
        "touchdown_relative_horizontal_velocity": float(np.linalg.norm(truth_final[3:5])),
        "touchdown_tilt": float(np.linalg.norm(rpy[:2])),
        "touchdown_angular_rate": float(np.linalg.norm(state["angular_velocity"])),
        "fov_loss_fraction": float(np.mean([not row["in_fov"] for row in rows])),
        "longest_visual_loss_s": float(longest_visual_loss * env.cfg.sim.dt),
        "visual_loss_estimation_error": float(np.mean(lost_errors)) if lost_errors else 0.0,
        "action_envelope_scale": action_scale,
        "touchdown_time_s": float(len(rows) * env.cfg.sim.dt),
        "battery_reserve_initial": float(initial_battery.get("reserve", 1.0)),
        "battery_reserve_final": _battery_reserve(state),
        "battery_energy_initial_j": float(initial_battery.get("remaining_j", 0.0)),
        "battery_energy_final_j": float(final_battery.get("remaining_j", 0.0)),
        "battery_energy_used_j": float(final_battery.get("energy_used_j", 0.0)),
        "battery_depleted": float(step.battery_depleted),
        "steps": len(rows), "status": ("success" if step.physical_contact else
                                         "battery_depleted" if step.battery_depleted else
                                         "timeout" if step.timeout else "failure"),
    }
    return rows, metric


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
    device = model.device
    losses = []
    estimator_parameters = [
        *model.encoder.parameters(), *model.estimator.parameters()]
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
            loss = model.estimator.auxiliary_loss(output.relative_state, truth)
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
                   target_kl=.03, minimum_learning_rate=5e-6):
    advantage, returns = _gae(rows, gamma, gae_lambda)
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    device = model.device
    metrics = []
    early_stop = False
    epochs_completed = 0
    target_kl = float(target_kl)
    for _ in range(int(epochs)):
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
                current_lr = float(optimizer.param_groups[0]["lr"])
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
        "ppo_epochs_completed": float(epochs_completed),
        "effective_learning_rate": float(optimizer.param_groups[0]["lr"]),
    })
    return summary


def training_health_issue(history, ppo, *, warmup_episodes=0) -> str | None:
    """Return why an unattended run is not learning, after a fair window."""
    window = max(1, int(ppo.get("health_window_episodes", 20)))
    policy_rows = [row for row in history
                   if int(float(row.get("episode", 0))) > int(warmup_episodes)]
    if len(policy_rows) < window:
        return None
    recent = policy_rows[-window:]
    successes = sum(float(row.get("paper_success", 0.0)) for row in recent)
    fov_loss = float(np.mean([
        float(row.get("fov_loss_fraction", 1.0)) for row in recent]))
    limit = float(ppo.get("health_max_fov_loss_fraction", 0.80))
    if successes == 0.0 and fov_loss > limit:
        return (f"no landing in the last {window} policy episodes and mean "
                f"FOV loss is {fov_loss:.1%} (limit {limit:.1%})")
    return None


def save_recurrent_checkpoint(path, model, optimizer, *, method, episode,
                              config_hash, curriculum, potential=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "shin2026-recurrent-v1", "method": method,
        "episode": int(episode), "config_hash": config_hash,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "curriculum": curriculum.state_dict(),
        "reward_design_id": getattr(potential, "design_id", None),
        "reward_design_sha256": getattr(potential, "sha256", None),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_live(env_factory: Callable, model, method, seeds, output_dir,
               *, config_hash, potential=None, ppo=None, curriculum_config=None,
               monitor=None, restart_incompatible=False):
    ppo = ppo or {}
    curriculum = PlatformMotionCurriculum(**(curriculum_config or {}))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo.get("learning_rate", 2e-4)))
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / f"{method}.pt"
    history_path = output_dir / f"{method}_training.csv"
    history = []
    completed = 0
    warmup_episodes = max(0, int(ppo.get("perception_warmup_episodes", 0)))
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
        incompatibility = None
        if saved.get("format") != "shin2026-recurrent-v1":
            incompatibility = "unsupported checkpoint format"
        elif saved.get("method") != method or saved.get("config_hash") != config_hash:
            incompatibility = "checkpoint method/config mismatch"
        expected_design = getattr(potential, "sha256", None)
        # The Shin arm never consumes the ontology potential. Older baseline
        # checkpoints may nevertheless carry the run-level artifact hash, so
        # only reward-shaped arms are coupled to a particular design artifact.
        if (incompatibility is None and method.startswith("ontoreward") and
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
            print(f"Archived incompatible {method} checkpoint as {archive.name}; "
                  "starting with the current control configuration.")
        else:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            curriculum.load_state_dict(saved["curriculum"])
            completed = int(saved["episode"])
            if history_path.is_file():
                with history_path.open(newline="", encoding="utf-8") as stream:
                    history = list(csv.DictReader(stream))
            print(f"Resuming {method} at episode {completed + 1} from {checkpoint_path}")
    seed_list = list(seeds)
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
        return history

    with env_factory() as env:
        for episode, seed in enumerate(seed_list[completed:], start=completed + 1):
            c = curriculum.update(episode - 1)
            perception_warmup = episode <= warmup_episodes
            rows, metric = collect_episode(
                env, model, method, seed, curriculum=c, potential=potential,
                gamma=float(ppo.get("gamma", .99)),
                shaping_lambda=float(ppo.get("shaping_lambda", 1.0)),
                monitor=monitor, phase=("perception warm-up" if perception_warmup
                                        else "training"),
                deterministic=perception_warmup)
            if perception_warmup:
                loss = update_estimator_episode(
                    model, optimizer, rows,
                    epochs=int(ppo.get("perception_warmup_epochs", 2)),
                    grad_clip=float(ppo.get("grad_clip", 5.0)),
                    sequence_length=int(ppo.get("sequence_length", 32)))
            else:
                loss = update_episode(
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
                        "minimum_learning_rate", 5e-6)))
            metric.update(loss)
            metric.update({"method": method, "scenario": "training_random_walk",
                           "episode": episode, "curriculum_level": curriculum.level,
                           "curriculum": float(c),
                           "optimization_phase": ("perception_warmup"
                                                  if perception_warmup else "ppo"),
                           "training_sample_efficiency": episode})
            history.append(metric)
            if monitor is not None:
                monitor.training_update(method, metric)
            save_recurrent_checkpoint(
                checkpoint_path, model, optimizer, method=method,
                episode=episode, config_hash=config_hash, curriculum=curriculum,
                potential=potential)
            persist_history()
            issue = training_health_issue(
                history, ppo, warmup_episodes=warmup_episodes)
            if issue is not None:
                raise RuntimeError(f"training health gate stopped {method}: {issue}")
            print(f"{method} episode {episode}/{len(seed_list)} "
                  f"return={metric['episode_return']:+.3f} "
                  f"success={int(metric['paper_success'])} c={c:.3f}")
    return history
