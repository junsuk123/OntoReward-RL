"""Live recurrent PPO training/evaluation for the controlled benchmark."""
from __future__ import annotations

import os
import csv
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


def _tensor_observation(model, observation):
    image = grayscale_image_tensor(observation.image, device=model.device)
    proprio = torch.as_tensor(observation.proprioception[None],
                              dtype=torch.float32, device=model.device)
    return image, proprio


def _reward(method, previous, following, estimate, next_estimate, potential,
            *, gamma=0.99, shaping_lambda=1.0):
    terminal = dict(
        physical_contact=following.physical_contact, crash=following.crash,
        excessive_drift=following.excessive_drift, terminal=following.terminal)
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
        {"estimated_relative_state": estimate},
        {"estimated_relative_state": next_estimate}, **terminal)
    if method == "ontoreward_plus_active" and not following.terminal:
        active = active_perception_reward(next_loss, ShinRewardConfig())
        value += active
        parts["active_perception"] = active
    return value, parts, next_loss


def collect_episode(env, model: ShinRecurrentActorCritic, method: str, seed: int,
                    *, curriculum=1.0, potential=None, deterministic=False,
                    gamma=0.99, shaping_lambda=1.0,
                    scenario="training_random_walk"):
    """Collect one true simulator episode without crossing the actor boundary."""
    step = env.reset(seed, curriculum, scenario=scenario)
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
                "reward_parts": parts,
                "hidden_h": hidden[0].cpu().numpy(),
                "hidden_c": hidden[1].cpu().numpy(),
            })
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
        "touchdown_time_s": float(len(rows) * env.cfg.sim.dt),
        "steps": len(rows), "status": ("success" if step.physical_contact else
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


def update_episode(model, optimizer, rows, *, gamma=.99, gae_lambda=.95,
                   epochs=5, clip=.2, value_coef=.5, entropy_coef=.003,
                   auxiliary_coef=1.0, grad_clip=5.0, sequence_length=32):
    advantage, returns = _gae(rows, gamma, gae_lambda)
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    device = model.device
    metrics = []
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            metrics.append({key: float(value) for key, value in values.items()})
    return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}


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
               *, config_hash, potential=None, ppo=None, curriculum_config=None):
    ppo = ppo or {}
    curriculum = PlatformMotionCurriculum(**(curriculum_config or {}))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo.get("learning_rate", 2e-4)))
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / f"{method}.pt"
    history_path = output_dir / f"{method}_training.csv"
    history = []
    completed = 0
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
        if saved.get("format") != "shin2026-recurrent-v1":
            raise ValueError(f"unsupported checkpoint format: {checkpoint_path}")
        if saved.get("method") != method or saved.get("config_hash") != config_hash:
            raise ValueError(f"checkpoint method/config mismatch: {checkpoint_path}")
        expected_design = getattr(potential, "sha256", None)
        if saved.get("reward_design_sha256") != expected_design:
            raise ValueError(f"checkpoint reward-design mismatch: {checkpoint_path}")
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
            rows, metric = collect_episode(
                env, model, method, seed, curriculum=c, potential=potential,
                gamma=float(ppo.get("gamma", .99)),
                shaping_lambda=float(ppo.get("shaping_lambda", 1.0)))
            loss = update_episode(
                model, optimizer, rows, gamma=float(ppo.get("gamma", .99)),
                gae_lambda=float(ppo.get("gae_lambda", .95)),
                epochs=int(ppo.get("epochs", 5)), clip=float(ppo.get("clip", .2)),
                value_coef=float(ppo.get("value_coef", .5)),
                entropy_coef=float(ppo.get("entropy_coef", .003)),
                auxiliary_coef=float(ppo.get("auxiliary_estimation_coefficient", 1.0)),
                grad_clip=float(ppo.get("grad_clip", 5.0)),
                sequence_length=int(ppo.get("sequence_length", 32)))
            metric.update(loss)
            metric.update({"method": method, "scenario": "training_random_walk",
                           "episode": episode, "curriculum_level": curriculum.level,
                           "training_sample_efficiency": episode})
            history.append(metric)
            save_recurrent_checkpoint(
                checkpoint_path, model, optimizer, method=method,
                episode=episode, config_hash=config_hash, curriculum=curriculum,
                potential=potential)
            persist_history()
            print(f"{method} episode {episode}/{len(seed_list)} "
                  f"return={metric['episode_return']:+.3f} "
                  f"success={int(metric['paper_success'])} c={c:.3f}")
    return history
