"""Train-split-only, masked same-time semantic reconstruction (no teacher)."""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch import nn
from .environment import TwoAxisLandingEnv


def pretrain_causal_encoder(agent, config, *, seed, env_factory=TwoAxisLandingEnv,
                           action_dimension=2):
    settings = config.ontology
    if not settings.pretrain_episodes or agent.mode != "ppo_ontology_rgat":
        return {"enabled": False, "environment_steps": 0}
    env = env_factory(config)
    rng = np.random.default_rng(seed+7_000_000)
    states, seeds = [], []
    try:
        for i in range(settings.pretrain_episodes):
            episode_seed = 5_000_000 + 10_000*seed + i
            seeds.append(episode_seed)
            observation, _ = env.reset(seed=episode_seed)
            for _ in range(settings.pretrain_decisions):
                states.append(observation.graph.X.copy())
                # Commands visit states only. Never retain them as targets.
                observation, _, done, _, _ = env.step(np.tanh(.5*rng.standard_normal(action_dimension)))
                if done:
                    break
    finally:
        close=getattr(env,'close',None)
        if close is not None:close()
    samples = torch.as_tensor(np.stack(states), dtype=torch.float32, device=agent.device)
    # Spatial v9 has two signed ENU projections, each retaining the same
    # 9x12 schema and shared encoder. They are same-time causal samples,
    # not additional environment interactions or cross-plane target labels.
    if samples.ndim == 4:
        samples = samples.reshape(-1,9,12)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+7_000_001)
        decoder = nn.Linear(settings.hidden_dimension, 9).to(agent.device)
    encoder = agent.actor.encoder
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    generator = torch.Generator(device=agent.device).manual_seed(seed+7_000_002)
    history = []
    for _ in range(settings.pretrain_epochs):
        losses = []
        order = torch.randperm(len(samples), generator=generator, device=agent.device)
        for indices in order.split(128):
            target = samples[indices]
            mask = torch.rand(target[:,:,:9].shape, generator=generator, device=agent.device) < .25
            mask[:,0,0] |= ~mask.flatten(1).any(1)
            masked = target.clone()
            masked[:,:,:9] = torch.where(mask, 0, target[:,:,:9])
            prediction = decoder(encoder.node_embeddings(masked))
            loss = ((prediction-target[:,:,:9])[mask]**2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    agent.critic.encoder.load_state_dict(encoder.state_dict())
    digest = hashlib.sha256()
    for name,value in encoder.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return {"enabled": True, "environment_steps": len(states), "seeds": seeds,
            "split": "train_only", "loss_history": history,
            "artifact_sha256": digest.hexdigest(),
            "objective": "masked_same_time_node_reconstruction",
            "uses_actions_as_targets": False, "uses_reward_or_truth_or_future": False}
