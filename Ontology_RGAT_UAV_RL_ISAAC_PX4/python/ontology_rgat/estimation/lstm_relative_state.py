"""Six-state recurrent visual estimator with explicit episode masks."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EstimatorOutput:
    relative_state: torch.Tensor
    latent: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor]


class LSTMRelativeStateEstimator(nn.Module):
    def __init__(self, image_embedding: int = 512, proprioception: int = 7,
                 hidden_size: int = 512, latent_size: int = 256,
                 output_size: int = 6):
        super().__init__()
        if output_size != 6:
            raise ValueError("the Shin-compatible estimator predicts exactly six values")
        self.hidden_size = int(hidden_size)
        self.lstm = nn.LSTM(image_embedding + proprioception, hidden_size,
                            batch_first=True)
        if latent_size <= output_size:
            raise ValueError("latent vector needs policy features after its six estimates")
        # Fig. 3--4: l_t, h_t and u_t are concatenated before the state-
        # estimation head; y[0:6] is the auxiliary estimate and y[6:N] is fed
        # to decision making.  Keeping that slicing literal prevents truth
        # targets from becoming actor features.
        self.latent_size = int(latent_size)
        self.output_size = int(output_size)
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_size + image_embedding + proprioception, latent_size),
            nn.Tanh())

    def initial_state(self, batch_size: int, *, device=None, dtype=None):
        shape = (1, int(batch_size), self.hidden_size)
        return (torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype))

    @staticmethod
    def reset_hidden(hidden, episode_start: torch.Tensor | None):
        if episode_start is None:
            return hidden
        mask = (~episode_start.bool()).to(hidden[0].dtype).reshape(1, -1, 1)
        return hidden[0] * mask, hidden[1] * mask

    def forward(self, image_embedding: torch.Tensor, proprioception: torch.Tensor,
                hidden=None, episode_start: torch.Tensor | None = None) -> EstimatorOutput:
        single_step = image_embedding.ndim == 2
        if single_step:
            image_embedding = image_embedding[:, None, :]
        if proprioception.ndim == 2:
            proprioception = proprioception[:, None, :]
        if image_embedding.shape[:2] != proprioception.shape[:2]:
            raise ValueError("image and proprioception sequence dimensions must match")
        batch = image_embedding.shape[0]
        if hidden is None:
            hidden = self.initial_state(batch, device=image_embedding.device,
                                        dtype=image_embedding.dtype)
        recurrent_input = torch.cat((image_embedding, proprioception), -1)
        # Reset at every environment boundary in a batched sequence.  A simple
        # one-shot mask is insufficient when a rollout contains multiple
        # episodes from the same environment.
        if episode_start is not None and episode_start.ndim == 2:
            outputs = []
            for step in range(recurrent_input.shape[1]):
                hidden = self.reset_hidden(hidden, episode_start[:, step])
                value, hidden = self.lstm(recurrent_input[:, step:step + 1], hidden)
                outputs.append(value)
            sequence = torch.cat(outputs, dim=1)
        else:
            hidden = self.reset_hidden(hidden, episode_start)
            sequence, hidden = self.lstm(recurrent_input, hidden)
        latent = self.latent_head(torch.cat((image_embedding, sequence, proprioception), -1))
        relative = latent[..., :self.output_size]
        return EstimatorOutput(relative, latent, hidden)

    @staticmethod
    def auxiliary_loss(prediction: torch.Tensor, privileged_target: torch.Tensor,
                       valid: torch.Tensor | None = None) -> torch.Tensor:
        error = (prediction - privileged_target).square().mean(-1)
        if valid is not None:
            weights = valid.to(error.dtype)
            return (error * weights).sum() / weights.sum().clamp_min(1.0)
        return error.mean()
