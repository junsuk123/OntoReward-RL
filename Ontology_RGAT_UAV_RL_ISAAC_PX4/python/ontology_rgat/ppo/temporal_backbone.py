"""Estimator-agnostic temporal visual representation backbone."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TemporalBackboneOutput:
    latent: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor]


class TemporalVisualBackbone(nn.Module):
    """Image/proprioception LSTM that knows nothing about relative-state truth."""

    def __init__(self, image_embedding: int = 512, proprioception: int = 7,
                 hidden_size: int = 512, latent_size: int = 256):
        super().__init__()
        if latent_size <= 6:
            raise ValueError("latent vector must retain policy features after six reserved values")
        self.hidden_size = int(hidden_size)
        self.latent_size = int(latent_size)
        self.lstm = nn.LSTM(image_embedding + proprioception, hidden_size,
                            batch_first=True)
        # The first six values are supervised relative-state coordinates.  A
        # global Tanh used to cap those physical predictions to [-1, 1], even
        # though the benchmark starts as high as 8 m and can move faster than
        # 1 m/s.  Keep that estimator channel unbounded; only the policy-only
        # representation is squashed.  Retaining the Sequential container
        # preserves the Linear parameter names for explicit checkpoint
        # incompatibility handling in recurrent_train.py.
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_size + image_embedding + proprioception, latent_size))

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
                hidden=None,
                episode_start: torch.Tensor | None = None) -> TemporalBackboneOutput:
        if image_embedding.ndim == 2:
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
        if episode_start is not None and episode_start.ndim == 2:
            values = []
            for index in range(recurrent_input.shape[1]):
                hidden = self.reset_hidden(hidden, episode_start[:, index])
                value, hidden = self.lstm(recurrent_input[:, index:index + 1], hidden)
                values.append(value)
            sequence = torch.cat(values, dim=1)
        else:
            hidden = self.reset_hidden(hidden, episode_start)
            sequence, hidden = self.lstm(recurrent_input, hidden)
        raw_latent = self.latent_head(
            torch.cat((image_embedding, sequence, proprioception), -1))
        latent = torch.cat((raw_latent[..., :6],
                            torch.tanh(raw_latent[..., 6:])), dim=-1)
        return TemporalBackboneOutput(latent=latent, hidden=hidden)
