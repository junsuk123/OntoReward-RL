"""Six-state recurrent visual estimator with explicit episode masks."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..ppo.temporal_backbone import TemporalVisualBackbone
from .relative_state_aux import RelativeStateAuxiliaryHead


@dataclass
class EstimatorOutput:
    relative_state: torch.Tensor
    latent: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor]


class LSTMRelativeStateEstimator(nn.Module):
    """Compatibility wrapper around the decoupled backbone and Shin head.

    New pipeline construction uses :class:`TemporalVisualBackbone` directly;
    this name remains for legacy callers and the Shin-specific smoke tests.
    """

    def __init__(self, image_embedding: int = 512, proprioception: int = 7,
                 hidden_size: int = 512, latent_size: int = 256,
                 output_size: int = 6):
        super().__init__()
        if output_size != 6:
            raise ValueError("the Shin-compatible estimator predicts exactly six values")
        if latent_size <= output_size:
            raise ValueError("latent vector needs policy features after its six estimates")
        self.backbone = TemporalVisualBackbone(
            image_embedding=image_embedding, proprioception=proprioception,
            hidden_size=hidden_size, latent_size=latent_size)
        self.auxiliary_head = RelativeStateAuxiliaryHead()
        self.hidden_size = int(hidden_size)
        self.latent_size = int(latent_size)
        self.output_size = int(output_size)

    def initial_state(self, batch_size: int, *, device=None, dtype=None):
        return self.backbone.initial_state(
            batch_size, device=device, dtype=dtype)

    @staticmethod
    def reset_hidden(hidden, episode_start: torch.Tensor | None):
        return TemporalVisualBackbone.reset_hidden(hidden, episode_start)

    def forward(self, image_embedding: torch.Tensor, proprioception: torch.Tensor,
                hidden=None, episode_start: torch.Tensor | None = None) -> EstimatorOutput:
        output = self.backbone(
            image_embedding, proprioception, hidden, episode_start)
        relative = self.auxiliary_head(output.latent)
        return EstimatorOutput(relative, output.latent, output.hidden)

    @staticmethod
    def auxiliary_loss(prediction: torch.Tensor, privileged_target: torch.Tensor,
                       valid: torch.Tensor | None = None) -> torch.Tensor:
        return RelativeStateAuxiliaryHead.loss(
            prediction, privileged_target, valid)
