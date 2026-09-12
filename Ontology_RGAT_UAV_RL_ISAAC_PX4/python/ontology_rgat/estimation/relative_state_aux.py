"""Optional Shin-style supervision attached to a generic temporal latent."""
from __future__ import annotations

import torch
from torch import nn


class RelativeStateAuxiliaryHead(nn.Module):
    """Expose and supervise the six reserved latent dimensions.

    This deliberately owns no temporal layers.  Consequently estimator-free
    pipelines instantiate no estimator head while retaining an identical LSTM
    backbone and latent dimensionality.
    """

    output_size = 6

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] <= self.output_size:
            raise ValueError("latent is too small for six-state auxiliary supervision")
        return latent[..., :self.output_size]

    @staticmethod
    def loss(prediction: torch.Tensor, privileged_target: torch.Tensor,
             valid: torch.Tensor | None = None) -> torch.Tensor:
        if prediction.shape != privileged_target.shape or prediction.shape[-1] != 6:
            raise ValueError("relative-state auxiliary target must match BxTx6 prediction")
        error = (prediction - privileged_target).square().mean(-1)
        if valid is not None:
            weights = valid.to(error.dtype)
            return (error * weights).sum() / weights.sum().clamp_min(1.0)
        return error.mean()
