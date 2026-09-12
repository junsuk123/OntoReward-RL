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

    def __init__(self, state_scale=(3.0, 3.0, 8.0, 3.0, 3.0, 2.0)):
        super().__init__()
        scale = torch.as_tensor(state_scale, dtype=torch.float32)
        if scale.shape != (self.output_size,) or not torch.isfinite(scale).all():
            raise ValueError("relative-state scale must contain six finite values")
        if torch.any(scale <= 0):
            raise ValueError("relative-state scale values must be positive")
        # Stored in checkpoints because it is part of the model/output unit
        # contract.  Values follow the configured Table-I position envelope
        # and the controller/platform velocity envelope.
        self.register_buffer("state_scale", scale)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] <= self.output_size:
            raise ValueError("latent is too small for six-state auxiliary supervision")
        return latent[..., :self.output_size] * self.state_scale

    def loss(self, prediction: torch.Tensor, privileged_target: torch.Tensor,
             valid: torch.Tensor | None = None) -> torch.Tensor:
        if prediction.shape != privileged_target.shape or prediction.shape[-1] != 6:
            raise ValueError("relative-state auxiliary target must match BxTx6 prediction")
        # Equation (1) is evaluated in dimensionless benchmark coordinates so
        # metres and metres/second contribute comparably.  forward() remains
        # physical-unit output for telemetry and downstream consumers.
        scale = self.state_scale.to(device=prediction.device,
                                    dtype=prediction.dtype)
        error = ((prediction - privileged_target) / scale).square().mean(-1)
        if valid is not None:
            weights = valid.to(error.dtype)
            return (error * weights).sum() / weights.sum().clamp_min(1.0)
        return error.mean()

    def numpy_loss(self, prediction, privileged_target) -> float:
        """Return the same normalized MSE for live NumPy reward telemetry."""
        prediction = torch.as_tensor(prediction, dtype=self.state_scale.dtype,
                                     device=self.state_scale.device)
        target = torch.as_tensor(privileged_target, dtype=self.state_scale.dtype,
                                 device=self.state_scale.device)
        if prediction.shape != (self.output_size,) or target.shape != (self.output_size,):
            raise ValueError("live relative-state loss expects two six-vectors")
        return float((((prediction - target) / self.state_scale) ** 2).mean().item())
