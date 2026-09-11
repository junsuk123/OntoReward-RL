"""Keypoint/image encoder used by the Shin-compatible approximation.

No official PACMAN implementation or compatible weights were available when
this benchmark was authored.  This independent network therefore deliberately
uses a different class name and records ``implementation='approximation'`` in
its checkpoints.  It must not be described as PACMAN.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class KeypointEncoderOutput:
    embedding: torch.Tensor
    keypoints: torch.Tensor
    heatmaps: torch.Tensor


class ShinKeypointEncoder(nn.Module):
    implementation = "shin-compatible-keypoint-approximation"

    def __init__(self, embedding_dim: int = 512, keypoints: int = 6):
        super().__init__()
        channels = (1, 32, 64, 128, 128)
        blocks = []
        for cin, cout in zip(channels[:-1], channels[1:]):
            blocks += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*blocks)
        self.heatmap = nn.Conv2d(channels[-1], keypoints, 1)
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(channels[-1] * 16, embedding_dim), nn.Tanh())

    @staticmethod
    def _soft_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
        batch, count, height, width = heatmaps.shape
        probabilities = heatmaps.reshape(batch, count, -1).softmax(-1)
        ys = torch.linspace(-1.0, 1.0, height, device=heatmaps.device,
                            dtype=heatmaps.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=heatmaps.device,
                            dtype=heatmaps.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        x = (probabilities * grid_x.reshape(1, 1, -1)).sum(-1)
        y = (probabilities * grid_y.reshape(1, 1, -1)).sum(-1)
        return torch.stack((x, y), dim=-1)

    def forward(self, image: torch.Tensor) -> KeypointEncoderOutput:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("keypoint encoder expects Bx1xHxW grayscale images")
        features = self.features(image)
        heatmaps = self.heatmap(features)
        return KeypointEncoderOutput(self.embedding(features),
                                     self._soft_argmax(heatmaps), heatmaps)
