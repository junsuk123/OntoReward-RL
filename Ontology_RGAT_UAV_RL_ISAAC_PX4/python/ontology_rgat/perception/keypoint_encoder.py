"""Keypoint/descriptor encoder used by the Shin-compatible approximation.

No official PACMAN implementation or compatible weights were available when
this benchmark was authored.  This independent network therefore deliberately
uses a different class name and records ``implementation='approximation'`` in
its checkpoints.  It must not be described as PACMAN.

Architecture (v4, 2026-09-20)
-----------------------------
The previous trunk was four stride-2 3x3 convolutions without normalisation:
a stride-16 heatmap cell saw a 31x31 px patch of a 512x320 frame.  Measured on
48 geometry-labelled Isaac frames it collapsed all six landmark predictions
onto the pad centre (predicted spread 12.9 px against 44.5 px true; PCK@20
15 %), because a cell that cannot see the hexagon as a whole cannot tell which
vertex it is looking at.

This version is a small encoder-decoder:

* per-image standardisation of the grayscale input, so the global brightness
  and contrast of the renderer (Isaac frames average 20-50/255) do not reach
  the convolutions;
* five stride-2 stages down to 10x16 (receptive field wider than the frame)
  with GroupNorm, plus a global-average context vector added back at the
  bottleneck;
* two decoder stages with skip connections up to stride 8 (40x64 heatmaps);
* coordinates decoded by a soft-argmax over a 9x9-cell window around each
  heatmap's peak, so background mass cannot pull a landmark toward the centre;
* the same descriptor path as before: a soft-argmax location per landmark, a
  descriptor pooled at each heatmap, a visibility head per landmark and one
  tanh embedding over the six descriptors, which the LSTM consumes.

Landmark *identity* is not what the six heatmap channels encode.  The
hexagonal target is 60-degree symmetric and its identity pips are one to
three pixels at approach altitude, so the supervision (see
``keypoint_pretrain``) orders the six landmarks canonically in the image
plane; channel ``k`` is "the k-th vertex counter-clockwise from the +x image
axis", not "pad-frame vertex k".
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class KeypointEncoderOutput:
    embedding: torch.Tensor
    keypoints: torch.Tensor
    heatmaps: torch.Tensor
    visibility: torch.Tensor


def _block(cin: int, cout: int, stride: int, groups: int = 8) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(groups, cout), cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(groups, cout), cout), nn.ReLU(inplace=True))


class ShinKeypointEncoder(nn.Module):
    """Grayscale frame -> six canonical-order heatmaps, descriptors, visibility."""

    implementation = "isaac-canonical-six-keypoint-unet-v4-stride8-window-visibility"
    # Output heatmaps are ``height // heatmap_stride`` x ``width // heatmap_stride``.
    heatmap_stride = 8
    # Coordinates are decoded by a soft-argmax over the (2r+1)x(2r+1)-cell
    # window around each heatmap's peak (r = 4 cells = 72 px at stride 8).  A
    # global soft-argmax is a mean over the whole frame, so any heatmap mass
    # on clutter drags the coordinate toward the frame centre; on the 48
    # surveyed Isaac frames the window halved the held-out error (14.9 px ->
    # 6.9 px, PCK@10 73 % -> 86 %).  Descriptor pooling and the visibility
    # peak features still use the global softmax.
    soft_argmax_radius_cells = 4
    widths = (32, 64, 128, 192, 256)

    def __init__(self, embedding_dim: int = 512, keypoints: int = 6):
        super().__init__()
        w1, w2, w3, w4, w5 = self.widths
        self.enc1 = _block(1, w1, 2)        # stride 2
        self.enc2 = _block(w1, w2, 2)       # stride 4
        self.enc3 = _block(w2, w3, 2)       # stride 8
        self.enc4 = _block(w3, w4, 2)       # stride 16
        self.enc5 = _block(w4, w5, 2)       # stride 32
        self.context = nn.Sequential(
            nn.Linear(w5, w5), nn.ReLU(inplace=True), nn.Linear(w5, w5))
        self.dec4 = _block(w5 + w4, w4, 1)  # stride 16
        self.dec3 = _block(w4 + w3, w3, 1)  # stride 8
        self.heatmap = nn.Conv2d(w3, keypoints, 1)
        # Descriptor plus the heatmap's peak height and peak-over-mean margin:
        # a landmark that is not in the frame has no peak to pool from.
        self.visibility_head = nn.Sequential(
            nn.Linear(w3 + 2, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        # The paper consumes descriptors attached to six keypoints.  Pooling a
        # descriptor at each learned heatmap location makes that information
        # path executable.
        self.embedding = nn.Sequential(
            nn.Linear(w3 * keypoints, embedding_dim), nn.Tanh())
        self.keypoint_count = int(keypoints)

    @classmethod
    def heatmap_shape(cls, height: int, width: int) -> tuple[int, int]:
        """Heatmap grid for an input of ``height`` x ``width`` pixels."""
        return int(height) // cls.heatmap_stride, int(width) // cls.heatmap_stride

    @staticmethod
    def standardize(image: torch.Tensor) -> torch.Tensor:
        """Zero-mean, unit-variance per frame; removes the renderer's exposure."""
        mean = image.mean(dim=(-1, -2), keepdim=True)
        std = image.std(dim=(-1, -2), keepdim=True)
        return (image - mean) / (std + 1e-3)

    @classmethod
    def _soft_argmax(cls, heatmaps: torch.Tensor) -> torch.Tensor:
        """Windowed soft-argmax: softmax over the cells within
        ``soft_argmax_radius_cells`` of each heatmap's peak (clamped at the
        border, so the peak is always inside and the softmax finite)."""
        batch, count, height, width = heatmaps.shape
        flat = heatmaps.reshape(batch, count, -1)
        radius = int(cls.soft_argmax_radius_cells)
        if radius > 0:
            peak = flat.detach().argmax(-1)
            peak_y = torch.div(peak, width, rounding_mode="floor")
            peak_x = peak - peak_y * width
            rows = torch.arange(height, device=heatmaps.device)
            columns = torch.arange(width, device=heatmaps.device)
            in_rows = (rows[None, None, :] - peak_y[..., None]).abs() <= radius
            in_columns = (columns[None, None, :] - peak_x[..., None]).abs() <= radius
            window = in_rows[..., :, None] & in_columns[..., None, :]
            flat = flat.masked_fill(~window.reshape(batch, count, -1), float("-inf"))
        probabilities = flat.float().softmax(-1)
        ys = torch.linspace(-1.0, 1.0, height, device=heatmaps.device,
                            dtype=probabilities.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=heatmaps.device,
                            dtype=probabilities.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        x = (probabilities * grid_x.reshape(1, 1, -1)).sum(-1)
        y = (probabilities * grid_y.reshape(1, 1, -1)).sum(-1)
        return torch.stack((x, y), dim=-1).to(heatmaps.dtype)

    def forward(self, image: torch.Tensor) -> KeypointEncoderOutput:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("keypoint encoder expects Bx1xHxW grayscale images")
        x = self.standardize(image)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e5 = e5 + self.context(e5.mean(dim=(-1, -2)))[:, :, None, None]
        d4 = self.dec4(torch.cat(
            (F.interpolate(e5, size=e4.shape[-2:], mode="nearest"), e4), 1))
        d3 = self.dec3(torch.cat(
            (F.interpolate(d4, size=e3.shape[-2:], mode="nearest"), e3), 1))
        heatmaps = self.heatmap(d3)
        batch, count = heatmaps.shape[:2]
        logits = heatmaps.reshape(batch, count, -1)
        probabilities = logits.softmax(-1)
        features = d3.reshape(batch, d3.shape[1], -1)
        descriptors = torch.einsum("bkn,bcn->bkc", probabilities, features)
        peak = logits.max(-1).values
        margin = peak - logits.mean(-1)
        visibility = torch.sigmoid(self.visibility_head(
            torch.cat((descriptors, peak[..., None], margin[..., None]), -1)
        ).squeeze(-1))
        embedding = self.embedding(descriptors.flatten(1))
        return KeypointEncoderOutput(
            embedding, self._soft_argmax(heatmaps), heatmaps, visibility)
