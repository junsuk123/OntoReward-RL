"""Strict grayscale preprocessing shared by training and deployment."""
from __future__ import annotations

import numpy as np
import torch


def grayscale_image_tensor(image, *, height: int = 320, width: int = 512,
                           device=None) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[None, None]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.moveaxis(array, -1, 0)[None]
    elif array.ndim == 3:
        array = array[:, None]
    elif array.ndim != 4:
        raise ValueError(f"image must be HxW, HxWx1, BxHxW or Bx1xHxW; got {array.shape}")
    if array.shape[1] != 1 or tuple(array.shape[-2:]) != (height, width):
        raise ValueError(f"expected grayscale image batch Bx1x{height}x{width}, got {array.shape}")
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
    if tensor.numel() and float(tensor.detach().max()) > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)
