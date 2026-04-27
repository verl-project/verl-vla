# Copyright 2026 Bytedance Ltd. and/or its affiliates

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def center_crop_chw(image: torch.Tensor, crop_size: int) -> torch.Tensor:
    image = image.detach().cpu().float().squeeze()
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor after squeeze, got shape={tuple(image.shape)}")

    _, height, width = image.shape
    crop_size = min(crop_size, height, width)
    top = max((height - crop_size) // 2, 0)
    left = max((width - crop_size) // 2, 0)
    return image[:, top : top + crop_size, left : left + crop_size]


def resize_image_chw(image: torch.Tensor, resize_size: tuple[int, int]) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape={tuple(image.shape)}")
    return F.interpolate(
        image.unsqueeze(0),
        size=resize_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def preprocess_image_to_uint8(image: torch.Tensor, crop_size: int, resize_size: tuple[int, int]) -> torch.Tensor:
    image = center_crop_chw(image, crop_size)
    image = resize_image_chw(image, resize_size)
    if image.max() <= 1.0:
        image = image * 255.0
    return image.clamp(0.0, 255.0).round().to(torch.uint8)


def preprocess_image_batch_to_uint8(
    images: torch.Tensor | np.ndarray | Any,
    crop_size: int,
    resize_size: tuple[int, int],
) -> torch.Tensor:
    images_t = images if torch.is_tensor(images) else torch.as_tensor(images)
    if images_t.ndim == 3:
        return preprocess_image_to_uint8(images_t, crop_size=crop_size, resize_size=resize_size)
    if images_t.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW image tensor, got shape={tuple(images_t.shape)}")

    return torch.stack(
        [preprocess_image_to_uint8(image, crop_size=crop_size, resize_size=resize_size) for image in images_t],
        dim=0,
    )


def is_int8_image_tensor(image: Any) -> bool:
    return torch.is_tensor(image) and image.dtype in (torch.uint8, torch.int8)


def image_to_float01(image: torch.Tensor | np.ndarray | Any) -> torch.Tensor:
    image_t = image if torch.is_tensor(image) else torch.as_tensor(image)
    src_dtype = image_t.dtype
    image_f = image_t.float()
    if src_dtype in (torch.uint8, torch.int8):
        image_f = image_f / 255.0
    elif image_f.numel() > 0 and image_f.max() > 1.0:
        image_f = image_f / 255.0
    return image_f.clamp(0.0, 1.0)
