# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preprocessing utilities for the GR00T policy.

Replaces the Isaac-GR00T processor stack (albumentations / cv2 / dm-tree /
Gr00tN1d7Processor) with lightweight torchvision + Qwen3VLProcessor equivalents.
"""

import re
from typing import Any

import torch
import torchvision.transforms.v2 as transforms

# Subset of Isaac-GR00T's EMBODIMENT_TAG_TO_PROJECTOR_INDEX that verl-vla uses.
EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    "libero_sim": 2,
    "new_embodiment": 10,
}


class MinMaxNormalize:
    """Normalize to [-1, 1] with q01/q99 percentile min-max and clipping.

    Mirrors Isaac-GR00T's StateActionProcessor (use_percentiles=True): values are
    mapped by (x - q01) / (q99 - q01) * 2 - 1 and clamped to [-1, 1].
    """

    def __init__(self, stats: dict[str, Any], *, clip: bool = True) -> None:
        self.EPSILON = 1e-6
        for attr in ("q01", "q99"):
            if attr not in stats:
                raise AttributeError(f"stats object is missing the following attribute: {attr}")
        self.q01 = torch.tensor(stats["q01"], dtype=torch.float32)
        self.q99 = torch.tensor(stats["q99"], dtype=torch.float32)
        self.clip = clip

    def to(self, device: torch.device | str) -> None:
        self.q01 = self.q01.to(device)
        self.q99 = self.q99.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_dim = x.shape[-1]
        normalized = (x - self.q01[..., :x_dim]) / (self.q99[..., :x_dim] - self.q01[..., :x_dim] + self.EPSILON)
        normalized = normalized * 2.0 - 1.0
        if self.clip:
            normalized = normalized.clamp(-1.0, 1.0)
        return normalized


class MinMaxUnnormalize:
    """Inverse of MinMaxNormalize (no clipping on the way out)."""

    def __init__(self, stats: dict[str, Any]) -> None:
        self.EPSILON = 1e-6
        for attr in ("q01", "q99"):
            if attr not in stats:
                raise AttributeError(f"stats object is missing the following attribute: {attr}")
        self.q01 = torch.tensor(stats["q01"], dtype=torch.float32)
        self.q99 = torch.tensor(stats["q99"], dtype=torch.float32)

    def to(self, device: torch.device | str) -> None:
        self.q01 = self.q01.to(device)
        self.q99 = self.q99.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_dim = x.shape[-1]
        return (x + 1.0) / 2.0 * (self.q99[..., :x_dim] - self.q01[..., :x_dim] + self.EPSILON) + self.q01[..., :x_dim]


class Gr00tImageTransform:
    """GR00T image pipeline: Resize(target) -> Random/CenterCrop(crop) -> Resize(target).

    Mirrors Isaac-GR00T's torchvision branch of build_image_transformations. Operates
    on uint8 image tensors and returns uint8 (the Qwen3-VL processor consumes raw
    uint8 images and does its own rescaling).
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (256, 256),
        crop_size: tuple[int, int] = (230, 230),
    ) -> None:
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(size=list(target_size)),
                transforms.RandomCrop(size=list(crop_size)),
                transforms.Resize(size=list(target_size)),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(size=list(target_size)),
                transforms.CenterCrop(size=list(crop_size)),
                transforms.Resize(size=list(target_size)),
            ]
        )

    def __call__(self, images: torch.Tensor, *, train: bool = False) -> torch.Tensor:
        """Apply the transform.

        Args:
            images: uint8 tensor of shape (..., C, H, W).
            train: use the RandomCrop pipeline instead of CenterCrop.
        """
        transform = self.train_transform if train else self.eval_transform
        return transform(images)


def formalize_language(language: str) -> str:
    """Lowercase and strip punctuation, as GR00T pretraining does."""
    return re.sub(r"[^\w\s]", "", language.lower())


class Gr00tVLMTransform:
    """Tokenize images + language through the Qwen3-VL processor.

    Builds the same single-turn chat conversation as Gr00tN1d7Processor
    (all camera views as images, then the task text) and runs one padded batch
    call through Qwen3VLProcessor.
    """

    def __init__(self, processor_path: str, *, formalize: bool = True) -> None:
        self.processor_path = processor_path
        self.formalize = formalize
        self._processor = None

    @property
    def processor(self):
        if self._processor is None:
            from transformers import Qwen3VLProcessor

            self._processor = Qwen3VLProcessor.from_pretrained(self.processor_path)
            # Left padding for Flash Attention compatibility (same as upstream).
            self._processor.tokenizer.padding_side = "left"
        return self._processor

    def call_batch(self, images: torch.Tensor, tasks: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a batch of multi-view images and task strings.

        Args:
            images: uint8 tensor of shape (B, V, C, H, W), already through
                Gr00tImageTransform.
            tasks: list of B task strings.

        Returns:
            dict with input_ids, attention_mask, pixel_values, image_grid_thw.
        """
        import numpy as np
        from PIL import Image

        assert images.ndim == 5, f"(B,V,C,H,W) expected, but got {images.shape}"
        batch_size = images.shape[0]
        assert len(tasks) == batch_size, f"got {len(tasks)} tasks for batch of {batch_size}"

        images_np = images.permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)

        texts: list[str] = []
        all_images: list[Image.Image] = []
        for i in range(batch_size):
            pil_images = [Image.fromarray(view) for view in images_np[i]]
            language = str(tasks[i])
            if self.formalize:
                language = formalize_language(language)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in pil_images],
                        {"type": "text", "text": language},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            all_images.extend(pil_images)

        tokenized = self.processor(text=texts, images=all_images, return_tensors="pt", padding=True)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "pixel_values": tokenized["pixel_values"],
            "image_grid_thw": tokenized["image_grid_thw"],
        }


def pad_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Zero-pad the last dimension of x to target_dim."""
    current_dim = x.shape[-1]
    if current_dim >= target_dim:
        return x
    shape = list(x.shape)
    shape[-1] = target_dim
    padded = torch.zeros(*shape, dtype=x.dtype, device=x.device)
    padded[..., :current_dim] = x
    return padded
