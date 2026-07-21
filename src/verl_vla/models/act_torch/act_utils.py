# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Normalize(nn.Module):
    def __init__(self, stats: dict):
        super().__init__()
        self.register_buffer("mean", torch.tensor(stats.get("mean", []), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(stats.get("std", []), dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / (self.std + 1e-6)


class Unnormalize(nn.Module):
    def __init__(self, stats: dict):
        super().__init__()
        self.register_buffer("mean", torch.tensor(stats.get("mean", []), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(stats.get("std", []), dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return x * (self.std + 1e-6) + self.mean


class ImageTransform:
    def __init__(self, resize_size: tuple[int, int] = (224, 224), norm_stats: dict | None = None):
        self.resize_size = tuple(resize_size)
        self.norm_stats = norm_stats or {}

    def _normalize(self, img: Tensor, key: str | None = None) -> Tensor:
        stats = self.norm_stats.get(key) if key is not None else None
        if stats is None:
            return (img - 0.5) * 2.0

        mean = torch.tensor(stats["mean"], dtype=img.dtype, device=img.device).view(1, -1, 1, 1)
        std = torch.tensor(stats["std"], dtype=img.dtype, device=img.device).view(1, -1, 1, 1)
        return (img - mean) / (std + 1e-6)

    def call_batch(self, images: list[Tensor], key: str | None = None) -> list[Tensor]:
        processed = []
        for img in images:
            if tuple(img.shape[-2:]) != self.resize_size:
                img = F.interpolate(img, size=self.resize_size, mode="bilinear", align_corners=False)
            img = self._normalize(img, key=key)
            processed.append(img)
        return processed


class IdentityTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x
