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

from abc import ABC, abstractmethod

import torch
from verl import DataProto


def prepare_act_image(image: torch.Tensor) -> torch.Tensor:
    """Convert live or recorded images to ACT's float32 BCHW contract."""
    if image.ndim == 5:
        image = image[:, -1]
    if image.ndim != 4:
        raise ValueError(f"ACT image must have four dimensions after time selection, got {tuple(image.shape)}")
    if image.shape[-1] == 3 and image.shape[1] != 3:
        image = image.permute(0, 3, 1, 2)
    if image.dtype == torch.uint8:
        return image.to(dtype=torch.float32).div_(255.0)
    return image.to(dtype=torch.float32)


class ActInput(ABC):
    def __init__(self):
        self.images: dict[str, torch.Tensor] = {}
        self.img_masks: list[torch.Tensor] = []
        self.task: list[str] = []
        self.state: torch.Tensor = None
        self.env_state: torch.Tensor = None

    @classmethod
    @abstractmethod
    def from_env_obs(cls, env_obs) -> "ActInput": ...


class ActOutput:
    def __init__(self):
        self.action: torch.Tensor = None
        self.log_prob: torch.Tensor = None

    @classmethod
    @abstractmethod
    def from_model_output(cls, model_output) -> "ActOutput": ...

    def to_data_proto(self) -> DataProto:
        tensor_batch = {"action": self.action.float()}
        if self.log_prob is not None:
            tensor_batch["log_prob"] = self.log_prob
        return DataProto.from_dict(tensors=tensor_batch)
