# Copyright 2026 Bytedance Ltd. and/or its affiliates
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


class Gr00tInput(ABC):
    def __init__(self):
        # Camera views stacked along a view dimension, shape (B, V, C, H, W), uint8.
        # View order must match the order used during SFT (and GR00T's modality
        # config for the embodiment).
        self.images: torch.Tensor = None

        # task description as a list of strings
        self.task: list[str] = []

        # robot state with shape (B, state_dim), kept at the raw env dimension —
        # normalization and padding to max_state_dim happen inside the model wrapper.
        self.state: torch.Tensor = None

    @classmethod
    @abstractmethod
    def from_env_obs(cls, env_obs) -> "Gr00tInput": ...


class Gr00tOutput:
    def __init__(self):
        self.action: torch.Tensor = None
        self.log_prob: torch.Tensor = None

    @classmethod
    @abstractmethod
    def from_model_output(cls, model_output) -> "Gr00tOutput": ...

    def to_data_proto(self) -> DataProto:
        tensor_batch = {"action": self.action}
        if self.log_prob is not None:
            tensor_batch["log_prob"] = self.log_prob
        return DataProto.from_dict(tensors=tensor_batch)
