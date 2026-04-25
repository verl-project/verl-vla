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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch


class CriticBackend(ABC):
    uses_task_ids = False

    @abstractmethod
    def init(self, model) -> None:
        pass

    @abstractmethod
    def forward(
        self,
        model,
        a: dict[str, torch.Tensor],
        state_features,
        task_ids: torch.Tensor | None = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_critic_parameters(self, model) -> list[torch.nn.Parameter]:
        pass

    @abstractmethod
    def update_target_network(self, model, tau: float) -> None:
        pass
