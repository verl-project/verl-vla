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

"""Base strategy for converting environment steps to LeRobot frames."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLeRobotStrategy(ABC):
    """Environment-specific LeRobot schema and frame conversion."""

    @property
    @abstractmethod
    def fps(self) -> int:
        """Recording FPS for this environment."""

    @property
    @abstractmethod
    def robot_type(self) -> str | None:
        """Robot type stored in LeRobot metadata."""

    @abstractmethod
    def features(self) -> dict[str, dict[str, Any]]:
        """Return the LeRobot feature schema."""

    @abstractmethod
    def make_frame(
        self,
        *,
        obs: dict[str, Any],
        state: Any,
        action: Any,
        task: str,
        reward: Any = 0.0,
        done: Any = False,
        truncated: Any = False,
        is_intervention: Any = False,
    ) -> dict[str, Any]:
        """Convert one environment step into one LeRobot frame."""
