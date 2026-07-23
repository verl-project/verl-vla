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

"""Base recorder interface for rollout side effects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseRecorder(ABC):
    """Common interface shared by dataset and video recorders."""

    @abstractmethod
    def record_once(
        self,
        *,
        env_id: int = 0,
        observation: dict[str, Any],
        extra: dict[str, Any] | None = None,
        action: Any,
        task: str,
        next_reward: Any = 0.0,
        next_terminated: Any = False,
        next_truncated: Any = False,
        next_success: Any = False,
        is_intervention: Any = False,
        critic_value: Any = None,
    ) -> None:
        """Record one environment step."""

    @abstractmethod
    def save_episode(self, env_id: int = 0) -> None:
        """Flush the current episode for one environment."""

    @abstractmethod
    def clear_episode(self, env_id: int = 0) -> None:
        """Discard buffered frames for one environment."""

    def pop_completed(self) -> Path | None:
        """Return a completed artifact root if this recorder owns one."""
        return None

    def set_mode(self, mode: str) -> None:  # noqa: B027
        """Switch the recording mode (e.g. ``train`` vs ``eval``).

        Recorders that separate artifacts by mode override this; the default
        implementation ignores the mode.
        """

    @abstractmethod
    def finalize(self) -> None:
        """Release recorder resources and clear temporary state."""
