# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verl.base_config import BaseConfig


@dataclass
class LiberoSimulatorConfig(BaseConfig):
    simulator_type: str = "libero"
    max_episode_steps: int = 512
    seed: int = 42
    task_suite_name: str = "libero_spatial"
    task_ids: list[int] | None = None
    num_trials_per_task: int | None = None
    specific_reset_id: int | None = None
    reset_warmup_steps: int = 10
    camera_depths: bool = False
    camera_heights: int = 256
    camera_widths: int = 256
    camera_names: list[str] = field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])

    def env_kwargs(self) -> dict[str, Any]:
        return {
            "camera_depths": self.camera_depths,
            "camera_heights": self.camera_heights,
            "camera_widths": self.camera_widths,
            "camera_names": list(self.camera_names),
        }
