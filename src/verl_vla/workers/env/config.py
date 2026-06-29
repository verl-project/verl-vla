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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from verl.base_config import BaseConfig

from verl_vla.recorder.config import RecorderConfig
from verl_vla.teleop.config import TeleopConfig

__all__ = ["EnvWorkerConfig", "SimulatorConfig"]


@dataclass
class SimulatorConfig(BaseConfig):
    """Simulator config consumed by environment workers."""

    simulator_type: str = "libero"
    seed: int = 42
    max_episode_steps: int = 512
    task_suite_name: str = "libero_spatial"
    task_ids: list[int] | None = None
    num_trials_per_task: int | None = None
    specific_reset_id: int | None = None
    reset_warmup_steps: int = 10
    init_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvWorkerConfig(BaseConfig):
    """Configuration for environment workers."""

    async_reset: bool = False
    modes: list[str] = field(default_factory=lambda: ["train"])
    num_envs: int = 1
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    device: str | None = None
    profiler: Any | None = None

    def __post_init__(self):
        if not isinstance(self.simulator, SimulatorConfig):
            object.__setattr__(self, "simulator", SimulatorConfig(**_to_dict(self.simulator)))
        if not isinstance(self.teleop, TeleopConfig):
            object.__setattr__(self, "teleop", instantiate(self.teleop))
        if not isinstance(self.recorder, RecorderConfig):
            object.__setattr__(self, "recorder", instantiate(self.recorder))
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        if not set(self.modes).issubset({"train", "eval"}):
            raise ValueError(f"Unsupported env worker modes: {self.modes}")


def _to_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True)
    return dict(raw or {})
