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
from verl.base_config import BaseConfig

from verl_vla.envs.arena_env.config import ArenaSimulatorConfig
from verl_vla.envs.libero_env.config import LiberoSimulatorConfig
from verl_vla.recorder.config import RecorderConfig
from verl_vla.teleop.config import TeleopConfig

__all__ = ["EnvWorkerConfig", "SimulatorConfig"]


@dataclass
class SimulatorConfig(BaseConfig):
    """Simulator config consumed by environment workers."""

    simulator_type: str = "libero"
    libero: LiberoSimulatorConfig = field(default_factory=LiberoSimulatorConfig)
    arena: ArenaSimulatorConfig = field(default_factory=ArenaSimulatorConfig)

    def __post_init__(self):
        if not isinstance(self.libero, LiberoSimulatorConfig):
            object.__setattr__(self, "libero", instantiate(self.libero))
        if not isinstance(self.arena, ArenaSimulatorConfig):
            object.__setattr__(self, "arena", instantiate(self.arena))
        if self.simulator_type not in {"libero", "arena"}:
            raise ValueError(f"Unsupported simulator_type: {self.simulator_type}")


@dataclass
class EnvWorkerConfig(BaseConfig):
    """Configuration for environment workers."""

    auto_reset: bool = False
    confirm_before_record: bool = False
    modes: list[str] = field(default_factory=lambda: ["train"])
    num_envs: int = 1
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    device: str | None = None
    profiler: Any | None = None
    simulator_start_timeout_s: int = 180

    def __post_init__(self):
        if not isinstance(self.simulator, SimulatorConfig):
            simulator = instantiate(self.simulator)
            if not isinstance(simulator, SimulatorConfig):
                raise TypeError(f"simulator config must instantiate to SimulatorConfig, got {type(simulator).__name__}")
            object.__setattr__(self, "simulator", simulator)
        if not isinstance(self.teleop, TeleopConfig):
            object.__setattr__(self, "teleop", instantiate(self.teleop))
        if not isinstance(self.recorder, RecorderConfig):
            object.__setattr__(self, "recorder", instantiate(self.recorder))
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        if self.simulator_start_timeout_s <= 0:
            raise ValueError(f"simulator_start_timeout_s must be positive, got {self.simulator_start_timeout_s}")
        if not set(self.modes).issubset({"train", "eval"}):
            raise ValueError(f"Unsupported env worker modes: {self.modes}")
