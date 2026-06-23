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

from dataclasses import dataclass

from verl.base_config import BaseConfig

__all__ = ["LeRobotDataLoaderConfig"]


@dataclass
class LeRobotDataLoaderConfig(BaseConfig):
    """Configuration for the SFT LeRobot dataset and PyTorch dataloader."""

    _target_: str = "verl_vla.utils.dataloader.LeRobotDataLoaderConfig"

    repo_id: str | None = None
    root: str | None = None
    revision: str | None = "main"
    batch_size: int = 32
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = True
    seed: int | None = 42
    video_backend: str | None = "pyav"
    action_delta_steps: int = 64

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        if self.prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be positive, got {self.prefetch_factor}")
        if self.action_delta_steps < 0:
            raise ValueError(f"action_delta_steps must be non-negative, got {self.action_delta_steps}")
