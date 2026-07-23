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

from dataclasses import dataclass, field

from verl.base_config import BaseConfig

from verl_vla.utils.rlpd import RLPDConfig

__all__ = ["RLPDConfig", "SACTrainerConfig"]


@dataclass
class SACTrainerConfig(BaseConfig):
    """Configuration for the VLA SAC trainer loop."""

    _target_: str = "verl_vla.trainer.sac.config.SACTrainerConfig"

    project_name: str = "vla-sac"
    experiment_name: str = "libero-preview"
    logger: list[str] = field(default_factory=lambda: ["console"])
    total_training_steps: int = 600
    rollout_interval: int = 20
    rollout_times: int = 1
    warm_rollout_steps: int = 22
    async_rollout: bool = False
    step_penalty: float = 0.0
    save_freq: int = -1
    test_freq: int = -1
    # Number of trajectories to aggregate per eval. <=0 falls back to the env's
    # eval benchmark size (1 for envs without a fixed benchmark, e.g. Arena,
    # which makes val/trajectory_success_rate a single 0/1 sample).
    eval_episodes: int = -1
    val_before_train: bool = True
    val_only: bool = False
    esi_redundant_time: int = 0
    device: str = "cuda"
    rlpd: RLPDConfig = field(default_factory=RLPDConfig)

    def __post_init__(self):
        if self.total_training_steps <= 0:
            raise ValueError(f"total_training_steps must be positive, got {self.total_training_steps}")
        if self.rollout_interval <= 0:
            raise ValueError(f"rollout_interval must be positive, got {self.rollout_interval}")
        if self.rollout_times < 0:
            raise ValueError(f"rollout_times must be non-negative, got {self.rollout_times}")
        if self.warm_rollout_steps < 0:
            raise ValueError(f"warm_rollout_steps must be non-negative, got {self.warm_rollout_steps}")
