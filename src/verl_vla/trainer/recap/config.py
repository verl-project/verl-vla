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
from typing import Any

from omegaconf import DictConfig, OmegaConf
from verl.base_config import BaseConfig

__all__ = ["MainReCapConfig"]


@dataclass
class MainReCapConfig(BaseConfig):
    """Top-level settings for the multi-stage ReCap loop."""

    _target_: str = "verl_vla.trainer.recap.config.MainReCapConfig"

    num_iterations: int = 1
    resume_iteration: int = 1
    resume_step: int = 1
    sft_dataset_path: str | None = None
    policy_eval: DictConfig | None = None
    collect_data: DictConfig | None = None
    compute_return: DictConfig | None = None
    train_value_model: DictConfig | None = None
    value_infer: DictConfig | None = None
    train_policy: DictConfig | None = None

    @classmethod
    def from_omega_conf(cls, config: DictConfig | Any) -> "MainReCapConfig":
        sft_dataset_path = OmegaConf.select(config, "sft_dataset_path", default=None)
        return cls(
            num_iterations=int(OmegaConf.select(config, "num_iterations", default=1)),
            resume_iteration=int(OmegaConf.select(config, "resume_iteration", default=1)),
            resume_step=int(OmegaConf.select(config, "resume_step", default=1)),
            sft_dataset_path=None if sft_dataset_path is None else str(sft_dataset_path),
            policy_eval=OmegaConf.select(config, "policy_eval", default=None),
            collect_data=OmegaConf.select(config, "collect_data", default=None),
            compute_return=OmegaConf.select(config, "compute_return", default=None),
            train_value_model=OmegaConf.select(config, "train_value_model", default=None),
            value_infer=OmegaConf.select(config, "value_infer", default=None),
            train_policy=OmegaConf.select(config, "train_policy", default=None),
        )

    def __post_init__(self):
        if self.num_iterations < 1:
            raise ValueError(f"recap.num_iterations must be at least 1, got {self.num_iterations}.")
        if self.resume_iteration < 1 or self.resume_iteration > self.num_iterations:
            raise ValueError(
                "recap.resume_iteration must be between 1 and recap.num_iterations, "
                f"got resume_iteration={self.resume_iteration}, num_iterations={self.num_iterations}."
            )
        if self.resume_step < 1 or self.resume_step > 6:
            raise ValueError(f"recap.resume_step must be between 1 and 6, got {self.resume_step}.")

    def should_run_stage(self, iteration: int, stage_step: int) -> bool:
        if iteration > self.resume_iteration:
            return True
        if iteration == self.resume_iteration:
            return stage_step >= self.resume_step
        return False

    def stage_enabled(self, stage_name: str, default: bool) -> bool:
        stage_config = getattr(self, stage_name)
        if stage_config is None:
            return default
        return bool(OmegaConf.select(stage_config, "enable", default=default))
