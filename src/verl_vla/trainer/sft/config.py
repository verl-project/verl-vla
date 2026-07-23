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

from verl_vla.utils.early_stopping import TrendEarlyStoppingConfig

__all__ = ["SFTTrainerConfig"]


@dataclass
class SFTTrainerConfig(BaseConfig):
    """Configuration for the VLA SFT trainer loop."""

    _target_: str = "verl_vla.trainer.sft.config.SFTTrainerConfig"

    project_name: str = "vla-sft"
    experiment_name: str = "lerobot-preview"
    logger: list[str] = field(default_factory=lambda: ["console"])
    total_epochs: int = 30
    save_freq: int = -1
    save_last: bool = True
    esi_redundant_time: int = 0
    resume_dataloader_state: bool = True
    early_stopping: TrendEarlyStoppingConfig = field(default_factory=TrendEarlyStoppingConfig)

    def __post_init__(self):
        if self.total_epochs <= 0:
            raise ValueError(f"total_epochs must be positive, got {self.total_epochs}")
