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

__all__ = ["SFTTrainerConfig"]


@dataclass
class SFTTrainerConfig(BaseConfig):
    """Configuration for the VLA SFT trainer loop and checkpoint behavior."""

    _target_: str = "verl_vla.trainer.sft.config.SFTTrainerConfig"

    project_name: str = "vla-sft"
    experiment_name: str = "lerobot-preview"
    logger: list[str] = field(default_factory=lambda: ["console"])
    nnodes: int = 1
    n_gpus_per_node: int = 1
    device: str = "cuda"
    total_epochs: int = 30
    save_freq: int = -1
    save_last: bool = False
    esi_redundant_time: int = 0
    resume_mode: str = "auto"
    resume_from_path: str | None = None
    default_hdfs_dir: str | None = None
    default_local_dir: str = "checkpoints/${trainer.project_name}/${trainer.experiment_name}"
    del_local_ckpt_after_load: bool = False
    max_actor_ckpt_to_keep: int | None = None
    remove_previous_ckpt_in_save: bool = False

    def __post_init__(self):
        if self.nnodes <= 0:
            raise ValueError(f"nnodes must be positive, got {self.nnodes}")
        if self.n_gpus_per_node <= 0:
            raise ValueError(f"n_gpus_per_node must be positive, got {self.n_gpus_per_node}")
        if self.total_epochs <= 0:
            raise ValueError(f"total_epochs must be positive, got {self.total_epochs}")
        if self.resume_mode not in {"auto", "disable", "resume_path"}:
            raise ValueError(f"Unsupported resume_mode: {self.resume_mode}")
        if self.max_actor_ckpt_to_keep is not None and self.max_actor_ckpt_to_keep <= 0:
            raise ValueError(
                f"max_actor_ckpt_to_keep must be positive when provided, got {self.max_actor_ckpt_to_keep}"
            )
