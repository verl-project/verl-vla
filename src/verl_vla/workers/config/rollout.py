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
from typing import Optional

from verl.base_config import BaseConfig
from verl.workers.config import CheckpointEngineConfig, MultiTurnConfig, SamplingConfig

__all__ = [
    "RolloutACPConfig",
    "RolloutActionChunkStitchingConfig",
    "RolloutActionInterpolationConfig",
    "RolloutConfig",
]


@dataclass
class RolloutACPConfig(BaseConfig):
    """ACP prompt tagging used during policy rollout."""

    enable: bool = False
    positive_tag: str = "Advantage: positive"


@dataclass
class RolloutActionInterpolationConfig(BaseConfig):
    """Temporal upsampling applied to rollout action chunks before execution."""

    enable: bool = False
    factor: int = 2

    def __post_init__(self):
        if self.factor < 2:
            raise ValueError(f"action interpolation factor must be at least 2, got {self.factor}")


@dataclass
class RolloutActionChunkStitchingConfig(BaseConfig):
    """Overlap and execution horizons for consecutive rollout action plans."""

    enable: bool = False
    execution_steps: int = 80
    blend_steps: int = 20

    def __post_init__(self):
        if self.execution_steps <= 0:
            raise ValueError(f"action chunk execution_steps must be positive, got {self.execution_steps}")
        if self.blend_steps < 2:
            raise ValueError(f"action chunk blend_steps must be at least 2, got {self.blend_steps}")


@dataclass
class RolloutConfig(BaseConfig):
    """VLA rollout config with only fields required by VLA trainers/workers."""

    _target_: str = "verl_vla.workers.config.RolloutConfig"
    name: Optional[str] = None
    mode: str = "async_envloop"
    nnodes: int = 0
    n_gpus_per_node: int = 8
    n: int = 1
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    prompt_length: int = 512
    response_length: int = 512
    gpu_memory_utilization: float = 0.5
    free_cache_engine: bool = True
    tensor_model_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384
    load_format: str = "dummy"
    layered_summon: bool = False
    output_critic_value: bool = False
    acp: RolloutACPConfig = field(default_factory=RolloutACPConfig)
    action_interpolation: RolloutActionInterpolationConfig = field(default_factory=RolloutActionInterpolationConfig)
    action_chunk_stitching: RolloutActionChunkStitchingConfig = field(default_factory=RolloutActionChunkStitchingConfig)
    val_kwargs: SamplingConfig = field(default_factory=SamplingConfig)
    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)
    checkpoint_engine: CheckpointEngineConfig = field(default_factory=CheckpointEngineConfig)
