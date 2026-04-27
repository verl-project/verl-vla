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
from typing import Any, Optional

from verl.base_config import BaseConfig
from verl.workers.config import QATEngineConfig

__all__ = [
    "EngineRouterReplayConfig",
    "EngineConfig",
    "FSDPEngineConfig",
]


@dataclass
class EngineRouterReplayConfig(BaseConfig):
    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class EngineConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {
        "use_dynamic_bsz",
        "max_token_len_per_gpu",
        "micro_batch_size_per_gpu",
        "infer_max_token_len_per_gpu",
        "infer_micro_batch_size_per_gpu",
        "use_fused_kernels",
        "use_remove_padding",
        "forward_only",
        "param_offload",
    }

    param_offload: bool = False
    optimizer_offload: bool = False
    grad_offload: bool = False
    forward_only: bool = False
    strategy: str | None = None
    dtype: str = "bfloat16"
    use_dynamic_bsz: bool = True
    max_token_len_per_gpu: int | None = None
    micro_batch_size_per_gpu: int | None = None
    infer_max_token_len_per_gpu: int | None = None
    infer_micro_batch_size_per_gpu: int | None = None
    use_fused_kernels: bool = False
    use_remove_padding: bool = True
    seed: int = 42
    full_determinism: bool = False
    router_replay: EngineRouterReplayConfig = field(default_factory=EngineRouterReplayConfig)


@dataclass
class FSDPEngineConfig(EngineConfig):
    _mutable_fields = EngineConfig._mutable_fields | {"ulysses_sequence_parallel_size"}

    wrap_policy: dict[str, Any] = field(default_factory=dict)
    offload_policy: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False
    model_dtype: str = "fp32"
    use_orig_params: bool = False
    mixed_precision: Optional[dict[str, Any]] = None
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    use_torch_compile: bool = True
    entropy_checkpointing: bool = False
    strategy: str = "fsdp"
    qat: QATEngineConfig = field(default_factory=QATEngineConfig)

    def __post_init__(self):
        if self.strategy not in ["fsdp", "fsdp2"]:
            raise ValueError(f"strategy {self.strategy} not supported")
