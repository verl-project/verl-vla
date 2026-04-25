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
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig
from verl.workers.config.model import HFModelConfig

from .engine import FSDPEngineConfig
from .optimizer import FSDPOptimizerConfig

__all__ = [
    "SACConfig",
    "ActorConfig",
]


@dataclass
class SACConfig(BaseConfig):
    """Configuration for Soft Actor-Critic specific training behavior."""

    gamma: float = 0.99
    tau: float = 0.25
    force_critic_tau_one_in_warmup: bool = True
    critic_target_ema_dynamic_enabled: bool = False
    critic_target_ema_strength_initial: float = 0.2
    critic_target_ema_strength_final: float = 0.98
    critic_target_ema_schedule_method: str = "square"
    bc_loss_coef: float = 0.0
    initial_alpha: float = 0.0
    critic_replay_positive_sample_ratio: float = 0.5
    actor_replay_positive_sample_ratio: float = 0.5
    auto_entropy: bool = False
    alpha_type: str = "exp"
    alpha_lr: float = 3e-4
    target_entropy: float = -64.0

    def __post_init__(self):
        valid_alpha_types = ["exp", "softplus"]
        if self.alpha_type not in valid_alpha_types:
            raise ValueError(f"Invalid alpha_type: {self.alpha_type}. Must be one of {valid_alpha_types}")


@dataclass
class ActorConfig(BaseConfig):
    """SAC actor config with local FSDP/optimizer config types."""

    _mutable_fields = BaseConfig._mutable_fields | {
        "engine",
        "model_config",
    }

    _target_: str = "verl_vla.workers.config.ActorConfig"

    strategy: str = "fsdp"
    use_kl_loss: bool = False

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: FSDPOptimizerConfig = field(default_factory=FSDPOptimizerConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    model_config: HFModelConfig = field(default_factory=BaseConfig)

    sac: SACConfig = field(default_factory=SACConfig)

    critic_lr: float = 1e-4
    critic_weight_decay: float = 0.0

    warm_rollout_steps: int = 0
    critic_warmup_steps: int = 0
    actor_update_interval: int = 1

    actor_ema_enabled: bool = True
    actor_ema_decay: float = 0.995
    actor_ema_dynamic_enabled: bool = False
    actor_ema_strength_initial: float = 0.4
    actor_ema_strength_final: float = 0.98
    actor_ema_schedule_method: str = "square"

    sac_mini_batch_size: int = 256
    sac_micro_batch_size_per_gpu: int = 16

    replay_pool_save_interval: int = 500
    replay_pool_single_size: int = 1000
    replay_pool_save_dir: str = "/tmp/replay_pools"

    def __post_init__(self):
        if self.strategy not in {"fsdp", "fsdp2"}:
            raise ValueError(f"Unsupported actor strategy: {self.strategy}")

        self.engine = self.fsdp_config

        if self.critic_lr <= 0:
            raise ValueError(f"critic_lr must be positive, got {self.critic_lr}")

        if self.actor_update_interval <= 0:
            raise ValueError(f"actor_update_interval must be positive, got {self.actor_update_interval}")

        if self.sac_mini_batch_size <= 0:
            raise ValueError(f"sac_mini_batch_size must be positive, got {self.sac_mini_batch_size}")

        if self.sac_micro_batch_size_per_gpu <= 0:
            raise ValueError(f"sac_micro_batch_size_per_gpu must be positive, got {self.sac_micro_batch_size_per_gpu}")
