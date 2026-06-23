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
    "EMAConfig",
    "SFTDataKeysConfig",
    "BaseVLAActorConfig",
    "ActorConfig",
    "SFTActorConfig",
]


@dataclass
class SACConfig(BaseConfig):
    """Configuration for Soft Actor-Critic specific training behavior."""

    gamma: float = 0.99
    tau: float = 0.25
    force_critic_tau_one_in_warmup: bool = True
    td3_enabled: bool = False
    td3_bc_alpha: float = 2.5
    cql_enabled: bool = False
    cql_alpha: float = 1.0
    cql_temperature: float = 1.0
    cql_noise_scale: float | None = None
    skip_critic_update_when_actor_update: bool = False
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
        if self.td3_bc_alpha <= 0:
            raise ValueError(f"td3_bc_alpha must be positive, got {self.td3_bc_alpha}")
        if self.cql_alpha < 0:
            raise ValueError(f"cql_alpha must be non-negative, got {self.cql_alpha}")
        if self.cql_temperature <= 0:
            raise ValueError(f"cql_temperature must be positive, got {self.cql_temperature}")
        if self.cql_noise_scale is not None and self.cql_noise_scale < 0:
            raise ValueError(f"cql_noise_scale must be non-negative when provided, got {self.cql_noise_scale}")


@dataclass
class EMAConfig(BaseConfig):
    """Configuration for actor exponential moving average weights."""

    enable: bool = False
    decay: float = 0.995

    def __post_init__(self):
        if not 0 < self.decay < 1:
            raise ValueError(f"EMA decay must be in (0, 1), got {self.decay}")


@dataclass
class SFTDataKeysConfig(BaseConfig):
    """Batch field names used by the SFT worker."""

    action: str = "action"
    action_mask: str | None = "action_is_pad"
    target_value: str | None = None


@dataclass
class BaseVLAActorConfig(BaseConfig):
    """Shared actor config used by algorithm-specific VLA actor configs."""

    _mutable_fields = BaseConfig._mutable_fields | {
        "engine",
        "model_config",
    }

    strategy: str = "fsdp"

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: FSDPOptimizerConfig = field(default_factory=FSDPOptimizerConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    model_config: HFModelConfig | None = None

    def __post_init__(self):
        if self.strategy not in {"fsdp", "fsdp2"}:
            raise ValueError(f"Unsupported actor strategy: {self.strategy}")
        self.engine = self.fsdp_config


@dataclass
class ActorConfig(BaseVLAActorConfig):
    """SAC actor config with local FSDP/optimizer config types."""

    _target_: str = "verl_vla.workers.config.ActorConfig"

    sac: SACConfig = field(default_factory=SACConfig)

    critic_lr: float = 1e-4
    critic_weight_decay: float = 0.0
    critic_grad_clip: float | None = None

    warm_rollout_steps: int = 0
    critic_warmup_steps: int = 0
    critic_only_steps_after_rollout: int = 0
    actor_update_interval: int = 1

    actor_ema_enabled: bool = True
    actor_ema_decay: float = 0.995

    sac_mini_batch_size: int = 256
    online_replay_sample_batch_size: int | None = None
    offline_replay_sample_batch_size: int | None = None
    sac_micro_batch_size_per_gpu: int = 16

    replay_pool_save_interval: int = 500
    replay_pool_single_size: int = 1000
    offline_replay_pool_single_size: int = 1000
    replay_pool_save_dir: str = "/tmp/replay_pools"

    def __post_init__(self):
        super().__post_init__()

        if self.critic_lr <= 0:
            raise ValueError(f"critic_lr must be positive, got {self.critic_lr}")

        if self.critic_grad_clip is not None and self.critic_grad_clip <= 0:
            raise ValueError(f"critic_grad_clip must be positive when provided, got {self.critic_grad_clip}")

        if self.actor_update_interval <= 0:
            raise ValueError(f"actor_update_interval must be positive, got {self.actor_update_interval}")

        if self.critic_only_steps_after_rollout < 0:
            raise ValueError(
                f"critic_only_steps_after_rollout must be non-negative, got {self.critic_only_steps_after_rollout}"
            )

        if self.sac_mini_batch_size <= 0:
            raise ValueError(f"sac_mini_batch_size must be positive, got {self.sac_mini_batch_size}")

        if self.online_replay_sample_batch_size is not None and self.online_replay_sample_batch_size < 0:
            raise ValueError(
                "online_replay_sample_batch_size must be non-negative when provided, "
                f"got {self.online_replay_sample_batch_size}"
            )

        if self.offline_replay_sample_batch_size is not None and self.offline_replay_sample_batch_size < 0:
            raise ValueError(
                "offline_replay_sample_batch_size must be non-negative when provided, "
                f"got {self.offline_replay_sample_batch_size}"
            )

        if self.sac_micro_batch_size_per_gpu <= 0:
            raise ValueError(f"sac_micro_batch_size_per_gpu must be positive, got {self.sac_micro_batch_size_per_gpu}")


@dataclass
class SFTActorConfig(BaseVLAActorConfig):
    """SFT actor config kept separate from SAC-specific fields."""

    _target_: str = "verl_vla.workers.config.SFTActorConfig"

    ema: EMAConfig = field(default_factory=EMAConfig)
    data_keys: SFTDataKeysConfig = field(default_factory=SFTDataKeysConfig)

    mini_batch_size: int = 256
    micro_batch_size: int | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.mini_batch_size <= 0:
            raise ValueError(f"mini_batch_size must be positive, got {self.mini_batch_size}")

        if self.micro_batch_size is not None and self.micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive when provided, got {self.micro_batch_size}")
