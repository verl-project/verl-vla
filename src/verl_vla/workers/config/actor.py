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
    "SACCriticConfig",
    "SACReplayConfig",
    "SACTD3Config",
    "SACCQLConfig",
    "EMAConfig",
    "ACPConfig",
    "ActorDataKeysConfig",
    "BaseVLAActorConfig",
    "ActorConfig",
    "SFTActorConfig",
]


@dataclass
class SACConfig(BaseConfig):
    """Configuration for Soft Actor-Critic specific training behavior."""

    initial_alpha: float = 0.0
    auto_entropy: bool = False
    alpha_type: str = "exp"
    alpha_lr: float = 3e-4
    target_entropy: float = -64.0

    def __post_init__(self):
        valid_alpha_types = ["exp", "softplus"]
        if self.alpha_type not in valid_alpha_types:
            raise ValueError(f"Invalid alpha_type: {self.alpha_type}. Must be one of {valid_alpha_types}")


@dataclass
class SACTD3Config(BaseConfig):
    """Configuration for optional TD3+BC actor loss."""

    enabled: bool = False
    bc_alpha: float = 2.5

    def __post_init__(self):
        if self.bc_alpha <= 0:
            raise ValueError(f"td3 bc_alpha must be positive, got {self.bc_alpha}")


@dataclass
class SACCQLConfig(BaseConfig):
    """Configuration for optional Conservative Q-Learning critic regularization."""

    enabled: bool = False
    alpha: float = 1.0
    temperature: float = 1.0
    noise_scale: float | None = None

    def __post_init__(self):
        if self.alpha < 0:
            raise ValueError(f"cql alpha must be non-negative, got {self.alpha}")
        if self.temperature <= 0:
            raise ValueError(f"cql temperature must be positive, got {self.temperature}")
        if self.noise_scale is not None and self.noise_scale < 0:
            raise ValueError(f"cql noise_scale must be non-negative when provided, got {self.noise_scale}")


@dataclass
class SACCriticConfig(BaseConfig):
    """Configuration for SAC critic optimizer and update schedule."""

    gamma: float = 0.99
    tau: float = 0.25
    force_target_tau_one_in_warmup: bool = True
    skip_update_when_actor_update: bool = False
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float | None = None
    warmup_steps: int = 0
    only_steps_after_rollout: int = 0

    def __post_init__(self):
        if self.gamma <= 0:
            raise ValueError(f"critic gamma must be positive, got {self.gamma}")
        if self.tau <= 0:
            raise ValueError(f"critic tau must be positive, got {self.tau}")
        if self.lr <= 0:
            raise ValueError(f"critic lr must be positive, got {self.lr}")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"critic grad_clip must be positive when provided, got {self.grad_clip}")
        if self.warmup_steps < 0:
            raise ValueError(f"critic warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.only_steps_after_rollout < 0:
            raise ValueError(
                f"critic only_steps_after_rollout must be non-negative, got {self.only_steps_after_rollout}"
            )


@dataclass
class SACReplayConfig(BaseConfig):
    """Configuration for SAC replay sampling, batching, and persistence."""

    critic_positive_sample_ratio: float = 0.5
    actor_positive_sample_ratio: float = 0.5
    online_sample_batch_size: int | None = None
    offline_sample_batch_size: int | None = None
    save_interval: int = 500
    online_single_size: int = 1000
    offline_single_size: int = 1000
    save_dir: str = "/tmp/replay_pools"

    def __post_init__(self):
        if not 0 <= self.critic_positive_sample_ratio <= 1:
            raise ValueError(
                f"replay critic_positive_sample_ratio must be in [0, 1], got {self.critic_positive_sample_ratio}"
            )
        if not 0 <= self.actor_positive_sample_ratio <= 1:
            raise ValueError(
                f"replay actor_positive_sample_ratio must be in [0, 1], got {self.actor_positive_sample_ratio}"
            )
        if self.online_sample_batch_size is not None and self.online_sample_batch_size < 0:
            raise ValueError(
                "replay online_sample_batch_size must be non-negative when provided, "
                f"got {self.online_sample_batch_size}"
            )
        if self.offline_sample_batch_size is not None and self.offline_sample_batch_size < 0:
            raise ValueError(
                "replay offline_sample_batch_size must be non-negative when provided, "
                f"got {self.offline_sample_batch_size}"
            )
        if self.save_interval <= 0:
            raise ValueError(f"replay save_interval must be positive, got {self.save_interval}")
        if self.online_single_size <= 0:
            raise ValueError(f"replay online_single_size must be positive, got {self.online_single_size}")
        if self.offline_single_size <= 0:
            raise ValueError(f"replay offline_single_size must be positive, got {self.offline_single_size}")


@dataclass
class EMAConfig(BaseConfig):
    """Configuration for actor exponential moving average weights."""

    enable: bool = False
    decay: float = 0.995

    def __post_init__(self):
        if not 0 < self.decay < 1:
            raise ValueError(f"EMA decay must be in (0, 1), got {self.decay}")


@dataclass
class ACPConfig(BaseConfig):
    """Configuration for advantage-conditioned prompt tagging."""

    enable: bool = False
    indicator_dropout_prob: float = 0.0
    positive_tag: str = "Advantage: positive"
    negative_tag: str = "Advantage: negative"

    def __post_init__(self):
        if not 0 <= self.indicator_dropout_prob <= 1:
            raise ValueError(f"ACP indicator_dropout_prob must be in [0, 1], got {self.indicator_dropout_prob}")


@dataclass
class ActorDataKeysConfig(BaseConfig):
    """Batch field names shared by actor training and rollout."""

    task: str = "task"
    action: str = "action"
    action_mask: str | None = "action_is_pad"
    indicator: str | None = None
    target_value: str | None = None


@dataclass
class BaseVLAActorConfig(BaseConfig):
    """Shared actor config used by algorithm-specific VLA actor configs."""

    _mutable_fields = BaseConfig._mutable_fields | {
        "engine",
        "data_keys",
        "model_config",
    }

    strategy: str = "fsdp"

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: FSDPOptimizerConfig = field(default_factory=FSDPOptimizerConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    data_keys: ActorDataKeysConfig = field(default_factory=ActorDataKeysConfig)
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
    td3: SACTD3Config = field(default_factory=SACTD3Config)
    cql: SACCQLConfig = field(default_factory=SACCQLConfig)
    critic: SACCriticConfig = field(default_factory=SACCriticConfig)
    replay: SACReplayConfig = field(default_factory=SACReplayConfig)

    actor_update_interval: int = 1
    ema_decay: float | None = None
    mini_batch_size: int = 256
    micro_batch_size: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.actor_update_interval <= 0:
            raise ValueError(f"actor_update_interval must be positive, got {self.actor_update_interval}")
        if self.ema_decay is not None and not 0 < self.ema_decay < 1:
            raise ValueError(f"ema_decay must be in (0, 1) when provided, got {self.ema_decay}")
        if self.mini_batch_size <= 0:
            raise ValueError(f"mini_batch_size must be positive, got {self.mini_batch_size}")
        if self.micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive, got {self.micro_batch_size}")


@dataclass
class SFTActorConfig(BaseVLAActorConfig):
    """SFT actor config kept separate from SAC-specific fields."""

    _target_: str = "verl_vla.workers.config.SFTActorConfig"

    ema: EMAConfig = field(default_factory=EMAConfig)
    acp: ACPConfig = field(default_factory=ACPConfig)

    mini_batch_size: int = 256
    micro_batch_size: int | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.mini_batch_size <= 0:
            raise ValueError(f"mini_batch_size must be positive, got {self.mini_batch_size}")

        if self.micro_batch_size is not None and self.micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive when provided, got {self.micro_batch_size}")
