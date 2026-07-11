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

"""Configuration models for TrainCluster resources and worker composition."""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.utils import instantiate
from omegaconf import DictConfig
from verl.base_config import BaseConfig
from verl.workers.config.model import HFModelConfig

from verl_vla.workers.config import ActorDataKeysConfig, RolloutConfig
from verl_vla.workers.env.config import EnvWorkerConfig

__all__ = [
    "ActorRolloutRefConfig",
    "EnvTrainClusterConfig",
    "EnvTrainConfig",
    "EnvTrainResourceConfig",
    "EnvWorkerConfig",
    "EnvLoopTrainClusterConfig",
    "EnvLoopTrainResourceConfig",
    "EnvLoopConfig",
    "OptionalResourceConfig",
    "ResourceConfig",
    "SFTTrainClusterConfig",
    "SFTTrainResourceConfig",
    "TrainClusterCheckpointConfig",
]


@dataclass
class EnvLoopConfig(BaseConfig):
    """Configuration for async env-loop rollout scheduling."""

    pipeline_stage_num: int = 1
    max_interactions: int = 1

    def __post_init__(self):
        if self.pipeline_stage_num <= 0:
            raise ValueError(f"pipeline_stage_num must be positive, got {self.pipeline_stage_num}")
        if self.max_interactions <= 0:
            raise ValueError(f"max_interactions must be positive, got {self.max_interactions}")


@dataclass
class ResourceConfig(BaseConfig):
    """Resource-pool placement config."""

    device: str = "cuda"
    resource_label: str | None = None
    nnodes: int = 1
    gpus_per_node: int = 1
    workers_per_node: int = 1

    def __post_init__(self):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported resource device: {self.device}")
        if self.gpus_per_node < 0:
            raise ValueError(f"gpus_per_node must be non-negative, got {self.gpus_per_node}")
        if self.workers_per_node <= 0:
            raise ValueError(f"workers_per_node must be positive, got {self.workers_per_node}")
        if self.nnodes <= 0:
            raise ValueError(f"nnodes must be positive, got {self.nnodes}")


@dataclass
class OptionalResourceConfig(ResourceConfig):
    """Optional resource-pool placement config."""

    enabled: bool = False


@dataclass
class SFTTrainResourceConfig(BaseConfig):
    """Resource placement for SFT-style training."""

    controller_label: str | None = None
    model: ResourceConfig = field(default_factory=ResourceConfig)

    def __post_init__(self):
        if not isinstance(self.model, ResourceConfig):
            object.__setattr__(self, "model", instantiate(self.model))


@dataclass
class EnvTrainResourceConfig(BaseConfig):
    """Resource placement for env-only clusters."""

    controller_label: str | None = None
    env: ResourceConfig = field(default_factory=ResourceConfig)

    def __post_init__(self):
        if not isinstance(self.env, ResourceConfig):
            object.__setattr__(self, "env", instantiate(self.env))


@dataclass
class EnvLoopTrainResourceConfig(BaseConfig):
    """Resource placement for env-loop training or evaluation."""

    controller_label: str | None = None
    env: ResourceConfig = field(default_factory=ResourceConfig)
    model: ResourceConfig = field(default_factory=ResourceConfig)
    separate_rollout_model: OptionalResourceConfig = field(default_factory=OptionalResourceConfig)

    def __post_init__(self):
        if not isinstance(self.env, ResourceConfig):
            object.__setattr__(self, "env", instantiate(self.env))
        if not isinstance(self.model, ResourceConfig):
            object.__setattr__(self, "model", instantiate(self.model))
        if not isinstance(self.separate_rollout_model, OptionalResourceConfig):
            object.__setattr__(self, "separate_rollout_model", instantiate(self.separate_rollout_model))


@dataclass
class EnvTrainConfig(BaseConfig):
    """Environment-side config used by TrainCluster."""

    env_loop: EnvLoopConfig = field(default_factory=EnvLoopConfig)
    env_worker: EnvWorkerConfig = field(default_factory=EnvWorkerConfig)

    def __post_init__(self):
        if not isinstance(self.env_loop, EnvLoopConfig):
            object.__setattr__(self, "env_loop", instantiate(self.env_loop))
        if not isinstance(self.env_worker, EnvWorkerConfig):
            object.__setattr__(self, "env_worker", instantiate(self.env_worker, _recursive_=False))

    def validate_worker_layout(self, env_resource: ResourceConfig) -> None:
        stage_num = self.env_loop.pipeline_stage_num
        num_envs_per_worker = self.env_worker.num_envs
        processes_per_node = (
            env_resource.workers_per_node if env_resource.device == "cpu" else env_resource.gpus_per_node
        )
        env_worker_world_size = env_resource.nnodes * processes_per_node
        total_envs = env_worker_world_size * num_envs_per_worker
        if total_envs % stage_num != 0:
            raise ValueError(f"Total envs ({total_envs}) must be divisible by stage_num ({stage_num})")


@dataclass
class ActorRolloutRefConfig(BaseConfig):
    """Model, actor, and rollout config for actor/rollout workers."""

    model: HFModelConfig = field(default_factory=HFModelConfig)
    data_keys: ActorDataKeysConfig = field(default_factory=ActorDataKeysConfig)
    actor: DictConfig | None = None
    rollout: RolloutConfig | None = None

    @property
    def has_actor_and_rollout(self) -> bool:
        return self.actor is not None and self.rollout is not None


@dataclass
class TrainClusterCheckpointConfig(BaseConfig):
    """Checkpoint path, retention, and resume config owned by TrainCluster."""

    resume_mode: str = "auto"
    resume_from_path: str | None = None
    default_hdfs_dir: str | None = None
    default_local_dir: str = "checkpoints/${trainer.project_name}/${trainer.experiment_name}"
    del_local_ckpt_after_load: bool = False
    max_actor_ckpt_to_keep: int | None = None

    def __post_init__(self):
        if self.resume_mode not in {"auto", "disable", "resume_path"}:
            raise ValueError(f"Unsupported resume_mode: {self.resume_mode}")
        if self.max_actor_ckpt_to_keep is not None and self.max_actor_ckpt_to_keep <= 0:
            raise ValueError(
                f"max_actor_ckpt_to_keep must be positive when provided, got {self.max_actor_ckpt_to_keep}"
            )


@dataclass
class SFTTrainClusterConfig(BaseConfig):
    """Cluster config for SFT-style training."""

    resource: SFTTrainResourceConfig = field(default_factory=SFTTrainResourceConfig)
    actor_rollout_ref: ActorRolloutRefConfig = field(default_factory=ActorRolloutRefConfig)
    checkpoint: TrainClusterCheckpointConfig | None = None

    def __post_init__(self):
        if not isinstance(self.resource, SFTTrainResourceConfig):
            object.__setattr__(self, "resource", instantiate(self.resource, _recursive_=False))
        if not isinstance(self.actor_rollout_ref, ActorRolloutRefConfig):
            object.__setattr__(self, "actor_rollout_ref", instantiate(self.actor_rollout_ref, _recursive_=False))
        if self.checkpoint is not None and not isinstance(self.checkpoint, TrainClusterCheckpointConfig):
            object.__setattr__(self, "checkpoint", instantiate(self.checkpoint))


@dataclass
class EnvTrainClusterConfig(BaseConfig):
    """Cluster config for env-only data recording or teleoperation."""

    resource: EnvTrainResourceConfig = field(default_factory=EnvTrainResourceConfig)
    env: EnvTrainConfig = field(default_factory=EnvTrainConfig)
    checkpoint: TrainClusterCheckpointConfig | None = None

    def __post_init__(self):
        if not isinstance(self.resource, EnvTrainResourceConfig):
            object.__setattr__(self, "resource", instantiate(self.resource, _recursive_=False))
        if not isinstance(self.env, EnvTrainConfig):
            object.__setattr__(self, "env", instantiate(self.env, _recursive_=False))
        if self.checkpoint is not None and not isinstance(self.checkpoint, TrainClusterCheckpointConfig):
            object.__setattr__(self, "checkpoint", instantiate(self.checkpoint))
        self.env.validate_worker_layout(self.resource.env)


@dataclass
class EnvLoopTrainClusterConfig(BaseConfig):
    """Cluster config for env-loop training or evaluation."""

    resource: EnvLoopTrainResourceConfig = field(default_factory=EnvLoopTrainResourceConfig)
    env: EnvTrainConfig = field(default_factory=EnvTrainConfig)
    actor_rollout_ref: ActorRolloutRefConfig = field(default_factory=ActorRolloutRefConfig)
    checkpoint: TrainClusterCheckpointConfig | None = None

    def __post_init__(self):
        if not isinstance(self.resource, EnvLoopTrainResourceConfig):
            object.__setattr__(self, "resource", instantiate(self.resource, _recursive_=False))
        if not isinstance(self.env, EnvTrainConfig):
            object.__setattr__(self, "env", instantiate(self.env, _recursive_=False))
        if not isinstance(self.actor_rollout_ref, ActorRolloutRefConfig):
            object.__setattr__(
                self,
                "actor_rollout_ref",
                instantiate(self.actor_rollout_ref, _recursive_=False),
            )
        if self.checkpoint is not None and not isinstance(self.checkpoint, TrainClusterCheckpointConfig):
            object.__setattr__(self, "checkpoint", instantiate(self.checkpoint))
        if self.env.env_worker.device is None:
            object.__setattr__(self.env.env_worker, "device", self.resource.env.device)
        self.env.validate_worker_layout(self.resource.env)
