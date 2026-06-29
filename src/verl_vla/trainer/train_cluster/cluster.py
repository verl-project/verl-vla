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

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import ray
import torch
from ray.util.placement_group import remove_placement_group
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.utils import Role

from verl_vla.env_loop.env_loop import EnvLoop
from verl_vla.trainer.train_cluster.checkpoint import CheckpointHelper
from verl_vla.trainer.train_cluster.config import EnvLoopTrainClusterConfig, ResourceConfig, SFTTrainClusterConfig
from verl_vla.trainer.train_cluster.resource_pool import VLAResourcePoolManager
from verl_vla.utils.recorder import merge_lerobot_datasets
from verl_vla.utils.recorder.lerobot import REQUIRED_LEROBOT_META_FILES
from verl_vla.workers.engine import VLAActorRolloutRefWorker, VLAActorWorker, VLARolloutWorker
from verl_vla.workers.env.env_worker import EnvWorker

__all__ = [
    "TrainCluster",
    "VLAResourcePoolManager",
]

ROLE_TO_WORKER_NAME = {
    Role.Actor: "actor",
    Role.Rollout: "rollout",
    Role.ActorRollout: "actor_rollout",
    Role.Env: "env",
}


def _reduce_time_tensor(value: torch.Tensor, *, reduction: str) -> torch.Tensor:
    """Reduce chunk/substep dimensions while preserving batch and rollout time."""
    if value.ndim <= 2:
        return value

    while value.ndim > 2:
        if reduction == "any":
            value = value.any(dim=-1)
        elif reduction == "sum":
            value = value.sum(dim=-1)
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
    return value


class TrainCluster:
    def __init__(self, config: SFTTrainClusterConfig | EnvLoopTrainClusterConfig):
        if not isinstance(config, SFTTrainClusterConfig | EnvLoopTrainClusterConfig):
            raise TypeError(
                "TrainCluster config must be SFTTrainClusterConfig or EnvLoopTrainClusterConfig, "
                f"got {type(config).__name__}."
            )
        self.config = config
        self.cluster_type = "sft" if isinstance(config, SFTTrainClusterConfig) else "env_loop"
        self.resource_pool_spec: dict[str, list[int]] = {}
        self.role_to_pool: dict[Role, str] = {}
        self.cpu_pool_names: set[str] = set()
        self.resource_labels: dict[str, str] = {}
        self.resource_pool_manager: VLAResourcePoolManager | None = None
        self.worker_groups: dict[str, Any] = {}
        self.env_loop: EnvLoop | None = None
        self.checkpoint_helper: CheckpointHelper | None = None
        self._lerobot_collected_once = False

    def start(self) -> None:
        self._build_resource_pool_plan()
        self.resource_pool_manager = VLAResourcePoolManager(
            resource_pool_spec=self.resource_pool_spec,
            mapping=cast(dict[int, str], self.role_to_pool),
            cpu_pool_names=self.cpu_pool_names,
            resource_labels=self.resource_labels,
        )
        self._init_workers()

        if self.cluster_type == "env_loop":
            env_wg = self.worker_groups[ROLE_TO_WORKER_NAME[Role.Env]]
            rollout_wg = (
                self.worker_groups.get(ROLE_TO_WORKER_NAME[Role.ActorRollout])
                or self.worker_groups[ROLE_TO_WORKER_NAME[Role.Rollout]]
            )
            self.env_loop = EnvLoop(
                config=self.config.env.env_loop,
                switch_actor_rollout_mode=(
                    self.config.actor_rollout_ref.actor is not None
                    and not self.config.resource.separate_rollout_model.enabled
                ),
                rollout_wg=rollout_wg,
                env_wg=env_wg,
            )

        if self.config.checkpoint is not None:
            actor_config = self.config.actor_rollout_ref.actor
            assert actor_config is not None
            assert ROLE_TO_WORKER_NAME[Role.Actor] in self.worker_groups
            self.checkpoint_helper = CheckpointHelper(
                config=self.config.checkpoint,
                actor_config=actor_config,
                actor_worker_group=self.worker_groups[ROLE_TO_WORKER_NAME[Role.Actor]],
            )

    def shutdown(self) -> None:
        seen_actor_ids: set[str] = set()
        for worker_group in self.worker_groups.values():
            for worker in getattr(worker_group, "_workers", []):
                actor_id = getattr(getattr(worker, "_actor_id", None), "hex", lambda: None)()
                if actor_id is not None and actor_id in seen_actor_ids:
                    continue
                if actor_id is not None:
                    seen_actor_ids.add(actor_id)
                try:
                    ray.kill(worker, no_restart=True)
                except Exception:
                    pass

        if self.resource_pool_manager is not None:
            for resource_pool in self.resource_pool_manager.resource_pool_dict.values():
                for pg in getattr(resource_pool, "pgs", None) or []:
                    try:
                        remove_placement_group(pg)
                    except Exception:
                        pass
                resource_pool.pgs = None

        self.worker_groups = {}
        self.env_loop = None
        self.checkpoint_helper = None
        self.resource_pool_manager = None

    def _init_workers(self) -> None:
        if self.resource_pool_manager is None:
            raise RuntimeError("Resource pool manager is not initialized. Call start() first.")
        assert self.resource_pool_manager is not None
        self.resource_pool_manager.create_resource_pool()
        self.worker_groups = {}

        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        role_worker_mapping = self._role_worker_mapping()
        for role, pool_name in self.role_to_pool.items():
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            worker_name = ROLE_TO_WORKER_NAME[role]
            worker_config = self._worker_config(role)
            ray_cls_with_init = RayClassWithInitArgs(
                cls=ray.remote(role_worker_mapping[role]),
                config=worker_config,
                role=worker_name,
            )
            resource_pool_to_cls[resource_pool][worker_name] = ray_cls_with_init

        for resource_pool, class_dict in resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.config.resource.model.device,
            )
            self.worker_groups.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        for worker_name in [
            ROLE_TO_WORKER_NAME[Role.Actor],
            ROLE_TO_WORKER_NAME[Role.Rollout],
            ROLE_TO_WORKER_NAME[Role.ActorRollout],
        ]:
            if worker_name in self.worker_groups:
                self.worker_groups[worker_name].init_model()
        if ROLE_TO_WORKER_NAME[Role.Env] in self.worker_groups:
            self.worker_groups[ROLE_TO_WORKER_NAME[Role.Env]].init_worker()

    def _build_resource_pool_plan(self) -> None:
        self.resource_pool_spec = {}
        self.role_to_pool = {}
        self.cpu_pool_names = set()
        self.resource_labels = {}

        if self.cluster_type == "sft":
            self._add_resource_pool(pool_name="train_rollout_pool", resource=self.config.resource.model)
            self.role_to_pool = {Role.Actor: "train_rollout_pool"}

        elif self.cluster_type == "env_loop":
            resource = self.config.resource
            env_pool_name = "env_cpu_pool" if resource.env.device == "cpu" else "env_gpu_pool"
            self._add_resource_pool(pool_name=env_pool_name, resource=resource.env)
            self.role_to_pool[Role.Env] = env_pool_name

            if resource.separate_rollout_model.enabled:
                if self.config.actor_rollout_ref.actor is None:
                    raise ValueError(
                        "Env-loop train cluster with separate_rollout_model enabled requires actor config."
                    )
                self._add_resource_pool(pool_name="train_pool", resource=resource.model)
                self._add_resource_pool(pool_name="rollout_pool", resource=resource.separate_rollout_model)
                self.role_to_pool[Role.Actor] = "train_pool"
                self.role_to_pool[Role.Rollout] = "rollout_pool"
            else:
                self._add_resource_pool(pool_name="train_rollout_pool", resource=resource.model)
                role = Role.ActorRollout if self.config.actor_rollout_ref.actor is not None else Role.Rollout
                self.role_to_pool[role] = "train_rollout_pool"
        else:
            raise ValueError(f"Unsupported train cluster type: {self.cluster_type}")

    def _add_resource_pool(self, pool_name: str, resource: ResourceConfig) -> None:
        processes_per_node = resource.workers_per_node if resource.device == "cpu" else resource.gpus_per_node
        self.resource_pool_spec[pool_name] = [processes_per_node] * resource.nnodes
        if resource.device == "cpu":
            self.cpu_pool_names.add(pool_name)
        if resource.resource_label is not None:
            self.resource_labels[pool_name] = resource.resource_label

    def _role_worker_mapping(self):
        if self.cluster_type == "sft":
            return {Role.Actor: VLAActorRolloutRefWorker}

        elif self.cluster_type == "env_loop":
            separate_rollout_model = self.config.resource.separate_rollout_model.enabled
            has_actor = self.config.actor_rollout_ref.actor is not None

            if separate_rollout_model:
                if not has_actor:
                    raise ValueError(
                        "Env-loop train cluster with separate_rollout_model enabled requires actor config."
                    )
                return {
                    Role.Actor: VLAActorWorker,
                    Role.Rollout: VLARolloutWorker,
                    Role.Env: EnvWorker,
                }

            if has_actor:
                return {
                    Role.ActorRollout: VLAActorRolloutRefWorker,
                    Role.Env: EnvWorker,
                }

            return {
                Role.Rollout: VLARolloutWorker,
                Role.Env: EnvWorker,
            }
        else:
            raise ValueError(f"Unsupported train cluster type: {self.cluster_type}")

    def _worker_config(self, role: Role):
        if role == Role.Env:
            if not isinstance(self.config, EnvLoopTrainClusterConfig) or self.config.env is None:
                raise ValueError("Env worker requires EnvLoopTrainClusterConfig.env.")
            return self.config.env
        elif role in {Role.Actor, Role.Rollout, Role.ActorRollout}:
            return self.config.actor_rollout_ref
        else:
            raise ValueError(f"Unsupported worker role: {role}")

    @property
    def train_world_size(self) -> int:
        return int(self.worker_groups[ROLE_TO_WORKER_NAME[Role.Actor]].world_size)

    def rollout(self) -> tuple[DataProto, dict[str, dict[str, Any]]]:
        if self.cluster_type != "env_loop":
            raise RuntimeError("rollout is only wired for env-loop train clusters.")

        reset_future = self.worker_groups["env"].reset_envs_to_state_ids(DataProto.from_dict())
        assert self.env_loop is not None
        output = self.env_loop.generate_sequences(reset_future)
        return output, self._collect_lerobot_datasets()

    def _collect_lerobot_datasets(self) -> dict[str, dict[str, Any]]:
        if not isinstance(self.config, EnvLoopTrainClusterConfig):
            return {}

        recorder_cfg = self.config.env.env_worker.recorder
        if not recorder_cfg.enable or not recorder_cfg.lerobot.enable:
            return {}

        env_wg = self.worker_groups[ROLE_TO_WORKER_NAME[Role.Env]]
        rank_datasets = [dataset for dataset in env_wg.pop_lerobot_dataset() if dataset is not None]
        root = Path(recorder_cfg.lerobot.root)
        repo_id = recorder_cfg.lerobot.repo_id
        existing_root = root / repo_id
        collected_datasets: dict[str, dict[str, Any]] = {}
        if all((existing_root / path).exists() for path in REQUIRED_LEROBOT_META_FILES):
            collected_datasets["existing_dataset"] = {"root": existing_root, "repo_id": repo_id}

        collected_dataset = self._merge_rank_lerobot_datasets(rank_datasets)
        if collected_dataset:
            collected_datasets["collected_dataset"] = collected_dataset
        return collected_datasets

    def _merge_rank_lerobot_datasets(self, rank_datasets: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rank_datasets:
            return None
        if not isinstance(self.config, EnvLoopTrainClusterConfig):
            return None

        recorder_cfg = self.config.env.env_worker.recorder
        repo_id = f"{recorder_cfg.lerobot.repo_id}_collected"
        collected_dataset = merge_lerobot_datasets(
            roots=[dataset["root"] for dataset in rank_datasets],
            output_root=Path(recorder_cfg.lerobot.root) / repo_id,
            repo_id=repo_id,
            repo_ids=[dataset["repo_id"] for dataset in rank_datasets],
            overwrite=not self._lerobot_collected_once,
            append=self._lerobot_collected_once,
            video_files_size_in_mb=recorder_cfg.lerobot.video_files_size_in_mb,
        )
        self._lerobot_collected_once = True
        return collected_dataset

    def train(self, data: DataProto, *, async_update: bool = True) -> Any:
        actor_wg = self.worker_groups[ROLE_TO_WORKER_NAME[Role.Actor]]
        if async_update:
            return actor_wg.update_actor_async(data)
        return actor_wg.update_actor(data)

    def eval(
        self,
        *,
        max_episodes: int | None = None,
    ) -> dict[str, float]:
        if self.cluster_type != "env_loop":
            raise RuntimeError("eval is only wired for env-loop train clusters.")

        env_wg = self.worker_groups[ROLE_TO_WORKER_NAME[Role.Env]]
        benchmark_size = int(env_wg.get_eval_benchmark_size()[0])
        target_episodes = benchmark_size if max_episodes is None else int(max_episodes)

        eval_step = 0
        rollout_metric_lists: dict[str, list[float]] = {}
        trajectory_records: list[dict[str, float | int | bool]] = []
        carry_state: dict[str, np.ndarray | None] = {"length": None, "reward": None, "task_id": None}
        while len(trajectory_records) < target_episodes:
            reset_eval = eval_step == 0
            reset_future = env_wg.reset_envs_to_state_ids(
                DataProto.from_dict(
                    meta_info={
                        "mode": "eval",
                        "reset_eval": reset_eval,
                    },
                )
            )
            rollout_output = self.env_loop.generate_sequences(reset_future, eval=True)
            for key, value in rollout_output.meta_info.get("metrics", {}).items():
                rollout_metric_lists.setdefault(key, []).append(float(value))

            trajectory_records.extend(
                self._collect_eval_trajectory_records(
                    rollout_output,
                    remaining=target_episodes - len(trajectory_records),
                    carry_state=carry_state,
                )
            )
            eval_step += 1

        metrics = self._eval_metrics_from_trajectory_records(
            trajectory_records,
            benchmark_size=benchmark_size,
        )
        for key, values in rollout_metric_lists.items():
            metrics[key] = float(np.mean(values)) if values else 0.0
        return metrics

    def _collect_eval_trajectory_records(
        self,
        output: DataProto,
        *,
        remaining: int,
        carry_state: dict[str, np.ndarray | None],
    ) -> list[dict[str, float | int | bool]]:
        if remaining <= 0:
            return []

        done_steps = _reduce_time_tensor(output.batch["next.done"].bool(), reduction="any")
        reward_steps = _reduce_time_tensor(output.batch["next.reward"].float(), reduction="sum")
        if done_steps.ndim == 1:
            done_steps = done_steps[:, None]
            reward_steps = reward_steps[:, None]
        task_id_steps = output.non_tensor_batch.get("obs.task_id")
        if task_id_steps is not None:
            task_id_steps = np.asarray(task_id_steps)
            if task_id_steps.ndim == 1:
                task_id_steps = task_id_steps[:, None]

        records: list[dict[str, float | int | bool]] = []
        batch_size = int(done_steps.shape[0])

        carry_lengths = carry_state.get("length")
        carry_rewards = carry_state.get("reward")
        carry_task_ids = carry_state.get("task_id")
        if carry_lengths is None or carry_lengths.shape[0] != batch_size:
            carry_lengths = np.zeros(batch_size, dtype=np.int64)
            carry_rewards = np.zeros(batch_size, dtype=np.float32)
            carry_task_ids = np.full(batch_size, -1, dtype=np.int64)
            carry_state["length"] = carry_lengths
            carry_state["reward"] = carry_rewards
            carry_state["task_id"] = carry_task_ids
        assert carry_rewards is not None
        assert carry_task_ids is not None

        for batch_idx in range(done_steps.shape[0]):
            if len(records) >= remaining:
                break

            done_row = done_steps[batch_idx].reshape(-1)
            reward_row = reward_steps[batch_idx].reshape(-1)
            done_indices = torch.nonzero(done_row, as_tuple=False).flatten().tolist()
            start_idx = 0
            segment_prefix_length = int(carry_lengths[batch_idx])
            segment_prefix_return = float(carry_rewards[batch_idx])
            segment_task_id = int(carry_task_ids[batch_idx])
            if segment_prefix_length == 0 and task_id_steps is not None and task_id_steps.shape[1] > 0:
                segment_task_id = int(task_id_steps[batch_idx, 0])

            for done_idx in done_indices:
                if len(records) >= remaining:
                    break

                segment = slice(start_idx, done_idx + 1)
                trajectory_return = segment_prefix_return + float(reward_row[segment].sum().item())
                trajectory_length = segment_prefix_length + done_idx - start_idx + 1
                records.append(
                    {
                        "length": int(trajectory_length),
                        "return": float(trajectory_return),
                        "success": bool(trajectory_return > 0.0),
                        "task_id": int(segment_task_id),
                    }
                )

                start_idx = done_idx + 1
                segment_prefix_length = 0
                segment_prefix_return = 0.0
                if task_id_steps is not None and start_idx < task_id_steps.shape[1]:
                    segment_task_id = int(task_id_steps[batch_idx, start_idx])

            if len(records) < remaining:
                remaining_steps = int(done_row.numel()) - start_idx
                carry_lengths[batch_idx] = segment_prefix_length + remaining_steps
                carry_rewards[batch_idx] = segment_prefix_return + float(reward_row[start_idx:].sum().item())
                carry_task_ids[batch_idx] = segment_task_id

        return records

    def _eval_metrics_from_trajectory_records(
        self,
        records: list[dict[str, float | int | bool]],
        *,
        benchmark_size: int,
    ) -> dict[str, float]:
        trajectory_count = len(records)
        success_records = [record for record in records if bool(record["success"])]
        success_count = len(success_records)
        metrics = {
            "val/benchmark_size": float(benchmark_size),
            "val/avg_return": (float(np.mean([float(record["return"]) for record in records])) if records else 0.0),
            "val/avg_success_trajectory_length": (
                float(np.mean([int(record["length"]) for record in success_records])) if success_records else 0.0
            ),
            "val/trajectory_count": float(trajectory_count),
            "val/success_trajectory_count": float(success_count),
            "val/failed_trajectory_count": float(trajectory_count - success_count),
            "val/trajectory_success_rate": (float(success_count / trajectory_count) if trajectory_count > 0 else 0.0),
        }
        task_ids = sorted({int(record["task_id"]) for record in records if int(record["task_id"]) >= 0})
        for task_id in task_ids:
            task_records = [record for record in records if int(record["task_id"]) == task_id]
            task_success_records = [record for record in task_records if bool(record["success"])]
            trajectory_count = len(task_records)
            success_count = len(task_success_records)
            metrics[f"val/per_task_trajectory_count/task_{task_id}"] = float(trajectory_count)
            metrics[f"val/per_task_success_trajectory_count/task_{task_id}"] = float(success_count)
            metrics[f"val/per_task_failed_trajectory_count/task_{task_id}"] = float(trajectory_count - success_count)
            metrics[f"val/per_task_success_rate/task_{task_id}"] = (
                float(success_count / trajectory_count) if trajectory_count > 0 else 0.0
            )
            metrics[f"val/per_task_avg_success_trajectory_length/task_{task_id}"] = (
                float(np.mean([int(record["length"]) for record in task_success_records]))
                if task_success_records
                else 0.0
            )
        return metrics

    def update_weights(self, *args: Any, **kwargs: Any) -> Any: ...

    def load_checkpoint(self) -> tuple[int, str] | None:
        assert self.checkpoint_helper is not None
        return self.checkpoint_helper.load()

    def save_checkpoint(
        self,
        global_step: int,
        save_extra_state: Callable[[str], None] | None = None,
    ) -> None:
        assert self.checkpoint_helper is not None
        self.checkpoint_helper.save(global_step, save_extra_state=save_extra_state)
