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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import ray
import torch
from ray.util.placement_group import remove_placement_group
from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.utils import Role
from verl.utils.config import omega_conf_to_dataclass

from verl_vla.env_loop.env_loop import EnvLoop
from verl_vla.trainer.train_cluster.checkpoint import CheckpointHelper
from verl_vla.trainer.train_cluster.config import EnvLoopTrainClusterConfig, ResourceConfig, SFTTrainClusterConfig
from verl_vla.trainer.train_cluster.resource_pool import VLAResourcePoolManager
from verl_vla.utils.recorder import merge_lerobot_datasets
from verl_vla.utils.recorder.lerobot import REQUIRED_LEROBOT_META_FILES
from verl_vla.workers.engine import VLAActorRolloutRefWorker, VLAActorWorker, VLARolloutWorker
from verl_vla.workers.env.env_worker import EnvWorker
from verl_vla.workers.rollout.vla_replica import VLARolloutReplica

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


@dataclass
class RolloutState:
    reset_future: Any | None = None
    carry_state: dict[str, np.ndarray | None] = field(
        default_factory=lambda: {"length": None, "reward": None, "task_id": None}
    )
    lerobot_collected_once: bool = False


@ray.remote
def ray_rollout_once(
    env_loop: EnvLoop,
    config: EnvLoopTrainClusterConfig,
    state: RolloutState,
):
    return TrainCluster._rollout_once(
        env_loop,
        config=config,
        state=state,
    )


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
        self.checkpoint_engine_manager: CheckpointEngineManager | None = None
        self.rollout_state = RolloutState()
        self._pending_rollout_ref: ray.ObjectRef | None = None

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
            self._init_checkpoint_engine_manager()

        if self.config.checkpoint is not None:
            actor_config = self.config.actor_rollout_ref.actor
            assert actor_config is not None
            self.checkpoint_helper = CheckpointHelper(
                config=self.config.checkpoint,
                actor_config=actor_config,
                actor_worker_group=self.actor_worker_group,
            )

    def shutdown(self) -> None:
        if self._pending_rollout_ref is not None:
            try:
                ray.cancel(self._pending_rollout_ref, force=True)
                ray.get(self._pending_rollout_ref, timeout=5)
            except Exception:
                pass
            self._pending_rollout_ref = None

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
        self.checkpoint_engine_manager = None
        self.resource_pool_manager = None
        self.rollout_state = RolloutState()
        self._pending_rollout_ref = None

    def _init_workers(self) -> None:
        if self.resource_pool_manager is None:
            raise RuntimeError("Resource pool manager is not initialized. Call start() first.")
        assert self.resource_pool_manager is not None
        self.resource_pool_manager.create_resource_pool()
        self.worker_groups = {}

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
            self.worker_groups[worker_name] = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                device_name=self.config.resource.model.device,
            )

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

    def _iter_unique_worker_groups(self):
        seen_worker_groups: set[int] = set()
        for worker_name, worker_group in self.worker_groups.items():
            worker_group_id = id(worker_group)
            if worker_group_id in seen_worker_groups:
                continue
            seen_worker_groups.add(worker_group_id)
            yield worker_name, worker_group

    def start_profiling(self, step: int) -> None:
        for worker_name, worker_group in self._iter_unique_worker_groups():
            start_profile = getattr(worker_group, "start_profile", None)
            if callable(start_profile):
                start_profile(role=worker_name, profile_step=step)

    def stop_profiling(self) -> None:
        for _worker_name, worker_group in self._iter_unique_worker_groups():
            stop_profile = getattr(worker_group, "stop_profile", None)
            if callable(stop_profile):
                stop_profile()

    def dump_memory_snapshot(self, *, tag: str, sub_dir: str) -> None:
        dump_memory_snapshot = getattr(self.actor_worker_group, "dump_memory_snapshot", None)
        if callable(dump_memory_snapshot):
            dump_memory_snapshot(tag=tag, sub_dir=sub_dir)

    def _init_checkpoint_engine_manager(self) -> None:
        if not isinstance(self.config, EnvLoopTrainClusterConfig):
            return
        if not self.config.resource.separate_rollout_model.enabled:
            return
        if ROLE_TO_WORKER_NAME[Role.Rollout] not in self.worker_groups:
            return

        rollout_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout)
        rollout_replicas = [
            VLARolloutReplica(
                replica_rank=0,
                config=rollout_config,
                model_config=self.config.actor_rollout_ref.model,
                rollout_workers=self.worker_groups[ROLE_TO_WORKER_NAME[Role.Rollout]].workers,
            )
        ]
        self.checkpoint_engine_manager = CheckpointEngineManager(
            config=omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine),
            trainer=self.actor_worker_group,
            replicas=rollout_replicas,
        )

    @property
    def actor_worker_group(self):
        actor_wg = self.worker_groups.get(ROLE_TO_WORKER_NAME[Role.Actor])
        if actor_wg is not None:
            return actor_wg
        return self.worker_groups[ROLE_TO_WORKER_NAME[Role.ActorRollout]]

    @property
    def train_world_size(self) -> int:
        return int(self.actor_worker_group.world_size)

    def rollout(
        self,
        *,
        async_rollout: bool = False,
    ) -> tuple[DataProto, dict[str, dict[str, Any]], dict[str, float]]:
        if self.cluster_type != "env_loop":
            raise RuntimeError("rollout is only wired for env-loop train clusters.")
        assert self.env_loop is not None

        if not async_rollout:
            output, collected_datasets, metrics, self.rollout_state = self._rollout_once(
                self.env_loop,
                config=self.config,
                state=self.rollout_state,
            )
            return output, collected_datasets, metrics

        else:
            if not self.config.resource.separate_rollout_model.enabled:
                raise ValueError("async_rollout requires separate actor and rollout workers.")

            if self._pending_rollout_ref is None:
                self._pending_rollout_ref = ray_rollout_once.remote(
                    self.env_loop,
                    self.config,
                    self.rollout_state,
                )

            assert self._pending_rollout_ref is not None
            output, collected_datasets, metrics, self.rollout_state = ray.get(self._pending_rollout_ref)

            self.update_weights()

            self._pending_rollout_ref = ray_rollout_once.remote(
                self.env_loop,
                self.config,
                self.rollout_state,
            )
            return output, collected_datasets, metrics

    @staticmethod
    def _rollout_once(
        env_loop: EnvLoop,
        *,
        config: EnvLoopTrainClusterConfig,
        state: RolloutState,
    ) -> tuple[
        DataProto,
        dict[str, dict[str, Any]],
        dict[str, float],
        RolloutState,
    ]:
        reset_future = state.reset_future
        if reset_future is None:
            reset_future = env_loop.env_wg.reset_env()
        output = env_loop.generate_sequences(reset_future)
        state.reset_future = env_loop.env_wg.reset_env()
        metrics = dict(output.meta_info.pop("metrics", {}))
        trajectory_records = TrainCluster._collect_trajectory_records(
            output,
            auto_reset=config.env.env_worker.auto_reset,
            carry_state=state.carry_state,
        )
        metrics.update(TrainCluster._trajectory_metrics_from_records(trajectory_records, metric_prefix="data"))
        collected_datasets, lerobot_collected_once = TrainCluster._collect_lerobot_datasets(
            env_loop.env_wg,
            config=config,
            lerobot_collected_once=state.lerobot_collected_once,
        )
        state.lerobot_collected_once = lerobot_collected_once
        return output, collected_datasets, metrics, state

    @staticmethod
    def _collect_lerobot_datasets(
        env_wg,
        *,
        config: EnvLoopTrainClusterConfig,
        lerobot_collected_once: bool,
    ) -> tuple[dict[str, dict[str, Any]], bool]:
        recorder_cfg = config.env.env_worker.recorder
        if not recorder_cfg.enable or not recorder_cfg.lerobot.enable:
            return {}, lerobot_collected_once

        rank_datasets = [dataset for dataset in env_wg.pop_lerobot_dataset() if dataset is not None]
        root = Path(recorder_cfg.lerobot.root)
        repo_id = recorder_cfg.lerobot.repo_id
        existing_root = root / repo_id
        collected_datasets: dict[str, dict[str, Any]] = {}
        if all((existing_root / path).exists() for path in REQUIRED_LEROBOT_META_FILES):
            collected_datasets["existing_dataset"] = {"root": existing_root, "repo_id": repo_id}

        if rank_datasets:
            collected_repo_id = f"{recorder_cfg.lerobot.repo_id}_collected"
            collected_datasets["collected_dataset"] = merge_lerobot_datasets(
                roots=[dataset["root"] for dataset in rank_datasets],
                output_root=Path(recorder_cfg.lerobot.root) / collected_repo_id,
                repo_id=collected_repo_id,
                repo_ids=[dataset["repo_id"] for dataset in rank_datasets],
                overwrite=not lerobot_collected_once,
                append=lerobot_collected_once,
                video_files_size_in_mb=recorder_cfg.lerobot.video_files_size_in_mb,
            )
            lerobot_collected_once = True
        return collected_datasets, lerobot_collected_once

    def train(self, data: DataProto, *, async_update: bool = True) -> Any:
        actor_wg = self.actor_worker_group
        if async_update:
            return actor_wg.update_actor_async(data)
        return actor_wg.update_actor(data)

    def update_weights(self) -> None:
        assert self.cluster_type == "env_loop"

        if self.config.resource.separate_rollout_model.enabled:
            assert self.checkpoint_engine_manager is not None
            self.checkpoint_engine_manager.update_weights()

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
            reset_future = env_wg.reset_env(mode="eval", reset_eval=eval_step == 0)
            rollout_output = self.env_loop.generate_sequences(reset_future, eval=True)
            for key, value in rollout_output.meta_info.get("metrics", {}).items():
                rollout_metric_lists.setdefault(key, []).append(float(value))

            trajectory_records.extend(
                self._collect_trajectory_records(
                    rollout_output,
                    auto_reset=self.config.env.env_worker.auto_reset,
                    remaining=target_episodes - len(trajectory_records),
                    carry_state=carry_state,
                )
            )
            eval_step += 1

        metrics = self._trajectory_metrics_from_records(
            trajectory_records,
            metric_prefix="val",
            benchmark_size=benchmark_size,
        )
        for key, values in rollout_metric_lists.items():
            metrics[key] = float(np.mean(values)) if values else 0.0
        return metrics

    @staticmethod
    def _collect_trajectory_records(
        output: DataProto,
        *,
        auto_reset: bool,
        remaining: int | None = None,
        carry_state: dict[str, np.ndarray | None],
    ) -> list[dict[str, float | int | bool]]:
        if remaining is not None and remaining <= 0:
            return []

        raw_done_steps = output.batch["next.terminated"].bool() | output.batch["next.truncated"].bool()
        raw_reward_steps = output.batch["next.reward"].float()
        raw_success_steps = output.batch["next.success"].bool()
        chunk_steps = int(raw_done_steps.shape[-1]) if raw_done_steps.ndim > 2 else 1
        done_steps = raw_done_steps.reshape(raw_done_steps.shape[0], -1)
        reward_steps = raw_reward_steps.reshape(raw_reward_steps.shape[0], -1)
        success_steps = raw_success_steps.reshape(raw_success_steps.shape[0], -1)
        if done_steps.ndim == 1:
            done_steps = done_steps[:, None]
            reward_steps = reward_steps[:, None]
            success_steps = success_steps[:, None]
        task_id_steps = np.asarray(output.non_tensor_batch["obs.task_id"])
        task_id_steps = np.repeat(task_id_steps, chunk_steps, axis=1)

        if not auto_reset:
            return TrainCluster._collect_non_auto_reset_trajectory_records(
                done_steps,
                reward_steps,
                success_steps,
                task_id_steps=task_id_steps,
                chunk_steps=chunk_steps,
                remaining=remaining,
            )

        records: list[dict[str, float | int | bool]] = []
        batch_size = int(done_steps.shape[0])

        carry_lengths = carry_state.get("length")
        carry_rewards = carry_state.get("reward")
        carry_successes = carry_state.get("success")
        carry_task_ids = carry_state.get("task_id")
        if carry_lengths is None or carry_lengths.shape[0] != batch_size:
            carry_lengths = np.zeros(batch_size, dtype=np.int64)
            carry_rewards = np.zeros(batch_size, dtype=np.float32)
            carry_successes = np.zeros(batch_size, dtype=bool)
            carry_task_ids = np.full(batch_size, -1, dtype=np.int64)
            carry_state["length"] = carry_lengths
            carry_state["reward"] = carry_rewards
            carry_state["success"] = carry_successes
            carry_state["task_id"] = carry_task_ids
        else:
            if carry_successes is None or carry_successes.shape[0] != batch_size:
                carry_successes = np.zeros(batch_size, dtype=bool)
                carry_state["success"] = carry_successes
            if not carry_lengths.flags.writeable:
                carry_lengths = carry_lengths.copy()
                carry_state["length"] = carry_lengths
            if carry_rewards is not None and not carry_rewards.flags.writeable:
                carry_rewards = carry_rewards.copy()
                carry_state["reward"] = carry_rewards
            if carry_successes is not None and not carry_successes.flags.writeable:
                carry_successes = carry_successes.copy()
                carry_state["success"] = carry_successes
            if carry_task_ids is not None and not carry_task_ids.flags.writeable:
                carry_task_ids = carry_task_ids.copy()
                carry_state["task_id"] = carry_task_ids
        assert carry_lengths is not None
        assert carry_rewards is not None
        assert carry_successes is not None
        assert carry_task_ids is not None

        for batch_idx in range(done_steps.shape[0]):
            if remaining is not None and len(records) >= remaining:
                break

            done_row = done_steps[batch_idx].reshape(-1)
            reward_row = reward_steps[batch_idx].reshape(-1)
            success_row = success_steps[batch_idx].reshape(-1)
            done_indices = torch.nonzero(done_row, as_tuple=False).flatten().tolist()
            start_idx = 0
            segment_prefix_length = int(carry_lengths[batch_idx])
            segment_prefix_return = float(carry_rewards[batch_idx])
            segment_prefix_success = bool(carry_successes[batch_idx])
            segment_task_id = int(carry_task_ids[batch_idx])
            if segment_prefix_length == 0 and task_id_steps.shape[1] > 0:
                segment_task_id = int(task_id_steps[batch_idx, 0])

            for done_idx in done_indices:
                if remaining is not None and len(records) >= remaining:
                    break

                segment = slice(start_idx, done_idx + 1)
                trajectory_return = segment_prefix_return + float(reward_row[segment].sum().item())
                trajectory_success = segment_prefix_success or bool(success_row[segment].any().item())
                trajectory_length = segment_prefix_length + done_idx - start_idx + 1
                trajectory_chunk_length = (trajectory_length + chunk_steps - 1) // chunk_steps
                records.append(
                    {
                        "length": int(trajectory_length),
                        "chunk_length": int(trajectory_chunk_length),
                        "return": float(trajectory_return),
                        "success": trajectory_success,
                        "task_id": int(segment_task_id),
                    }
                )

                start_idx = done_idx + 1
                segment_prefix_length = 0
                segment_prefix_return = 0.0
                segment_prefix_success = False
                if start_idx < task_id_steps.shape[1]:
                    segment_task_id = int(task_id_steps[batch_idx, start_idx])

            if remaining is None or len(records) < remaining:
                remaining_steps = int(done_row.numel()) - start_idx
                carry_lengths[batch_idx] = segment_prefix_length + remaining_steps
                carry_rewards[batch_idx] = segment_prefix_return + float(reward_row[start_idx:].sum().item())
                carry_successes[batch_idx] = segment_prefix_success or bool(success_row[start_idx:].any().item())
                carry_task_ids[batch_idx] = segment_task_id

        return records

    @staticmethod
    def _collect_non_auto_reset_trajectory_records(
        done_steps: torch.Tensor,
        reward_steps: torch.Tensor,
        success_steps: torch.Tensor,
        *,
        task_id_steps: np.ndarray,
        chunk_steps: int,
        remaining: int | None,
    ) -> list[dict[str, float | int | bool]]:
        records: list[dict[str, float | int | bool]] = []
        for batch_idx in range(done_steps.shape[0]):
            if remaining is not None and len(records) >= remaining:
                break

            done_row = done_steps[batch_idx].reshape(-1)
            reward_row = reward_steps[batch_idx].reshape(-1)
            success_row = success_steps[batch_idx].reshape(-1)
            done_indices = torch.nonzero(done_row, as_tuple=False).flatten().tolist()
            done_idx = int(done_indices[0]) if done_indices else int(done_row.numel()) - 1
            segment = slice(0, done_idx + 1)
            task_id = int(task_id_steps[batch_idx, 0])

            trajectory_return = float(reward_row[segment].sum().item())
            records.append(
                {
                    "length": int(done_idx + 1),
                    "chunk_length": int(done_idx // chunk_steps + 1),
                    "return": trajectory_return,
                    "success": bool(success_row[segment].any().item()),
                    "task_id": task_id,
                }
            )
        return records

    @staticmethod
    def _trajectory_metrics_from_records(
        records: list[dict[str, float | int | bool]],
        *,
        metric_prefix: str,
        benchmark_size: int | None = None,
    ) -> dict[str, float]:
        trajectory_count = len(records)
        success_records = [record for record in records if bool(record["success"])]
        success_count = len(success_records)
        metrics = {
            f"{metric_prefix}/avg_return": (
                float(np.mean([float(record["return"]) for record in records])) if records else 0.0
            ),
            f"{metric_prefix}/avg_success_trajectory_length": (
                float(np.mean([int(record["length"]) for record in success_records])) if success_records else 0.0
            ),
            f"{metric_prefix}/avg_success_trajectory_chunk_length": (
                float(np.mean([int(record["chunk_length"]) for record in success_records])) if success_records else 0.0
            ),
            f"{metric_prefix}/trajectory_count": float(trajectory_count),
            f"{metric_prefix}/success_trajectory_count": float(success_count),
            f"{metric_prefix}/failed_trajectory_count": float(trajectory_count - success_count),
            f"{metric_prefix}/trajectory_success_rate": (
                float(success_count / trajectory_count) if trajectory_count > 0 else 0.0
            ),
        }
        if benchmark_size is not None:
            metrics[f"{metric_prefix}/benchmark_size"] = float(benchmark_size)
        task_ids = sorted({int(record["task_id"]) for record in records if int(record["task_id"]) >= 0})
        for task_id in task_ids:
            task_records = [record for record in records if int(record["task_id"]) == task_id]
            task_success_records = [record for record in task_records if bool(record["success"])]
            trajectory_count = len(task_records)
            success_count = len(task_success_records)
            metrics[f"{metric_prefix}/per_task_trajectory_count/task_{task_id}"] = float(trajectory_count)
            metrics[f"{metric_prefix}/per_task_success_trajectory_count/task_{task_id}"] = float(success_count)
            metrics[f"{metric_prefix}/per_task_failed_trajectory_count/task_{task_id}"] = float(
                trajectory_count - success_count
            )
            metrics[f"{metric_prefix}/per_task_success_rate/task_{task_id}"] = (
                float(success_count / trajectory_count) if trajectory_count > 0 else 0.0
            )
            metrics[f"{metric_prefix}/per_task_avg_success_trajectory_length/task_{task_id}"] = (
                float(np.mean([int(record["length"]) for record in task_success_records]))
                if task_success_records
                else 0.0
            )
            metrics[f"{metric_prefix}/per_task_avg_success_trajectory_chunk_length/task_{task_id}"] = (
                float(np.mean([int(record["chunk_length"]) for record in task_success_records]))
                if task_success_records
                else 0.0
            )
        return metrics

    def load_checkpoint(self) -> tuple[int, str] | None:
        assert self.checkpoint_helper is not None
        checkpoint_state = self.checkpoint_helper.load()
        if self.cluster_type == "env_loop" and self.config.resource.separate_rollout_model.enabled:
            self.update_weights()
        return checkpoint_state

    def save_checkpoint(
        self,
        global_step: int,
        save_extra_state: Callable[[str], None] | None = None,
    ) -> None:
        assert self.checkpoint_helper is not None
        self.checkpoint_helper.save(global_step, save_extra_state=save_extra_state)
