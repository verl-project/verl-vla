# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import asyncio
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup

from verl_vla.utils.data import get_dataproto_from_prefix, stack_dataproto_with_padding
from verl_vla.utils.keys import ACTION_KEY, FEEDBACK_KEY, OBS_KEY
from verl_vla.utils.recorder import merge_lerobot_datasets

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EnvLoop:
    """An env loop manages interactions between models and vectorized environments."""

    def __init__(self, env_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, config: DictConfig):
        self.env_wg = env_wg
        self.rollout_wg = rollout_wg
        self.config = config

        self.stage_num = config.env.rollout.pipeline_stage_num
        self.num_envs_per_worker = config.env.train.num_envs
        self.action_dim = config.env.actor.model.action_dim
        self.num_action_chunks = config.env.actor.model.num_action_chunks
        self.single_env_rollout = bool(config.env.train.get("single_env_rollout", False))
        if self.single_env_rollout:
            assert self.stage_num == 1, "single_env_rollout only supports pipeline_stage_num == 1"

        self.total_envs = self.env_wg.world_size * self.num_envs_per_worker
        if self.total_envs % self.stage_num != 0:
            raise ValueError(f"Total envs ({self.total_envs}) must be divisible by stage_num ({self.stage_num})")
        self.envs_per_stage = self.total_envs // self.stage_num

        self.default_max_interactions = config.env.train.max_episode_steps // config.env.actor.model.num_action_chunks
        self.configured_max_interactions = config.env.train.get("max_interactions", self.default_max_interactions)
        self.max_interactions = self.configured_max_interactions
        self.warmup_max_interactions = False

        self.env_wg.init_worker()
        self.env_wg.init_simulator()

    def _strip_meta_info(self, data: DataProto) -> DataProto:
        return DataProto(
            batch=data.batch,
            non_tensor_batch=data.non_tensor_batch,
            meta_info={},
        )

    def generate_sequences(self, prompts: DataProto, reset_future: asyncio.Future) -> DataProto:
        total_start_t = time.perf_counter()
        reset_wait_start_t = time.perf_counter()
        reset_results = reset_future.get()
        reset_wait_s = time.perf_counter() - reset_wait_start_t

        if self.warmup_max_interactions:
            self.max_interactions = self.default_max_interactions
        else:
            self.max_interactions = self.configured_max_interactions

        loop = asyncio.get_event_loop()

        if not self.config.trainer.separate_train_inference:
            self.rollout_wg.switch_to_rollout()
            run_start_t = time.perf_counter()
            output, run_metrics = loop.run_until_complete(self.run(prompts, reset_results))
            run_s = time.perf_counter() - run_start_t
            self.rollout_wg.switch_to_train()
        else:
            run_start_t = time.perf_counter()
            output, run_metrics = loop.run_until_complete(self.run(prompts, reset_results))
            run_s = time.perf_counter() - run_start_t

        total_s = time.perf_counter() - total_start_t
        metrics = dict(output.meta_info.get("metrics", {}))
        metrics.update(
            {
                "timing_s/env_loop_total": total_s,
                "timing_s/env_loop_reset_wait": reset_wait_s,
                "timing_s/env_loop_run": run_s,
            }
        )
        metrics.update(run_metrics)
        output.meta_info["metrics"] = metrics
        return output

    async def run(self, prompts: DataProto, reset_results: DataProto) -> tuple[DataProto, dict[str, float]]:
        trajectories = {i: [] for i in range(self.stage_num)}
        initial_state_ids = np.asarray(prompts.non_tensor_batch["state_ids"])
        staged_task_ids = self._restructure_prompt_task_ids(prompts)
        staged_state_ids = self._restructure_prompt_values(initial_state_ids)
        if self.single_env_rollout:
            initial_state_ids = initial_state_ids[:1]
            staged_state_ids = [staged_state_ids[0][:1]]

        staged_obs = self._restructure_obs_data(reset_results)
        for stage_id in range(self.stage_num):
            trajectories[stage_id].append({OBS_KEY: self._strip_meta_info(staged_obs[stage_id])})
        if self.single_env_rollout:
            staged_obs = [staged_obs[0].repeat(repeat_times=len(prompts), interleave=True)]
            staged_task_ids = [staged_task_ids[0]]

        rollout_futures = {}
        for stage_id in range(self.stage_num):
            vla_input = staged_obs[stage_id]
            vla_input.meta_info = prompts.meta_info
            if staged_task_ids[stage_id] is not None:
                vla_input.non_tensor_batch["task_ids"] = staged_task_ids[stage_id]
            rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

        stage_timing = {
            stage_id: {
                "stage_wall_s": 0.0,
                "rollout_wait_s": 0.0,
                "env_wait_s": 0.0,
                "rollout_wait_calls": 0.0,
                "env_wait_calls": 0.0,
                "effective_steps": 0.0,
            }
            for stage_id in range(self.stage_num)
        }

        async def _stage_loop(stage_id: int):
            stage_start_t = time.perf_counter()
            step_idx = 0
            while step_idx < self.max_interactions:
                if stage_id == 0:
                    logger.info(f"[{step_idx}/{self.max_interactions - 1}] rollout step")

                rollout_wait_start_t = time.perf_counter()
                action_result: DataProto = await asyncio.to_thread(rollout_futures[stage_id].get)
                stage_timing[stage_id]["rollout_wait_s"] += time.perf_counter() - rollout_wait_start_t
                stage_timing[stage_id]["rollout_wait_calls"] += 1.0

                action_batch_size = len(action_result)
                action_result = action_result[:1] if self.single_env_rollout else action_result
                trajectories[stage_id][-1][ACTION_KEY] = self._strip_meta_info(action_result)
                action_result.meta_info["stage_id"] = stage_id
                env_ref = self.env_wg.env_interact_step(action_result)

                env_wait_start_t = time.perf_counter()
                env_result: DataProto = await asyncio.to_thread(env_ref.get)
                stage_timing[stage_id]["env_wait_s"] += time.perf_counter() - env_wait_start_t
                stage_timing[stage_id]["env_wait_calls"] += 1.0

                next_step = self._strip_meta_info(get_dataproto_from_prefix(env_result, FEEDBACK_KEY, "."))
                next_obs = self._strip_meta_info(get_dataproto_from_prefix(env_result, OBS_KEY, "."))

                current_slot = trajectories[stage_id].pop()
                current_slot[FEEDBACK_KEY] = next_step
                trajectories[stage_id].append(current_slot)

                stage_timing[stage_id]["effective_steps"] += 1.0
                step_idx += 1
                if step_idx < self.max_interactions:
                    trajectories[stage_id].append({OBS_KEY: next_obs})

                if step_idx < self.max_interactions:
                    vla_input = next_obs
                    if self.single_env_rollout:
                        vla_input = vla_input.repeat(repeat_times=action_batch_size, interleave=True)
                    vla_input.meta_info = prompts.meta_info
                    if staged_task_ids[stage_id] is not None:
                        vla_input.non_tensor_batch["task_ids"] = staged_task_ids[stage_id]
                    rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

            stage_timing[stage_id]["stage_wall_s"] = time.perf_counter() - stage_start_t

        await asyncio.gather(*[asyncio.create_task(_stage_loop(sid)) for sid in range(self.stage_num)])
        self.env_wg.finish_rollout()
        # lerobot_dataset = self._pop_lerobot_dataset()

        collated_state_ids = np.concatenate(staged_state_ids, axis=0)
        collated_meta_info = dict(prompts.meta_info)
        if any(task_ids is not None for task_ids in staged_task_ids):
            collated_meta_info["task_ids"] = np.concatenate(
                [task_ids for task_ids in staged_task_ids if task_ids is not None],
                axis=0,
            )
        output = self._collate_trajectories(trajectories, collated_state_ids, meta_info=collated_meta_info)
        stage_wall_max_s = max(stage_timing[sid]["stage_wall_s"] for sid in range(self.stage_num))
        rollout_wait_sum_s = sum(stage_timing[sid]["rollout_wait_s"] for sid in range(self.stage_num))
        env_wait_sum_s = sum(stage_timing[sid]["env_wait_s"] for sid in range(self.stage_num))
        rollout_wait_calls = sum(stage_timing[sid]["rollout_wait_calls"] for sid in range(self.stage_num))
        env_wait_calls = sum(stage_timing[sid]["env_wait_calls"] for sid in range(self.stage_num))
        effective_steps = sum(stage_timing[sid]["effective_steps"] for sid in range(self.stage_num))

        run_metrics = {
            "timing_s/env_loop_stage_wall_max": stage_wall_max_s,
            "timing_s/env_loop_rollout_wait_sum": rollout_wait_sum_s,
            "timing_s/env_loop_env_wait_sum": env_wait_sum_s,
            "timing_s/env_loop_rollout_wait_avg": rollout_wait_sum_s / max(1.0, rollout_wait_calls),
            "timing_s/env_loop_env_wait_avg": env_wait_sum_s / max(1.0, env_wait_calls),
            "throughput/env_loop_effective_steps_per_s": effective_steps / max(1e-12, stage_wall_max_s),
            "throughput/env_loop_env_rpc_per_s": env_wait_calls / max(1e-12, stage_wall_max_s),
            "count/env_loop_effective_steps": effective_steps,
            "count/env_loop_env_rpc_calls": env_wait_calls,
            "count/env_loop_rollout_wait_calls": rollout_wait_calls,
        }
        return output, run_metrics

    def _pop_lerobot_dataset(self):
        recorder_cfg = self.config.env.train.get("dataset_recorder", {})
        if not recorder_cfg.get("enable", False):
            return None

        rank_datasets = [dataset for dataset in self.env_wg.pop_lerobot_dataset() if dataset is not None]
        if not rank_datasets:
            return None

        root = Path(recorder_cfg.get("root", "/tmp/verl_vla_lerobot_records"))
        repo_id = recorder_cfg.get("repo_id", "local/verl_vla_libero")
        return merge_lerobot_datasets(
            roots=[dataset["root"] for dataset in rank_datasets],
            output_root=root / repo_id,
            repo_id=repo_id,
            repo_ids=[dataset["repo_id"] for dataset in rank_datasets],
            append=True,
        )

    def _restructure_obs_data(self, data_proto: DataProto) -> list[DataProto]:
        num_workers = self.env_wg.world_size
        staged_data = [[] for _ in range(self.stage_num)]
        chunks = data_proto.chunk(num_workers)
        for worker_chunk in chunks:
            stage_chunks = worker_chunk.chunk(self.stage_num)
            for stage_id, data in enumerate(stage_chunks):
                staged_data[stage_id].append(data)
        return [DataProto.concat(data_list) for data_list in staged_data]

    def _restructure_prompt_values(self, values: np.ndarray) -> list[np.ndarray]:
        staged_values = [[] for _ in range(self.stage_num)]
        num_workers = self.env_wg.world_size
        envs_per_worker = len(values) // num_workers
        if envs_per_worker * num_workers != len(values):
            raise ValueError(f"value length {len(values)} is not divisible by env worker count {num_workers}.")

        for worker_id in range(num_workers):
            worker_start = worker_id * envs_per_worker
            worker_end = worker_start + envs_per_worker
            worker_values = values[worker_start:worker_end]
            if len(worker_values) % self.stage_num != 0:
                raise ValueError(
                    f"worker value length {len(worker_values)} is not divisible by stage_num {self.stage_num}."
                )
            stage_size = len(worker_values) // self.stage_num
            for stage_id in range(self.stage_num):
                stage_start = stage_id * stage_size
                stage_end = stage_start + stage_size
                staged_values[stage_id].append(worker_values[stage_start:stage_end])

        return [
            np.concatenate(value_list, axis=0) if value_list else np.array([], dtype=values.dtype)
            for value_list in staged_values
        ]

    def _restructure_prompt_task_ids(self, prompts: DataProto) -> list[np.ndarray | None]:
        if "task_ids" not in prompts.meta_info:
            return [None for _ in range(self.stage_num)]

        task_ids = np.asarray(prompts.meta_info["task_ids"])
        return self._restructure_prompt_values(task_ids)

    def _collate_trajectories(self, trajectories: dict, initial_state_ids: np.ndarray, meta_info) -> DataProto:
        flat_trajs = [{} for _ in range(len(trajectories[0]))]
        mergeable_keys = {OBS_KEY, ACTION_KEY, FEEDBACK_KEY}
        for stage_id in range(self.stage_num):
            for step_idx, step_data in enumerate(trajectories[stage_id]):
                if not flat_trajs[step_idx]:
                    flat_trajs[step_idx] = step_data
                else:
                    for key, value in step_data.items():
                        if key not in mergeable_keys:
                            continue
                        if isinstance(value, DataProto):
                            left = self._strip_meta_info(flat_trajs[step_idx][key])
                            right = self._strip_meta_info(value)
                            flat_trajs[step_idx][key] = DataProto.concat([left, right])
                        elif isinstance(value, torch.Tensor):
                            flat_trajs[step_idx][key] = torch.cat([flat_trajs[step_idx][key], value], dim=0)

        batch_dict = {}
        for field_key in [OBS_KEY, ACTION_KEY, FEEDBACK_KEY]:
            batch_dict.update(stack_dataproto_with_padding([step[field_key] for step in flat_trajs], field_key))
        if "next.done" in batch_dict:
            batch_dict["complete"] = batch_dict["next.done"].clone()
        batch_dict["env_state_id"] = torch.from_numpy(initial_state_ids.astype(int))

        return DataProto.from_single_dict(batch_dict, meta_info=meta_info)
