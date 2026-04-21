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

import numpy as np
import torch
from omegaconf import DictConfig
from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup

from verl_vla.utils.data import get_dataproto_from_prefix, slice_dataproto_batch, stack_dataproto_with_padding
from verl_vla.workers.actor_critic.base import ACTION_KEY, FEEDBACK_KEY, INTERVENTION_INFO_KEY, OBS_KEY

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

INTERVENTION_ACTION_KEY = "action"


class EnvLoop:
    """An env loop manages interactions between models and vectorized environments."""

    def __init__(self, env_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, config: DictConfig):
        self.env_wg = env_wg
        self.rollout_wg = rollout_wg
        self.config = config

        self.max_interactions = config.env.train.max_episode_steps // config.env.actor.model.num_action_chunks
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

        self.env_wg.init_worker()
        self.env_wg.init_simulator()

    def _extract_intervention_obs_at(self, intervention_info: DataProto, obs_idx: int) -> DataProto:
        tensor_batch = {
            key.removeprefix(f"{OBS_KEY}."): value[:, obs_idx]
            for key, value in intervention_info.batch.items()
            if key.startswith(f"{OBS_KEY}.")
        }
        non_tensor_batch = {
            key.removeprefix(f"{OBS_KEY}."): value[:, obs_idx]
            for key, value in intervention_info.non_tensor_batch.items()
            if key.startswith(f"{OBS_KEY}.")
        }
        return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch)

    def _strip_meta_info(self, data: DataProto) -> DataProto:
        return DataProto(
            batch=data.batch,
            non_tensor_batch=data.non_tensor_batch,
            meta_info={},
        )

    def _expand_single_env_intervention_steps(
        self,
        current_obs: DataProto,
        rollout_action: DataProto,
        feedback: DataProto,
        intervention_info: DataProto,
        max_chunks: int,
    ) -> list[dict]:
        total_steps = intervention_info.batch[INTERVENTION_ACTION_KEY].shape[1]
        aligned_steps = (total_steps // self.num_action_chunks) * self.num_action_chunks
        if aligned_steps == 0 or max_chunks <= 0:
            return []

        num_chunks = min(aligned_steps // self.num_action_chunks, max_chunks)
        expanded_steps = []
        chunk_obs = current_obs
        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.num_action_chunks
            end = start + self.num_action_chunks

            action_chunk = intervention_info.batch[INTERVENTION_ACTION_KEY][:, start:end].clone()
            if start == 0:
                intervention_mask = intervention_info.batch["is_intervention"][:, start:end].to(torch.bool)
                rollout_action_chunk = rollout_action.batch[ACTION_KEY][
                    :1, : action_chunk.shape[1], : action_chunk.shape[2]
                ]
                action_chunk = torch.where(
                    intervention_mask.unsqueeze(-1).to(action_chunk.device),
                    action_chunk,
                    rollout_action_chunk.to(device=action_chunk.device, dtype=action_chunk.dtype),
                )
            action_dp = DataProto.from_dict(tensors={INTERVENTION_ACTION_KEY: action_chunk})
            feedback_dp = slice_dataproto_batch(feedback, start, end)
            expanded_steps.append({OBS_KEY: chunk_obs, ACTION_KEY: action_dp, FEEDBACK_KEY: feedback_dp})

            if chunk_idx < num_chunks - 1:
                chunk_obs = self._extract_intervention_obs_at(intervention_info, chunk_idx)

        return expanded_steps

    def generate_sequences(self, prompts: DataProto, reset_future: asyncio.Future) -> DataProto:
        total_start_t = time.perf_counter()
        reset_wait_start_t = time.perf_counter()
        reset_results = reset_future.get()
        reset_wait_s = time.perf_counter() - reset_wait_start_t

        loop = asyncio.get_event_loop()
        self.rollout_wg.switch_to_rollout()
        run_start_t = time.perf_counter()
        output, run_metrics = loop.run_until_complete(self.run(prompts, reset_results))
        run_s = time.perf_counter() - run_start_t
        self.rollout_wg.switch_to_train()

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
        initial_state_ids = prompts.non_tensor_batch["state_ids"]

        staged_obs = self._restructure_obs_data(reset_results)
        for stage_id in range(self.stage_num):
            trajectories[stage_id].append({OBS_KEY: self._strip_meta_info(staged_obs[stage_id])})
        if self.single_env_rollout:
            staged_obs = [staged_obs[0].repeat(repeat_times=len(prompts), interleave=True)]

        rollout_futures = {}
        for stage_id in range(self.stage_num):
            vla_input = staged_obs[stage_id]
            vla_input.meta_info = prompts.meta_info
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

                feedback = self._strip_meta_info(get_dataproto_from_prefix(env_result, FEEDBACK_KEY, "."))
                intervention_info = self._strip_meta_info(
                    get_dataproto_from_prefix(env_result, INTERVENTION_INFO_KEY, ".")
                )
                next_obs = self._strip_meta_info(get_dataproto_from_prefix(env_result, OBS_KEY, "."))

                expanded_steps = []
                effective_steps = 1
                current_slot = trajectories[stage_id].pop()
                has_intervention = (
                    self.single_env_rollout
                    and intervention_info.batch is not None
                    and INTERVENTION_ACTION_KEY in intervention_info.batch.keys()
                )
                if has_intervention:
                    expanded_steps = self._expand_single_env_intervention_steps(
                        current_obs=current_slot[OBS_KEY],
                        rollout_action=action_result,
                        feedback=feedback,
                        intervention_info=intervention_info,
                        max_chunks=self.max_interactions - step_idx,
                    )

                if expanded_steps:
                    trajectories[stage_id].extend(expanded_steps)
                    effective_steps = len(expanded_steps)
                else:
                    current_slot[FEEDBACK_KEY] = feedback
                    current_slot[INTERVENTION_INFO_KEY] = intervention_info
                    trajectories[stage_id].append(current_slot)

                stage_timing[stage_id]["effective_steps"] += float(effective_steps)
                step_idx += effective_steps
                if step_idx < self.max_interactions:
                    trajectories[stage_id].append({OBS_KEY: next_obs})

                if step_idx < self.max_interactions:
                    vla_input = next_obs
                    if self.single_env_rollout:
                        vla_input = vla_input.repeat(repeat_times=action_batch_size, interleave=True)
                    vla_input.meta_info = prompts.meta_info
                    rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

            stage_timing[stage_id]["stage_wall_s"] = time.perf_counter() - stage_start_t

        await asyncio.gather(*[asyncio.create_task(_stage_loop(sid)) for sid in range(self.stage_num)])
        self.env_wg.finish_rollout()

        output = self._collate_trajectories(trajectories, initial_state_ids, meta_info=prompts.meta_info)
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

    def _restructure_obs_data(self, data_proto: DataProto) -> list[DataProto]:
        num_workers = self.env_wg.world_size
        staged_data = [[] for _ in range(self.stage_num)]
        chunks = data_proto.chunk(num_workers)
        for worker_chunk in chunks:
            stage_chunks = worker_chunk.chunk(self.stage_num)
            for stage_id, data in enumerate(stage_chunks):
                staged_data[stage_id].append(data)
        return [DataProto.concat(data_list) for data_list in staged_data]

    def _collate_trajectories(self, trajectories: dict, initial_state_ids: np.ndarray, meta_info) -> DataProto:
        flat_trajs = [{} for _ in range(len(trajectories[0]))]
        mergeable_keys = {OBS_KEY, ACTION_KEY, FEEDBACK_KEY, INTERVENTION_INFO_KEY}
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
        if "feedback.terminations" in batch_dict:
            batch_dict["complete"] = batch_dict["feedback.terminations"].clone()
        batch_dict["env_state_id"] = torch.from_numpy(initial_state_ids.astype(int))

        return DataProto.from_single_dict(batch_dict, meta_info=meta_info)
