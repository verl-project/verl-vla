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

import torch
from tqdm import tqdm
from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup

from verl_vla.train_cluster.config import EnvLoopConfig
from verl_vla.utils.data import (
    get_dataproto_from_prefix,
    stack_dataproto_with_padding,
    update_progress_trajectory_counts,
)
from verl_vla.utils.keys import ACTION_KEY, FEEDBACK_KEY, OBS_KEY

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EnvLoop:
    """An env loop manages interactions between models and vectorized environments."""

    def __init__(
        self,
        env_wg: RayWorkerGroup,
        rollout_wg: RayWorkerGroup,
        config: EnvLoopConfig,
        switch_actor_rollout_mode: bool,
    ):
        self.env_wg = env_wg
        self.rollout_wg = rollout_wg

        self.stage_num = config.pipeline_stage_num
        self.max_interactions = config.max_interactions
        self.switch_actor_rollout_mode = switch_actor_rollout_mode

    def _strip_meta_info(self, data: DataProto) -> DataProto:
        return DataProto(
            batch=data.batch,
            non_tensor_batch=data.non_tensor_batch,
            meta_info={},
        )

    def generate_sequences(
        self,
        reset_future: asyncio.Future,
        *,
        eval: bool = False,
    ) -> DataProto:
        total_start_t = time.perf_counter()
        reset_wait_start_t = time.perf_counter()
        reset_results = reset_future.get()
        reset_wait_s = time.perf_counter() - reset_wait_start_t
        rollout_meta_info = {"eval": True} if eval else {}
        env_mode = "eval" if eval else "train"

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.switch_actor_rollout_mode:
            self.rollout_wg.switch_to_rollout()
            run_start_t = time.perf_counter()
            output, run_metrics = loop.run_until_complete(self.run(reset_results, rollout_meta_info, env_mode=env_mode))
            run_s = time.perf_counter() - run_start_t
            self.rollout_wg.switch_to_train()
        else:
            run_start_t = time.perf_counter()
            output, run_metrics = loop.run_until_complete(self.run(reset_results, rollout_meta_info, env_mode=env_mode))
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

    async def run(
        self,
        reset_results: DataProto,
        rollout_meta_info: dict,
        *,
        env_mode: str,
    ) -> tuple[DataProto, dict[str, float]]:
        trajectories = {i: [] for i in range(self.stage_num)}

        staged_obs = self._restructure_obs_data(reset_results)
        for stage_id in range(self.stage_num):
            trajectories[stage_id].append({OBS_KEY: self._strip_meta_info(staged_obs[stage_id])})

        rollout_futures = {}
        for stage_id in range(self.stage_num):
            vla_input = staged_obs[stage_id]
            vla_input.meta_info = rollout_meta_info
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

        progress_bar = tqdm(total=self.max_interactions, desc="Rollout Progress", leave=True)
        progress_counts = {"done_eps": 0, "succ_eps": 0}
        progress_lane_state: dict[int, dict[str, torch.Tensor]] = {}

        async def _stage_loop(stage_id: int):
            stage_start_t = time.perf_counter()
            step_idx = 0
            while step_idx < self.max_interactions:
                rollout_wait_start_t = time.perf_counter()
                action_result: DataProto = await asyncio.to_thread(rollout_futures[stage_id].get)
                stage_timing[stage_id]["rollout_wait_s"] += time.perf_counter() - rollout_wait_start_t
                stage_timing[stage_id]["rollout_wait_calls"] += 1.0

                trajectories[stage_id][-1][ACTION_KEY] = self._strip_meta_info(action_result)
                action_result.meta_info["stage_id"] = stage_id
                env_ref = self.env_wg.env_interact_step(action_result, mode=env_mode)

                env_wait_start_t = time.perf_counter()
                env_result: DataProto = await asyncio.to_thread(env_ref.get)
                stage_timing[stage_id]["env_wait_s"] += time.perf_counter() - env_wait_start_t
                stage_timing[stage_id]["env_wait_calls"] += 1.0
                update_progress_trajectory_counts(
                    env_result,
                    stage_id=stage_id,
                    progress_counts=progress_counts,
                    progress_lane_state=progress_lane_state,
                )
                progress_bar.set_postfix(progress_counts, refresh=False)

                next_step = self._strip_meta_info(get_dataproto_from_prefix(env_result, FEEDBACK_KEY, "."))
                next_obs = self._strip_meta_info(get_dataproto_from_prefix(env_result, OBS_KEY, "."))

                current_slot = trajectories[stage_id].pop()
                current_slot[FEEDBACK_KEY] = next_step
                trajectories[stage_id].append(current_slot)

                stage_timing[stage_id]["effective_steps"] += 1.0
                step_idx += 1
                if stage_id == 0:
                    progress_bar.update(1)
                if step_idx < self.max_interactions:
                    trajectories[stage_id].append({OBS_KEY: next_obs})

                if step_idx < self.max_interactions:
                    vla_input = next_obs
                    vla_input.meta_info = rollout_meta_info
                    rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

            stage_timing[stage_id]["stage_wall_s"] = time.perf_counter() - stage_start_t

        try:
            await asyncio.gather(*[asyncio.create_task(_stage_loop(sid)) for sid in range(self.stage_num)])
        finally:
            progress_bar.close()
        success_count = progress_counts["succ_eps"]
        trajectory_count = progress_counts["done_eps"]
        print(
            f"Rollout collected trajectories: success={success_count}, "
            f"failed={trajectory_count - success_count}, total={trajectory_count}"
        )
        self.env_wg.finish_rollout()
        collated_meta_info = dict(rollout_meta_info)
        output = self._collate_trajectories(trajectories, meta_info=collated_meta_info)
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

    def _collate_trajectories(self, trajectories: dict, meta_info) -> DataProto:
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

        return DataProto.from_single_dict(batch_dict, meta_info=meta_info)
