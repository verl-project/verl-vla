# Copyright 2025 The RLinf Authors.
# Copyright 2024 Bytedance Ltd. and/or its affiliates

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import torch
from libero.libero import get_libero_path
from libero.libero.benchmark import Benchmark, get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from omegaconf import OmegaConf

from verl_vla.envs.base import BaseEnv
from verl_vla.envs.libero_env.utils import get_libero_image, get_libero_wrist_image, quat2axisangle
from verl_vla.envs.libero_env.venv import ReconfigureSubprocEnv
from verl_vla.utils.envs.action import to_tensor
from verl_vla.utils.random import compose_seed

LIBERO_ACTION_DIM = 7


def patched_get_task_init_states(self, i):
    init_states_path = os.path.join(
        get_libero_path("init_states"),
        self.tasks[i].problem_folder,
        self.tasks[i].init_states_file,
    )
    init_states = torch.load(init_states_path, weights_only=False)
    return init_states


Benchmark.get_task_init_states = patched_get_task_init_states


class LiberoResetStatePlanner:
    """Owns LIBERO reset-id filtering, eval ordering, and process sharding."""

    def __init__(
        self,
        *,
        task_suite: Benchmark,
        libero_cfg,
        global_process_id: int,
        global_process_count: int,
        envs_per_process: int,
        generator: np.random.Generator,
    ):
        self.task_suite = task_suite
        self.libero_cfg = libero_cfg
        self.global_process_id = int(global_process_id)
        self.global_process_count = int(global_process_count)
        self.envs_per_process = int(envs_per_process)
        self.generator = generator
        self.eval_cursor = 0

        self.reset_states_by_task = {}
        next_reset_state_id = 0
        for task_id in range(self.task_suite.get_num_tasks()):
            num_trials = len(self.task_suite.get_task_init_states(task_id))
            self.reset_states_by_task[task_id] = np.column_stack(
                (
                    np.full(num_trials, task_id, dtype=np.int64),  # task_id
                    np.arange(num_trials, dtype=np.int64),  # trial_id
                    np.arange(next_reset_state_id, next_reset_state_id + num_trials, dtype=np.int64),  # reset_state_id
                )
            )
            next_reset_state_id += num_trials

        self.valid_reset_states_by_task = self._filter_reset_states_by_task()
        self.valid_reset_states = np.concatenate(list(self.valid_reset_states_by_task.values()), axis=0)
        self.benchmark_size = int(self.valid_reset_states.shape[0])
        self.valid_eval_states = np.column_stack(
            (
                self.valid_reset_states,
                np.arange(self.benchmark_size, dtype=np.int64),
            )
        )
        self.eval_process_queue = self._build_process_queue(self.valid_eval_states)

    def sample_train_states(self, count: int) -> np.ndarray:
        indices = self.generator.integers(low=0, high=len(self.valid_reset_states), size=(count,))
        return np.column_stack(
            (
                self.valid_reset_states[indices],
                np.full(count, -1, dtype=np.int64),
            )
        ).astype(np.int64, copy=False)

    def reset_eval_cursor(self) -> None:
        self.eval_cursor = 0
        self.eval_process_queue = self._build_process_queue(self.valid_eval_states)

    def next_eval_states(self, count: int) -> np.ndarray:
        process_states = self.eval_process_queue[self.global_process_id]
        indices = (self.eval_cursor + np.arange(count)) % len(process_states)
        self.eval_cursor = (self.eval_cursor + count) % len(process_states)
        return process_states[indices].astype(np.int64, copy=False)

    def _filter_reset_states_by_task(self) -> dict[int, np.ndarray]:
        reset_states_by_task = self.reset_states_by_task
        if self.libero_cfg.task_ids:
            reset_states_by_task = {task_id: reset_states_by_task[task_id] for task_id in self.libero_cfg.task_ids}

        if self.libero_cfg.num_trials_per_task is not None:
            num_trials_per_task = int(self.libero_cfg.num_trials_per_task)
            reset_states_by_task = {
                task_id: reset_states[:num_trials_per_task] for task_id, reset_states in reset_states_by_task.items()
            }

        if self.libero_cfg.specific_reset_id is not None:
            specific_reset_id = int(self.libero_cfg.specific_reset_id)
            reset_states_by_task = {
                task_id: reset_states[reset_states[:, 2] == specific_reset_id]
                for task_id, reset_states in reset_states_by_task.items()
                if np.any(reset_states[:, 2] == specific_reset_id)
            }
            if not reset_states_by_task:
                raise ValueError(f"specific_reset_id {specific_reset_id} is outside the configured LIBERO reset ids.")

        if not any(len(reset_states) for reset_states in reset_states_by_task.values()):
            raise ValueError("LIBERO reset id filter produced no states.")
        return reset_states_by_task

    def _build_process_queue(self, reset_states: np.ndarray) -> np.ndarray:
        reset_states = np.asarray(reset_states, dtype=np.int64)
        state_width = int(reset_states.shape[1])
        reset_states = reset_states.reshape(-1, state_width)
        if len(reset_states) == 0:
            raise ValueError("LIBERO reset queue is empty.")

        total_env_slots = self.global_process_count * self.envs_per_process
        num_batches = (len(reset_states) + total_env_slots - 1) // total_env_slots
        padded_size = max(total_env_slots, num_batches * total_env_slots)
        reset_states = reset_states[np.resize(np.arange(len(reset_states)), padded_size)]
        return (
            reset_states.reshape(-1, self.global_process_count, self.envs_per_process, state_width)
            .transpose(1, 0, 2, 3)
            .reshape(self.global_process_count, -1, state_width)
        )


class LiberoResetStateMixin:
    """LIBERO benchmark reset-state and task reconfiguration helpers."""

    def init_libero_env(self, cfg):
        del cfg
        self.init_random()
        self.task_suite: Benchmark = get_benchmark(self.libero_cfg.task_suite_name)()
        self.init_reset_states()
        self._init_env()

    def init_random(self):
        self.seed = int(self.libero_cfg.seed)
        self.rollout_id = 0
        self._generator = np.random.default_rng(seed=compose_seed(self.seed, self.rank, self.stage_id, 0, 0, 0))

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []
        for env_fn_param in env_fn_params:

            def env_fn(param=env_fn_param):
                seed = param.pop("seed")
                env = OffScreenRenderEnv(**param)
                env.seed(seed)
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        base_env_args = self.libero_cfg.env_kwargs()

        task_descriptions = []
        if env_idx is None:
            env_idx = np.arange(self.cfg.num_envs)
        for env_id in range(self.cfg.num_envs):
            if env_id not in env_idx:
                task_descriptions.append(self.task_descriptions[env_id])
                continue
            task = self.task_suite.get_task(self.task_ids[env_id])
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env_fn_params.append(
                {
                    **base_env_args,
                    "bddl_file_name": task_bddl_file,
                    "seed": compose_seed(self.seed, self.rank, self.stage_id, self.rollout_id, env_id, 0),
                }
            )
            task_descriptions.append(task.language)
        self.task_descriptions = task_descriptions
        return env_fn_params

    ### Reset State Helpers ###

    def init_reset_states(self):
        self.reset_planner = LiberoResetStatePlanner(
            task_suite=self.task_suite,
            libero_cfg=self.libero_cfg,
            global_process_id=self.global_process_id,
            global_process_count=self.global_process_count,
            envs_per_process=self.num_envs,
            generator=self._generator,
        )
        reset_states = (
            self.reset_planner.next_eval_states(self.num_envs)
            if self.only_eval
            else self.reset_planner.sample_train_states(self.num_envs)
        )
        self.task_ids = reset_states[:, 0].copy()
        self.trial_ids = reset_states[:, 1].copy()
        self.reset_state_ids = reset_states[:, 2].copy()
        self.eval_episode_ids = reset_states[:, 3].copy()
        if self.only_eval:
            self.reset_planner.reset_eval_cursor()

    def _reset_to_states(self, reset_states, env_idx):
        reset_states = np.asarray(reset_states, dtype=np.int64).reshape(-1, 4)
        old_task_ids = self.task_ids[env_idx].copy()
        self.task_ids[env_idx] = reset_states[:, 0]
        self.trial_ids[env_idx] = reset_states[:, 1]
        self.reset_state_ids[env_idx] = reset_states[:, 2]
        self.eval_episode_ids[env_idx] = reset_states[:, 3]

        reconfig_env_idx = env_idx[old_task_ids != self.task_ids[env_idx]]
        if len(reconfig_env_idx):
            self.env.reconfigure_env_fns(self.get_env_fn_params(reconfig_env_idx), reconfig_env_idx)

        self.env.reset(id=env_idx)
        self.env.set_init_state(
            init_state=[
                self.task_suite.get_task_init_states(self.task_ids[env_id])[self.trial_ids[env_id]]
                for env_id in env_idx
            ],
            id=env_idx,
        )


class LiberoEnv(LiberoResetStateMixin, BaseEnv):
    env_type = "libero"

    def __init__(
        self,
        cfg,
        rank,
        world_size,
        stage_id: int = 0,
        stage_num: int = 1,
        only_eval: bool = False,
    ):
        self.only_eval = only_eval
        self.libero_cfg = OmegaConf.to_object(cfg.simulator.libero)
        self.stage_num = int(stage_num)
        self.global_process_id = int(rank) * self.stage_num + int(stage_id)
        self.global_process_count = int(world_size) * self.stage_num
        super().__init__(cfg, rank, world_size, stage_id=stage_id)
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

    def _reset_elapsed_steps(self, env_ids=None):
        if env_ids is None:
            self._elapsed_steps[:] = 0
            return
        self._elapsed_steps[np.asarray(env_ids, dtype=np.int64)] = 0

    ### Observation Formatting ###

    def _make_observations(self, raw_obs):
        observations = []
        for obs in raw_obs:
            observations.append(
                {
                    "observation.images.image": get_libero_image(obs),
                    "observation.images.wrist_image": get_libero_wrist_image(obs),
                    "observation.state": np.concatenate(
                        [
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]
                    ),
                }
            )
        return observations

    ### Main API ###

    def env_init(self) -> None:
        self.init_libero_env(self.cfg)

    def env_benchmark_size(self) -> int:
        return self.reset_planner.benchmark_size

    def get_recorder_strategy_kwargs(self):
        return {
            "image_shape": (
                int(self.libero_cfg.camera_heights),
                int(self.libero_cfg.camera_widths),
                3,
            )
        }

    def env_reset(
        self,
        *,
        env_ids,
        reset_eval: bool = False,
    ):
        """Reset LIBERO envs.

        LIBERO chooses task and trial ids internally from its configured reset
        queues. Partial resets are applied to the requested env ids; eval uses
        the ordered eval queue.
        """

        # Configure envs with internally selected reset state ids, then reset them and set their init states.
        env_ids = np.asarray(env_ids, dtype=np.int64)
        if reset_eval:
            self.reset_planner.reset_eval_cursor()

        self.rollout_id += 1
        reset_states = (
            self.reset_planner.next_eval_states(len(env_ids))
            if self.only_eval
            else self.reset_planner.sample_train_states(len(env_ids))
        )
        self._reset_to_states(reset_states, env_ids)

        # Perform extra warmup steps after reset to let the observations settle.
        raw_obs = None
        reset_warmup_steps = int(self.libero_cfg.reset_warmup_steps)
        partial_reset = len(env_ids) != self.num_envs
        step_env_ids = env_ids if partial_reset else None
        action_count = len(env_ids) if partial_reset else self.num_envs
        zero_actions = np.zeros((action_count, LIBERO_ACTION_DIM))
        for _ in range(reset_warmup_steps):
            if step_env_ids is None:
                raw_obs, _reward, _terminations, _info_lists = self.env.step(zero_actions)
            else:
                raw_obs, _reward, _terminations, _info_lists = self.env.step(zero_actions, id=step_env_ids)

        obs_env_ids = env_ids if partial_reset else np.arange(self.num_envs)
        tasks = [self.task_descriptions[env_id] for env_id in obs_env_ids]
        task_id = self.task_ids[obs_env_ids].astype(np.int64, copy=False)
        eval_episode_id = self.eval_episode_ids[obs_env_ids].astype(np.int64, copy=False)

        obs = {
            "observation": self._make_observations(raw_obs),
            "task": tasks,
            "task_id": task_id,
            "eval_episode_id": eval_episode_id,
        }
        self._reset_elapsed_steps(env_ids)

        return obs

    def env_step(self, actions, *, env_ids):
        env_ids = np.asarray(env_ids, dtype=np.int64)
        self._elapsed_steps[env_ids] += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions, id=env_ids)
        del info_lists
        truncations = self._elapsed_steps[env_ids] >= self.libero_cfg.max_episode_steps

        step_reward = np.asarray(_reward)

        return {
            "observation": self._make_observations(raw_obs),
            "task": [self.task_descriptions[env_id] for env_id in env_ids],
            "task_id": self.task_ids[env_ids].astype(np.int64, copy=False),
            "eval_episode_id": self.eval_episode_ids[env_ids].astype(np.int64, copy=False),
            "next.reward": to_tensor(step_reward),
            "next.terminated": to_tensor(np.asarray(terminations, dtype=bool)),
            "next.truncated": to_tensor(np.asarray(truncations, dtype=bool)),
            "next.success": to_tensor(np.asarray(terminations, dtype=bool)),
        }

    def env_close(self):
        self.env.close()
