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

import logging
import os

import numpy as np
import torch
from libero.libero import get_libero_path
from libero.libero.benchmark import Benchmark, get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from omegaconf.omegaconf import OmegaConf

from verl_vla.envs.base import BaseEnv
from verl_vla.envs.libero_env.utils import get_libero_image, get_libero_wrist_image, quat2axisangle
from verl_vla.envs.libero_env.venv import ReconfigureSubprocEnv
from verl_vla.utils.envs.action import (
    list_of_dict_to_dict_of_list,
    to_tensor,
)
from verl_vla.utils.random import compose_seed

logger = logging.getLogger(__name__)

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


class LiberoResetStateMixin:
    """LIBERO benchmark reset-state and task reconfiguration helpers."""

    def init_libero_env(self, cfg):
        self.init_random()
        self.task_suite: Benchmark = get_benchmark(cfg.task_suite_name)()
        self.init_reset_states()
        self._init_env()

    def init_random(self):
        self.seed = int(self.cfg.seed)
        self.rollout_id = 0
        self._generator = np.random.default_rng(seed=compose_seed(self.seed, self.rank, self.stage_id, 0, 0, 0))
        self._generator_ordered = np.random.default_rng(seed=compose_seed(self.seed, self.rank, self.stage_id, 0, 0, 1))
        self.start_idx = 0

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
        raw_base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)
        if raw_base_env_args is None:
            base_env_args = {}
        elif isinstance(raw_base_env_args, dict):
            base_env_args = raw_base_env_args
        else:
            raise TypeError(f"Expected init_params to be a mapping, got {type(raw_base_env_args)}")

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
        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        self.reset_state_ids = self._get_ordered_reset_state_ids(self.num_envs)
        self._init_task_and_trial_ids()

    def get_all_state_ids(self):
        """Returns all possible state IDs from the entire benchmark."""
        return np.arange(self.total_num_group_envs)

    def load_state(self, state_buffer: bytes):
        self.env.load_state(state_buffer)

    def _compute_total_num_group_envs(self):
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        for task_id in range(self.task_suite.get_num_tasks()):
            task_num_trials = len(self.task_suite.get_task_init_states(task_id))
            self.trial_id_bins.append(task_num_trials)

            self.total_num_group_envs += task_num_trials

        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)

    def _get_random_reset_state_ids(self, num_reset_states):
        reset_state_ids = self._generator.integers(low=0, high=self.total_num_group_envs, size=(num_reset_states,))
        return reset_state_ids

    def get_reset_state_ids_all(self):
        reset_state_ids = np.arange(self.total_num_group_envs)
        valid_size = len(reset_state_ids) - (len(reset_state_ids) % self.world_size)
        if not self.cfg.only_eval:
            self._generator_ordered.shuffle(reset_state_ids)
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.world_size, -1)
        return reset_state_ids

    def _get_ordered_reset_state_ids(self, num_reset_states):
        reset_state_ids = self.reset_state_ids_all[self.rank][self.start_idx : self.start_idx + num_reset_states]
        self.start_idx = self.start_idx + num_reset_states
        if self.start_idx >= len(self.reset_state_ids_all[0]):
            self.reset_state_ids_all = self.get_reset_state_ids_all()
            self.start_idx = 0
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
        # get task id and trial id from reset state ids
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if reset_state_id < end_pivot and reset_state_id >= start_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot
        logger.debug(
            "get task and trial id",
            self.cumsum_trial_id_bins,
            reset_state_ids,
            task_ids,
            trial_ids,
        )
        return np.array(task_ids), np.array(trial_ids)

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        init_state = [
            self.task_suite.get_task_init_states(self.task_ids[env_id])[self.trial_ids[env_id]] for env_id in env_idx
        ]
        return init_state

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(reset_state_ids)
        for j, env_id in enumerate(env_idx):
            if self.task_ids[env_id] != task_ids[j]:
                reconfig_env_idx.append(env_id)
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)

        seed_list = [
            compose_seed(self.seed, self.rank, self.stage_id, self.rollout_id, int(env_id), 0) for env_id in env_idx
        ]
        self.env.seed(seed_list)
        self.env.reset(id=env_idx)
        init_state = self._get_reset_states(env_idx=env_idx)
        self.env.set_init_state(init_state=init_state, id=env_idx)


class LiberoEnv(LiberoResetStateMixin, BaseEnv):
    env_type = "libero"

    def __init__(self, cfg, rank, world_size, stage_id: int = 0):
        super().__init__(cfg, rank, world_size, stage_id=stage_id)

        self.init_libero_env(cfg)
        self._init_metrics()

    ### Metrics ###

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos, env_ids):
        episode_info = {}
        self.returns[env_ids] += step_reward
        self.success_once[env_ids] = self.success_once[env_ids] | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self._elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = to_tensor(episode_info)
        return infos

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

    def make_policy_observation(self, step_result):
        images_and_states = [
            {
                "full_image": observation["observation.images.image"],
                "wrist_image": observation["observation.images.wrist_image"],
                "state": observation["observation.state"],
            }
            for observation in step_result["observation"]
        ]
        return {
            "images_and_states": to_tensor(list_of_dict_to_dict_of_list(images_and_states)),
            "task_descriptions": step_result["task"],
        }

    def get_recorder_strategy_kwargs(self):
        return {
            "image_shape": (
                int(self.cfg.init_params.camera_heights),
                int(self.cfg.init_params.camera_widths),
                3,
            )
        }

    ### Main API ###

    def env_reset(
        self,
        *,
        env_ids,
        reset_state_ids=None,
        task_ids=None,
    ):
        """Reset LIBERO envs.

        LIBERO reset state ids are global benchmark ids. Each id already
        determines both the task id and the trial id, so the separately supplied
        task ids are redundant for this environment.
        """
        del task_ids

        # configure envs with the given reset state ids, then reset them and set their init states.
        self.rollout_id += 1
        env_ids = np.asarray(env_ids, dtype=np.int64)
        if reset_state_ids is None:
            reset_state_ids = self._get_random_reset_state_ids(len(env_ids))
        self._reconfigure(reset_state_ids, env_ids)

        # Perform extra warmup steps after reset to let the observations settle.
        raw_obs = None
        reset_warmup_steps = int(getattr(self.cfg, "reset_warmup_steps", 10))
        for _ in range(reset_warmup_steps):
            zero_actions = np.zeros((self.num_envs, LIBERO_ACTION_DIM))
            raw_obs, _reward, terminations, info_lists = self.env.step(zero_actions)
        obs = self.make_policy_observation(
            {
                "observation": self._make_observations(raw_obs),
                "task": self.task_descriptions,
            }
        )
        self._reset_metrics(env_ids)

        return obs, {}

    def env_step(self, actions, *, env_ids):
        env_ids = np.asarray(env_ids, dtype=np.int64)
        self._elapsed_steps[env_ids] += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions, id=env_ids)
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self._elapsed_steps[env_ids] >= self.cfg.max_episode_steps

        step_reward = np.asarray(_reward)
        infos = self._record_metrics(step_reward, terminations, infos, env_ids)

        return {
            "observation": self._make_observations(raw_obs),
            "task": [self.task_descriptions[env_id] for env_id in env_ids],
            "next.reward": to_tensor(step_reward),
            "next.done": to_tensor(np.asarray(terminations, dtype=bool)),
            "next.truncated": to_tensor(np.asarray(truncations, dtype=bool)),
            "info": infos,
        }

    def env_close(self):
        self.env.close()
