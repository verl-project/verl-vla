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

"""Isaac Lab Arena environment adapted to the shared BaseEnv interface."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import torch
from typing_extensions import override

from verl_vla.envs.arena_env.config import load_arena_config
from verl_vla.envs.arena_env.utils import (
    apply_rl_reward_and_disable_autoreset,
    build_env_cfg_without_recorder,
    disable_lightwheel_ssl_verify,
)
from verl_vla.envs.base import BaseEnv
from verl_vla.utils.envs.action import to_tensor

logger = logging.getLogger(__name__)


class IsaacLabArenaEnv(BaseEnv):
    """Arena vector environment with BaseEnv-owned chunking, recording and teleop."""

    env_type = "arena"
    USE_POLICY_ACTION = False
    _DEFAULT_BASE_HEIGHT_COMMAND = 0.75

    def __init__(self, cfg, rank: int, world_size: int, stage_id: int = 0):
        disable_lightwheel_ssl_verify()

        self.arena_cfg = load_arena_config(cfg)
        self.seed = int(cfg.seed) + int(rank)
        self.device = cfg.get("device", "cuda:0")
        self.enable_cameras = self.arena_cfg.enable_cameras
        self.camera_names = list(self.arena_cfg.camera_names)
        self.task_description = self.arena_cfg.task_description
        self.subtask_reward = self.arena_cfg.subtask_reward
        self._elapsed_steps = np.zeros(int(cfg.num_envs), dtype=np.int32)
        self.max_episode_steps = int(cfg.max_episode_steps)

        self.action_dim = int(cfg.get("action_dim", 50))
        self.state_dim = int(cfg.get("state_dim", self.action_dim))
        self.env = None
        self.app = None
        self._stable_actions = np.zeros((int(cfg.num_envs), self.action_dim), dtype=np.float32)
        self._stable_actions[:, 46] = self._DEFAULT_BASE_HEIGHT_COMMAND

        from isaaclab.app import AppLauncher

        self.app = AppLauncher(headless=True, enable_cameras=self.enable_cameras).app
        super().__init__(cfg, rank, world_size, stage_id=stage_id)
        self._init_metrics()

    @override
    def env_init(self, *, async_reset: bool) -> None:
        del async_reset
        self._init_env()

    def _build_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            num_envs=self.num_envs,
            env_spacing=self.arena_cfg.env_spacing,
            disable_fabric=self.arena_cfg.disable_fabric,
            device=self.device,
            seed=self.seed,
            solve_relations=self.arena_cfg.solve_relations,
            mimic=False,
            enable_pinocchio=self.arena_cfg.enable_pinocchio,
            placement_seed=self.arena_cfg.placement_seed,
            resolve_on_reset=self.arena_cfg.resolve_on_reset,
            presets=self.arena_cfg.presets,
            object=self.arena_cfg.object,
            embodiment=self.arena_cfg.embodiment,
            enable_cameras=self.enable_cameras,
            teleop_device=None,
            kitchen_style=self.arena_cfg.kitchen_style,
            object_set=self.arena_cfg.object_set,
        )

    def _init_env(self) -> None:
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena_environments.cli import ExampleEnvironments

        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                logger.exception("Failed to close previous Arena env")
            import omni

            omni.usd.get_context().new_stage()

        disable_lightwheel_ssl_verify()
        args = self._build_args()
        if self.arena_cfg.env_name not in ExampleEnvironments:
            raise ValueError(
                f"Arena env '{self.arena_cfg.env_name}' not found. Available: {sorted(ExampleEnvironments.keys())}"
            )

        arena_env = ExampleEnvironments[self.arena_cfg.env_name]().get_env(args)
        task = getattr(arena_env, "task", None)
        if task is not None and hasattr(task, "get_task_description"):
            desc = task.get_task_description()
            if desc:
                self.task_description = desc

        env_builder = ArenaEnvBuilder(arena_env, args)
        env_cfg = build_env_cfg_without_recorder(env_builder)
        if self.arena_cfg.rl_success_reward:
            apply_rl_reward_and_disable_autoreset(env_cfg, subtask_reward=self.subtask_reward)
        self.env = env_builder.make_registered(env_cfg=env_cfg)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        base = getattr(self.env, "unwrapped", self.env)
        action_mgr = getattr(base, "action_manager", None)
        if action_mgr is not None:
            self.action_dim = int(action_mgr.total_action_dim)
        joint_pos_space = self.observation_space["policy"]["robot_joint_pos"]
        self.state_dim = int(np.prod(joint_pos_space.shape))
        logger.info(
            "Arena environment initialised: action_dim=%d state_dim=%d cameras=%s",
            self.action_dim,
            self.state_dim,
            self.camera_names,
        )

    @property
    def _raw_env(self):
        return getattr(self.env, "unwrapped", self.env)

    ### Metrics ###

    @property
    def _success_reward_thresh(self) -> float:
        return 1.0 - 1e-6 if self.subtask_reward else 0.0

    def _init_metrics(self) -> None:
        self.success_once = np.zeros(int(self.cfg.num_envs), dtype=bool)
        self.returns = np.zeros(int(self.cfg.num_envs), dtype=np.float32)

    def _reset_metrics(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        env_ids = np.asarray(env_ids, dtype=np.int64)
        self.success_once[env_ids] = False
        self.returns[env_ids] = 0.0
        self._elapsed_steps[env_ids] = 0
        self._stable_actions[env_ids] = 0.0
        self._stable_actions[env_ids, 46] = self._DEFAULT_BASE_HEIGHT_COMMAND

    def _record_metrics(self, step_reward, infos, env_ids):
        env_ids = np.asarray(env_ids, dtype=np.int64)
        self.returns[env_ids] += step_reward
        self.success_once[env_ids] |= step_reward > self._success_reward_thresh
        episode_info = {
            "success_once": self.success_once.copy(),
            "return": self.returns.copy(),
            "episode_len": self._elapsed_steps.copy(),
            "reward": np.divide(
                self.returns,
                self._elapsed_steps,
                out=np.zeros_like(self.returns),
                where=self._elapsed_steps != 0,
            ),
        }
        infos["episode"] = to_tensor(episode_info)
        return infos

    ### BaseEnv hooks ###

    @override
    def env_reset(
        self,
        *,
        env_ids,
        reset_state_ids=None,
        task_ids=None,
        async_reset: bool = False,
        reset_eval: bool = False,
    ):
        del reset_state_ids, task_ids, async_reset, reset_eval
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        reset_env_ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        raw_obs, infos = self._raw_env.reset(env_ids=reset_env_ids)
        self._reset_metrics(env_ids)
        obs = self._make_obs(raw_obs, env_ids=env_ids)
        if not self.USE_POLICY_ACTION:
            self._update_stable_actions_from_obs(obs["observation"], env_ids)
        return obs, infos

    @override
    def env_step(self, action, *, env_ids):
        env_ids = np.asarray(env_ids, dtype=np.int64)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self._elapsed_steps[env_ids] += 1

        raw_obs, reward, _terminations, _truncations, infos = self._raw_env.step(action)
        step_reward = self._to_numpy(reward).astype(np.float32)
        timeouts = self._elapsed_steps[env_ids] >= self.max_episode_steps
        dones = np.logical_or(step_reward > self._success_reward_thresh, timeouts)
        infos = self._record_metrics(step_reward, infos, env_ids)

        obs = self._make_obs(raw_obs, env_ids=env_ids)
        return {
            "observation": obs["observation"],
            "task": obs["task"],
            "next.reward": to_tensor(step_reward),
            "next.done": to_tensor(np.asarray(dones, dtype=bool)),
            "next.truncated": to_tensor(np.asarray(timeouts, dtype=bool)),
            "info": infos,
        }

    # Stable-action adapter: temporarily replace policy actions with a held pose.

    @override
    def step_with_teleop_and_recording(self, action, critic_value=None):
        if not self.USE_POLICY_ACTION:
            action = self._replace_with_stable_actions(action)
        return super().step_with_teleop_and_recording(action, critic_value=critic_value)

    def _replace_with_stable_actions(self, action) -> np.ndarray:
        action = np.asarray(action).copy()
        n = min(self.num_envs, action.shape[0])
        action[:n] = self._stable_actions[:n]
        return action

    @override
    def apply_teleop_action(self, action):
        action = action if self.USE_POLICY_ACTION else self._replace_with_stable_actions(action)
        action, intervention_mask = super().apply_teleop_action(action)
        if not self.USE_POLICY_ACTION:
            self._stable_actions[intervention_mask, :43] = action[intervention_mask, :43]
            self._stable_actions[intervention_mask, 46] = action[intervention_mask, 46]
        return action, intervention_mask

    def _update_stable_actions_from_obs(self, observations: list[dict[str, Any]], env_ids: np.ndarray) -> None:
        for obs, env_id in zip(observations, np.asarray(env_ids, dtype=np.int64), strict=True):
            state = np.asarray(obs["observation.state"], dtype=np.float32)
            self._stable_actions[int(env_id), :43] = state[:43]

    # End stable-action adapter.

    @override
    def env_close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
        if self.app is not None:
            self.app.close()
            self.app = None

    @override
    def get_recorder_strategy_kwargs(self) -> dict[str, Any]:
        return {
            "camera_names": tuple(self.camera_names),
            "image_shape": self._image_shape(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "fps": int(self.cfg.get("recorder", {}).get("video", {}).get("fps", 50))
            if hasattr(self.cfg, "get")
            else 50,
            "robot_type": self.arena_cfg.embodiment,
        }

    ### Observation formatting ###

    def _make_obs(self, raw_obs, *, env_ids):
        observations = self._make_observations(raw_obs, env_ids=env_ids)
        tasks = [self.task_description] * len(observations)
        return {"observation": observations, "task": tasks}

    def _make_observations(self, raw_obs, *, env_ids) -> list[dict[str, Any]]:
        env_ids = np.asarray(env_ids, dtype=np.int64)
        camera_images = self._extract_camera_images(raw_obs)
        state = self._extract_state(raw_obs)
        self.state_dim = int(state.shape[-1])

        observations = []
        for local_id, env_id in enumerate(env_ids):
            item = {key: value[env_id] for key, value in camera_images.items()}
            item["observation.state"] = state[env_id].astype(np.float32)
            observations.append(item)
        return observations

    def _extract_camera_images(self, raw_obs) -> dict[str, np.ndarray]:
        camera_obs = raw_obs.get("camera_obs", {}) if isinstance(raw_obs, dict) else {}
        images = {}
        available = list(camera_obs.keys())

        selected = self.camera_names
        if not available and self.enable_cameras:
            raise KeyError("Camera observations are missing although enable_cameras=True")
        if available and any(name not in camera_obs for name in selected):
            missing = [name for name in selected if name not in camera_obs]
            logger.warning("Arena camera(s) %s not found; available=%s", missing, available)
            selected = [name for name in selected if name in camera_obs] or [available[0]]

        for name in selected:
            rgb = camera_obs[name]
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.detach().cpu().numpy()
            rgb = np.asarray(rgb)
            images[f"observation.images.{name}"] = self._to_uint8_rgb(rgb)

        if not images:
            images["observation.images.image"] = np.zeros((self.num_envs, 1, 1, 3), dtype=np.uint8)
        return images

    def _extract_state(self, raw_obs) -> np.ndarray:
        policy_obs = raw_obs.get("policy", {}) if isinstance(raw_obs, dict) else {}
        if "robot_joint_pos" in policy_obs:
            state = policy_obs["robot_joint_pos"]
        else:
            parts = list(policy_obs.values())
            if not parts:
                return np.zeros((self.num_envs, self.state_dim), dtype=np.float32)
            tensors = [part if isinstance(part, torch.Tensor) else torch.as_tensor(part) for part in parts]
            state = torch.cat([tensor.reshape(tensor.shape[0], -1) for tensor in tensors], dim=-1)

        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        return np.asarray(state, dtype=np.float32)

    def _image_shape(self) -> tuple[int, int, int]:
        if self.enable_cameras:
            return self.arena_cfg.image_shape
        return (1, 1, 3)

    @staticmethod
    def _normalize_camera_names(camera_names) -> list[str]:
        if isinstance(camera_names, str):
            return [camera_names]
        return [str(name) for name in camera_names]

    @staticmethod
    def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.shape[-1] > 3:
            image = image[..., :3]
        if image.dtype != np.uint8:
            if image.max(initial=0) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)
