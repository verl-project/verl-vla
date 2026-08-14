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

from typing import Any

import gymnasium as gym
import numpy as np
from typing_extensions import override

from verl_vla.envs.base import BaseEnv
from verl_vla.envs.piper.backend import PiperBackend


class PiperEnv(BaseEnv):
    """Configured Piper arms controlled directly in absolute joint space."""

    env_type = "piper"

    def __init__(
        self,
        cfg,
        rank: int,
        world_size: int,
        stage_id: int = 0,
        stage_num: int = 1,
        only_eval: bool = False,
    ) -> None:
        del stage_num, only_eval
        self.piper_cfg = cfg.simulator.piper
        if int(cfg.num_envs) != 1:
            raise ValueError(f"PiperEnv only supports num_envs=1, got {cfg.num_envs}")
        if int(world_size) != 1:
            raise ValueError(f"PiperEnv requires one EnvWorker to own the CAN lifecycle, got {world_size}")

        self.action_dim = int(self.piper_cfg.action_dim)
        self.state_dim = int(self.piper_cfg.state_dim)
        self.task_description = str(self.piper_cfg.task_description)
        self.task_descriptions = [self.task_description]
        self._backend = PiperBackend(self.piper_cfg)
        self._step_id = 0
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "observation.state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.state_dim,),
                    dtype=np.float32,
                )
            }
        )
        super().__init__(cfg, rank, world_size, stage_id=stage_id)

    @override
    def env_init(self) -> None:
        self._backend.start()

    @override
    def env_reset(self, *, env_ids, reset_eval: bool = False, extra=None):
        del reset_eval, extra
        self._validate_env_ids(env_ids)
        self._step_id = 0
        self._backend.reset()
        return self._step_result(
            reward=np.zeros(1, dtype=np.float32),
            terminated=np.zeros(1, dtype=bool),
            truncated=np.zeros(1, dtype=bool),
            success=np.zeros(1, dtype=bool),
        )

    @override
    def env_step(self, action, *, env_ids):
        self._validate_env_ids(env_ids)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (1, self.action_dim):
            raise ValueError(f"Piper action must have shape [1, {self.action_dim}], got {action.shape}")
        action[0] = self._backend.apply_action(action[0])
        self._step_id += 1
        return self._step_result(
            reward=np.zeros(1, dtype=np.float32),
            terminated=np.zeros(1, dtype=bool),
            truncated=np.zeros(1, dtype=bool),
            success=np.zeros(1, dtype=bool),
        )

    @override
    def env_close(self) -> None:
        self._backend.close()

    @override
    def get_teleop_strategy_kwargs(self, device_type: str) -> dict[str, Any]:
        kwargs = {
            "task_delta_to_action": self._backend.task_delta_to_action,
            "task_target_sync": self._backend.sync_task_target,
        }
        if device_type == "xr_controller":
            kwargs["arm_rotation_reader"] = self._backend.read_arm_rotations
        return kwargs

    @override
    def get_recorder_strategy_kwargs(self) -> dict[str, Any]:
        return {
            "camera_names": tuple(camera.name for camera in self.piper_cfg.cameras),
            "image_shapes": {
                camera.name: (int(camera.height), int(camera.width), 3) for camera in self.piper_cfg.cameras
            },
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "fps": int(self.cfg.recorder.video.fps),
            "robot_type": "piper",
        }

    def _validate_env_ids(self, env_ids) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if len(env_ids) != 1 or int(env_ids[0]) != 0:
            raise ValueError(f"PiperEnv only supports env_id 0, got {env_ids.tolist()}")

    def _step_result(self, *, reward, terminated, truncated, success) -> dict[str, Any]:
        return {
            "observation": [self._observation()],
            "task": [self.task_description],
            "task_id": np.zeros(1, dtype=np.int64),
            "next.reward": reward,
            "next.terminated": terminated,
            "next.truncated": truncated,
            "next.success": success,
        }

    def _observation(self) -> dict[str, np.ndarray]:
        obs = {"observation.state": self._backend.read_state()}
        for name, image in self._backend.read_images().items():
            obs[f"observation.images.{name}"] = image
        return obs
