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

"""Base environment scaffolding for shared teleop and recorder utilities."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from typing_extensions import override

from verl_vla.recorder import MultiRecorder, load_recorder_config
from verl_vla.teleop import TeleopController, load_teleop_config


class BaseEnv(gym.Env):
    """Shared vector-env wrapper for teleop and recording.

    Public interfaces exposed to callers:
        step(action, **kwargs): Gym-style API that accepts chunked actions with
            shape ``[B, T, D]``. It executes each chunk step, applies active
            teleop interventions, publishes observations to teleop servers, and
            records completed frames.
        reset(seed=None, options=None): Gym-style reset wrapper. It calls the
            subclass reset hook, then resets teleop devices and recorder
            buffers for the selected env ids.
        close(): Gym-style close wrapper. It finalizes recorder state, closes
            teleop servers, and then closes subclass-owned simulator resources.
        finish_rollout(): Flush any buffered recorder episodes for all envs.
        pop_completed_dataset(): Return a completed recorder dataset root for
            worker-side aggregation, if one exists.

    Hooks implemented by subclasses:
        env_type: Class attribute used for teleop and recorder strategy lookup.
        env_reset(env_ids, reset_state_ids=None, task_ids=None): Reset
            simulator-specific state and return ``(obs, infos)``.
        env_step(action, env_ids): Step simulator-specific state once with
            action shape ``[B, D]`` and return the standard step-result dict
            documented on ``env_step``.
        env_close(): Close simulator-specific resources.
        get_recorder_strategy_kwargs(): Optional hook for recorder strategy
            configuration such as image shape.
    """

    env_type: str

    def __init__(self, cfg, rank: int, world_size: int, stage_id: int = 0) -> None:
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.stage_id = stage_id
        self.num_envs = int(cfg.num_envs)

        self.teleops = self.create_teleops()
        self.recorder = self.create_recorder()
        self._recorder_episode_done = np.zeros(self.num_envs, dtype=bool)

    ### Gym Environment API ###
    @override
    def step(self, action, **kwargs) -> Any:
        """Step the environment with chunked actions.

        Args:
            action: Action array or tensor with shape ``[B, T, D]``.
            **kwargs: Optional gym-compatible side inputs. ``chunk_values`` is
                consumed as per-env values passed to video and teleop.
        """
        action = self._to_numpy(action, copy=True)
        chunk_values = self._to_numpy(kwargs.pop("chunk_values", None))
        if chunk_values is None:
            chunk_values = np.zeros(action.shape[0], dtype=np.float32)
        num_chunk_steps = action.shape[1]

        reward_chunks = []
        done_chunks = []
        truncated_chunks = []
        obs = None
        infos = None
        for step_idx in range(num_chunk_steps):
            step_actions = action[:, step_idx]
            step_values = chunk_values if chunk_values.ndim <= 1 else chunk_values[:, step_idx]

            step_result = self.step_with_teleop_and_recording(step_actions, critic_value=step_values)
            obs = {
                "observation": step_result["observation"],
                "task": step_result["task"],
            }
            infos = step_result.get("info", {})

            reward_chunks.append(step_result["next.reward"])
            done_chunks.append(step_result["next.done"])
            truncated_chunks.append(step_result["next.truncated"])

        return (
            obs,
            torch.stack([torch.as_tensor(chunk) for chunk in reward_chunks], dim=1),
            torch.stack([torch.as_tensor(chunk) for chunk in done_chunks], dim=1),
            torch.stack([torch.as_tensor(chunk) for chunk in truncated_chunks], dim=1),
            infos,
        )

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        del seed
        reset_kwargs = self._reset_kwargs_from_options(options)
        obs, infos = self.env_reset(**reset_kwargs)
        env_ids = reset_kwargs["env_ids"]
        self.reset_teleops()
        self.reset_recorder_envs(env_ids)
        return obs, infos

    @override
    def close(self) -> None:
        self.close_recorder()
        self.close_teleops()
        self.env_close()

    ### Step Control ###

    def env_reset(self, *, env_ids, reset_state_ids=None, task_ids=None):
        """Reset the underlying simulator and return ``(obs, infos)``.

        Args:
            env_ids: Environment ids to reset.
            reset_state_ids: Optional environment-specific reset state ids.
            task_ids: Optional task ids from the rollout sampler. Some envs use
                them directly; others, such as LIBERO, infer task ids from
                ``reset_state_ids`` and may intentionally ignore this argument.
        """
        raise NotImplementedError

    def env_step(self, action, *, env_ids):
        """Step the underlying vectorized simulator once.

        Subclasses should implement this method. ``BaseEnv.step`` handles chunk
        execution, teleop publishing, and recorder side effects around it.

        Args:
            action: Action array with shape ``[B, D]`` for the envs stepped in
                this call.
            env_ids: Env ids being stepped when this is a partial step.

        Returns:
            A step-result dict with at least these keys:
                observation: Sequence or batch of per-env observations. Each
                    per-env observation should contain normalized keys such as
                    ``observation.images.*`` and ``observation.state``.
                task: Sequence with shape ``[B]`` containing task descriptions
                    or ids for each stepped env.
                next.reward: Array or tensor with shape ``[B]`` containing the
                    reward after stepping.
                next.done: Boolean array or tensor with shape ``[B]`` indicating
                    whether each stepped env is done.
                next.truncated: Boolean array or tensor with shape ``[B]``
                    indicating whether each stepped env is truncated.
        """
        raise NotImplementedError

    def env_close(self) -> None:
        """Close subclass-owned simulator resources."""

    def _reset_kwargs_from_options(self, options: dict[str, Any] | None) -> dict[str, Any]:
        options = options or {}
        env_ids = options.get("env_idx")
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        return {
            "env_ids": env_ids,
            "reset_state_ids": options.get("reset_state_ids"),
            "task_ids": options.get("task_ids"),
        }

    def mask_step(self, action, execute_mask, is_intervention, critic_value=None):
        is_intervention = np.asarray(is_intervention, dtype=bool)
        env_ids = np.flatnonzero(execute_mask)
        action = action[env_ids]
        result = self.env_step(action, env_ids=env_ids)
        critic_value = critic_value[env_ids]
        self.publish_to_teleop(result, env_ids=env_ids, critic_value=critic_value)
        self.record_step_result(
            result,
            action,
            env_ids=env_ids,
            is_intervention=is_intervention,
            critic_value=critic_value,
        )
        return result

    def step_with_teleop_and_recording(self, action, critic_value=None):
        critic_value = self._to_numpy(critic_value)
        if critic_value is None:
            critic_value = np.zeros(self.num_envs, dtype=np.float32)
        is_intervened = np.zeros(self.num_envs, dtype=bool)

        while self.is_intervening():
            next_action, intervention_mask = self.apply_teleop_action(action)
            need_execute = is_intervened & intervention_mask
            if need_execute.any():
                self.mask_step(
                    action,
                    need_execute,
                    is_intervention=intervention_mask,
                    critic_value=critic_value,
                )

            action[intervention_mask] = next_action[intervention_mask]
            is_intervened |= intervention_mask

        execute_mask = np.ones(self.num_envs, dtype=bool)
        return self.mask_step(action, execute_mask, is_intervention=is_intervened, critic_value=critic_value)

    @staticmethod
    def _to_numpy(value, *, copy: bool = False):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        else:
            value = np.asarray(value)
        return value.copy() if copy else value

    ### Teleop Control ###

    def create_teleops(self):
        teleop_cfg = load_teleop_config(self.cfg)
        if not teleop_cfg.enable or not teleop_cfg.devices:
            return []

        return [
            TeleopController.create(
                self.cfg,
                rank=self.rank,
                stage_id=self.stage_id,
                env_id=env_id,
                env_type=self.env_type,
                device=teleop_cfg.device or "keyboard",
            )
            for env_id in range(self.num_envs)
        ]

    def is_intervening(self) -> bool:
        return any(teleop.is_intervening() for teleop in self.teleops)

    def apply_teleop_action(self, action):
        action = np.asarray(action).copy()
        intervention_mask = np.zeros(self.num_envs, dtype=bool)
        for env_id, teleop in enumerate(self.teleops):
            if teleop.is_intervening():
                intervention_mask[env_id] = True
                action[env_id] = teleop.apply_action(action[env_id])
        return action, intervention_mask

    def reset_teleops(self) -> None:
        for teleop in self.teleops:
            teleop.reset()

    def close_teleops(self) -> None:
        for teleop in self.teleops:
            teleop.close()
        self.teleops = []

    def publish_to_teleop(self, step_result, env_ids, critic_value=None) -> None:
        if not self.teleops:
            return

        observations = step_result["observation"]
        tasks = step_result["task"]
        rewards = step_result["next.reward"]
        dones = step_result["next.done"]
        truncations = step_result["next.truncated"]

        for local_id, env_id in enumerate(env_ids):
            teleop = self.teleops[env_id]

            observation = observations[local_id]
            images = {key: value for key, value in observation.items() if key.startswith("observation.images.")}
            state = observation.get("observation.state")
            extra = {
                "rank": self.rank,
                "stage_id": self.stage_id,
                "reward": rewards[local_id],
                "done": dones[local_id],
                "truncated": truncations[local_id],
                "critic_value": critic_value[local_id],
            }
            teleop.publish_obs(
                images=images,
                state=state,
                extra=extra,
                task_description=str(tasks[local_id]),
            )

    ### Recorder Control ###

    def create_recorder(self):
        recorder_cfg = load_recorder_config(self.cfg)
        if not recorder_cfg.enable:
            return None

        return MultiRecorder.from_cfg(
            cfg=recorder_cfg,
            env_type=self.env_type,
            rank=self.rank,
            stage_id=self.stage_id,
            num_envs=self.num_envs,
            strategy_kwargs=self.get_recorder_strategy_kwargs(),
        )

    def get_recorder_strategy_kwargs(self) -> dict[str, Any]:
        return {}

    def reset_recorder_envs(self, env_ids) -> None:
        if self.recorder is None:
            return
        for env_id in np.asarray(env_ids, dtype=np.int64).reshape(-1):
            self._recorder_episode_done[int(env_id)] = False
            self.recorder.clear_episode(int(env_id))

    def record_step_result(self, step_result, actions, env_ids, is_intervention, critic_value=None) -> None:
        if self.recorder is None:
            return

        observations = step_result["observation"]
        tasks = step_result["task"]
        rewards = step_result["next.reward"]
        dones = step_result["next.done"]
        truncations = step_result["next.truncated"]

        for local_id, env_id in enumerate(env_ids):
            if self._recorder_episode_done[env_id]:
                continue
            done = bool(dones[local_id])
            self.recorder.record_once(
                env_id=env_id,
                observation=observations[local_id],
                action=actions[local_id],
                task=tasks[local_id],
                next_reward=rewards[local_id],
                next_done=done,
                next_truncated=truncations[local_id],
                is_intervention=is_intervention[env_id],
                critic_value=critic_value[local_id],
            )
            if done:
                self.recorder.save_episode(env_id)
                self._recorder_episode_done[env_id] = True

    def finish_rollout(self) -> None:
        if self.recorder is None:
            return
        for env_id in range(self.num_envs):
            if not self._recorder_episode_done[env_id]:
                self.recorder.save_episode(env_id)
                self._recorder_episode_done[env_id] = True

    def close_recorder(self) -> None:
        if self.recorder is None:
            return

        completed_root = self.recorder.pop_completed()
        if completed_root is None:
            self.recorder.finalize()
        self.recorder = None

    def pop_completed_dataset(self):
        if self.recorder is None:
            return None
        root = self.recorder.pop_completed()
        if root is None:
            return None
        recorder_cfg = load_recorder_config(self.cfg)
        return {
            "root": root,
            "repo_id": f"{recorder_cfg.lerobot.repo_id}_rank_{self.rank}_stage_{self.stage_id}",
        }
