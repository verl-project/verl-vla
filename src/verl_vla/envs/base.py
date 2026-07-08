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

from verl_vla.recorder import MultiRecorder
from verl_vla.teleop import TeleopController


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
        env_init(): Initialize simulator-specific resources.
        env_reset(env_ids, reset_eval=False):
            Reset simulator-specific state and return observations.
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
        self.auto_reset_enabled = bool(cfg.get("auto_reset", False))
        self._latest_obs = None

        self.env_init()
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
        if num_chunk_steps <= 0:
            raise ValueError(f"action chunk must contain at least one step, got shape {action.shape}")

        reward_chunks = []
        terminated_chunks = []
        truncated_chunks = []
        success_chunks = []
        step_result = None
        for step_idx in range(num_chunk_steps):
            step_actions = action[:, step_idx]
            step_values = chunk_values if chunk_values.ndim <= 1 else chunk_values[:, step_idx]

            step_result = self.step_with_teleop_and_recording(step_actions, critic_value=step_values)

            reward_chunks.append(step_result["next.reward"])
            terminated_chunks.append(step_result["next.terminated"])
            truncated_chunks.append(step_result["next.truncated"])
            success_chunks.append(step_result["next.success"])

        reward_steps = torch.stack([torch.as_tensor(chunk) for chunk in reward_chunks], dim=1)
        terminated_steps = torch.stack([torch.as_tensor(chunk) for chunk in terminated_chunks], dim=1)
        truncated_steps = torch.stack([torch.as_tensor(chunk) for chunk in truncated_chunks], dim=1)
        success_steps = torch.stack([torch.as_tensor(chunk) for chunk in success_chunks], dim=1)

        done_mask = (terminated_steps.bool() | truncated_steps.bool()).any(dim=1).numpy()
        step_result = self._reset_done_envs(step_result, np.arange(self.num_envs), done_mask=done_mask)
        obs = {
            "observation": step_result["observation"],
            "task": step_result["task"],
            "task_id": step_result["task_id"],
        }
        if "eval_episode_id" in step_result:
            obs["eval_episode_id"] = step_result["eval_episode_id"]

        self._latest_obs = obs
        return (
            obs,
            reward_steps,
            terminated_steps,
            truncated_steps,
            success_steps,
        )

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        del seed
        reset_kwargs = self._reset_kwargs_from_options(options)
        reset_eval = bool(reset_kwargs.pop("reset_eval", False))
        if self.auto_reset_enabled and self._latest_obs is not None and not reset_eval:
            return self._latest_obs, {}

        obs = self.env_reset(reset_eval=reset_eval, **reset_kwargs)
        env_ids = reset_kwargs["env_ids"]
        self._latest_obs = obs
        self.reset_teleops()
        self.reset_recorder_envs(env_ids)
        return obs, {}

    @override
    def close(self) -> None:
        self.close_recorder()
        self.close_teleops()
        self.env_close()

    def record(self) -> None:
        assert int(getattr(self, "stage_num", 1)) == 1
        assert self.num_envs == 1
        assert self.world_size == 1
        assert len(self.teleops) == 1 and self.teleops[0] is not None

        self.reset(options={"env_idx": [0]})
        done = False
        while not done:
            action = np.asarray(self.teleops[0].get_action(), dtype=np.float32).reshape(1, -1)
            step_result = self.mask_step(
                action,
                np.ones(self.num_envs, dtype=bool),
                is_intervention=np.ones(self.num_envs, dtype=bool),
                critic_value=np.zeros(self.num_envs, dtype=np.float32),
            )
            done = bool(
                np.asarray(step_result["next.terminated"], dtype=bool).any()
                or np.asarray(step_result["next.truncated"], dtype=bool).any()
            )

    ### Step Control ###

    def env_init(self) -> None:
        """Initialize subclass-owned simulator resources."""

    def env_reset(
        self,
        *,
        env_ids,
        reset_eval: bool = False,
    ):
        """Reset the underlying simulator and return observations.

        Args:
            env_ids: Environment ids to reset.
            reset_eval: If True, start a fresh evaluation queue and force a
                real reset even when auto reset has a cached latest obs.
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
                task_id: Array with shape ``[B]`` containing task ids.
                eval_episode_id: Optional array with shape ``[B]`` containing
                    eval benchmark ids for fixed-case deduplication and repeat
                    quotas.
                next.reward: Array or tensor with shape ``[B]`` containing the
                    reward after stepping.
                next.terminated: Boolean array or tensor with shape ``[B]``
                    indicating whether each stepped env terminated naturally.
                next.truncated: Boolean array or tensor with shape ``[B]``
                    indicating whether each stepped env is truncated.
                next.success: Boolean array or tensor with shape ``[B]``
                    indicating whether each stepped env achieved task success.
        """
        raise NotImplementedError

    def env_close(self) -> None:
        """Close subclass-owned simulator resources."""

    def env_benchmark_size(self) -> int:
        """Return the number of episodes in the environment's eval benchmark."""
        return 1

    def _reset_kwargs_from_options(self, options: dict[str, Any] | None) -> dict[str, Any]:
        options = options or {}
        env_ids = options.get("env_idx")
        reset_eval = bool(options.get("reset_eval", False))
        if env_ids is None or reset_eval:
            env_ids = np.arange(self.num_envs)
        return {
            "env_ids": env_ids,
            "reset_eval": reset_eval,
        }

    def mask_step(self, action, execute_mask, is_intervention, critic_value):
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
        self._update_latest_obs(env_ids, result)
        return result

    def step_with_teleop_and_recording(self, action, critic_value=None):
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

    def _reset_done_envs(self, step_result, env_ids, done_mask):
        if not self.auto_reset_enabled:
            return step_result

        reset_local_ids = np.flatnonzero(done_mask)
        if len(reset_local_ids) == 0:
            return step_result

        reset_obs = self.env_reset(
            env_ids=env_ids[reset_local_ids],
        )
        for key in ("observation", "task", "task_id", "eval_episode_id"):
            if key not in step_result or key not in reset_obs:
                continue
            for reset_idx, local_id in enumerate(reset_local_ids):
                step_result[key][local_id] = reset_obs[key][reset_idx]
        self.reset_recorder_envs(env_ids[reset_local_ids])
        return step_result

    def _slice_latest_obs(self, env_ids):
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        current_obs = {
            "observation": [self._latest_obs["observation"][env_id] for env_id in env_ids],
            "task": [self._latest_obs["task"][env_id] for env_id in env_ids],
            "task_id": [self._latest_obs["task_id"][env_id] for env_id in env_ids],
        }
        if "eval_episode_id" in self._latest_obs:
            current_obs["eval_episode_id"] = [self._latest_obs["eval_episode_id"][env_id] for env_id in env_ids]
        return current_obs

    def _update_latest_obs(self, env_ids, step_result) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        for local_id, env_id in enumerate(env_ids):
            self._latest_obs["observation"][env_id] = step_result["observation"][local_id]
            self._latest_obs["task"][env_id] = step_result["task"][local_id]
            self._latest_obs["task_id"][env_id] = step_result["task_id"][local_id]
            if "eval_episode_id" in step_result and "eval_episode_id" in self._latest_obs:
                self._latest_obs["eval_episode_id"][env_id] = step_result["eval_episode_id"][local_id]

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
        teleop_cfg = self.cfg.teleop
        if not teleop_cfg.enable:
            return []

        return [
            TeleopController.create(
                teleop_cfg,
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

    def publish_to_teleop(self, step_result, env_ids, critic_value) -> None:
        if not self.teleops:
            return

        observations = step_result["observation"]
        tasks = step_result["task"]
        rewards = step_result["next.reward"]
        terminations = step_result["next.terminated"]
        truncations = step_result["next.truncated"]
        successes = step_result["next.success"]

        for local_id, env_id in enumerate(env_ids):
            teleop = self.teleops[env_id]

            observation = observations[local_id]
            images = {key: value for key, value in observation.items() if key.startswith("observation.images.")}
            state = observation.get("observation.state")
            extra = {
                "rank": self.rank,
                "stage_id": self.stage_id,
                "reward": rewards[local_id],
                "terminated": terminations[local_id],
                "truncated": truncations[local_id],
                "success": successes[local_id],
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
        recorder_cfg = self.cfg.recorder
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

    def record_step_result(
        self,
        step_result,
        actions,
        env_ids,
        is_intervention,
        critic_value,
    ) -> None:
        if self.recorder is None:
            return

        source_obs = self._slice_latest_obs(env_ids)
        observations = source_obs["observation"]
        tasks = source_obs["task"]
        rewards = step_result["next.reward"]
        terminations = step_result["next.terminated"]
        truncations = step_result["next.truncated"]
        successes = step_result["next.success"]

        for local_id, env_id in enumerate(env_ids):
            if self._recorder_episode_done[env_id]:
                continue
            terminated = bool(terminations[local_id])
            truncated = bool(truncations[local_id])
            episode_done = terminated or truncated
            self.recorder.record_once(
                env_id=env_id,
                observation=observations[local_id],
                action=actions[local_id],
                task=tasks[local_id],
                next_reward=rewards[local_id],
                next_terminated=terminated,
                next_truncated=truncated,
                next_success=successes[local_id],
                is_intervention=is_intervention[env_id],
                critic_value=critic_value[local_id],
            )
            if episode_done:
                self.recorder.save_episode(env_id)
                self._recorder_episode_done[env_id] = True

    def finish_rollout(self) -> None:
        if self.auto_reset_enabled:
            return
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
        recorder_cfg = self.cfg.recorder
        return {
            "root": root,
            "repo_id": f"{recorder_cfg.lerobot.repo_id}_rank_{self.rank}_stage_{self.stage_id}",
        }
