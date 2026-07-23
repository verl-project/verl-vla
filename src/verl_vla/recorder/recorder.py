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

"""Composite recorder wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typing_extensions import override

from verl_vla.recorder.async_recorder import AsyncRecorder
from verl_vla.recorder.base import BaseRecorder
from verl_vla.recorder.config import RecorderConfig
from verl_vla.recorder.impl.lerobot import LeRobotDatasetRecorder
from verl_vla.recorder.impl.video import VideoRecorder


class MultiRecorder(BaseRecorder):
    """Fan out recorder calls to all enabled concrete recorders."""

    def __init__(
        self,
        recorders: list[BaseRecorder],
    ) -> None:
        self.recorders = recorders

    @classmethod
    def from_cfg(
        cls,
        *,
        cfg: RecorderConfig,
        env_type: str,
        rank: int = 0,
        stage_id: int = 0,
        num_envs: int = 1,
        strategy_kwargs: dict[str, Any] | None = None,
    ) -> MultiRecorder | None:
        if not cfg.enable:
            return None

        recorders: list[BaseRecorder] = []
        if cfg.lerobot.enable:
            recorders.append(
                LeRobotDatasetRecorder(
                    env_type=env_type,
                    cfg=cfg.lerobot,
                    repo_id=f"{cfg.lerobot.repo_id}_rank_{rank}_stage_{stage_id}",
                    num_envs=num_envs,
                    strategy_kwargs=strategy_kwargs,
                )
            )
        if cfg.video.enable:
            recorders.append(
                VideoRecorder(
                    cfg=cfg.video,
                    env_type=env_type,
                    rank=rank,
                    stage_id=stage_id,
                    num_envs=num_envs,
                    strategy_kwargs=strategy_kwargs,
                )
            )
        if not recorders:
            return None
        recorder = cls(recorders)
        if cfg.async_enable:
            return AsyncRecorder(recorder, queue_size=cfg.async_queue_size)
        return recorder

    @override
    def record_once(
        self,
        *,
        env_id: int = 0,
        observation: dict[str, Any],
        extra: dict[str, Any] | None = None,
        action: Any,
        task: str,
        next_reward: Any = 0.0,
        next_terminated: Any = False,
        next_truncated: Any = False,
        next_success: Any = False,
        is_intervention: Any = False,
        critic_value: Any = None,
    ) -> None:
        for recorder in self.recorders:
            recorder.record_once(
                env_id=env_id,
                observation=observation,
                extra=extra,
                action=action,
                task=task,
                next_reward=next_reward,
                next_terminated=next_terminated,
                next_truncated=next_truncated,
                next_success=next_success,
                is_intervention=is_intervention,
                critic_value=critic_value,
            )

    @override
    def save_episode(self, env_id: int = 0) -> None:
        for recorder in self.recorders:
            recorder.save_episode(env_id)

    @override
    def clear_episode(self, env_id: int = 0) -> None:
        for recorder in self.recorders:
            recorder.clear_episode(env_id)

    @override
    def set_mode(self, mode: str) -> bool:
        changed = False
        for recorder in self.recorders:
            changed = recorder.set_mode(mode) or changed
        return changed

    @override
    def pop_completed(self) -> Path | None:
        for recorder in self.recorders:
            completed = recorder.pop_completed()
            if completed is not None:
                return completed
        return None

    @override
    def finalize(self) -> None:
        for recorder in self.recorders:
            recorder.finalize()
