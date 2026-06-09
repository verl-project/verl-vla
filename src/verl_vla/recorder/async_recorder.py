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

"""Asynchronous recorder wrapper."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from typing_extensions import override

from verl_vla.recorder.base import BaseRecorder


@dataclass(frozen=True)
class _RecorderCommand:
    name: str
    kwargs: dict[str, Any]
    response: queue.Queue | None = None


class AsyncRecorder(BaseRecorder):
    """Run a recorder on a background thread with the same public interface.

    ``record_once()``, ``save_episode()``, and ``clear_episode()`` enqueue work and
    return quickly. ``pop_completed()`` and ``finalize()`` are synchronization
    points: they wait for all previous commands to finish before touching the
    underlying recorder.
    """

    def __init__(self, recorder: BaseRecorder, *, queue_size: int = 256) -> None:
        self.recorder = recorder
        self._commands: queue.Queue[_RecorderCommand | None] = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="async-recorder", daemon=True)
        self._thread.start()

    @override
    def record_once(
        self,
        *,
        env_id: int = 0,
        observation: dict[str, Any],
        action: Any,
        task: str,
        next_reward: Any = 0.0,
        next_done: Any = False,
        next_truncated: Any = False,
        is_intervention: Any = False,
        critic_value: Any = None,
    ) -> None:
        self._enqueue(
            "record_once",
            env_id=env_id,
            observation=_copy_value(observation),
            action=_copy_value(action),
            task=task,
            next_reward=_copy_value(next_reward),
            next_done=_copy_value(next_done),
            next_truncated=_copy_value(next_truncated),
            is_intervention=_copy_value(is_intervention),
            critic_value=_copy_value(critic_value),
        )

    @override
    def save_episode(self, env_id: int = 0) -> None:
        self._enqueue("save_episode", env_id=env_id)

    @override
    def clear_episode(self, env_id: int = 0) -> None:
        self._enqueue("clear_episode", env_id=env_id)

    @override
    def pop_completed(self) -> Path | None:
        return self._run_sync("pop_completed")

    @override
    def finalize(self) -> None:
        if self._closed:
            return
        try:
            self._run_sync("finalize")
        finally:
            self._closed = True
            self._commands.put(None)
            self._thread.join()

    def _enqueue(self, name: str, **kwargs) -> None:
        self._raise_worker_error()
        if self._closed:
            raise RuntimeError("AsyncRecorder is already finalized.")
        self._commands.put(_RecorderCommand(name=name, kwargs=kwargs))

    def _run_sync(self, name: str, **kwargs):
        self._raise_worker_error()
        if self._closed:
            raise RuntimeError("AsyncRecorder is already finalized.")
        response: queue.Queue = queue.Queue(maxsize=1)
        self._commands.put(_RecorderCommand(name=name, kwargs=kwargs, response=response))
        result = response.get()
        if isinstance(result, BaseException):
            raise result
        self._raise_worker_error()
        return result

    def _run(self) -> None:
        while True:
            command = self._commands.get()
            try:
                if command is None:
                    return
                result = getattr(self.recorder, command.name)(**command.kwargs)
                if command.response is not None:
                    command.response.put(result)
            except BaseException as exc:
                self._error = exc
                if command is not None and command.response is not None:
                    command.response.put(exc)
            finally:
                self._commands.task_done()

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("AsyncRecorder worker failed.") from self._error


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    return value
