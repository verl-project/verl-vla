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

"""Action scheduling for synchronous and continuously executing environments."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from verl.base_config import BaseConfig


@dataclass
class ActionExecutionConfig(BaseConfig):
    """Environment-owned action scheduling configuration."""

    mode: str = "serial"
    replan_after_steps: int = 1
    smooth_overlapping_actions: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"serial", "async"}:
            raise ValueError(f"Unsupported action execution mode: {self.mode}")
        if self.replan_after_steps <= 0:
            raise ValueError(f"replan_after_steps must be positive, got {self.replan_after_steps}")


@dataclass(frozen=True)
class ExecutedStep:
    """One committed control tick and its environment-owned feedback."""

    feedback: Any


@dataclass(frozen=True)
class QueuedAction:
    """One control tick waiting to be consumed by the async executor."""

    action: np.ndarray
    value: np.ndarray
    chunk_started: bool


@dataclass(frozen=True)
class ExecutionSlice:
    """A completed interval returned to the environment caller."""

    result: Any


StepFn = Callable[[np.ndarray, np.ndarray, bool], ExecutedStep]
FinishFn = Callable[[Sequence[ExecutedStep]], Any]
SnapshotFn = Callable[[tuple[int, int]], tuple[Any, bool]]


class ActionExecutor(ABC):
    """Exchange action chunks for completed execution intervals."""

    @abstractmethod
    def exchange(self, actions: np.ndarray, values: np.ndarray) -> ExecutionSlice:
        """Submit an action chunk and return the next execution interval."""

    @abstractmethod
    def reset(self) -> None:
        """Stop execution at a tick boundary and discard all queued state."""

    @abstractmethod
    def close(self) -> None:
        """Stop the executor and release its thread, if any."""


class SerialActionExecutor(ActionExecutor):
    """Execute each submitted action chunk completely in the caller thread."""

    def __init__(self, step_fn: StepFn, finish_fn: FinishFn) -> None:
        self._step_fn = step_fn
        self._finish_fn = finish_fn

    def exchange(self, actions: np.ndarray, values: np.ndarray) -> ExecutionSlice:
        actions, values = _normalize_action_chunk(actions, values)
        steps = [
            self._step_fn(
                actions[:, step_idx].copy(),
                values[:, step_idx].copy(),
                step_idx == 0,
            )
            for step_idx in range(actions.shape[1])
        ]
        return _finish_slice(steps, self._finish_fn)

    def reset(self) -> None:
        return

    def close(self) -> None:
        return


class AsyncActionExecutor(ActionExecutor):
    """Execute continuously and replace queued actions at control-tick boundaries.

    Only the control thread removes actions from the queue. A caller replaces
    the queued actions under the same lock. An action already removed by the
    control thread is therefore committed, and the replacement takes effect on
    the following tick.
    """

    def __init__(
        self,
        step_fn: StepFn,
        snapshot_fn: SnapshotFn,
        *,
        replan_after_steps: int,
        smooth_overlapping_actions: bool = False,
    ) -> None:
        if replan_after_steps <= 0:
            raise ValueError(f"replan_after_steps must be positive, got {replan_after_steps}")
        self._step_fn = step_fn
        self._snapshot_fn = snapshot_fn
        self._replan_after_steps = int(replan_after_steps)
        self._smooth_overlapping_actions = bool(smooth_overlapping_actions)
        self._condition = threading.Condition()
        self._action_queue: deque[QueuedAction] = deque()
        self._completed_steps = 0
        self._chunk_origin_step = 0
        self._return_target: int | None = None
        self._paused = True
        self._in_step = False
        self._closed = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="action-executor", daemon=True)
        self._thread.start()

    def exchange(self, actions: np.ndarray, values: np.ndarray) -> ExecutionSlice:
        actions, values = _normalize_action_chunk(actions, values)
        if actions.shape[1] < self._replan_after_steps:
            raise ValueError(
                "async action chunks must cover one replan interval, "
                f"got {actions.shape[1]} < {self._replan_after_steps}"
            )
        with self._condition:
            self._raise_if_failed()
            if self._return_target is not None:
                raise RuntimeError("async action executor supports one exchange call at a time")
            replacement_step = self._completed_steps + int(self._in_step)
            elapsed_steps = max(0, replacement_step - self._chunk_origin_step)
            replacement_actions = actions[:, elapsed_steps:]
            if self._smooth_overlapping_actions and self._action_queue:
                queued_actions = np.stack([queued.action for queued in self._action_queue], axis=1)
                replacement_actions = smooth_action_chunk_overlap(queued_actions, replacement_actions)
            action_queue = [
                QueuedAction(
                    action=replacement_actions[:, queue_idx].copy(),
                    value=values[:, elapsed_steps + queue_idx].copy(),
                    chunk_started=queue_idx == 0,
                )
                for queue_idx in range(replacement_actions.shape[1])
            ]
            self._action_queue.clear()
            self._action_queue.extend(action_queue)
            self._paused = False
            self._return_target = self._chunk_origin_step + self._replan_after_steps
            self._condition.notify_all()
            while self._completed_steps < self._return_target:
                self._raise_if_failed()
                self._condition.wait()
            try:
                result, episode_started = self._snapshot_fn(actions.shape[:2])
                if episode_started:
                    self._action_queue.clear()
                self._chunk_origin_step = self._completed_steps
                return ExecutionSlice(result=result)
            finally:
                self._return_target = None
                self._condition.notify_all()

    def reset(self) -> None:
        with self._condition:
            self._paused = True
            self._action_queue.clear()
            self._return_target = None
            while self._in_step:
                self._condition.wait()
            self._completed_steps = 0
            self._chunk_origin_step = 0
            self._raise_if_failed()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._closed and (
                        self._paused
                        or not self._action_queue
                        or (self._return_target is not None and self._completed_steps >= self._return_target)
                    ):
                        self._condition.wait()
                    if self._closed:
                        return
                    queued_action = self._action_queue.popleft()
                    self._in_step = True

                self._step_fn(
                    queued_action.action,
                    queued_action.value,
                    queued_action.chunk_started,
                )

                with self._condition:
                    self._completed_steps += 1
                    self._in_step = False
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._in_step = False
                self._error = error
                self._paused = True
                self._condition.notify_all()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("async action executor failed") from self._error


def _normalize_action_chunk(actions: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions).copy()
    values = np.asarray(values).copy()
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [batch, time, action_dim], got {actions.shape}")
    if actions.shape[1] <= 0:
        raise ValueError(f"action chunk must contain at least one step, got {actions.shape}")
    if values.shape != actions.shape[:2]:
        raise ValueError(f"values must have shape {actions.shape[:2]}, got {values.shape}")
    actions.setflags(write=False)
    values.setflags(write=False)
    return actions, values


def smooth_action_chunk_overlap(previous_actions: np.ndarray, next_actions: np.ndarray) -> np.ndarray:
    """Blend the queued plan tail into the temporally aligned new plan."""
    overlap_steps = min(previous_actions.shape[1], next_actions.shape[1])
    smoothed_actions = next_actions.copy()
    if overlap_steps == 0:
        return smoothed_actions

    weight = np.linspace(0.0, 1.0, overlap_steps, dtype=next_actions.dtype)
    weight = (weight * weight * (3.0 - 2.0 * weight)).reshape(1, overlap_steps, 1)
    smoothed_actions[:, :overlap_steps] = (1.0 - weight) * previous_actions[:, :overlap_steps] + weight * (
        next_actions[:, :overlap_steps]
    )
    return smoothed_actions


def _finish_slice(steps: Sequence[ExecutedStep], finish_fn: FinishFn) -> ExecutionSlice:
    result = finish_fn(steps)
    return ExecutionSlice(result=result)
