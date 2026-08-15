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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from verl.base_config import BaseConfig


@dataclass
class ActionInterpolationConfig(BaseConfig):
    """Temporal upsampling shared by all action executors."""

    enable: bool = False
    factor: int = 2

    def __post_init__(self) -> None:
        if self.factor < 2:
            raise ValueError(f"action interpolation factor must be at least 2, got {self.factor}")


@dataclass
class SerialActionSmoothingConfig(BaseConfig):
    """Plan overlap smoothing owned by the serial executor."""

    enable: bool = False
    execution_steps: int = 80
    blend_steps: int = 20

    def __post_init__(self) -> None:
        if self.execution_steps <= 0:
            raise ValueError(f"serial execution_steps must be positive, got {self.execution_steps}")
        if self.blend_steps < 2:
            raise ValueError(f"serial blend_steps must be at least 2, got {self.blend_steps}")


@dataclass
class ActionExecutionConfig(BaseConfig):
    """Environment-owned action transformation and scheduling configuration."""

    mode: str = "serial"
    replan_after_steps: int = 1
    smooth_overlapping_actions: bool = False
    interpolation: ActionInterpolationConfig = field(default_factory=ActionInterpolationConfig)
    serial_smoothing: SerialActionSmoothingConfig = field(default_factory=SerialActionSmoothingConfig)

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
FinishFn = Callable[[Sequence[ExecutedStep]], tuple[Any, np.ndarray]]
SnapshotFn = Callable[[tuple[int, int]], tuple[Any, np.ndarray]]


class ActionExecutor(ABC):
    """Exchange action chunks for completed execution intervals."""

    def __init__(self, interpolation: ActionInterpolationConfig) -> None:
        self._interpolation_factor = interpolation.factor if interpolation.enable else 1

    def _prepare(self, actions: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        actions, values = _normalize_action_chunk(actions, values)
        return interpolate_action_chunk(actions, values, factor=self._interpolation_factor)

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
    """Synchronously execute complete chunks or smoothed plan prefixes."""

    def __init__(
        self,
        step_fn: StepFn,
        finish_fn: FinishFn,
        *,
        interpolation: ActionInterpolationConfig | None = None,
        smoothing: SerialActionSmoothingConfig | None = None,
    ) -> None:
        super().__init__(interpolation or ActionInterpolationConfig())
        self._step_fn = step_fn
        self._finish_fn = finish_fn
        self._smoothing = smoothing or SerialActionSmoothingConfig()
        self._previous_actions: np.ndarray | None = None
        self._previous_valid: np.ndarray | None = None

    def exchange(self, actions: np.ndarray, values: np.ndarray) -> ExecutionSlice:
        actions, values = self._prepare(actions, values)
        if self._smoothing.enable:
            actions = smooth_serial_action_chunks(
                self._previous_actions,
                actions,
                execution_steps=self._smoothing.execution_steps,
                blend_steps=self._smoothing.blend_steps,
                previous_valid=self._previous_valid,
            )
            self._previous_actions = actions.copy()
            self._previous_valid = np.ones(actions.shape[0], dtype=bool)
            actions = actions[:, : self._smoothing.execution_steps]
            values = values[:, : self._smoothing.execution_steps]
        steps = [
            self._step_fn(
                actions[:, step_idx].copy(),
                values[:, step_idx].copy(),
                step_idx == 0,
            )
            for step_idx in range(actions.shape[1])
        ]
        result, reset_mask = self._finish_fn(steps)
        if self._previous_valid is not None:
            self._previous_valid &= ~np.asarray(reset_mask, dtype=bool).reshape(-1)
        return ExecutionSlice(result=result)

    def reset(self) -> None:
        self._previous_actions = None
        self._previous_valid = None

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
        interpolation: ActionInterpolationConfig | None = None,
    ) -> None:
        super().__init__(interpolation or ActionInterpolationConfig())
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
        feedback_shape = actions.shape[:2]
        actions, values = self._prepare(actions, values)
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
                result, reset_mask = self._snapshot_fn(feedback_shape)
                if np.asarray(reset_mask, dtype=bool).any():
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


def interpolate_action_chunk(
    actions: np.ndarray,
    values: np.ndarray,
    *,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly upsample actions and their per-step values along time."""
    if factor == 1:
        return actions, values

    output_steps = actions.shape[1] * factor
    positions = np.linspace(0.0, actions.shape[1] - 1, output_steps)
    left = np.floor(positions).astype(np.int64)
    right = np.ceil(positions).astype(np.int64)
    weight = positions - left
    interpolated_actions = (1.0 - weight.reshape(1, -1, 1)) * actions[:, left] + weight.reshape(1, -1, 1) * actions[
        :, right
    ]
    interpolated_values = (1.0 - weight.reshape(1, -1)) * values[:, left] + weight.reshape(1, -1) * values[:, right]
    return interpolated_actions.astype(actions.dtype, copy=False), interpolated_values.astype(values.dtype, copy=False)


def smooth_action_chunk_overlap(previous_actions: np.ndarray, next_actions: np.ndarray) -> np.ndarray:
    """Blend the actual queued tail into the temporally aligned new async plan."""
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


def smooth_serial_action_chunks(
    previous_actions: np.ndarray | None,
    next_actions: np.ndarray,
    *,
    execution_steps: int,
    blend_steps: int,
    previous_valid: np.ndarray | None,
) -> np.ndarray:
    """Blend the previous unexecuted tail into the next serial action plan."""
    plan_steps = next_actions.shape[1]
    if execution_steps + blend_steps > plan_steps:
        raise ValueError(
            "serial action smoothing requires execution_steps + blend_steps <= action plan steps, "
            f"got {execution_steps} + {blend_steps} > {plan_steps}"
        )

    smoothed_actions = next_actions.copy()
    if previous_actions is None:
        return smoothed_actions
    if previous_actions.shape != next_actions.shape:
        raise ValueError(
            "consecutive serial action plans must preserve shape, "
            f"got {previous_actions.shape} then {next_actions.shape}"
        )

    valid = np.ones(next_actions.shape[0], dtype=bool) if previous_valid is None else previous_valid
    if valid.any():
        weight = np.linspace(0.0, 1.0, blend_steps, dtype=next_actions.dtype)
        weight = (weight * weight * (3.0 - 2.0 * weight)).reshape(1, blend_steps, 1)
        previous_tail = previous_actions[valid, execution_steps : execution_steps + blend_steps]
        smoothed_actions[valid, :blend_steps] = (1.0 - weight) * previous_tail + weight * smoothed_actions[
            valid, :blend_steps
        ]
    return smoothed_actions
