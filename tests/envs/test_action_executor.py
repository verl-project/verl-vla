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

import threading
import time

import numpy as np

from verl_vla.envs.action_executor import (
    ActionInterpolationConfig,
    AsyncActionExecutor,
    ExecutedStep,
    SerialActionExecutor,
    SerialActionSmoothingConfig,
    interpolate_action_chunk,
    smooth_action_chunk_overlap,
)


def test_action_chunk_interpolation_upsamples_actions_and_values() -> None:
    actions = np.asarray([[[0.0], [3.0], [6.0]]], dtype=np.float32)
    values = np.asarray([[0.0, 3.0, 6.0]], dtype=np.float32)

    interpolated_actions, interpolated_values = interpolate_action_chunk(actions, values, factor=2)

    np.testing.assert_allclose(interpolated_actions[0, :, 0], np.linspace(0.0, 6.0, 6))
    np.testing.assert_allclose(interpolated_values[0], np.linspace(0.0, 6.0, 6))


def test_smooth_action_chunk_overlap_blends_only_shared_steps() -> None:
    previous = np.zeros((1, 3, 1), dtype=np.float32)
    next_actions = np.full((1, 4, 1), 10.0, dtype=np.float32)

    smoothed = smooth_action_chunk_overlap(previous, next_actions)

    np.testing.assert_allclose(smoothed[0, :, 0], [0.0, 5.0, 10.0, 10.0])


def test_async_executor_replaces_queue_after_committed_tick() -> None:
    selected_actions: list[float] = []
    selected = threading.Condition()
    release_third_action = threading.Event()

    def execute_step(action, value, chunk_started):
        del value, chunk_started
        selected_action = float(action[0, 0])
        with selected:
            selected_actions.append(selected_action)
            selected.notify_all()
        if selected_action == 3.0:
            release_third_action.wait(timeout=2.0)
        else:
            time.sleep(0.005)
        return ExecutedStep(feedback=selected_action)

    executor = AsyncActionExecutor(
        execute_step,
        lambda feedback_shape: (feedback_shape, False),
        replan_after_steps=2,
    )
    first_result = {}

    first_thread = threading.Thread(
        target=lambda: first_result.setdefault(
            "slice",
            executor.exchange(
                np.asarray([[[1.0], [2.0], [3.0]]]),
                np.zeros((1, 3)),
            ),
        )
    )
    first_thread.start()
    first_thread.join(timeout=2.0)
    assert not first_thread.is_alive()
    assert first_result["slice"].result == (1, 3)

    with selected:
        selected.wait_for(lambda: selected_actions[:3] == [1.0, 2.0, 3.0], timeout=2.0)

    second_result = {}
    second_thread = threading.Thread(
        target=lambda: second_result.setdefault(
            "slice",
            executor.exchange(
                np.asarray([[[10.0], [11.0]]]),
                np.zeros((1, 2)),
            ),
        )
    )
    second_thread.start()
    release_third_action.set()
    second_thread.join(timeout=2.0)
    assert not second_thread.is_alive()
    assert second_result["slice"].result == (1, 2)
    assert selected_actions[:4] == [1.0, 2.0, 3.0, 11.0]

    executor.close()


def test_async_executor_interpolates_actions_but_preserves_input_feedback_shape() -> None:
    selected_actions: list[float] = []

    def execute_step(action, value, chunk_started):
        del value, chunk_started
        selected_actions.append(float(action[0, 0]))
        return ExecutedStep(feedback=None)

    executor = AsyncActionExecutor(
        execute_step,
        lambda feedback_shape: (feedback_shape, np.zeros(1, dtype=bool)),
        replan_after_steps=3,
        interpolation=ActionInterpolationConfig(enable=True, factor=2),
    )

    execution = executor.exchange(
        np.asarray([[[0.0], [3.0]]]),
        np.zeros((1, 2)),
    )

    assert execution.result == (1, 2)
    np.testing.assert_allclose(selected_actions[:3], [0.0, 1.0, 2.0])
    executor.close()


def test_async_executor_smooths_actual_queue_overlap_when_enabled() -> None:
    selected_actions: list[float] = []
    selected = threading.Condition()
    release_third_action = threading.Event()

    def execute_step(action, value, chunk_started):
        del value, chunk_started
        selected_action = float(action[0, 0])
        with selected:
            selected_actions.append(selected_action)
            selected.notify_all()
        if selected_action == 3.0:
            release_third_action.wait(timeout=2.0)
        return ExecutedStep(feedback=None)

    executor = AsyncActionExecutor(
        execute_step,
        lambda feedback_shape: (feedback_shape, False),
        replan_after_steps=2,
        smooth_overlapping_actions=True,
    )
    executor.exchange(
        np.asarray([[[1.0], [2.0], [3.0], [4.0], [5.0]]]),
        np.zeros((1, 5)),
    )

    with selected:
        selected.wait_for(lambda: selected_actions[:3] == [1.0, 2.0, 3.0], timeout=2.0)

    second_result = {}
    second_thread = threading.Thread(
        target=lambda: second_result.setdefault(
            "slice",
            executor.exchange(
                np.asarray([[[10.0], [11.0], [12.0], [13.0]]]),
                np.zeros((1, 4)),
            ),
        )
    )
    second_thread.start()
    release_third_action.set()
    second_thread.join(timeout=2.0)
    assert not second_thread.is_alive()

    with selected:
        selected.wait_for(lambda: len(selected_actions) == 6, timeout=2.0)
    assert selected_actions == [1.0, 2.0, 3.0, 4.0, 12.0, 13.0]

    executor.close()


def test_serial_executor_interpolates_then_smooths_consecutive_plans() -> None:
    selected_actions: list[float] = []
    finished_slices = 0

    def execute_step(action, value, chunk_started):
        del value, chunk_started
        selected_action = float(action[0, 0])
        selected_actions.append(selected_action)
        return ExecutedStep(feedback=selected_action)

    def finish_slice(steps):
        nonlocal finished_slices
        finished_slices += 1
        return tuple(step.feedback for step in steps), np.asarray([finished_slices == 2])

    executor = SerialActionExecutor(
        execute_step,
        finish_slice,
        interpolation=ActionInterpolationConfig(enable=True, factor=2),
        smoothing=SerialActionSmoothingConfig(enable=True, execution_steps=4, blend_steps=2),
    )
    first = executor.exchange(
        np.asarray([[[0.0], [3.0], [6.0]]]),
        np.zeros((1, 3)),
    )
    second = executor.exchange(
        np.full((1, 3, 1), 10.0),
        np.zeros((1, 3)),
    )
    third = executor.exchange(
        np.full((1, 3, 1), 20.0),
        np.zeros((1, 3)),
    )

    np.testing.assert_allclose(first.result, [0.0, 1.2, 2.4, 3.6])
    np.testing.assert_allclose(second.result, [4.8, 10.0, 10.0, 10.0])
    np.testing.assert_allclose(third.result, [20.0, 20.0, 20.0, 20.0])

    executor.close()


def test_async_executor_aligns_chunk_to_previous_observation_step() -> None:
    selected_actions: list[float] = []

    def execute_step(action, value, chunk_started):
        del value, chunk_started
        selected_actions.append(float(action[0, 0]))
        return ExecutedStep(feedback=None)

    executor = AsyncActionExecutor(
        execute_step,
        lambda feedback_shape: (tuple(selected_actions), False),
        replan_after_steps=2,
    )
    first = executor.exchange(
        np.asarray([[[1.0], [2.0], [3.0]]]),
        np.zeros((1, 3)),
    )
    assert first.result == (1.0, 2.0)

    deadline = time.monotonic() + 2.0
    while selected_actions != [1.0, 2.0, 3.0] and time.monotonic() < deadline:
        time.sleep(0.005)
    assert selected_actions == [1.0, 2.0, 3.0]

    second = executor.exchange(
        np.asarray([[[10.0], [11.0], [12.0]]]),
        np.zeros((1, 3)),
    )
    assert second.result == (1.0, 2.0, 3.0, 11.0)

    executor.close()


def test_async_executor_discards_remaining_actions_after_episode_reset() -> None:
    selected_actions: list[float] = []
    snapshot_count = 0

    def execute_step(action, value, chunk_started):
        del value, chunk_started
        selected_actions.append(float(action[0, 0]))
        return ExecutedStep(feedback=None)

    def snapshot(feedback_shape):
        nonlocal snapshot_count
        del feedback_shape
        snapshot_count += 1
        return tuple(selected_actions), snapshot_count == 1

    executor = AsyncActionExecutor(execute_step, snapshot, replan_after_steps=2)

    first = executor.exchange(
        np.asarray([[[1.0], [2.0], [3.0], [4.0]]]),
        np.zeros((1, 4)),
    )
    assert first.result == (1.0, 2.0)

    time.sleep(0.02)
    assert selected_actions == [1.0, 2.0]

    second = executor.exchange(
        np.asarray([[[10.0], [11.0], [12.0]]]),
        np.zeros((1, 3)),
    )
    assert second.result == (1.0, 2.0, 10.0, 11.0)

    executor.close()
