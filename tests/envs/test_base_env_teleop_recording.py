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

from typing import Any

import numpy as np
import pytest

from verl_vla.envs.base import BaseEnv


def _mask(num_envs: int, env_ids: list[int] | None = None) -> np.ndarray:
    mask = np.zeros(num_envs, dtype=bool)
    if env_ids:
        mask[np.asarray(env_ids, dtype=np.int64)] = True
    return mask


class FakeBaseEnv(BaseEnv):
    env_type = "fake"

    def __init__(
        self,
        *,
        num_envs: int,
        events: list[dict[str, Any]] | None = None,
        terminate_env_ids: list[int] | None = None,
        auto_reset: bool = True,
    ) -> None:
        self.num_envs = num_envs
        self.auto_reset_enabled = auto_reset
        self._latest_obs = self._make_obs("initial", np.arange(num_envs))
        self.teleops = []
        self.recorder = None
        self._confirm_before_record_enabled = False
        self.events = list(events or [])
        self.terminate_env_ids = set(terminate_env_ids or [])
        self.step_calls: list[list[int]] = []
        self.action_calls: list[np.ndarray] = []
        self.reset_calls: list[list[int]] = []
        self.recorder_reset_calls: list[list[int]] = []
        self._step_count = 0

    def apply_teleop_action(self, action):
        event = self.events.pop(0) if self.events else {}
        next_action = np.asarray(event.get("next_action", action)).copy()
        return (
            next_action,
            _mask(self.num_envs, event.get("intervention")),
            _mask(self.num_envs, event.get("manual_reward")),
            _mask(self.num_envs, event.get("restart_episode")),
            _mask(self.num_envs, event.get("stop_episode")),
        )

    def env_step(self, action, *, env_ids):
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        self._step_count += 1
        self.step_calls.append(env_ids.tolist())
        self.action_calls.append(np.asarray(action, dtype=np.float32).copy())
        terminated = np.asarray([int(env_id) in self.terminate_env_ids for env_id in env_ids], dtype=bool)
        return {
            **self._make_obs(f"step-{self._step_count}", env_ids),
            "next.reward": np.zeros(len(env_ids), dtype=np.float32),
            "next.terminated": terminated,
            "next.truncated": np.zeros(len(env_ids), dtype=bool),
            "next.success": terminated.copy(),
        }

    def env_reset(self, *, env_ids, reset_eval: bool = False):
        del reset_eval
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        self.reset_calls.append(env_ids.tolist())
        return self._make_obs("reset", env_ids)

    def reset_recorder_envs(self, env_ids) -> None:
        self.recorder_reset_calls.append(np.asarray(env_ids, dtype=np.int64).reshape(-1).tolist())

    def publish_reset_obs_to_teleop(self, obs, env_ids) -> None:
        del obs, env_ids

    def _confirm_before_record(self, env_ids) -> None:
        del env_ids

    @staticmethod
    def _make_obs(prefix: str, env_ids) -> dict[str, Any]:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        return {
            "observation": [f"{prefix}-obs-{int(env_id)}" for env_id in env_ids],
            "task": [f"{prefix}-task-{int(env_id)}" for env_id in env_ids],
            "task_id": env_ids.astype(np.int64, copy=False),
        }


@pytest.mark.parametrize(
    ("event_key", "expected_reward", "expected_terminated", "expected_truncated", "expected_success"),
    [
        ("manual_reward", [[1.0]], [[True]], [[False]], [[True]]),
        ("stop_episode", [[0.0]], [[False]], [[True]], [[False]]),
    ],
)
def test_record_control_overrides_step_result(
    event_key: str,
    expected_reward: list[list[float]],
    expected_terminated: list[list[bool]],
    expected_truncated: list[list[bool]],
    expected_success: list[list[bool]],
) -> None:
    env = FakeBaseEnv(num_envs=1, events=[{event_key: [0]}])

    obs, reward, terminated, truncated, success = env.step(np.zeros((1, 1, 1), dtype=np.float32))

    assert obs["observation"] == ["reset-obs-0"]
    assert reward.tolist() == expected_reward
    assert terminated.tolist() == expected_terminated
    assert truncated.tolist() == expected_truncated
    assert success.tolist() == expected_success
    assert env.reset_calls == [[0]]
    assert env.recorder_reset_calls == [[0]]


@pytest.mark.parametrize(
    ("event_key", "expected_reward", "expected_terminated", "expected_truncated", "expected_success"),
    [
        ("manual_reward", [[1.0, 0.0]], [[True, False]], [[False, False]], [[True, False]]),
        ("stop_episode", [[0.0, 0.0]], [[False, False]], [[True, False]], [[False, False]]),
    ],
)
def test_record_done_control_is_not_overwritten_by_later_chunk_step(
    event_key: str,
    expected_reward: list[list[float]],
    expected_terminated: list[list[bool]],
    expected_truncated: list[list[bool]],
    expected_success: list[list[bool]],
) -> None:
    env = FakeBaseEnv(
        num_envs=1,
        events=[
            {event_key: [0]},
            {},
        ],
    )

    obs, reward, terminated, truncated, success = env.step(np.asarray([[[100.0], [101.0]]], dtype=np.float32))

    assert env.step_calls == [[0], [0]]
    assert [action.tolist() for action in env.action_calls] == [
        [[100.0]],
        [[101.0]],
    ]
    assert obs["observation"] == ["reset-obs-0"]
    assert reward.tolist() == expected_reward
    assert terminated.tolist() == expected_terminated
    assert truncated.tolist() == expected_truncated
    assert success.tolist() == expected_success
    assert env.reset_calls == [[0]]
    assert env.recorder_reset_calls == [[0]]


def test_restart_episode_resets_without_marking_transition_done() -> None:
    env = FakeBaseEnv(num_envs=1, events=[{"restart_episode": [0]}])

    obs, reward, terminated, truncated, success = env.step(np.zeros((1, 1, 1), dtype=np.float32))

    assert obs["observation"] == ["reset-obs-0"]
    assert reward.tolist() == [[0.0]]
    assert terminated.tolist() == [[False]]
    assert truncated.tolist() == [[False]]
    assert success.tolist() == [[False]]
    assert env.reset_calls == [[0]]
    assert env.recorder_reset_calls == [[0]]


def test_intervention_done_env_is_not_stepped_again() -> None:
    env = FakeBaseEnv(
        num_envs=2,
        events=[
            {"intervention": [0], "next_action": np.asarray([[1.0], [0.0]], dtype=np.float32)},
            {"intervention": [0], "next_action": np.asarray([[2.0], [0.0]], dtype=np.float32)},
        ],
        terminate_env_ids=[0],
    )

    step_result, restart_episode, chunk_intervened = env.step_with_teleop_and_recording(
        np.zeros((2, 1), dtype=np.float32),
        chunk_intervened=np.zeros(2, dtype=bool),
        merged_step_result=None,
    )

    assert env.step_calls == [[0], [1]]
    assert restart_episode.tolist() == [False, False]
    assert chunk_intervened.tolist() == [True, False]
    assert step_result["observation"] == ["step-1-obs-0", "step-2-obs-1"]
    assert step_result["next.terminated"].tolist() == [True, False]


def test_multi_env_intervention_waits_until_all_envs_stop_or_finish() -> None:
    env = FakeBaseEnv(
        num_envs=3,
        events=[
            {"intervention": [0], "next_action": np.asarray([[10.0], [0.0], [0.0]], dtype=np.float32)},
            {"intervention": [0, 2], "next_action": np.asarray([[11.0], [0.0], [20.0]], dtype=np.float32)},
            {"intervention": [1, 2], "next_action": np.asarray([[0.0], [30.0], [21.0]], dtype=np.float32)},
            {"intervention": [1], "next_action": np.asarray([[0.0], [31.0], [0.0]], dtype=np.float32)},
            {"intervention": []},
        ],
        terminate_env_ids=[2],
    )

    step_result, restart_episode, chunk_intervened = env.step_with_teleop_and_recording(
        np.zeros((3, 1), dtype=np.float32),
        chunk_intervened=np.zeros(3, dtype=bool),
        merged_step_result=None,
    )

    assert env.events == []
    assert env.step_calls == [[0], [2], [1], [0, 1]]
    assert [action.tolist() for action in env.action_calls] == [
        [[10.0]],
        [[20.0]],
        [[30.0]],
        [[11.0], [31.0]],
    ]
    assert restart_episode.tolist() == [False, False, False]
    assert chunk_intervened.tolist() == [True, True, True]
    assert step_result["observation"] == ["step-4-obs-0", "step-4-obs-1", "step-2-obs-2"]
    assert step_result["next.terminated"].tolist() == [False, False, True]


def test_chunk_intervened_env_skips_remaining_policy_actions() -> None:
    env = FakeBaseEnv(
        num_envs=2,
        events=[
            {"intervention": [0], "next_action": np.asarray([[10.0], [0.0]], dtype=np.float32)},
            {"intervention": []},
        ],
    )

    action = np.asarray(
        [
            [[100.0], [101.0], [102.0]],
            [[200.0], [201.0], [202.0]],
        ],
        dtype=np.float32,
    )
    obs, reward, terminated, truncated, success = env.step(action)

    assert env.step_calls == [[0, 1], [1], [1]]
    assert [action.tolist() for action in env.action_calls] == [
        [[10.0], [200.0]],
        [[201.0]],
        [[202.0]],
    ]
    assert obs["observation"] == ["step-1-obs-0", "step-3-obs-1"]
    assert reward.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert terminated.tolist() == [[False, False, False], [False, False, False]]
    assert truncated.tolist() == [[False, False, False], [False, False, False]]
    assert success.tolist() == [[False, False, False], [False, False, False]]


def test_execute_mask_keeps_policy_and_fresh_teleop_but_skips_old_intervention() -> None:
    env = FakeBaseEnv(
        num_envs=3,
        events=[
            {"intervention": [2], "next_action": np.asarray([[100.0], [200.0], [30.0]], dtype=np.float32)},
            {"intervention": []},
        ],
    )
    merged_step_result = {
        "observation": ["previous-obs-0", "previous-obs-1", "previous-obs-2"],
        "task": ["previous-task-0", "previous-task-1", "previous-task-2"],
        "task_id": np.asarray([0, 1, 2], dtype=np.int64),
        "next.reward": np.zeros(3, dtype=np.float32),
        "next.terminated": np.zeros(3, dtype=bool),
        "next.truncated": np.zeros(3, dtype=bool),
        "next.success": np.zeros(3, dtype=bool),
    }

    step_result, restart_episode, chunk_intervened = env.step_with_teleop_and_recording(
        np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32),
        chunk_intervened=np.asarray([False, True, True], dtype=bool),
        merged_step_result=merged_step_result,
    )

    assert env.events == []
    assert env.step_calls == [[0, 2]]
    assert [action.tolist() for action in env.action_calls] == [
        [[10.0], [30.0]],
    ]
    assert restart_episode.tolist() == [False, False, False]
    assert chunk_intervened.tolist() == [False, True, True]
    assert step_result["observation"] == ["step-1-obs-0", "previous-obs-1", "step-1-obs-2"]
    assert step_result["next.terminated"].tolist() == [False, False, False]


def test_all_chunk_intervened_envs_can_skip_remaining_policy_actions() -> None:
    env = FakeBaseEnv(
        num_envs=2,
        events=[
            {"intervention": [0, 1], "next_action": np.asarray([[10.0], [20.0]], dtype=np.float32)},
            {"intervention": []},
        ],
    )

    action = np.asarray(
        [
            [[100.0], [101.0]],
            [[200.0], [201.0]],
        ],
        dtype=np.float32,
    )
    obs, reward, terminated, truncated, success = env.step(action)

    assert env.step_calls == [[0, 1]]
    assert [action.tolist() for action in env.action_calls] == [
        [[10.0], [20.0]],
    ]
    assert obs["observation"] == ["step-1-obs-0", "step-1-obs-1"]
    assert reward.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert terminated.tolist() == [[False, False], [False, False]]
    assert truncated.tolist() == [[False, False], [False, False]]
    assert success.tolist() == [[False, False], [False, False]]
