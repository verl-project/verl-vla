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

"""Unit tests for trajectory metric parsing in TrainCluster.

These tests use synthetic DataProto objects and do not start Ray, models, or
environments. The goal is to lock down how low-level env-step done/success
signals are converted into per-trajectory records for eval metrics.
"""

import numpy as np
import torch
from verl import DataProto

from verl_vla.trainer.train_cluster.cluster import TrainCluster


def _make_output(done, *, truncated=None, success=None, reward=None, task_id=1, eval_episode_id=None) -> DataProto:
    """Build the minimum rollout output consumed by _collect_trajectory_records.

    `done` maps to next.terminated. `truncated` is optional because production
    code treats a step as finished when either next.terminated or next.truncated
    is true. Inputs use the rollout shape [B, T, K].
    """

    terminated_tensor = torch.as_tensor(done, dtype=torch.bool)

    if truncated is None:
        truncated_tensor = torch.zeros_like(terminated_tensor)
    else:
        truncated_tensor = torch.as_tensor(truncated, dtype=torch.bool)

    if success is None:
        success_tensor = torch.zeros_like(terminated_tensor)
    else:
        success_tensor = torch.as_tensor(success, dtype=torch.bool)

    if reward is None:
        reward_tensor = torch.zeros(terminated_tensor.shape, dtype=torch.float32)
    else:
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32)

    task_ids = np.full(terminated_tensor.shape[:2], task_id, dtype=np.int64)
    non_tensors = {"obs.task_id": task_ids}
    if eval_episode_id is not None:
        non_tensors["obs.eval_episode_id"] = np.asarray(eval_episode_id, dtype=np.int64)
    return DataProto.from_dict(
        tensors={
            "next.terminated": terminated_tensor,
            "next.truncated": truncated_tensor,
            "next.success": success_tensor,
            "next.reward": reward_tensor,
        },
        non_tensors=non_tensors,
    )


def _empty_carry_state() -> dict[str, np.ndarray | None]:
    """Create the mutable length/return state for partial auto-reset episodes."""

    return {"length": None, "reward": None}


class TestAutoResetTrajectoryRecords:
    def test_ignores_done_tail_within_action_chunk(self):
        """After the first done in a chunk, the remaining low-level steps are ignored."""

        carry_state = _empty_carry_state()

        records = TrainCluster._collect_trajectory_records(
            _make_output(
                [[[0, 0, 1, 1, 0, 1, 0, 0]]],
                success=[[[0, 0, 1, 1, 0, 1, 0, 0]]],
                reward=[[[0, 0, 1, 1, 0, 1, 0, 0]]],
            ),
            auto_reset=True,
            carry_state=carry_state,
        )

        assert records == [{"length": 3, "chunk_length": 1, "return": 1.0, "success": True, "task_id": 1}]
        assert carry_state["length"][0] == 0

    def test_next_chunk_starts_new_episode_after_done(self):
        """Auto reset happens at chunk boundaries, so the next chunk starts a fresh episode."""

        carry_state = _empty_carry_state()

        records = TrainCluster._collect_trajectory_records(
            _make_output(
                [[[0, 0, 1, 1], [0, 1, 1, 0]]],
                success=[[[0, 0, 1, 1], [0, 1, 1, 0]]],
                reward=[[[0, 0, 1, 1], [0, 1, 1, 0]]],
            ),
            auto_reset=True,
            carry_state=carry_state,
        )

        assert records == [
            {"length": 3, "chunk_length": 1, "return": 1.0, "success": True, "task_id": 1},
            {"length": 2, "chunk_length": 1, "return": 1.0, "success": True, "task_id": 1},
        ]

    def test_carries_unfinished_episode_across_batches(self):
        """Partial episodes should keep length and reward prefixes."""

        carry_state = _empty_carry_state()

        records = TrainCluster._collect_trajectory_records(
            _make_output([[[0, 0]]], success=[[[0, 0]]], reward=[[[1, 2]]]),
            auto_reset=True,
            carry_state=carry_state,
        )
        assert records == []
        assert carry_state["length"][0] == 2
        assert carry_state["reward"][0] == 3.0

        records = TrainCluster._collect_trajectory_records(
            _make_output([[[0, 1]]], success=[[[0, 1]]], reward=[[[3, 4]]]),
            auto_reset=True,
            carry_state=carry_state,
        )
        assert records == [{"length": 4, "chunk_length": 2, "return": 10.0, "success": True, "task_id": 1}]

    def test_chunk_shape_reports_low_level_length_and_chunk_length(self):
        """For [B, T, K] tensors, length is low-level steps and chunk_length is ceil(length / K)."""

        carry_state = _empty_carry_state()

        records = TrainCluster._collect_trajectory_records(
            _make_output(
                [[[0, 0, 0, 1], [1, 1, 0, 1]]],
                success=[[[0, 0, 0, 1], [1, 1, 0, 1]]],
                reward=[[[0, 0, 0, 1], [1, 1, 0, 1]]],
                task_id=7,
            ),
            auto_reset=True,
            carry_state=carry_state,
        )

        assert records == [
            {"length": 4, "chunk_length": 1, "return": 1.0, "success": True, "task_id": 7},
            {"length": 1, "chunk_length": 1, "return": 1.0, "success": True, "task_id": 7},
        ]

    def test_remaining_caps_records(self):
        """Eval can request only the number of records needed to reach max_episodes."""

        carry_state = _empty_carry_state()

        records = TrainCluster._collect_trajectory_records(
            _make_output([[[0, 1], [0, 1]]], success=[[[0, 1], [0, 1]]], reward=[[[0, 1], [0, 1]]]),
            auto_reset=True,
            remaining=1,
            carry_state=carry_state,
        )

        assert records == [{"length": 2, "chunk_length": 1, "return": 1.0, "success": True, "task_id": 1}]

    def test_records_eval_episode_id_from_done_chunk(self):
        """Completed auto-reset records keep the benchmark id from their start chunk."""

        records = TrainCluster._collect_trajectory_records(
            _make_output(
                [[[0, 1], [0, 1]]],
                success=[[[0, 1], [0, 1]]],
                reward=[[[0, 1], [0, 1]]],
                eval_episode_id=[[10, 11]],
            ),
            auto_reset=True,
            carry_state=_empty_carry_state(),
        )

        assert [record["eval_episode_id"] for record in records] == [10, 11]


class TestNonAutoResetTrajectoryRecords:
    def test_uses_first_done_in_each_row(self):
        """Without auto-reset, each row contributes one trajectory ending at its first done."""

        records = TrainCluster._collect_trajectory_records(
            _make_output(
                [[[0, 0, 1], [1, 1, 0]]],
                success=[[[0, 0, 0], [1, 1, 1]]],
                reward=[[[1, 2, 3], [4, 5, 6]]],
                task_id=3,
            ),
            auto_reset=False,
            carry_state=_empty_carry_state(),
        )

        assert records == [{"length": 3, "chunk_length": 1, "return": 6.0, "success": False, "task_id": 3}]

    def test_no_done_uses_full_row(self):
        """A row with no done is still summarized as one in-progress trajectory."""

        records = TrainCluster._collect_trajectory_records(
            _make_output([[[0, 0, 0]]], success=[[[0, 1, 0]]], reward=[[[1, 2, 3]]]),
            auto_reset=False,
            carry_state=_empty_carry_state(),
        )

        assert records == [{"length": 3, "chunk_length": 1, "return": 6.0, "success": True, "task_id": 1}]

    def test_truncated_counts_as_done(self):
        """next.truncated should terminate a trajectory even when next.terminated is false."""

        records = TrainCluster._collect_trajectory_records(
            _make_output([[[0, 0, 0, 0]]], truncated=[[[0, 0, 1, 0]]], success=[[[0, 0, 0, 0]]]),
            auto_reset=False,
            carry_state=_empty_carry_state(),
        )

        assert records == [{"length": 3, "chunk_length": 1, "return": 0.0, "success": False, "task_id": 1}]

    def test_chunk_shape_reports_first_done_chunk(self):
        """Chunked non-auto rows report the chunk containing the first done."""

        records = TrainCluster._collect_trajectory_records(
            _make_output(
                [[[0, 0, 0, 0], [0, 1, 1, 1]]],
                success=[[[0, 0, 0, 0], [0, 1, 1, 1]]],
                reward=[[[1, 1, 1, 1], [1, 1, 1, 1]]],
                task_id=9,
            ),
            auto_reset=False,
            carry_state=_empty_carry_state(),
        )

        assert records == [{"length": 6, "chunk_length": 2, "return": 6.0, "success": True, "task_id": 9}]

    def test_records_eval_episode_id_from_first_chunk(self):
        """Non-auto-reset records keep the benchmark id from the trajectory start chunk."""

        records = TrainCluster._collect_trajectory_records(
            _make_output(
                [[[0, 0, 0, 0], [0, 1, 1, 1]]],
                success=[[[0, 0, 0, 0], [0, 1, 1, 1]]],
                reward=[[[1, 1, 1, 1], [1, 1, 1, 1]]],
                eval_episode_id=[[20, 21]],
            ),
            auto_reset=False,
            carry_state=_empty_carry_state(),
        )

        assert records[0]["eval_episode_id"] == 20
