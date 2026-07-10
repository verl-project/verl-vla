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

import torch
from verl import DataProto

from verl_vla.utils.data import update_progress_trajectory_counts


def _make_env_result(done, *, success=None) -> DataProto:
    done_tensor = torch.as_tensor(done, dtype=torch.bool)
    success_tensor = torch.zeros_like(done_tensor) if success is None else torch.as_tensor(success, dtype=torch.bool)
    return DataProto.from_dict(
        tensors={
            "next.terminated": done_tensor,
            "next.truncated": torch.zeros_like(done_tensor),
            "next.success": success_tensor,
        }
    )


def test_progress_counts_at_most_one_trajectory_per_chunk_lane():
    progress_counts = {"done_eps": 0, "succ_eps": 0}

    update_progress_trajectory_counts(
        _make_env_result(
            [[False, False, True, True, False, True, False]],
            success=[[False, False, True, True, False, True, False]],
        ),
        stage_id=0,
        progress_counts=progress_counts,
        progress_lane_state={},
    )

    assert progress_counts == {"done_eps": 1, "succ_eps": 1}


def test_progress_ignores_success_after_first_done_in_chunk():
    progress_counts = {"done_eps": 0, "succ_eps": 0}

    update_progress_trajectory_counts(
        _make_env_result(
            [[False, True, False, False]],
            success=[[False, False, True, False]],
        ),
        stage_id=0,
        progress_counts=progress_counts,
        progress_lane_state={},
    )

    assert progress_counts == {"done_eps": 1, "succ_eps": 0}
