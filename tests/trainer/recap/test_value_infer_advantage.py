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

import numpy as np
import pytest

from verl_vla.trainer.recap.value_infer import _compute_n_step_advantages, _smooth_future_scores


def _record(
    index: int,
    frame_index: int,
    recap_return: float,
    recap_value: float,
    *,
    episode_index: int = 0,
) -> dict[str, object]:
    return {
        "index": index,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "task_index": 0,
        "return": recap_return,
        "value": recap_value,
        "is_intervention": False,
    }


def test_n_step_advantage_uses_future_return_and_value_when_in_bounds():
    records = [
        _record(0, 0, 10.0, 1.0),
        _record(1, 1, 7.0, 2.0),
        _record(2, 2, 4.0, 3.0),
        _record(3, 3, 1.0, 5.0),
    ]

    advantages = _compute_n_step_advantages(records, n_step=2)

    np.testing.assert_allclose(
        advantages[:2],
        np.asarray(
            [
                (10.0 - 4.0) + (3.0 - 1.0),
                (7.0 - 1.0) + (5.0 - 2.0),
            ],
            dtype=np.float32,
        ),
    )


def test_n_step_advantage_clamps_future_return_and_value_to_episode_tail_when_out_of_bounds():
    records = [
        _record(0, 0, 10.0, 1.0),
        _record(1, 1, 7.0, 2.0),
        _record(2, 2, 4.0, 3.0),
        _record(3, 3, 1.0, 5.0),
        _record(4, 0, 100.0, 100.0, episode_index=1),
    ]

    advantages = _compute_n_step_advantages(records, n_step=5)

    np.testing.assert_allclose(
        advantages,
        np.asarray(
            [
                (10.0 - 7.0) + (7.0 - 4.0) + (4.0 - 1.0) + (1.0 - 1.0) + (1.0 - 1.0) + 5.0 - 1.0,
                (7.0 - 4.0) + (4.0 - 1.0) + (1.0 - 1.0) + (1.0 - 1.0) + (1.0 - 1.0) + 5.0 - 2.0,
                (4.0 - 1.0) + (1.0 - 1.0) + (1.0 - 1.0) + (1.0 - 1.0) + (1.0 - 1.0) + 5.0 - 3.0,
                (1.0 - 1.0) + (1.0 - 1.0) + (1.0 - 1.0) + (1.0 - 1.0) + (1.0 - 1.0) + 5.0 - 5.0,
                (100.0 - 100.0) + (100.0 - 100.0) + (100.0 - 100.0) + (100.0 - 100.0) + (100.0 - 100.0) + 100.0 - 100.0,
            ],
            dtype=np.float32,
        ),
    )


def test_future_smoothed_scores_use_decayed_future_window_without_crossing_episode_boundary():
    records = [
        _record(0, 0, 0.0, 0.0),
        _record(1, 1, 0.0, 0.0),
        _record(2, 2, 0.0, 0.0),
        _record(3, 0, 0.0, 0.0, episode_index=1),
    ]
    scores = np.asarray([1.0, 3.0, 5.0, 100.0], dtype=np.float32)

    smoothed = _smooth_future_scores(records=records, scores=scores, window=3, decay=0.5)

    assert smoothed[0] == pytest.approx((1.0 + 0.5 * 3.0 + 0.25 * 5.0) / (1.0 + 0.5 + 0.25))
    assert smoothed[1] == pytest.approx((3.0 + 0.5 * 5.0) / (1.0 + 0.5))
    assert smoothed[2] == pytest.approx(5.0)
    assert smoothed[3] == pytest.approx(100.0)
