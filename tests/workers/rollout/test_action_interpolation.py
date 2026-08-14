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

from types import SimpleNamespace

import torch

from verl_vla.workers.rollout.hf_rollout import HFRollout


def test_hf_rollout_linearly_upsamples_action_chunk() -> None:
    rollout = object.__new__(HFRollout)
    rollout.config = SimpleNamespace(action_interpolation=SimpleNamespace(enable=True, factor=2))
    actions = torch.tensor([[[0.0, 2.0], [3.0, 5.0], [6.0, 8.0]]])

    interpolated = rollout._interpolate_action_chunk(actions)

    assert interpolated.shape == (1, 6, 2)
    torch.testing.assert_close(interpolated[:, 0], actions[:, 0])
    torch.testing.assert_close(interpolated[:, -1], actions[:, -1])
    torch.testing.assert_close(interpolated[0, :, 0], torch.linspace(0.0, 6.0, 6))
