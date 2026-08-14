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
import torch
from verl.protocol import DataProto

from verl_vla.models.act_torch.policy.piper_policy import (
    PiperActInput,
    PiperActOutput,
)


def test_piper_act_policy_preserves_joint_state_and_action_chunks():
    joint_state = torch.arange(14, dtype=torch.float32).unsqueeze(0)
    observation = DataProto.from_dict(
        tensors={
            "observation.state": joint_state,
            "observation.images.height": torch.full((1, 2, 4, 5, 3), 255, dtype=torch.uint8),
            "observation.images.left_wrist": torch.zeros((1, 3, 4, 5), dtype=torch.float32),
        },
        non_tensors={"task": np.asarray(["test"], dtype=object)},
    )

    policy_input = PiperActInput.from_env_obs(observation)

    torch.testing.assert_close(policy_input.state, joint_state)
    assert tuple(policy_input.images["observation.images.height"].shape) == (1, 3, 4, 5)
    assert policy_input.images["observation.images.height"].max().item() == 1.0
    assert tuple(policy_input.images["observation.images.left_wrist"].shape) == (1, 3, 4, 5)

    full_action = torch.arange(2 * 50 * 14, dtype=torch.float32).reshape(2, 50, 14)
    output = PiperActOutput.from_model_output({"full_action": full_action, "action_chunk_size": 40})
    torch.testing.assert_close(output.action, full_action[:, :40])
