# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from typing_extensions import override
from verl.protocol import DataProto

from .base import ActInput, ActOutput, prepare_act_image

LIBERO_ACTION_DIM = 7


class LiberoActInput(ActInput):
    @override
    @classmethod
    def from_env_obs(cls, env_obs: DataProto) -> "LiberoActInput":
        input = cls()

        images = env_obs.batch["observation.images.image"]
        wrist_images = env_obs.batch["observation.images.wrist_image"]
        device = images.device

        images = prepare_act_image(images)
        wrist_images = prepare_act_image(wrist_images)

        batch_size = images.shape[0]
        input.images = {
            "observation.images.image": images,
            "observation.images.wrist_image": wrist_images,
        }
        input.img_masks = [
            torch.ones((batch_size,), device=device, dtype=torch.bool),
            torch.ones((batch_size,), device=device, dtype=torch.bool),
        ]

        input.task = list(env_obs.non_tensor_batch.get("task", ["" for _ in range(batch_size)]))

        state = env_obs.batch["observation.state"]
        input.state = state.to(device=device, dtype=torch.float32)

        if "observation.environment_state" in env_obs.batch:
            input.env_state = env_obs.batch["observation.environment_state"].to(device=device, dtype=torch.float32)

        return input


class LiberoActOutput(ActOutput):
    @override
    @classmethod
    def from_model_output(cls, model_output: dict) -> "LiberoActOutput":
        output = cls()
        action_chunk_size = int(model_output.get("action_chunk_size", model_output["full_action"].shape[1]))
        output.action = model_output["full_action"][:, :action_chunk_size, :LIBERO_ACTION_DIM]
        output.log_prob = model_output.get("log_probs")
        return output
