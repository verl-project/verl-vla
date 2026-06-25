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

from .base import Pi0Input, Pi0Output

PI0_MAX_STATE_DIM = 32
LIBERO_ACTION_DIM = 7


class LiberoPi0Input(Pi0Input):
    @staticmethod
    def _to_bchw(image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected batched image tensor with 4 dims, got shape={tuple(image.shape)}.")
        if image.shape[1] == 3:
            return image
        if image.shape[-1] == 3:
            return image.permute(0, 3, 1, 2)
        raise ValueError(f"Expected image tensor in BCHW or BHWC format, got shape={tuple(image.shape)}.")

    @override
    @classmethod
    def from_env_obs(cls, env_obs: DataProto) -> "LiberoPi0Input":
        input = cls()

        # Process images
        images = env_obs.batch["observation.images.image"]
        wrist_images = env_obs.batch["observation.images.wrist_image"]
        device = images.device

        batch_size = images.shape[0]
        cam_high = cls._to_bchw(images)
        left_wrist = cls._to_bchw(wrist_images)
        empty_images = torch.zeros(
            (batch_size, 3, cam_high.shape[2], cam_high.shape[3]),
            device=device,
            dtype=torch.bfloat16,
        )

        input.images = {
            "observation.images.cam_high": cam_high.to(torch.bfloat16),
            "observation.images.cam_left_wrist": left_wrist.to(torch.bfloat16),
            "observation.images.cam_right_wrist": empty_images,
        }
        input.img_masks = [
            torch.ones((batch_size,), device=device, dtype=torch.bool),
            torch.ones((batch_size,), device=device, dtype=torch.bool),
            torch.zeros((batch_size,), device=device, dtype=torch.bool),
        ]

        # Process other data
        input.task = list(env_obs.non_tensor_batch["task"])

        state = env_obs.batch["observation.state"]
        input.state = torch.nn.functional.pad(
            state, (0, max(0, PI0_MAX_STATE_DIM - state.shape[-1])), "constant", 0
        ).to(device=device, dtype=torch.float32)

        return input


class LiberoPi0Output(Pi0Output):
    @override
    @classmethod
    def from_model_output(cls, model_output: dict) -> "LiberoPi0Output":
        output = cls()
        action_chunk_size = int(model_output.get("action_chunk_size", model_output["full_action"].shape[1]))
        output.action = model_output["full_action"][:, :action_chunk_size, :LIBERO_ACTION_DIM]
        output.log_prob = model_output.get("log_probs")
        return output
