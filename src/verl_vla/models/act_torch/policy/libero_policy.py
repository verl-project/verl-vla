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

from .base import ActInput, ActOutput

ACT_MAX_STATE_DIM = 8
LIBERO_ACTION_DIM = 7


def _normalize_image_range(image: torch.Tensor) -> torch.Tensor:
    """Convert LIBERO images to the [0, 1] range used by LeRobot datasets.

    Offline LeRobot video decoding returns float tensors in [0, 1], while
    live LIBERO observations arrive as uint8 tensors in [0, 255]. Keeping
    both paths on the same scale is required before ACT's mean/std image
    normalization is applied.
    """
    if image.dtype == torch.uint8:
        return image.to(dtype=torch.float32).div_(255.0)

    return image.to(dtype=torch.float32)


class LiberoActInput(ActInput):
    @override
    @classmethod
    def from_env_obs(cls, env_obs: DataProto) -> "LiberoActInput":
        input = cls()

        images = env_obs.batch["observation.images.image"]
        wrist_images = env_obs.batch["observation.images.wrist_image"]
        device = images.device

        if images.ndim == 5:
            images = images[:, -1, :, :, :]
            wrist_images = wrist_images[:, -1, :, :, :]

        # Ensure images are in (B, C, H, W) format - convert from (B, H, W, C) if needed
        if images.shape[-1] == 3 and images.shape[-3] != 3:
            images = images.permute(0, 3, 1, 2)
        if wrist_images.shape[-1] == 3 and wrist_images.shape[-3] != 3:
            wrist_images = wrist_images.permute(0, 3, 1, 2)

        images = _normalize_image_range(images)
        wrist_images = _normalize_image_range(wrist_images)

        batch_size = images.shape[0]
        cam_high = images
        left_wrist = wrist_images
        input.images = {
            "observation.images.cam_high": cam_high,
            "observation.images.cam_left_wrist": left_wrist,
        }
        input.img_masks = [
            torch.ones((batch_size,), device=device, dtype=torch.bool),
            torch.ones((batch_size,), device=device, dtype=torch.bool),
        ]

        input.task = list(env_obs.non_tensor_batch.get("task", ["" for _ in range(batch_size)]))

        state = env_obs.batch["observation.state"]
        input.state = torch.nn.functional.pad(
            state, (0, max(0, ACT_MAX_STATE_DIM - state.shape[-1])), "constant", 0
        ).to(device=device, dtype=torch.float32)

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
