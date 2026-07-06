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
from typing_extensions import override
from verl.protocol import DataProto

from .base import Gr00tInput, Gr00tOutput

LIBERO_ACTION_DIM = 7


class LiberoGr00tInput(Gr00tInput):
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
    def from_env_obs(cls, env_obs: DataProto) -> "LiberoGr00tInput":
        input = cls()

        # View order [agent view, wrist view]: must stay consistent between SFT
        # and rollout.
        images = cls._to_bchw(env_obs.batch["observation.images.image"])
        wrist_images = cls._to_bchw(env_obs.batch["observation.images.wrist_image"])
        input.images = torch.stack([images, wrist_images], dim=1).to(torch.uint8)

        input.task = [str(t) for t in env_obs.non_tensor_batch["task"]]

        input.state = env_obs.batch["observation.state"].to(dtype=torch.float32)

        return input


class LiberoGr00tOutput(Gr00tOutput):
    @override
    @classmethod
    def from_model_output(cls, model_output: dict) -> "LiberoGr00tOutput":
        output = cls()
        action_chunk_size = int(model_output.get("action_chunk_size", model_output["full_action"].shape[1]))
        action_dim = int(model_output.get("action_dim", LIBERO_ACTION_DIM))
        output.action = model_output["full_action"][:, :action_chunk_size, :action_dim]
        output.log_prob = model_output.get("log_probs")
        return output
