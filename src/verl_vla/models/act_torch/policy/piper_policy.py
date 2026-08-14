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

"""ACT policy I/O for Piper absolute joint observations and actions."""

from __future__ import annotations

import torch
from lerobot.utils.constants import OBS_STATE
from typing_extensions import override
from verl.protocol import DataProto

from .base import ActInput, ActOutput, prepare_act_image

PIPER_ACTION_DIM_PER_ARM = 7


class PiperActInput(ActInput):
    @override
    @classmethod
    def from_env_obs(cls, env_obs: DataProto) -> PiperActInput:
        input = cls()
        image_keys = sorted(str(key) for key in env_obs.batch.keys() if str(key).startswith("observation.images."))
        input.images = {key: prepare_act_image(env_obs.batch[key]) for key in image_keys}

        input.state = env_obs.batch[OBS_STATE].to(dtype=torch.float32)
        batch_size = int(input.state.shape[0])
        input.img_masks = [torch.ones((batch_size,), device=input.state.device, dtype=torch.bool) for _ in image_keys]
        input.task = list(env_obs.non_tensor_batch.get("task", [""] * batch_size))
        return input


class PiperActOutput(ActOutput):
    @override
    @classmethod
    def from_model_output(cls, model_output: dict) -> PiperActOutput:
        output = cls()
        full_action = model_output["full_action"]
        action_dim = int(full_action.shape[-1])
        if action_dim not in {PIPER_ACTION_DIM_PER_ARM, 2 * PIPER_ACTION_DIM_PER_ARM}:
            raise ValueError(f"Piper ACT action dim must be 7 or 14, got {action_dim}")
        action_chunk_size = int(model_output.get("action_chunk_size", full_action.shape[1]))
        output.action = full_action[:, :action_chunk_size]
        output.log_prob = model_output.get("log_probs")
        return output


__all__ = [
    "PiperActInput",
    "PiperActOutput",
]
