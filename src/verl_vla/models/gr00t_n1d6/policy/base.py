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

"""GR00T policy IO contract (mirrors ``pi0_torch/policy/base.py``).

``Gr00tInput`` collects the raw tensors the :class:`GR00TN16Adapter` needs:
``observation.images.*`` frames, flat policy-order state, and the task string.
Optional ``actions`` carry already-converted (processor-space) demos for SFT.
``Gr00tOutput`` wraps a model rollout so it can be turned into a ``DataProto``
for the env / replay buffer.

Both classes are gr00t-package-free so this module imports without the gr00t
checkpoint / package installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from verl import DataProto

from ...base import ModelOutput


class Gr00tInput(ABC):
    def __init__(self):
        # ``{observation.images.<name>: (B, H, W, C) uint8}`` from the env.
        self.images: dict[str, torch.Tensor] = {}

        # Flat policy-order robot state, (B, state_dim). The adapter splits it into
        # the checkpoint's per-modality state groups.
        self.state: Optional[torch.Tensor] = None

        # Task description, one string per batch element.
        self.task: list[str] = []

        # Optional demo actions already in processor/GR00T space, (B, T, action_dim).
        # Embodiment IO (e.g. LIBERO gripper) must convert before attaching.
        self.actions: Optional[torch.Tensor] = None

    @classmethod
    @abstractmethod
    def from_env_obs(cls, env_obs: DataProto) -> Gr00tInput: ...

    @classmethod
    def actions_to_processor_space(cls, actions: torch.Tensor) -> torch.Tensor:
        """Map env-space demo actions into the processor / GR00T action space.

        Embodiment subclasses override this for gripper / joint remapping.
        :class:`~verl_vla.models.gr00t_n1d6.gr00t_adapter.GR00TN16Adapter` then
        performs relative-action conversion and statistical normalisation.
        """
        return actions.to(dtype=torch.float32)

    @classmethod
    def from_data_proto(
        cls,
        obs: DataProto,
        actions: torch.Tensor | None = None,
    ) -> Gr00tInput:
        """Build from a shared VLA ``DataProto``; optional actions for SFT."""
        model_input = cls.from_env_obs(obs)
        if actions is not None:
            model_input.actions = cls.actions_to_processor_space(actions)
        return model_input


class Gr00tOutput(ModelOutput):
    """Wraps a GR00T rollout for downstream (env / replay / critic) consumption.

    Attributes:
        action:      env-facing DECODED action chunk, (B, num_action_chunks, action_dim).
                     This is what the simulator steps with; consumed via ``to_data_proto``
                     under the ``action`` key.
        full_action: NORMALISED model action, (B, action_horizon, max_action_dim). This
                     is the differentiable action space the SAC actor/critic operate in
                     (decoding is non-differentiable), stored in replay under
                     ``full_action`` so the critic sees the same space it is trained on.
                     In DSRL runs (``adapter.dsrl.enabled``) this instead holds the
                     steering noise ``x0`` — the latent SAC action that seeds the frozen
                     flow head — with the same shape.
        log_prob:    optional per-sample Flow-SDE log-prob, (B,).
    """

    def __init__(self):
        self.action: Optional[torch.Tensor] = None
        self.full_action: Optional[torch.Tensor] = None
        self.log_prob: Optional[torch.Tensor] = None

    @classmethod
    @abstractmethod
    def from_model_output(cls, model_output: dict) -> Gr00tOutput: ...

    def to_data_proto(self) -> DataProto:
        tensor_batch = {"action": self.action}
        if self.full_action is not None:
            tensor_batch["full_action"] = self.full_action
        if self.log_prob is not None:
            tensor_batch["log_prob"] = self.log_prob
        return DataProto.from_dict(tensors=tensor_batch)


__all__ = [
    "Gr00tInput",
    "Gr00tOutput",
]
