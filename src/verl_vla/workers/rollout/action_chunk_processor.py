# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from verl_vla.workers.config import RolloutConfig


class ActionChunkProcessor(ABC):
    """Transform a newly predicted action chunk before environment execution."""

    @abstractmethod
    def process(
        self,
        actions: torch.Tensor,
        *,
        stage_id: int,
        episode_start: Any | None,
    ) -> torch.Tensor:
        """Return the processed action chunk."""


class ActionInterpolationProcessor(ActionChunkProcessor):
    """Temporally upsample each action chunk with linear interpolation."""

    def __init__(self, factor: int) -> None:
        self.factor = int(factor)

    def process(
        self,
        actions: torch.Tensor,
        *,
        stage_id: int,
        episode_start: Any | None,
    ) -> torch.Tensor:
        del stage_id, episode_start
        if actions.ndim != 3:
            raise ValueError(f"rollout action chunk must have shape [batch, time, action_dim], got {actions.shape}")

        output_steps = int(actions.shape[1]) * self.factor
        return F.interpolate(
            actions.transpose(1, 2),
            size=output_steps,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)


class ActionChunkStitchingProcessor(ActionChunkProcessor):
    """Blend a new action plan with the unexecuted tail of its previous plan."""

    def __init__(self, execution_steps: int, blend_steps: int) -> None:
        self.execution_steps = int(execution_steps)
        self.blend_steps = int(blend_steps)
        self._action_plans: dict[int, torch.Tensor] = {}

    def process(
        self,
        actions: torch.Tensor,
        *,
        stage_id: int,
        episode_start: Any | None,
    ) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(f"rollout action plan must have shape [batch, time, action_dim], got {actions.shape}")
        if episode_start is None:
            raise ValueError("action chunk stitching requires episode_start in the observation metadata")

        plan_steps = int(actions.shape[1])
        if self.execution_steps + self.blend_steps > plan_steps:
            raise ValueError(
                "action chunk stitching requires execution_steps + blend_steps <= interpolated plan steps, "
                f"got {self.execution_steps} + {self.blend_steps} > {plan_steps}"
            )

        reset_mask = torch.as_tensor(episode_start, device=actions.device, dtype=torch.bool).reshape(-1)
        if reset_mask.shape[0] != actions.shape[0]:
            raise ValueError(
                f"action plan reset mask must have batch size {actions.shape[0]}, got {reset_mask.shape[0]}"
            )

        merged = actions.clone()
        previous = self._action_plans.get(stage_id)
        if previous is not None:
            if previous.shape[0] != actions.shape[0] or previous.shape[2] != actions.shape[2]:
                raise ValueError(
                    "consecutive action plans must preserve batch and action dimensions, "
                    f"got {tuple(previous.shape)} then {tuple(actions.shape)}"
                )
            valid = ~reset_mask
            if valid.any():
                old_prefix = previous[
                    valid,
                    self.execution_steps : self.execution_steps + self.blend_steps,
                ]
                weight = torch.linspace(
                    0,
                    1,
                    self.blend_steps,
                    device=actions.device,
                    dtype=actions.dtype,
                )
                weight = (weight.square() * (3 - 2 * weight)).reshape(1, self.blend_steps, 1)
                merged[valid, : self.blend_steps] = (1 - weight) * old_prefix + weight * actions[
                    valid, : self.blend_steps
                ]

        self._action_plans[stage_id] = merged.detach().clone()
        return merged[:, : self.execution_steps]


class CompositeActionChunkProcessor(ActionChunkProcessor):
    """Apply an ordered collection of action chunk processors."""

    def __init__(self, processors: Sequence[ActionChunkProcessor]) -> None:
        self.processors = tuple(processors)

    @classmethod
    def from_config(cls, config: RolloutConfig) -> "CompositeActionChunkProcessor":
        processors: list[ActionChunkProcessor] = []
        if config.action_interpolation.enable:
            processors.append(ActionInterpolationProcessor(config.action_interpolation.factor))
        if config.action_chunk_stitching.enable:
            processors.append(
                ActionChunkStitchingProcessor(
                    execution_steps=config.action_chunk_stitching.execution_steps,
                    blend_steps=config.action_chunk_stitching.blend_steps,
                )
            )
        return cls(processors)

    def process(
        self,
        actions: torch.Tensor,
        *,
        stage_id: int,
        episode_start: Any | None,
    ) -> torch.Tensor:
        for processor in self.processors:
            actions = processor.process(
                actions,
                stage_id=stage_id,
                episode_start=episode_start,
            )
        return actions
