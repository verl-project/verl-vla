# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Common verl-facing input/output contract for GR00T policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from verl import DataProto

from verl_vla.models.base import ModelOutput


class Gr00tPolicyInput(ABC):
    """Raw VLA inputs converted from verl's shared ``DataProto`` schema."""

    @classmethod
    @abstractmethod
    def from_data_proto(
        cls,
        obs: DataProto,
        actions: torch.Tensor | None = None,
    ) -> Gr00tPolicyInput:
        """Build an environment-specific input without exposing GR00T tensors to DataProto."""

    @abstractmethod
    def collate(
        self,
        processor: Any,
        *,
        action_valid_mask: torch.Tensor | None = None,
    ) -> Any:
        """Run the official processor and collator."""


class Gr00tPolicyOutput(ModelOutput, ABC):
    """Environment-ready action output returned to verl rollout workers."""

    def __init__(self, action: torch.Tensor):
        self.action = action

    def to_data_proto(self) -> DataProto:
        return DataProto.from_dict(tensors={"action": self.action})

    @classmethod
    @abstractmethod
    def from_model_output(cls, model_output: Any, **kwargs) -> Gr00tPolicyOutput:
        """Decode one upstream model output into simulator-ready actions."""
