# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Thin verl training/rollout adapter around the official GR00T N1.6 model."""

from __future__ import annotations

from threading import Lock
from typing import Any

import torch

# Nothing imports this module unless the external ``gr00t`` distribution is installed.
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from torch.distributed.fsdp import register_fsdp_forward_method
from transformers import AutoModel
from typing_extensions import override
from verl import DataProto

from verl_vla.models.base import SupportSFTTraining

from .policy.libero_policy import LiberoGr00tInput, LiberoGr00tOutput, load_gr00t_processor


class _CpuBeta(torch.distributions.Beta):
    """Construct N1.6's non-parameter distribution outside FSDP's meta device."""

    def __init__(self, concentration1: float, concentration0: float):
        super().__init__(
            torch.tensor(float(concentration1), dtype=torch.float32, device="cpu"),
            torch.tensor(float(concentration0), dtype=torch.float32, device="cpu"),
        )


_BETA_PATCH_LOCK = Lock()


def _rec_to_device_dtype(
    value: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(
            device=device,
            dtype=dtype if torch.is_floating_point(value) else value.dtype,
        )
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _rec_to_device_dtype(item, device=device, dtype=dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_rec_to_device_dtype(item, device=device, dtype=dtype) for item in value]
    return value


class VerlGr00tN1d6(Gr00tN1d6, SupportSFTTraining):
    """Official N1.6 parameters plus verl's SFT and deterministic rollout APIs.

    No model layers or parameters are introduced, so saved state-dict keys stay
    directly loadable by the upstream ``Gr00tN1d6`` implementation.
    """

    config_class = Gr00tN1d6Config

    def __init__(self, config: Gr00tN1d6Config, *args, **kwargs):
        # N1.6 constructs Beta from Python floats. Under FSDP2 meta init,
        # torch.distributions validates them via Tensor.item(), which is illegal
        # on meta tensors. The distribution is not a module/parameter and N1.6
        # samples it on CPU anyway, so construct just this object explicitly on CPU.
        import gr00t.model.gr00t_n1d6.gr00t_n1d6 as upstream_model

        with _BETA_PATCH_LOCK:
            original_beta = upstream_model.Beta
            upstream_model.Beta = _CpuBeta
            try:
                super().__init__(config, *args, **kwargs)
            finally:
                upstream_model.Beta = original_beta
        SupportSFTTraining.__init__(self, config)
        self._verl_processor = None
        self._verl_processor_training: bool | None = None

    @override
    def sft_init(self):
        self.sft_metrics = {}
        register_fsdp_forward_method(self, "sft_loss")

    @override
    def sft_loss(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module | None,
        actions: dict[str, torch.Tensor],
        valids: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        target_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del tokenizer, target_values
        processor = self._get_verl_processor(training=True)
        policy_input = LiberoGr00tInput.from_data_proto(obs, actions=actions["action"])
        collated = policy_input.collate(processor, action_valid_mask=action_mask)
        inputs = _rec_to_device_dtype(
            collated["inputs"],
            device=self.device,
            dtype=torch.bfloat16,
        )

        output = self(inputs)
        element_loss = output["action_loss"]
        element_mask = output["action_mask"].to(element_loss.dtype)
        per_sample = element_loss.flatten(1).sum(-1) / element_mask.flatten(1).sum(-1).clamp_min(1.0)
        valid_weights = valids.to(device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
        loss = (per_sample * valid_weights).sum() / valid_weights.sum().clamp_min(1.0)
        self.sft_metrics = {
            "sft/action_loss": loss.detach(),
            "sft/valid_action_fraction": element_mask.float().mean().detach(),
        }
        return loss

    def sac_init(self):
        """Only the rollout-shaped method is supported; GR00T SAC is out of scope."""
        register_fsdp_forward_method(self, "sac_sample_actions")

    def _get_verl_processor(self, *, training: bool):
        if self._verl_processor is None:
            processor_path = getattr(self.config, "verl_processor_path", None)
            if not processor_path:
                processor_path = getattr(self.config, "_name_or_path", None)
            if not processor_path:
                raise ValueError(
                    "GR00T rollout needs config.verl_processor_path or a checkpoint directory "
                    "containing processor_config.json."
                )
            self._verl_processor = load_gr00t_processor(
                str(processor_path),
                getattr(self.config, "verl_norm_stats_path", None),
                training=training,
            )
        elif self._verl_processor_training != training:
            self._verl_processor.train() if training else self._verl_processor.eval()
        self._verl_processor_training = training
        return self._verl_processor

    @torch.no_grad()
    def sac_sample_actions(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module | None = None,
        eval: bool = False,
    ) -> LiberoGr00tOutput:
        """Deterministic N1.6 inference exposed through verl's rollout interface."""
        del tokenizer, eval
        processor = self._get_verl_processor(training=False)
        policy_input = LiberoGr00tInput.from_data_proto(obs)
        collated = _rec_to_device_dtype(
            policy_input.collate(processor),
            device=self.device,
            dtype=torch.bfloat16,
        )
        model_pred = self.get_action(**collated)
        action_chunk_size = int(getattr(self.config, "verl_action_chunk_size", 8))
        return LiberoGr00tOutput.from_model_output(
            model_pred,
            processor=processor,
            policy_input=policy_input,
            action_chunk_size=action_chunk_size,
            device=self.device,
        )


def register_with_transformers() -> None:
    """Replace only the N1.6 AutoModel mapping with the verl-capable subclass."""
    AutoModel.register(Gr00tN1d6Config, VerlGr00tN1d6, exist_ok=True)


__all__ = ["VerlGr00tN1d6", "register_with_transformers"]
