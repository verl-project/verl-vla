# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""FSDP checkpoint manager with native policy export support."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fsdp_utils import (
    fsdp_version,
    get_fsdp_full_state_dict,
    merged_lora_context,
    normalize_peft_param_name,
)


def _contains_peft_model(model: torch.nn.Module) -> bool:
    from peft import PeftModel

    return any(isinstance(module, PeftModel) for module in model.modules())


def _unwrap_trainable_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._fsdp_wrapped_module if fsdp_version(model) == 1 else model


def _save_lora_adapter(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    output_dir: str | Path,
) -> None:
    """Save the nested native policy adapter through PEFT's standard API."""

    from peft import PeftModel

    trainable_model = _unwrap_trainable_model(model)
    policy = trainable_model.policy
    if not isinstance(policy, PeftModel):
        raise TypeError(f"Expected a PEFT policy, got {type(policy).__name__}")

    policy_state_dict = trainable_model.extract_policy_state_dict(state_dict)
    policy.save_pretrained(
        output_dir,
        state_dict=policy_state_dict,
        safe_serialization=True,
    )


class NativePolicyFSDPCheckpointManager(FSDPCheckpointManager):
    """Delegate ``hf_model`` export to the VLA adapter instead of AutoModel."""

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        should_export = self.should_save_hf_model
        original_contents = self.checkpoint_save_contents
        if should_export:
            self.checkpoint_save_contents = [item for item in original_contents if item != "hf_model"]
        try:
            super().save_checkpoint(
                local_path,
                hdfs_path=hdfs_path,
                global_step=global_step,
                max_ckpt_to_keep=max_ckpt_to_keep,
            )
        finally:
            self.checkpoint_save_contents = original_contents

        if not should_export:
            return

        has_lora = _contains_peft_model(self.model)
        if has_lora:
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)
            if self.rank == 0:
                _save_lora_adapter(
                    self.model,
                    state_dict,
                    Path(local_path) / "lora_adapter",
                )
                del state_dict
            torch.distributed.barrier()

            with merged_lora_context(self.model, backup_adapters=True):
                state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)
                state_dict = {name: tensor.clone() for name, tensor in state_dict.items()}
            state_dict = normalize_peft_param_name(state_dict)
        else:
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)
        if self.rank == 0:
            adapter = _unwrap_trainable_model(self.model)
            export_policy = getattr(adapter, "export_policy", None)
            output_dir = os.path.join(local_path, "huggingface")
            if callable(export_policy):
                export_policy(output_dir, state_dict=state_dict)
            else:
                save_pretrained = getattr(adapter, "save_pretrained", None)
                if not callable(save_pretrained):
                    raise TypeError(
                        f"{type(adapter).__name__} implements neither export_policy() nor save_pretrained()"
                    )
                save_pretrained(output_dir, state_dict=state_dict)
            del state_dict
        torch.distributed.barrier()
