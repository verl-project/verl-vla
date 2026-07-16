# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path

import torch
from torch import nn
from verl.utils.fsdp_utils import normalize_peft_param_name

from verl_vla.models.base import TrainableVLAModelMixin
from verl_vla.workers.engine.fsdp.native_policy_checkpoint_manager import _save_lora_adapter


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.projection(inputs)

    def save_pretrained(self, output_dir, *, state_dict=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict() if state_dict is None else state_dict, output_dir / "policy.pt")


class _TinyTrainableModel(nn.Module, TrainableVLAModelMixin):
    def __init__(self):
        super().__init__()
        self.init_trainable_model(policy=_TinyPolicy())
        self.critic = nn.Linear(4, 1)


def _apply_lora(model: _TinyTrainableModel):
    model.apply_lora(
        rank=2,
        alpha=4,
        target_modules=["projection"],
    )


def test_lora_adapts_only_native_policy_and_preserves_auxiliary_training():
    model = _TinyTrainableModel()

    _apply_lora(model)

    policy_parameters = dict(model.policy.named_parameters())
    assert not policy_parameters["base_model.model.projection.base_layer.weight"].requires_grad
    assert policy_parameters["base_model.model.projection.lora_A.default.weight"].requires_grad
    assert policy_parameters["base_model.model.projection.lora_B.default.weight"].requires_grad
    assert all(parameter.requires_grad for parameter in model.critic.parameters())

    inputs = torch.randn(2, 4)
    loss = model.policy(inputs).sum() + model.critic(inputs).sum()
    loss.backward()

    assert policy_parameters["base_model.model.projection.lora_B.default.weight"].grad is not None
    assert model.critic.weight.grad is not None


def test_lora_export_uses_merged_native_policy_state(tmp_path):
    model = _TinyTrainableModel()
    _apply_lora(model)
    with torch.no_grad():
        model.policy.base_model.model.projection.lora_B.default.weight.fill_(0.5)

    model.policy.merge_adapter()
    merged_state = normalize_peft_param_name({name: tensor.clone() for name, tensor in model.state_dict().items()})
    model.policy.unmerge_adapter()
    model.export_policy(tmp_path, state_dict=merged_state)

    exported_state = torch.load(tmp_path / "policy.pt", weights_only=True)
    assert set(exported_state) == {"projection.weight", "projection.bias"}
    assert not torch.equal(exported_state["projection.weight"], model.native_policy.projection.weight)


def test_lora_adapter_can_be_loaded_for_continued_training(tmp_path):
    source = _TinyTrainableModel()
    _apply_lora(source)
    _save_lora_adapter(source, source.state_dict(), tmp_path)

    assert (tmp_path / "adapter_config.json").is_file()
    assert (tmp_path / "adapter_model.safetensors").is_file()

    resumed = _TinyTrainableModel()
    resumed.apply_lora(
        rank=2,
        alpha=4,
        target_modules=["projection"],
        adapter_path=str(tmp_path),
    )

    assert resumed.has_lora
    assert all(parameter.requires_grad for name, parameter in resumed.policy.named_parameters() if "lora_" in name)
