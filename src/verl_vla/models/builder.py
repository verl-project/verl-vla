# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Explicit VLA model construction without Transformers AutoClass registration."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _apply_overrides(config, overrides: Mapping) -> None:
    for name, value in overrides.items():
        setattr(config, name, value)


def build_vla_model(model_config, *, torch_dtype: torch.dtype):
    architecture = model_config.native_architecture
    path = model_config.local_path
    overrides = dict(model_config.override_config)
    if "model_config" in overrides:
        overrides = dict(overrides["model_config"])

    if architecture == "pi0":
        from .pi0_torch import PI0TrainableModel

        return PI0TrainableModel.from_pretrained(
            path,
            adapter_config=dict(model_config.adapter),
            policy_config_overrides=overrides,
            torch_dtype=torch_dtype,
        )

    if architecture == "act":
        from .act_torch import ACTTorchConfig, ACTTrainableModel

        config = ACTTorchConfig.from_pretrained(path)
        _apply_overrides(config, overrides)
        return ACTTrainableModel.from_pretrained(
            path,
            policy_config=config,
            adapter_config=dict(model_config.adapter),
            torch_dtype=torch_dtype,
        )

    if architecture == "gr00t_n1d6":
        from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

        from .gr00t_n1d6.adapter_config import Gr00tAdapterConfig
        from .gr00t_n1d6.trainable_model import Gr00tN1d6TrainableModel, load_gr00t_n1d6_policy

        config = Gr00tN1d6Config.from_pretrained(path)
        _apply_overrides(config, overrides)
        adapter_config = Gr00tAdapterConfig(model_path=path, **dict(model_config.adapter))

        policy = load_gr00t_n1d6_policy(path, config=config, torch_dtype=torch_dtype)
        return Gr00tN1d6TrainableModel(policy, adapter_config=adapter_config)

    if architecture == "openvla_oft":
        from .openvla_oft.configuration_prismatic import OpenVLAConfig
        from .openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
        from .openvla_oft.trainable_model import OpenVLATrainableModel

        config = OpenVLAConfig.from_pretrained(path)
        _apply_overrides(config, overrides)
        policy = OpenVLAForActionPrediction.from_pretrained(path, config=config, torch_dtype=torch_dtype)
        return OpenVLATrainableModel(policy)

    if architecture == "recap_value_critic":
        from .recap_value_critic import ReCapValueCriticConfig, ReCapValueCriticTrainableModel

        config = ReCapValueCriticConfig.from_pretrained(path)
        _apply_overrides(config, overrides)
        return ReCapValueCriticTrainableModel.from_pretrained(path, config=config, torch_dtype=torch_dtype)

    raise ValueError(f"Unsupported VLA architecture: {architecture!r}")


__all__ = ["build_vla_model"]
