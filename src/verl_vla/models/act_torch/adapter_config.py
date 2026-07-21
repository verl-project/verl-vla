# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Framework-side configuration for the ACT trainable wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ACTCriticConfig:
    DEFAULTS = {
        "enabled": False,
        "type": "mean_pool",
        "head_num": 2,
        "prefix_embed_dim": 512,
        "input_dim": 0,
        "hidden_dims": [1024, 512, 256],
    }

    def __init__(self, **values: Any) -> None:
        for name, value in {**self.DEFAULTS, **values}.items():
            setattr(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class ACTAdapterConfig:
    DEFAULTS = {
        "policy_type": "libero",
        "n_action_steps": 100,
        "temporal_ensemble_coeff": None,
        "state_norm_stats": None,
        "action_norm_stats": None,
        "image_norm_stats": None,
        "sac_action_noise_scale": 0.1,
        "sac_action_noise_schedule_enabled": False,
        "sac_action_noise_schedule_initial": None,
        "sac_action_noise_schedule_final": None,
        "sac_action_noise_schedule_method": "cos",
        "freeze_vision_tower": True,
        "optimizer_lr_backbone": 1e-5,
    }

    def __init__(
        self,
        *,
        policy_config: Mapping[str, Any],
        model_path: str | Path | None = None,
        **overrides: Any,
    ) -> None:
        critic = ACTCriticConfig(**dict(overrides.pop("critic", {})))
        values = {**self.DEFAULTS, **dict(policy_config), **overrides}
        self.model_path = str(model_path) if model_path is not None else None
        self.critic = critic
        for name, value in values.items():
            setattr(self, name, value)

    @property
    def sac_enable(self) -> bool:
        return bool(self.critic.enabled)

    @property
    def critic_type(self) -> str:
        return str(self.critic.type)

    @property
    def critic_head_num(self) -> int:
        return int(self.critic.head_num)

    @property
    def critic_prefix_embed_dim(self) -> int:
        return int(self.critic.prefix_embed_dim)

    @property
    def critic_input_dim(self) -> int:
        return int(self.critic.input_dim)

    @critic_input_dim.setter
    def critic_input_dim(self, value: int) -> None:
        self.critic.input_dim = value

    @property
    def critic_hidden_dims(self) -> list[int]:
        return list(self.critic.hidden_dims)

    def to_dict(self) -> dict[str, Any]:
        config = {name: value for name, value in vars(self).items() if name not in {"critic", "model_path"}}
        config["critic"] = self.critic.to_dict()
        return config

    def save_pretrained(self, save_directory: str | Path) -> None:
        del save_directory
