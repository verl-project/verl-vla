# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Framework-side PI0 adapter configuration.

This is deliberately not a Transformers ``PretrainedConfig``.  The native PI0
policy owns ``config.json``; this object only carries verl-vla training and
environment adaptation settings.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..dsrl.config import DSRLSteeringConfig


class PI0CriticConfig:
    DEFAULTS = {
        "enabled": False,
        "type": "cross_attn",
        "num": 1,
        "head_num": 2,
        "prefix_attn_heads": 8,
        "input_dim": 2150,
        "hidden_dims": [1024, 512, 256],
        "prefix_embed_dim": 2048,
        "task_to_critic": None,
    }

    def __init__(self, **values: Any) -> None:
        for name, value in {**self.DEFAULTS, **values}.items():
            setattr(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class PI0AdapterConfig:
    DEFAULTS = {
        "embodiment": "libero",
        "action_chunk_size": 10,
        "norm_stats_path": None,
        "state_norm_stats": {},
        "action_norm_stats": {},
        "flow_sde_enable": True,
        "flow_sde_noise_level": 0.5,
        "flow_sde_noise_schedule_enabled": False,
        "flow_sde_noise_schedule_initial": None,
        "flow_sde_noise_schedule_final": None,
        "flow_sde_noise_schedule_method": "cos",
        "flow_sde_task_noise_level": {},
        "flow_sde_rollout_noise_scale": 1.0,
        "flow_sde_train_noise_scale": 1.0,
        "flow_sde_initial_beta": 1.0,
        "flow_sde_beta_min": 0.02,
        "flow_sde_beta_schedule_T": 2000,
    }

    def __init__(
        self,
        *,
        policy_config: Mapping[str, Any] | None = None,
        model_path: str | Path | None = None,
        **overrides: Any,
    ) -> None:
        critic_values = dict(overrides.pop("critic", {}))
        legacy_critic_fields = {
            "sac_enable": "enabled",
            "critic_type": "type",
            "critic_num": "num",
            "critic_head_num": "head_num",
            "critic_prefix_attn_heads": "prefix_attn_heads",
            "critic_input_dim": "input_dim",
            "critic_hidden_dims": "hidden_dims",
            "critic_prefix_embed_dim": "prefix_embed_dim",
            "critic_task_to_critic": "task_to_critic",
        }
        for old_name, new_name in legacy_critic_fields.items():
            if old_name in overrides:
                critic_values[new_name] = overrides.pop(old_name)

        dsrl_values = dict(overrides.pop("dsrl", {}) or {})
        values = {**self.DEFAULTS, **dict(policy_config or {}), **overrides}
        dsrl_values = {**dict(values.pop("dsrl", {}) or {}), **dsrl_values}
        self.model_path = str(model_path) if model_path is not None else None
        self.critic = PI0CriticConfig(**critic_values)
        self.dsrl = DSRLSteeringConfig(**dsrl_values)
        for name, value in values.items():
            setattr(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        private_runtime_fields = {
            "state_norm_stats",
            "action_norm_stats",
            "model_path",
            "norm_stats_path",
        }
        config = {
            name: value
            for name, value in vars(self).items()
            if name not in private_runtime_fields and not name.startswith("_") and name not in ("critic", "dsrl")
        }
        config["critic"] = self.critic.to_dict()
        config["dsrl"] = self.dsrl.to_dict()
        return config

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Satisfy verl's config hook without polluting the native export.

        The adapter settings come from the workflow configuration when a full
        training checkpoint is resumed. The native PI0 policy owns every file
        under ``huggingface/`` and writes its config during policy export.
        """
        del save_directory
