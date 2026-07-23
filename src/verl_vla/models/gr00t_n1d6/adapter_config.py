# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Framework-side GR00T N1.6 adapter configuration.

This is deliberately not a Transformers ``PretrainedConfig``.  The native GR00T
policy owns ``config.json``; this object only carries verl-vla training and
environment adaptation settings (policy IO, critic, Flow-SDE).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..dsrl.config import DSRLSteeringConfig


class Gr00tCriticConfig:
    DEFAULTS = {
        "enabled": False,
        "type": "cross_attn",
        "head_num": 10,
        "prefix_attn_heads": 8,
        "action_dim": None,
        "action_horizon": None,
        "layernorm": False,
        "pooling": None,  # derived from type when unset: cross_attn->attn, mean_pool->mean
        "use_encoded_state": False,
        "pool_proj_dim": 0,
        "state_real_dim": None,
        "mask_frozen_action": False,
        "privileged_obs": False,
        "privileged_obs_dim": 0,
        "input_dim": None,  # resolved at model init when unset
    }

    def __init__(self, **values: Any) -> None:
        for name, value in {**self.DEFAULTS, **values}.items():
            setattr(self, name, value)
        if self.pooling is None:
            self.pooling = {"cross_attn": "attn", "mean_pool": "mean"}.get(str(self.type).lower(), "attn")

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class Gr00tAdapterConfig:
    DEFAULTS = {
        "policy_type": "libero",
        "embodiment_tag": "libero_panda",
        "action_dim": 7,
        "embodiment_id": 2,
        "num_action_chunks": 16,
        "state_horizon": 1,
        "freeze_vision_tower": True,
        "freeze_action_io": False,
        "flow_sde_enable": False,
        "flow_sde_noise_level": 0.065,
        "flow_sde_noise_level_per_dim": None,
        "flow_sde_rollout_noise_scale": 1.0,
        "flow_sde_train_noise_scale": 1.0,
        "flow_sde_initial_beta": 1.0,
        "flow_sde_beta_min": 0.02,
        "flow_sde_beta_schedule_T": 4000,
        "flow_sde_std_head": False,
        "flow_sde_std_head_hidden": 256,
        "sac_action_train_dims": None,
        "adapter_model_path": None,
        # Processor bridge (shared by SFT / SAC). None = leave checkpoint defaults.
        "norm_stats_path": None,
        "use_relative_action": None,
        # When True, inject MODALITY_CONFIGS[embodiment_tag] at processor load
        # (needed for LIBERO SFT on a base GR1 checkpoint).
        "override_modality_configs": None,
    }

    # Flat override_config / yaml keys that map into nested critic.*.
    _LEGACY_CRITIC_FIELDS = {
        "sac_enable": "enabled",
        "critic_type": "type",
        "critic_head_num": "head_num",
        "critic_prefix_attn_heads": "prefix_attn_heads",
        "critic_action_dim": "action_dim",
        "critic_action_horizon": "action_horizon",
        "critic_layernorm": "layernorm",
        "critic_pooling": "pooling",
        "critic_use_encoded_state": "use_encoded_state",
        "critic_pool_proj_dim": "pool_proj_dim",
        "critic_state_real_dim": "state_real_dim",
        "critic_mask_frozen_action": "mask_frozen_action",
        "critic_privileged_obs": "privileged_obs",
        "critic_privileged_obs_dim": "privileged_obs_dim",
        "critic_input_dim": "input_dim",
    }

    def __init__(
        self,
        *,
        policy_config: Mapping[str, Any] | None = None,
        model_path: str | Path | None = None,
        **overrides: Any,
    ) -> None:
        critic_values = dict(overrides.pop("critic", {}) or {})
        for old_name, new_name in self._LEGACY_CRITIC_FIELDS.items():
            if old_name in overrides:
                critic_values[new_name] = overrides.pop(old_name)
        dsrl_values = dict(overrides.pop("dsrl", {}) or {})

        values = {**self.DEFAULTS, **dict(policy_config or {}), **overrides}
        dsrl_values = {**dict(values.pop("dsrl", {}) or {}), **dsrl_values}
        if "embodiment" in values and "policy_type" not in overrides:
            # Accept pi0-style wording without writing it back.
            values["policy_type"] = values.pop("embodiment")
        self.model_path = str(model_path) if model_path is not None else None
        self.critic = Gr00tCriticConfig(**critic_values)
        self.dsrl = DSRLSteeringConfig(**dsrl_values)
        for name, value in values.items():
            setattr(self, name, value)

    @property
    def sac_enable(self) -> bool:
        """Legacy alias: SAC heads are enabled when the critic is enabled."""
        return bool(self.critic.enabled)

    def to_dict(self) -> dict[str, Any]:
        private_runtime_fields = {"model_path", "adapter_model_path"}
        config = {
            name: value
            for name, value in vars(self).items()
            if name not in private_runtime_fields and not name.startswith("_") and name not in ("critic", "dsrl")
        }
        config["critic"] = self.critic.to_dict()
        config["dsrl"] = self.dsrl.to_dict()
        return config


__all__ = ["Gr00tAdapterConfig", "Gr00tCriticConfig"]
