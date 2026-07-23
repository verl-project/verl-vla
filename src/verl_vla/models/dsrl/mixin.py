# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Shared DSRL wiring for trainable VLA models.

:class:`DSRLSteeringMixin` owns everything about DSRL that is identical across
VLA integrations (GR00T, pi0/pi05): validation, noise-actor construction,
whole-policy freezing, actor-parameter enumeration, the SAC actor forward, and
checkpoint sidecar export. A trainable model opts in by

1. listing the mixin BEFORE ``nn.Module`` in its bases (so the ``train``
   override participates in the MRO),
2. calling :meth:`init_dsrl` at the end of ``__init__`` with its model-derived
   dimensions, and
3. implementing :meth:`_dsrl_actor_inputs` to map its ``state_features``
   structure to the ``(features, state)`` pair the noise actor consumes.

The model keeps ownership of the two genuinely model-specific pieces: seeding
its flow sampler with the steering noise, and where the noise is stored in its
rollout output (``full_action``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

from .noise_actor import DSRLNoiseActor

logger = logging.getLogger(__name__)

_DSRL_STATE_PREFIX = "dsrl_noise_actor."


class DSRLSteeringMixin:
    """Frozen-VLA DSRL support (arXiv:2506.15799) shared by trainable models.

    Hosts assign ``self.policy`` (the native VLA), ``self.config.critic`` and
    ``self.flow_sde_enable`` before calling :meth:`init_dsrl`.
    """

    dsrl_enable: bool = False
    # Annotation only — a class-level default of None would shadow the
    # nn.Module.__getattr__ submodule lookup once the actor is registered.
    dsrl_noise_actor: Optional[DSRLNoiseActor]

    # ------------------------------------------------------------------
    # Construction / freezing
    # ------------------------------------------------------------------

    def init_dsrl(
        self,
        dsrl_cfg: Any,
        *,
        feature_dim: int,
        state_dim: int,
        noise_dim: int,
        noise_horizon: int,
    ) -> None:
        """Validate the config and build the noise actor; freeze the VLA.

        ``feature_dim`` / ``state_dim`` are the model-derived defaults; the
        ``dsrl_cfg`` overrides win when set. No-op when DSRL is disabled.
        """
        self.dsrl_enable = bool(getattr(dsrl_cfg, "enabled", False))
        self.dsrl_noise_actor = None
        if not self.dsrl_enable:
            return
        if getattr(self, "flow_sde_enable", False):
            raise ValueError("DSRL noise steering and Flow-SDE are mutually exclusive; set flow_sde_enable=False.")
        if not self.config.critic.enabled:
            raise ValueError("DSRL requires the SAC critic; set adapter.critic.enabled=True.")
        self.dsrl_noise_actor = DSRLNoiseActor(
            feature_dim=int(dsrl_cfg.feature_dim or feature_dim),
            state_dim=int(dsrl_cfg.state_dim or state_dim),
            noise_dim=int(noise_dim),
            noise_horizon=int(noise_horizon),
            config=dsrl_cfg,
        )
        self.freeze_policy_for_dsrl()

    def freeze_policy_for_dsrl(self) -> None:
        """Freeze the entire native policy (backbone + action head + projections).

        DSRL trains only the noise actor and the SAC critic; the VLA is a fixed
        noise→action decoder, so it is kept in eval mode (see :meth:`train`).
        """
        self.policy.requires_grad_(False)
        self.policy.eval()
        logger.info("[dsrl] full VLA policy frozen; only the DSRL noise actor + SAC critic train")

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "dsrl_enable", False):
            # The frozen VLA must stay in eval mode (no dropout noise in the
            # deterministic noise→action decoding path).
            self.policy.eval()
        return self

    # ------------------------------------------------------------------
    # SAC actor plumbing
    # ------------------------------------------------------------------

    def _dsrl_actor_inputs(self, state_features: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Map the model's ``state_features`` to the noise actor's ``(features, state)``."""
        raise NotImplementedError("DSRL hosts must implement _dsrl_actor_inputs.")

    def dsrl_forward_actor(
        self, state_features: Any, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample steering noise — the SAC action — from the noise actor.

        The frozen flow head never runs here: during actor/critic updates the
        noise itself is the action, so this is all DSRL training needs.
        """
        features, state = self._dsrl_actor_inputs(state_features)
        noise, log_probs = self.dsrl_noise_actor.sample(features, state, deterministic=deterministic)
        return noise, log_probs, {}

    def dsrl_named_actor_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [
            (f"{_DSRL_STATE_PREFIX}{name}", param)
            for name, param in self.dsrl_noise_actor.named_parameters()
            if param.requires_grad
        ]

    def dsrl_select_critic_noise(self, a: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Rewrite an action dict so the critic scores the steering noise.

        Replay transitions store the noise under ``full_action``; actor-side
        calls already pass it under ``action``. A shared (non-per-step) noise
        vector is sliced to one horizon step to avoid scoring broadcast copies.
        """
        noise = a.get("full_action")
        if noise is None:
            noise = a["action"]
        if noise.dim() == 3 and not self.dsrl_noise_actor.noise_per_step:
            noise = noise[:, :1, :]
        return {"action": noise}

    # ------------------------------------------------------------------
    # Checkpoint export
    # ------------------------------------------------------------------

    def dsrl_export_sidecar(self, output_dir, *, state_dict=None) -> None:
        """Write ``dsrl_noise_actor.pt`` next to the exported (frozen) policy."""
        dsrl_state = None
        if state_dict is not None:
            dsrl_state = {
                name.removeprefix(_DSRL_STATE_PREFIX): value
                for name, value in state_dict.items()
                if name.startswith(_DSRL_STATE_PREFIX)
            }
        elif self.dsrl_noise_actor is not None:
            dsrl_state = self.dsrl_noise_actor.state_dict()
        if dsrl_state:
            torch.save(dsrl_state, Path(output_dir) / "dsrl_noise_actor.pt")


__all__ = ["DSRLSteeringMixin"]
