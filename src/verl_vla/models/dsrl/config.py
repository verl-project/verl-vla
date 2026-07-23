# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Shared DSRL (latent-noise steering) configuration.

DSRL (Diffusion Steering via Reinforcement Learning, arXiv:2506.15799) keeps
the whole VLA frozen and trains a small SAC policy over the *initial noise*
``x0`` of the flow-matching action head. This config is model-agnostic and is
embedded by both the GR00T and pi0/pi05 adapter configs under the ``dsrl`` key;
model-derived dimensions (feature/state/noise widths, action horizon) are
resolved by each trainable model at build time, not stored here.
"""

from __future__ import annotations

from typing import Any


class DSRLSteeringConfig:
    DEFAULTS = {
        # Master switch. When True the VLA policy is fully frozen and SAC
        # trains only the noise actor (+ critic).
        "enabled": False,
        # Optional overrides for the model-derived actor input widths. None
        # resolves from the model (backbone feature dim / processor state dim).
        "feature_dim": None,
        "state_dim": None,
        # MLP trunk widths of the noise actor.
        "hidden_dims": [256, 256, 256],
        # Width the (frozen) backbone feature vector is projected to.
        "feature_latent_dim": 128,
        # Width the raw robot state is projected to.
        "state_latent_dim": 64,
        # False (RLinf parity): one noise vector shared by every step of the
        # action chunk. True: an independent noise vector per horizon step.
        "noise_per_step": False,
        # tanh output bound; x0 lives in [-noise_bound, noise_bound]^d.
        "noise_bound": 1.0,
        # Pre-tanh Gaussian log-std clamp range.
        "log_std_min": -20.0,
        "log_std_max": 2.0,
    }

    def __init__(self, **values: Any) -> None:
        for name, value in {**self.DEFAULTS, **values}.items():
            setattr(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


__all__ = ["DSRLSteeringConfig"]
