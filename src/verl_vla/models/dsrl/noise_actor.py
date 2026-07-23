# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""DSRL noise actor: a small tanh-Gaussian SAC policy over flow-matching noise.

The actor maps a frozen-backbone feature vector plus the raw robot state to the
initial noise ``x0`` of the flow-matching sampler. The frozen VLA head then
integrates ``x0`` into an env action with the deterministic Euler ODE, so
"steering" the noise is the only control channel — the SAC action space *is*
``x0`` (RLinf DSRL design, arXiv:2506.15799).

Model-agnostic by construction: GR00T feeds its pooled VL embedding
(``pooled``) and processor state; pi0/pi05 feeds its mean-pooled prefix
embedding and normalized state.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import DSRLSteeringConfig

_TANH_EPS = 1e-6


class DSRLNoiseActor(nn.Module):
    """Tanh-squashed Gaussian policy over the initial flow noise.

    Args:
        feature_dim: Width of the frozen backbone feature vector (e.g. 2048).
        state_dim: Flattened raw/normalized state width fed to the actor.
        noise_dim: Per-step noise width (= the model's padded ``max_action_dim``).
        noise_horizon: Flow action horizon the noise is broadcast/reshaped to.
        config: Shared :class:`DSRLSteeringConfig`.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        state_dim: int,
        noise_dim: int,
        noise_horizon: int,
        config: DSRLSteeringConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or DSRLSteeringConfig()
        self.feature_dim = int(feature_dim)
        self.state_dim = int(state_dim)
        self.noise_dim = int(noise_dim)
        self.noise_horizon = int(noise_horizon)
        self.noise_per_step = bool(config.noise_per_step)
        self.noise_bound = float(config.noise_bound)
        self.log_std_min = float(config.log_std_min)
        self.log_std_max = float(config.log_std_max)
        if self.noise_bound <= 0:
            raise ValueError(f"dsrl noise_bound must be positive, got {self.noise_bound}")

        feature_latent_dim = int(config.feature_latent_dim)
        state_latent_dim = int(config.state_latent_dim)
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, feature_latent_dim),
            nn.LayerNorm(feature_latent_dim),
            nn.Tanh(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, state_latent_dim),
            nn.LayerNorm(state_latent_dim),
            nn.Tanh(),
        )

        trunk: list[nn.Module] = []
        in_dim = feature_latent_dim + state_latent_dim
        for hidden_dim in (int(h) for h in config.hidden_dims):
            trunk += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk)

        self.out_dim = self.noise_dim * (self.noise_horizon if self.noise_per_step else 1)
        self.mean_head = nn.Linear(in_dim, self.out_dim)
        self.log_std_head = nn.Linear(in_dim, self.out_dim)
        # Near-zero head init (RLinf parity): the initial policy is ~N(0, 1)
        # pre-tanh, i.e. close to the prior the frozen flow head was trained on.
        for head in (self.mean_head, self.log_std_head):
            nn.init.xavier_uniform_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, features: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the pre-tanh Gaussian ``(mean, log_std)`` in float32."""
        features = features.reshape(features.shape[0], -1).float()
        state = state.reshape(state.shape[0], -1).float()
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"DSRL noise actor expected feature_dim={self.feature_dim}, got {features.shape[-1]}")
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"DSRL noise actor expected state_dim={self.state_dim}, got {state.shape[-1]}")
        latent = torch.cat([self.feature_encoder(features), self.state_encoder(state)], dim=-1)
        hidden = self.trunk(latent)
        mean = self.mean_head(hidden).float()
        log_std = self.log_std_head(hidden).float().clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        features: torch.Tensor,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample steering noise for the flow head.

        Returns:
            noise: (B, noise_horizon, noise_dim) float32 in
                ``[-noise_bound, noise_bound]``, ready to seed the flow ODE.
            log_prob: (B,) float32, summed over noise dims (0 when
                ``deterministic``), tanh-corrected.
        """
        mean, log_std = self(features, state)
        if deterministic:
            pre_tanh = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device, dtype=mean.dtype)
        else:
            normal = torch.distributions.Normal(mean, log_std.exp())
            pre_tanh = normal.rsample()
            squashed = torch.tanh(pre_tanh)
            log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - squashed.pow(2) + _TANH_EPS)
            log_prob = log_prob.sum(dim=-1)
            if self.noise_bound != 1.0:
                log_prob = log_prob - self.out_dim * math.log(self.noise_bound)
        noise_flat = torch.tanh(pre_tanh) * self.noise_bound

        batch_size = noise_flat.shape[0]
        if self.noise_per_step:
            noise = noise_flat.view(batch_size, self.noise_horizon, self.noise_dim)
        else:
            noise = noise_flat.unsqueeze(1).expand(batch_size, self.noise_horizon, self.noise_dim).contiguous()
        return noise, log_prob


__all__ = ["DSRLNoiseActor"]
