# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the shared DSRL noise actor and its adapter-config plumbing."""

from __future__ import annotations

import pytest
import torch

from verl_vla.models.dsrl import DSRLNoiseActor, DSRLSteeringConfig
from verl_vla.models.gr00t_n1d6.adapter_config import Gr00tAdapterConfig
from verl_vla.models.pi0_torch.adapter_config import PI0AdapterConfig

FEATURE_DIM = 32
STATE_DIM = 8
NOISE_DIM = 12
HORIZON = 5


def _make_actor(**config_overrides) -> DSRLNoiseActor:
    config = DSRLSteeringConfig(hidden_dims=[16, 16], feature_latent_dim=8, state_latent_dim=4, **config_overrides)
    return DSRLNoiseActor(
        feature_dim=FEATURE_DIM,
        state_dim=STATE_DIM,
        noise_dim=NOISE_DIM,
        noise_horizon=HORIZON,
        config=config,
    )


def test_sample_shapes_and_bounds():
    actor = _make_actor()
    noise, log_prob = actor.sample(torch.randn(3, FEATURE_DIM), torch.randn(3, STATE_DIM))
    assert noise.shape == (3, HORIZON, NOISE_DIM)
    assert log_prob.shape == (3,)
    assert noise.dtype == torch.float32
    assert noise.abs().max() <= 1.0


def test_shared_noise_is_broadcast_across_horizon():
    actor = _make_actor()
    noise, _ = actor.sample(torch.randn(2, FEATURE_DIM), torch.randn(2, STATE_DIM))
    torch.testing.assert_close(noise, noise[:, :1].expand_as(noise))


def test_deterministic_sample_is_tanh_mean_with_zero_logprob():
    actor = _make_actor()
    features, state = torch.randn(4, FEATURE_DIM), torch.randn(4, STATE_DIM)
    noise, log_prob = actor.sample(features, state, deterministic=True)
    mean, _ = actor(features, state)
    torch.testing.assert_close(noise[:, 0], torch.tanh(mean))
    assert torch.all(log_prob == 0)


def test_noise_bound_scales_output():
    actor = _make_actor(noise_bound=2.0)
    noise, _ = actor.sample(torch.randn(64, FEATURE_DIM), torch.randn(64, STATE_DIM))
    assert noise.abs().max() <= 2.0


def test_per_step_noise_shape():
    actor = _make_actor(noise_per_step=True)
    noise, log_prob = actor.sample(torch.randn(2, FEATURE_DIM), torch.randn(2, STATE_DIM))
    assert noise.shape == (2, HORIZON, NOISE_DIM)
    assert log_prob.shape == (2,)
    # Per-step noise must not be a broadcast copy.
    assert not torch.equal(noise[:, 0], noise[:, 1])


def test_sample_is_reparameterized():
    actor = _make_actor()
    noise, log_prob = actor.sample(torch.randn(2, FEATURE_DIM), torch.randn(2, STATE_DIM))
    assert noise.requires_grad
    assert log_prob.requires_grad
    (noise.sum() + log_prob.sum()).backward()
    assert actor.mean_head.weight.grad is not None
    assert actor.log_std_head.weight.grad is not None


def test_state_flattening_and_dim_check():
    actor = _make_actor()
    noise, _ = actor.sample(torch.randn(2, FEATURE_DIM), torch.randn(2, 1, STATE_DIM))
    assert noise.shape == (2, HORIZON, NOISE_DIM)
    with pytest.raises(ValueError, match="state_dim"):
        actor.sample(torch.randn(2, FEATURE_DIM), torch.randn(2, STATE_DIM + 1))
    with pytest.raises(ValueError, match="feature_dim"):
        actor.sample(torch.randn(2, FEATURE_DIM + 1), torch.randn(2, STATE_DIM))


def test_bfloat16_inputs_produce_float32_outputs():
    actor = _make_actor()
    noise, log_prob = actor.sample(
        torch.randn(2, FEATURE_DIM, dtype=torch.bfloat16),
        torch.randn(2, STATE_DIM, dtype=torch.bfloat16),
    )
    assert noise.dtype == torch.float32
    assert log_prob.dtype == torch.float32


def test_gr00t_adapter_config_parses_and_roundtrips_dsrl():
    cfg = Gr00tAdapterConfig(dsrl={"enabled": True, "hidden_dims": [64, 64], "noise_bound": 1.5})
    assert cfg.dsrl.enabled is True
    assert cfg.dsrl.hidden_dims == [64, 64]
    assert cfg.dsrl.noise_bound == 1.5
    payload = cfg.to_dict()
    assert payload["dsrl"]["enabled"] is True
    # Reload from the serialized payload (adapter_config.json roundtrip).
    reloaded = Gr00tAdapterConfig(**payload)
    assert reloaded.dsrl.enabled is True
    assert reloaded.dsrl.hidden_dims == [64, 64]


def test_gr00t_adapter_config_dsrl_defaults_off():
    cfg = Gr00tAdapterConfig()
    assert cfg.dsrl.enabled is False
    assert cfg.to_dict()["dsrl"]["enabled"] is False


def test_pi0_adapter_config_parses_and_roundtrips_dsrl():
    cfg = PI0AdapterConfig(dsrl={"enabled": True, "state_dim": 8})
    assert cfg.dsrl.enabled is True
    assert cfg.dsrl.state_dim == 8
    payload = cfg.to_dict()
    assert payload["dsrl"]["enabled"] is True
    reloaded = PI0AdapterConfig(**payload)
    assert reloaded.dsrl.enabled is True
    assert reloaded.dsrl.state_dim == 8
