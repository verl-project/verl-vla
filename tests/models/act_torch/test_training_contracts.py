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

from __future__ import annotations

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION
from verl import DataProto

from verl_vla.models.act_torch import ACTTrainableModel
from verl_vla.models.act_torch.policy import LiberoActOutput


def _libero_policy() -> ACTPolicy:
    config = ACTConfig(
        device="cpu",
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.images.wrist_image": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        chunk_size=2,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        pretrained_backbone_weights=None,
    )
    return ACTPolicy(config)


def _observations() -> DataProto:
    return DataProto.from_dict(
        tensors={
            "observation.images.image": torch.rand(2, 3, 64, 64),
            "observation.images.wrist_image": torch.rand(2, 3, 64, 64),
            "observation.state": torch.rand(2, 7),
        }
    )


def _model(*, adapter_config=None) -> ACTTrainableModel:
    policy = _libero_policy()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        dataset_stats={
            "observation.images.image": {
                "mean": [[[0.485]], [[0.456]], [[0.406]]],
                "std": [[[0.229]], [[0.224]], [[0.225]]],
            },
            "observation.images.wrist_image": {
                "mean": [[[0.485]], [[0.456]], [[0.406]]],
                "std": [[[0.229]], [[0.224]], [[0.225]]],
            },
            "observation.state": {"mean": [0.0] * 7, "std": [1.0] * 7},
            "action": {"mean": [0.0] * 7, "std": [1.0] * 7},
        },
    )
    return ACTTrainableModel(
        policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        adapter_config=adapter_config,
    )


def test_act_sft_optimizer_step() -> None:
    model = _model(adapter_config={"freeze_vision_tower": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    original_head = model.policy.model.action_head.weight.detach().clone()

    loss = model.sft_loss(
        _observations(),
        None,
        {"action": torch.rand(2, 2, 7)},
        torch.ones(2),
        action_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.float32),
    )
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(model.policy.model.action_head.weight, original_head)


def test_act_sft_padding_matches_native_lerobot_reduction() -> None:
    model = _model()
    model.policy.model.forward = lambda batch: (torch.zeros_like(batch[ACTION]), (None, None))
    actions = torch.ones(2, 2, 7)
    actions[1, 1] = 100.0

    loss = model.sft_loss(
        _observations(),
        None,
        {"action": actions},
        torch.ones(2),
        action_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.float32),
    )

    # LeRobot masks padded values but retains the full chunk in the mean:
    # sample losses are 1.0 and 0.5, respectively.
    torch.testing.assert_close(loss, torch.tensor(0.75))
    torch.testing.assert_close(model.sft_metrics["l1_loss"], torch.tensor(0.75))


def test_act_sft_treats_rollout_action_chunks_as_fully_valid() -> None:
    model = _model()
    model.policy.model.forward = lambda batch: (torch.zeros_like(batch[ACTION]), (None, None))

    loss = model.sft_loss(
        _observations(),
        None,
        {"action": torch.ones(2, 2, 7)},
        torch.ones(2),
    )

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_act_rollout_initializes_without_critic() -> None:
    model = _model(adapter_config={"sac_rollout_noise_scale": 0.1})
    model.eval()

    model.sac_init()
    observations = _observations()
    eval_output = model.sac_sample_actions(observations, eval=True)
    rollout_output = model.sac_sample_actions(observations, eval=False)

    assert eval_output.action.shape == (2, 2, 7)
    assert eval_output.log_prob is None
    assert not torch.equal(eval_output.action, rollout_output.action)
    assert not hasattr(model, "critic_backend")


def test_act_rollout_action_chunk_size_replans_before_native_chunk_end() -> None:
    model = _model(adapter_config={"action_chunk_size": 1})

    output, _, actions = model.sample_actions(_observations())

    assert output.action.shape == (2, 1, 7)
    assert actions["full_action"].shape == (2, 1, 7)


def test_act_sac_freezes_the_configured_vision_tower() -> None:
    model = _model(adapter_config={"freeze_vision_tower": True})

    model.sac_init()

    assert all(not parameter.requires_grad for parameter in model.policy.model.backbone.parameters())


def test_act_rollout_transports_environment_actions_as_float32() -> None:
    output = LiberoActOutput.from_model_output(
        {
            "full_action": torch.zeros(2, 2, 7, dtype=torch.bfloat16),
            "action_chunk_size": 2,
        }
    )

    assert output.to_data_proto().batch["action"].dtype == torch.float32


def test_act_sac_actor_and_critic_forward_for_batched_images() -> None:
    model = _model(
        adapter_config={
            "critic": {"enabled": True, "prefix_embed_dim": 32, "hidden_dims": [16]},
            "sac_train_noise_scale": 0.0,
        },
    )

    features = model.sac_forward_state_features(_observations(), None)
    actions, log_probs, metrics = model.sac_forward_actor(features)
    q_values = model.sac_forward_critic({"action": actions}, features, method="cat", requires_grad=True)

    assert all(feature.shape[0] == 2 for feature in features)
    assert actions.shape == (2, 2, 7)
    assert log_probs is None
    assert metrics == {}
    assert q_values.shape == (2, 2)


def test_act_sac_critic_detaches_state_features_but_preserves_action_gradient() -> None:
    model = _model(
        adapter_config={
            "critic": {"enabled": True, "prefix_embed_dim": 32, "hidden_dims": [16]},
            "sac_train_noise_scale": 0.0,
        },
    )
    prefix, states, positions = model.sac_forward_state_features(_observations(), None)
    detached_features = (
        prefix.detach().requires_grad_(),
        states.detach().requires_grad_(),
        positions.detach(),
    )
    actions = torch.zeros(2, 2, 7, requires_grad=True)

    q_values = model.sac_forward_critic(
        {"action": actions},
        detached_features,
        method="min",
        requires_grad=False,
    )
    q_values.sum().backward()

    assert detached_features[0].grad is None
    assert detached_features[1].grad is None
    assert actions.grad is not None
