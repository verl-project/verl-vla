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

import json
from types import SimpleNamespace

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from verl_vla.models.act_torch import ACTTrainableModel
from verl_vla.models.builder import build_vla_model


def _tiny_policy() -> ACTPolicy:
    config = ACTConfig(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (3,)),
            "observation.environment_state": PolicyFeature(FeatureType.ENV, (4,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
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


def _processors(policy: ACTPolicy):
    return make_pre_post_processors(
        policy.config,
        dataset_stats={
            "observation.state": {"mean": [1.0, 2.0, 3.0], "std": [2.0, 2.0, 2.0]},
            "action": {"mean": [0.5, -0.5], "std": [0.25, 0.5]},
        },
    )


def test_native_act_artifact_round_trip(tmp_path) -> None:
    policy = _tiny_policy()
    preprocessor, postprocessor = _processors(policy)
    model = ACTTrainableModel(policy, preprocessor=preprocessor, postprocessor=postprocessor)
    expected_state = {name: value.detach().clone() for name, value in policy.state_dict().items()}

    model.export_policy(tmp_path, state_dict=model.state_dict())

    artifact_names = {path.name for path in tmp_path.iterdir()}
    assert {"config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"} <= (
        artifact_names
    )
    reloaded = ACTPolicy.from_pretrained(tmp_path, strict=True)
    assert reloaded.state_dict().keys() == expected_state.keys()
    assert all(torch.equal(reloaded.state_dict()[name], value) for name, value in expected_state.items())
    actions = reloaded.predict_action_chunk(
        {
            "observation.state": torch.zeros(1, 3),
            "observation.environment_state": torch.zeros(1, 4),
        }
    )
    assert actions.shape == (1, 2, 2)

    built = build_vla_model(
        SimpleNamespace(
            native_architecture="act",
            local_path=str(tmp_path),
            override_config={},
            adapter={},
        ),
        torch_dtype=torch.float32,
    )
    assert isinstance(built.policy, ACTPolicy)


def test_build_native_act_from_config_without_policy_weights(tmp_path) -> None:
    config_dir = tmp_path / "config_only"
    _tiny_policy().config.save_pretrained(config_dir)

    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {"mean": [1.0, 2.0, 3.0], "std": [2.0, 2.0, 2.0]},
                "action": {"mean": [0.5, -0.5], "std": [0.25, 0.5]},
            }
        ),
        encoding="utf-8",
    )
    model_config = SimpleNamespace(
        native_architecture="act",
        local_path=str(config_dir),
        override_config={},
        adapter={"processor_dataset_root": str(dataset_root)},
    )
    with pytest.raises(FileNotFoundError, match="initialization.json"):
        build_vla_model(model_config, torch_dtype=torch.float32)

    (config_dir / "initialization.json").write_text('{"type": "act_config"}', encoding="utf-8")
    built = build_vla_model(
        model_config,
        torch_dtype=torch.float32,
    )

    assert isinstance(built.policy, ACTPolicy)
    assert built.policy.config.chunk_size == 2
    assert not (config_dir / "model.safetensors").exists()


def test_build_native_act_rejects_dataset_state_from_an_old_schema(tmp_path) -> None:
    config_dir = tmp_path / "config_only"
    _tiny_policy().config.save_pretrained(config_dir)
    (config_dir / "initialization.json").write_text('{"type": "act_config"}', encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {"mean": [0.0] * 6, "std": [1.0] * 6},
                "action": {"mean": [0.0] * 2, "std": [1.0] * 2},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dimension 6, but the checkpoint expects 3"):
        build_vla_model(
            SimpleNamespace(
                native_architecture="act",
                local_path=str(config_dir),
                override_config={},
                adapter={"processor_dataset_root": str(dataset_root)},
            ),
            torch_dtype=torch.float32,
        )
