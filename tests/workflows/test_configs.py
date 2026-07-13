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

import pytest
from hydra import compose, initialize_config_module


@pytest.mark.parametrize(
    ("config_name", "required_keys"),
    [
        ("ppo", {"actor_rollout_ref", "env", "trainer"}),
        ("recap", {"ray_kwargs", "recap"}),
        ("sac", {"cluster", "ray_kwargs", "trainer"}),
        ("sft", {"cluster", "data", "ray_kwargs", "trainer"}),
    ],
)
def test_train_workflow_config_composes(config_name: str, required_keys: set[str]) -> None:
    with initialize_config_module(config_module="verl_vla.workflows.config.train", version_base=None):
        config = compose(config_name=config_name)

    assert required_keys <= set(config.keys())


@pytest.mark.parametrize(
    ("config_name", "required_keys"),
    [
        ("dagger", {"cluster", "max_episodes", "ray_kwargs"}),
        ("eval", {"cluster", "max_episodes", "output_dir", "ray_kwargs"}),
        ("record", {"cluster", "num_episodes", "ray_kwargs"}),
        ("teleop", {"cluster", "ray_kwargs"}),
    ],
)
def test_operational_workflow_config_composes(config_name: str, required_keys: set[str]) -> None:
    with initialize_config_module(config_module="verl_vla.workflows.config", version_base=None):
        config = compose(config_name=config_name)

    assert required_keys <= set(config.keys())
