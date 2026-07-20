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

import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]


def _compose_sft_config(*args: str) -> DictConfig:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "verl_vla.entrypoints.train.sft",
            *args,
            "--cfg",
            "job",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    return OmegaConf.create(result.stdout)


def test_pi05_sft_example_composes_through_training_entrypoint():
    config = _compose_sft_config(
        "cluster.actor_rollout_ref.model.path=Miical/pi05-base",
        "+cluster.actor_rollout_ref.model.override_config.n_action_steps=10",
        "cluster.actor_rollout_ref.model.adapter.embodiment=libero",
        "cluster.actor_rollout_ref.model.adapter.critic.enabled=False",
        "data.repo_id=lerobot/libero_spatial_image",
    )

    assert config.ray_kwargs.ray_init is not None
    assert config.cluster.actor_rollout_ref.model.path == "Miical/pi05-base"
    assert config.cluster.actor_rollout_ref.model.adapter.embodiment == "libero"
    assert config.data.repo_id == "lerobot/libero_spatial_image"


def test_gr00t_sft_example_composes_through_training_entrypoint():
    config = _compose_sft_config(
        "--config-path",
        str(REPO_ROOT / "examples/gr00t_sft"),
        "--config-name",
        "main_gr00t_sft",
        f"hydra.searchpath=[file://{REPO_ROOT}/src/verl_vla/workflows/config]",
    )

    assert config.ray_kwargs.ray_init is not None
    assert config.cluster.actor_rollout_ref.model.adapter.policy_type == "libero"
    assert config.cluster.actor_rollout_ref.model.override_config.load_bf16 is True
    assert config.data.repo_id == "lerobot/libero_spatial_image"
