# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass

from verl.base_config import BaseConfig


@dataclass
class ArenaSimulatorConfig(BaseConfig):
    """Simulator config for Isaac Lab Arena environments."""

    simulator_type: str = "arena"
    max_episode_steps: int = 20
    seed: int = 42
    action_dim: int = 50
    state_dim: int | None = None

    env_name: str = "galileo_g1_locomanip_pick_and_place"
    object: str = "brown_box"
    embodiment: str = "g1_wbc_joint"
    object_set: str | None = None
    kitchen_style: int = 2
    task_description: str = "Pick and place the brown box."
    enable_cameras: bool = True
    camera_names: tuple[str, ...] = ("robot_head_cam_rgb",)
    image_shape: tuple[int, int, int] = (480, 640, 3)
    rl_success_reward: bool = True
    subtask_reward: bool = False
    env_spacing: float = 30.0
    disable_fabric: bool = False
    solve_relations: bool = True
    enable_pinocchio: bool = True
    placement_seed: int | None = None
    resolve_on_reset: bool | None = None
    presets: str | None = None
