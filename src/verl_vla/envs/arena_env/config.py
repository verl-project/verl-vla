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
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class ArenaConfig:
    env_name: str = "put_item_in_fridge_and_close_door"
    object: str = "ranch_dressing_hope_robolab"
    embodiment: str = "gr1_joint"
    object_set: str | None = None
    kitchen_style: int = 2
    task_description: str = "Place the sauce bottle on the top shelf of the fridge, and close the fridge door."
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


def load_arena_config(cfg: DictConfig | Any) -> ArenaConfig:
    raw = _to_dict(cfg.get("arena", {}) if hasattr(cfg, "get") else {})
    camera_names = _as_tuple(raw.get("camera_names", ArenaConfig.camera_names), item_type=str)
    image_shape = _as_tuple(raw.get("image_shape", ArenaConfig.image_shape), item_type=int)

    return ArenaConfig(
        env_name=str(raw.get("env_name", ArenaConfig.env_name)),
        object=str(raw.get("object", ArenaConfig.object)),
        embodiment=str(raw.get("embodiment", ArenaConfig.embodiment)),
        object_set=_optional_str(raw.get("object_set", ArenaConfig.object_set)),
        kitchen_style=int(raw.get("kitchen_style", ArenaConfig.kitchen_style)),
        task_description=str(raw.get("task_description", ArenaConfig.task_description)),
        enable_cameras=bool(raw.get("enable_cameras", ArenaConfig.enable_cameras)),
        camera_names=camera_names,
        image_shape=image_shape,
        rl_success_reward=bool(raw.get("rl_success_reward", ArenaConfig.rl_success_reward)),
        subtask_reward=bool(raw.get("subtask_reward", ArenaConfig.subtask_reward)),
        env_spacing=float(raw.get("env_spacing", ArenaConfig.env_spacing)),
        disable_fabric=bool(raw.get("disable_fabric", ArenaConfig.disable_fabric)),
        solve_relations=bool(raw.get("solve_relations", ArenaConfig.solve_relations)),
        enable_pinocchio=bool(raw.get("enable_pinocchio", ArenaConfig.enable_pinocchio)),
        placement_seed=_optional_int(raw.get("placement_seed", ArenaConfig.placement_seed)),
        resolve_on_reset=_optional_bool(raw.get("resolve_on_reset", ArenaConfig.resolve_on_reset)),
        presets=_optional_str(raw.get("presets", ArenaConfig.presets)),
    )


def _to_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True)
    return dict(raw or {})


def _as_tuple(raw: Any, *, item_type) -> tuple:
    if isinstance(raw, str):
        return (item_type(raw),)
    return tuple(item_type(item) for item in raw)


def _optional_str(raw: Any) -> str | None:
    return None if raw is None else str(raw)


def _optional_int(raw: Any) -> int | None:
    return None if raw is None else int(raw)


def _optional_bool(raw: Any) -> bool | None:
    return None if raw is None else bool(raw)
