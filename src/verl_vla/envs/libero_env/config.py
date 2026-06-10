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

from dataclasses import asdict, dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class LiberoInitParamsConfig:
    camera_depths: bool = False
    camera_heights: int = 256
    camera_widths: int = 256
    camera_names: tuple[str, ...] = ("agentview", "robot0_eye_in_hand")

    def to_env_kwargs(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["camera_names"] = list(self.camera_names)
        return raw


@dataclass(frozen=True)
class LiberoConfig:
    task_suite_name: str = "libero_spatial"
    reset_warmup_steps: int = 10
    init_params: LiberoInitParamsConfig = field(default_factory=LiberoInitParamsConfig)


def load_libero_config(cfg: DictConfig | Any) -> LiberoConfig:
    raw = {}
    if hasattr(cfg, "get"):
        raw = cfg.get("libero", {}) or {}
    raw = _to_dict(raw)

    init_raw = _to_dict(raw.get("init_params", {}))
    camera_names = init_raw.get("camera_names")
    if isinstance(camera_names, str):
        init_raw["camera_names"] = (camera_names,)
    elif camera_names is not None:
        init_raw["camera_names"] = tuple(camera_names)

    init_cfg = LiberoInitParamsConfig(
        **{key: init_raw[key] for key in LiberoInitParamsConfig.__annotations__ if key in init_raw}
    )
    return LiberoConfig(
        task_suite_name=str(raw.get("task_suite_name", LiberoConfig.task_suite_name)),
        reset_warmup_steps=int(raw.get("reset_warmup_steps", LiberoConfig.reset_warmup_steps)),
        init_params=init_cfg,
    )


def _to_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True)
    return dict(raw or {})
