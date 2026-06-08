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

"""Configuration helpers for LeRobot dataset recording."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class LeRobotRecorderConfig:
    enable: bool = True
    root: str = "/tmp/verl_vla_lerobot_records"
    repo_id: str = "local/verl_vla_libero"
    use_videos: bool = True
    image_writer_processes: int = 0
    image_writer_threads: int = 0
    batch_encoding_size: int = 1
    vcodec: str = "libsvtav1"
    video_files_size_in_mb: float = 1e-6
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoRecorderConfig:
    enable: bool = True
    root: str = "/tmp/verl_vla_videos"
    fps: int = 30
    font_size: int = 14


@dataclass(frozen=True)
class RecorderConfig:
    enable: bool = True
    async_enable: bool = False
    async_queue_size: int = 256
    recorders: tuple[str, ...] = ("lerobot", "video")
    lerobot: LeRobotRecorderConfig = field(default_factory=LeRobotRecorderConfig)
    video: VideoRecorderConfig = field(default_factory=VideoRecorderConfig)


def load_recorder_config(cfg: DictConfig | Any) -> RecorderConfig:
    raw = _to_dict(cfg.get("recorder", {}) if hasattr(cfg, "get") else {})
    legacy_lerobot_raw = _to_dict(cfg.get("dataset_recorder", {}) if hasattr(cfg, "get") else {})

    lerobot_raw = {**legacy_lerobot_raw, **_to_dict(raw.get("lerobot", {}))}
    video_raw = _to_dict(raw.get("video", {}))
    recorders = _normalize_recorders(raw.get("recorders", RecorderConfig.recorders))
    lerobot_raw["enable"] = "lerobot" in recorders
    video_raw["enable"] = "video" in recorders

    return RecorderConfig(
        enable=bool(raw.get("enable", RecorderConfig.enable)),
        async_enable=bool(raw.get("async_enable", RecorderConfig.async_enable)),
        async_queue_size=int(raw.get("async_queue_size", RecorderConfig.async_queue_size)),
        recorders=recorders,
        lerobot=_load_lerobot_recorder_config_from_raw(lerobot_raw),
        video=VideoRecorderConfig(
            **{key: video_raw[key] for key in VideoRecorderConfig.__annotations__ if key in video_raw}
        ),
    )


def load_lerobot_recorder_config(cfg: DictConfig | Any) -> LeRobotRecorderConfig:
    raw = _to_dict(cfg.get("dataset_recorder", {}) if hasattr(cfg, "get") else {})
    return _load_lerobot_recorder_config_from_raw(raw)


def _load_lerobot_recorder_config_from_raw(raw: dict[str, Any]) -> LeRobotRecorderConfig:
    raw = dict(raw)
    strategy_kwargs = dict(raw.pop("strategy_kwargs", {}))
    return LeRobotRecorderConfig(
        **{key: raw[key] for key in LeRobotRecorderConfig.__annotations__ if key in raw},
        strategy_kwargs=strategy_kwargs,
    )


def _to_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True)
    return dict(raw or {})


def _normalize_recorders(recorders: Any) -> tuple[str, ...]:
    if isinstance(recorders, str):
        recorders = [recorders]
    return tuple(str(recorder).strip().lower() for recorder in recorders if str(recorder).strip())
