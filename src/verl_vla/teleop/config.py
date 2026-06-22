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

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class TeleopServerConfig:
    host: str = "0.0.0.0"
    base_port: int = 18000
    rank_stride: int = 10000
    stage_stride: int = 1000
    jpeg_quality: int = 80
    log_level: str = "warning"
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None


@dataclass(frozen=True)
class KeyboardTeleopConfig:
    pos_sensitivity: float = 0.05
    rot_sensitivity: float = 0.12


@dataclass(frozen=True)
class XRControllerTeleopConfig:
    hand: str = "right"
    pos_sensitivity: float = 25.0
    rot_sensitivity: float = 2.0
    intervention_button: str = "squeeze"
    gripper_button: str = "trigger"
    button_threshold: float = 0.5
    max_events: int = 256


@dataclass(frozen=True)
class GamepadTeleopConfig:
    pos_sensitivity: float = 0.5
    rot_sensitivity: float = 1.0
    intervention_button: str = "RT"
    gripper_button: str = "X"
    button_threshold: float = 0.5
    max_events: int = 256
    left_stick_x_axis: str = "axis_0"
    left_stick_y_axis: str = "axis_1"
    right_stick_y_axis: str = "axis_3"
    right_stick_x_axis: str = "axis_2"
    dpad_up_button: str = "DUp"
    dpad_down_button: str = "DDown"
    dpad_left_button: str = "DLeft"
    dpad_right_button: str = "DRight"


@dataclass(frozen=True)
class TeleopConfig:
    enable: bool = False
    device: str | None = "keyboard"
    devices: tuple[str, ...] = ("keyboard",)
    server: TeleopServerConfig = field(default_factory=TeleopServerConfig)
    keyboard: KeyboardTeleopConfig = field(default_factory=KeyboardTeleopConfig)
    xr_controller: XRControllerTeleopConfig = field(default_factory=XRControllerTeleopConfig)
    gamepad: GamepadTeleopConfig = field(default_factory=GamepadTeleopConfig)

    def __post_init__(self):
        if isinstance(self.devices, str):
            devices = (self.devices,)
        else:
            devices = tuple(self.devices)
        object.__setattr__(self, "devices", devices)


def load_teleop_config(cfg: DictConfig | Any, device: str | None = None) -> TeleopConfig:
    raw = {}
    if hasattr(cfg, "get"):
        raw = cfg.get("teleop", {}) or {}
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw)
    if device is not None:
        raw["device"] = device

    server_raw = raw.get("server", {})
    if isinstance(server_raw, DictConfig):
        server_raw = OmegaConf.to_container(server_raw, resolve=True)
    server_raw = dict(server_raw or {})
    for key in TeleopServerConfig.__annotations__:
        if key in raw and key not in server_raw:
            server_raw[key] = raw[key]
    server_cfg = TeleopServerConfig(
        **{key: server_raw[key] for key in TeleopServerConfig.__annotations__ if key in server_raw}
    )

    keyboard_raw = raw.get("keyboard", {})
    if isinstance(keyboard_raw, DictConfig):
        keyboard_raw = OmegaConf.to_container(keyboard_raw, resolve=True)
    keyboard_raw = dict(keyboard_raw or {})
    if "keyboard_pos_sensitivity" in raw and "pos_sensitivity" not in keyboard_raw:
        keyboard_raw["pos_sensitivity"] = raw["keyboard_pos_sensitivity"]
    if "keyboard_rot_sensitivity" in raw and "rot_sensitivity" not in keyboard_raw:
        keyboard_raw["rot_sensitivity"] = raw["keyboard_rot_sensitivity"]
    keyboard_cfg = KeyboardTeleopConfig(
        **{key: keyboard_raw[key] for key in KeyboardTeleopConfig.__annotations__ if key in keyboard_raw}
    )
    xr_controller_raw = raw.get("xr_controller", {})
    if isinstance(xr_controller_raw, DictConfig):
        xr_controller_raw = OmegaConf.to_container(xr_controller_raw, resolve=True)
    xr_controller_raw = dict(xr_controller_raw or {})
    xr_controller_cfg = XRControllerTeleopConfig(
        **{key: xr_controller_raw[key] for key in XRControllerTeleopConfig.__annotations__ if key in xr_controller_raw}
    )
    gamepad_raw = raw.get("gamepad", {})
    if isinstance(gamepad_raw, DictConfig):
        gamepad_raw = OmegaConf.to_container(gamepad_raw, resolve=True)
    gamepad_raw = dict(gamepad_raw or {})
    gamepad_cfg = GamepadTeleopConfig(
        **{key: gamepad_raw[key] for key in GamepadTeleopConfig.__annotations__ if key in gamepad_raw}
    )
    devices = raw.get("devices")
    if devices is None:
        devices = [raw.get("device", TeleopConfig.device)]
    elif isinstance(devices, str):
        devices = [devices]
    devices = tuple(
        str(item).strip().lower()
        for item in devices
        if item is not None and str(item).strip().lower() not in {"", "none", "null"}
    )

    return TeleopConfig(
        enable=bool(raw.get("enable", TeleopConfig.enable)),
        device=raw.get("device", TeleopConfig.device),
        devices=devices,
        server=server_cfg,
        keyboard=keyboard_cfg,
        xr_controller=xr_controller_cfg,
        gamepad=gamepad_cfg,
    )
