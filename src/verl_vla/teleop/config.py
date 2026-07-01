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
    rot_sensitivity: float = 0.5
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
