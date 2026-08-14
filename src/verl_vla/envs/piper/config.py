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

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from verl.base_config import BaseConfig

_PIPER_MODELS = {"piper", "piper_h", "piper_l", "piper_x"}
_ARM_NAMES = {"left", "right"}


@dataclass
class PiperArmConfig:
    """One physical Piper and its logical teleoperation hand."""

    name: str
    can_channel: str
    model: str = "piper_x"
    firmware_version: str = "v188"
    initial_action: list[float] | None = None

    def __post_init__(self) -> None:
        if self.name not in _ARM_NAMES:
            raise ValueError(f"Piper arm name must be left or right, got {self.name!r}")
        if self.model not in _PIPER_MODELS:
            raise ValueError(f"Unsupported Piper model {self.model!r}; choose from {sorted(_PIPER_MODELS)}")
        if not self.can_channel:
            raise ValueError("Piper can_channel must not be empty")
        if self.firmware_version not in {"default", "v183", "v188", "v189"}:
            raise ValueError(f"Unsupported Piper firmware version {self.firmware_version!r}")
        if self.initial_action is not None:
            action = np.asarray(self.initial_action, dtype=float)
            if action.shape != (7,) or not np.all(np.isfinite(action)):
                raise ValueError(f"{self.name} initial_action must contain seven finite values")


@dataclass
class PiperCameraConfig:
    """One V4L2 camera exposed as a named observation."""

    name: str
    device: str
    fps: int = 30
    width: int = 640
    height: int = 480
    fourcc: str = "YUYV"
    warmup_s: int = 3

    def __post_init__(self) -> None:
        if not self.name or not self.device:
            raise ValueError("Piper camera name and device must not be empty")
        if self.fps <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError(f"Piper camera dimensions must be positive, got {self.width}x{self.height}")
        if len(self.fourcc) != 4:
            raise ValueError(f"Piper camera fourcc must contain four characters, got {self.fourcc!r}")
        if self.warmup_s < 0:
            raise ValueError("Piper camera warmup_s must be non-negative")


@dataclass
class PiperConfig(BaseConfig):
    """One- or two-arm Piper environment backed directly by pyAgxArm."""

    simulator_type: str = "piper"
    arms: list[PiperArmConfig] = field(
        default_factory=lambda: [
            PiperArmConfig(name="left", can_channel="can0"),
            PiperArmConfig(name="right", can_channel="can1"),
        ]
    )
    cameras: list[PiperCameraConfig] = field(default_factory=list)
    action_dim: int = field(init=False)
    state_dim: int = field(init=False)
    task_description: str = "Teleoperate the Piper arms."
    urdf_root: str = field(
        default_factory=lambda: str(
            Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
            / "verl-vla/agx_arm_description/agx_arm_urdf"
        )
    )
    startup_timeout_s: float = 10.0
    control_hz: float = 30.0
    speed_percent: int | None = None
    driver_tcp_length: float = 0.13
    tool_length: float = 0.1325
    ik_position_weight: float = 2.0
    ik_orientation_weight: float = 2.0
    ik_regularization_weight: float = 0.01
    ik_smooth_weight: float = 0.1
    ik_max_iterations: int = 50
    ik_tolerance: float = 1e-4
    camera_stale_timeout_s: float = 0.5
    reset_duration_s: float = 3.0
    reset_timeout_s: float = 15.0
    reset_joint_tolerance: float = 0.03
    reset_gripper_tolerance: float = 0.002
    gripper_open_width: float = 0.1
    gripper_close_width: float = 0.0
    gripper_width_step: float = 0.005
    gripper_force: float = 1.0

    def __post_init__(self) -> None:
        arms = [arm if isinstance(arm, PiperArmConfig) else PiperArmConfig(**dict(arm)) for arm in self.arms]
        cameras = [
            camera if isinstance(camera, PiperCameraConfig) else PiperCameraConfig(**dict(camera))
            for camera in self.cameras
        ]
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "cameras", cameras)
        if not 1 <= len(arms) <= 2:
            raise ValueError(f"Piper WebXR/keyboard teleoperation supports one or two arms, got {len(arms)}")
        arm_names = [arm.name for arm in arms]
        if len(set(arm_names)) != len(arm_names):
            raise ValueError(f"Piper arm names must be unique, got {arm_names}")
        camera_names = [camera.name for camera in cameras]
        if len(set(camera_names)) != len(camera_names):
            raise ValueError(f"Piper camera names must be unique, got {camera_names}")
        object.__setattr__(self, "action_dim", 7 * len(arms))
        object.__setattr__(self, "state_dim", 7 * len(arms))
        if not self.urdf_root:
            raise ValueError("Piper urdf_root must not be empty")
        if self.startup_timeout_s <= 0 or self.control_hz <= 0:
            raise ValueError("Piper startup timeout and control rate must be positive")
        if self.speed_percent is not None and not 1 <= self.speed_percent <= 100:
            raise ValueError("Piper speed_percent must be null or between 1 and 100")
        if self.driver_tcp_length < 0 or self.tool_length < 0:
            raise ValueError("Piper TCP lengths must be non-negative")
        if self.ik_position_weight <= 0 or self.ik_orientation_weight <= 0 or self.ik_smooth_weight < 0:
            raise ValueError("IK pose weights must be positive and smooth weight must be non-negative")
        if self.ik_regularization_weight < 0 or self.ik_max_iterations <= 0 or self.ik_tolerance <= 0:
            raise ValueError("IK regularization, iteration, and tolerance settings are invalid")
        if self.camera_stale_timeout_s <= 0:
            raise ValueError("Piper camera_stale_timeout_s must be positive")
        if self.reset_duration_s <= 0 or self.reset_timeout_s <= self.reset_duration_s:
            raise ValueError("reset_timeout_s must be greater than the positive reset_duration_s")
        if self.reset_joint_tolerance <= 0 or self.reset_gripper_tolerance <= 0:
            raise ValueError("Piper reset tolerances must be positive")
        if self.gripper_close_width > self.gripper_open_width:
            raise ValueError("gripper_close_width must not exceed gripper_open_width")
        for arm in arms:
            if arm.initial_action is not None and not (
                self.gripper_close_width <= arm.initial_action[6] <= self.gripper_open_width
            ):
                raise ValueError(
                    f"{arm.name} initial_action gripper width must be between "
                    f"{self.gripper_close_width} and {self.gripper_open_width} meters"
                )
        if self.gripper_width_step <= 0 or not 0.5 <= self.gripper_force <= 3.0:
            raise ValueError("gripper_width_step must be positive and gripper_force must be between 0.5 and 3.0 N")
