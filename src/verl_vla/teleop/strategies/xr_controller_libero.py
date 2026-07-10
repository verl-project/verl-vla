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

from typing import Any, cast

import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import override

from verl_vla.teleop.config import XRControllerTeleopConfig
from verl_vla.teleop.devices import DeviceBase, XRControllerDevice
from verl_vla.teleop.strategies.base import InterventionStrategyBase


class LiberoXRControllerStrategy(InterventionStrategyBase):
    env_type = "libero"
    device_type = "xr_controller"
    _POSITION_AXES = (2, 0, 1)
    _POSITION_SIGNS = (1.0, -1.0, 1.0)
    _ROTATION_AXES = (2, 0, 1)
    _ROTATION_SIGNS = (1.0, -1.0, 1.0)
    _ROTATION_DELTA_FRAME = "local"
    _GRIPPER_TERM = True

    def __init__(self, cfg: XRControllerTeleopConfig | None = None):
        xr_cfg = cfg or XRControllerTeleopConfig()
        super().__init__(xr_cfg)
        self.cfg = xr_cfg
        self._reference_pos: np.ndarray | None = None
        self._reference_quat: np.ndarray | None = None
        self._last_pos: np.ndarray | None = None
        self._last_quat: np.ndarray | None = None
        self._last_command = self._default_command()
        self._active = False

    @override
    def reset(self) -> None:
        self._reference_pos = None
        self._reference_quat = None
        self._last_pos = None
        self._last_quat = None
        self._last_command = self._default_command()
        self._active = False

    @override
    def is_intervening(self, device: DeviceBase) -> bool:
        xr_device = cast(XRControllerDevice, device)
        frame = xr_device.latest_frame()
        controller = self._controller_from_frame(frame)
        active = self._button_value(controller, str(self.cfg.intervention_button)) >= self.cfg.button_threshold
        if active and not self._active:
            self._set_reference(controller)
        elif not active:
            self._reference_pos = None
            self._reference_quat = None
            self._last_pos = None
            self._last_quat = None
            self._last_command = self._default_command()
        self._active = active
        return active

    @override
    def apply_action(self, action: Any, device: DeviceBase) -> Any:
        action_array = np.asarray(action)
        if not self.is_intervening(device):
            command = self._default_command()
            if action_array.shape != command.shape:
                return self._default_command()
            return action
        command = self._consume_command_from_device(device)
        if action_array.shape == command.shape:
            return command.astype(action_array.dtype, copy=False)
        if action_array.ndim > 0 and action_array.shape[-1] == command.shape[-1]:
            overridden = action_array.copy()
            overridden[...] = command.astype(action_array.dtype, copy=False)
            return overridden
        return command

    @override
    def get_action(self, device: DeviceBase) -> Any:
        if not self.is_intervening(device):
            return self._default_command()
        return self._consume_command_from_device(device)

    @override
    def snapshot(self, device: DeviceBase) -> dict[str, Any]:
        return {
            "strategy": f"{self.env_type}:{self.device_type}",
            "is_intervening": self._active,
            "active": self._active,
            "command": self._last_command.astype(float).tolist(),
        }

    def _consume_command_from_device(self, device: DeviceBase) -> np.ndarray:
        xr_device = cast(XRControllerDevice, device)
        return self._consume_command_from_frame(xr_device.latest_frame())

    def _consume_command_from_frame(self, frame: dict[str, Any]) -> np.ndarray:
        controller = self._controller_from_frame(frame)
        pose = self._pose_from_controller(controller)
        if pose is None or self._last_pos is None or self._last_quat is None:
            self._last_command = self._default_command()
            return self._default_command()
        position, quat = pose
        web_delta_pos = position - self._last_pos
        web_delta_rot = self._relative_rotation(Rotation.from_quat(self._last_quat), Rotation.from_quat(quat))
        self._last_pos = position
        self._last_quat = quat
        delta_pos = self._remap_vector(web_delta_pos, self._POSITION_AXES, self._POSITION_SIGNS)
        delta_rot = self._remap_rotation(web_delta_rot, self._ROTATION_AXES, self._ROTATION_SIGNS).as_rotvec()
        delta_pos = delta_pos * self.cfg.pos_sensitivity
        delta_rot = delta_rot * self.cfg.rot_sensitivity
        command = np.concatenate([delta_pos, delta_rot]).astype(np.float32)
        if self._GRIPPER_TERM:
            gripper_value = self._button_value(controller, str(self.cfg.gripper_button))
            gripper = 1.0 if gripper_value >= self.cfg.button_threshold else -1.0
            command = np.append(command, gripper).astype(np.float32)
        self._last_command = command
        return command

    def _set_reference(self, controller: dict[str, Any]) -> None:
        pose = self._pose_from_controller(controller)
        if pose is None:
            self._reference_pos = None
            self._reference_quat = None
            self._last_pos = None
            self._last_quat = None
            self._last_command = self._default_command()
            return
        self._reference_pos, self._reference_quat = pose
        self._last_pos = self._reference_pos.copy()
        self._last_quat = self._reference_quat.copy()
        self._last_command = self._default_command()

    def _controller_from_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        controllers = frame.get("controllers", {})
        controller = controllers.get(self.cfg.hand, {})
        return controller if isinstance(controller, dict) else {}

    def _pose_from_controller(self, controller: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
        pose = controller.get("grip_pose") or controller.get("target_ray_pose") or {}
        position = pose.get("position")
        orientation = pose.get("orientation")
        if position is None or orientation is None:
            return None
        position_array = np.asarray(position, dtype=np.float32)
        quat_array = np.asarray(orientation, dtype=np.float32)
        if position_array.shape != (3,) or quat_array.shape != (4,):
            return None
        return position_array, quat_array

    def _button_value(self, controller: dict[str, Any], button_name: str) -> float:
        buttons = controller.get("buttons", {})
        button = buttons.get(button_name, {})
        if isinstance(button, dict):
            return float(button.get("value", 0.0))
        return 0.0

    def _default_command(self) -> np.ndarray:
        command = np.zeros(6, dtype=np.float32)
        if self._GRIPPER_TERM:
            command = np.append(command, 0.0).astype(np.float32)
        return command

    @staticmethod
    def _remap_vector(vector: np.ndarray, axes: tuple[int, int, int], signs: tuple[float, float, float]) -> np.ndarray:
        return np.asarray([vector[axis] * sign for axis, sign in zip(axes, signs, strict=True)], dtype=np.float32)

    def _relative_rotation(self, reference: Rotation, current: Rotation) -> Rotation:
        if self._ROTATION_DELTA_FRAME == "local":
            return reference.inv() * current
        return current * reference.inv()

    @staticmethod
    def _remap_rotation(rotation: Rotation, axes: tuple[int, int, int], signs: tuple[float, float, float]) -> Rotation:
        transform = LiberoXRControllerStrategy._axis_transform(axes, signs)
        return Rotation.from_matrix(transform @ rotation.as_matrix() @ transform.T)

    @staticmethod
    def _axis_transform(axes: tuple[int, int, int], signs: tuple[float, float, float]) -> np.ndarray:
        transform = np.zeros((3, 3), dtype=np.float32)
        for output_axis, (input_axis, sign) in enumerate(zip(axes, signs, strict=True)):
            transform[output_axis, input_axis] = sign
        return transform
