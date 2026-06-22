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

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import override

from verl_vla.teleop.config import GamepadTeleopConfig
from verl_vla.teleop.devices import DeviceBase
from verl_vla.teleop.strategies.base import InterventionStrategyBase


class LiberoGamepadStrategy(InterventionStrategyBase):
    env_type = "libero"
    device_type = "gamepad"
    _GRIPPER_TERM = True

    def __init__(self, cfg: GamepadTeleopConfig | None = None):
        gamepad_cfg = cfg or GamepadTeleopConfig()
        super().__init__(gamepad_cfg)
        self.cfg = gamepad_cfg
        self._active = False
        self._close_gripper = False
        self._button_states: dict[str, bool] = {}
        self._axis_values: dict[str, float] = {}
        self._prev_gripper_pressed = False

    @override
    def reset(self) -> None:
        self._active = False
        self._close_gripper = False
        self._button_states.clear()
        self._axis_values.clear()
        self._prev_gripper_pressed = False

    @override
    def is_intervening(self, device: DeviceBase) -> bool:
        self._process_events(device)
        return self._active

    @override
    def apply_action(self, action: Any, device: DeviceBase) -> Any:
        command = self._command_from_device(device)
        action_array = np.asarray(action)
        if not self.is_intervening(device):
            if action_array.shape != command.shape:
                return self._default_command()
            return action
        command = self._state_to_action(action, command)
        if action_array.shape == command.shape:
            return command.astype(action_array.dtype, copy=False)
        if action_array.ndim > 0 and action_array.shape[-1] == command.shape[-1]:
            overridden = action_array.copy()
            overridden[...] = command.astype(action_array.dtype, copy=False)
            return overridden
        return command

    @override
    def snapshot(self, device: DeviceBase) -> dict[str, Any]:
        command = self._command_from_device(device)
        return {
            "strategy": f"{self.env_type}:{self.device_type}",
            "is_intervening": self.is_intervening(device),
            "active": self._active,
            "close_gripper": self._close_gripper,
            "command": command.astype(float).tolist(),
        }

    def _command_from_device(self, device: DeviceBase) -> np.ndarray:
        self._process_events(device)

        left_x = self._axis_values.get(self.cfg.left_stick_x_axis, 0.0)
        left_y = self._axis_values.get(self.cfg.left_stick_y_axis, 0.0)
        right_y = self._axis_values.get(self.cfg.right_stick_y_axis, 0.0)

        delta_pos = np.zeros(3, dtype=np.float32)
        delta_pos[0] = left_y * self.cfg.pos_sensitivity
        delta_pos[1] = -left_x * self.cfg.pos_sensitivity
        delta_pos[2] = -right_y * self.cfg.pos_sensitivity

        dpad_up = self._button_states.get(self.cfg.dpad_up_button, False)
        dpad_down = self._button_states.get(self.cfg.dpad_down_button, False)
        dpad_left = self._button_states.get(self.cfg.dpad_left_button, False)
        dpad_right = self._button_states.get(self.cfg.dpad_right_button, False)
        right_x = self._axis_values.get(self.cfg.right_stick_x_axis, 0.0)

        delta_rot = np.zeros(3, dtype=np.float32)
        delta_rot[0] = (dpad_left - dpad_right) * self.cfg.rot_sensitivity
        delta_rot[1] = (dpad_up - dpad_down) * self.cfg.rot_sensitivity
        delta_rot[2] = -right_x * self.cfg.rot_sensitivity

        rot_vec = Rotation.from_euler("XYZ", delta_rot).as_rotvec()
        command = np.concatenate([delta_pos, rot_vec]).astype(np.float32)

        if self._GRIPPER_TERM:
            gripper = 1.0 if self._close_gripper else -1.0
            command = np.append(command, gripper).astype(np.float32)

        return command

    def _state_to_action(self, action: Any, relative_command: np.ndarray) -> np.ndarray:
        return relative_command.astype(np.float32)

    def _process_events(self, device: DeviceBase) -> None:
        try:
            events = device.drain_events()
        except Exception:
            events = []

        for event in events:
            if not isinstance(event, dict):
                continue

            event_type = str(event.get("event_type", "")).lower()
            if event_type == "gamepad_update":
                raw = event.get("raw", {})
                if not isinstance(raw, dict):
                    raw = {}

                buttons_raw = raw.get("buttons", {})
                if not isinstance(buttons_raw, dict):
                    buttons_raw = {}

                axes_raw = raw.get("axes", {})
                if not isinstance(axes_raw, dict):
                    axes_raw = {}

                for key, value in buttons_raw.items():
                    if isinstance(value, dict):
                        self._button_states[key] = bool(value.get("pressed", False))
                    else:
                        self._button_states[key] = bool(value)

                for key, value in axes_raw.items():
                    self._axis_values[key] = float(value) if isinstance(value, int | float) else 0.0

                intervention_pressed = self._button_states.get(self.cfg.intervention_button, False)
                self._active = bool(intervention_pressed)

                gripper_pressed = self._button_states.get(self.cfg.gripper_button, False)
                if gripper_pressed and not self._prev_gripper_pressed:
                    self._close_gripper = not self._close_gripper
                self._prev_gripper_pressed = gripper_pressed

            elif event_type == "gamepad_disconnect":
                self._active = False
                self._button_states.clear()
                self._axis_values.clear()
                self._prev_gripper_pressed = False

    def _default_command(self) -> np.ndarray:
        command = np.zeros(6, dtype=np.float32)
        if self._GRIPPER_TERM:
            command = np.append(command, 0.0).astype(np.float32)
        return command
