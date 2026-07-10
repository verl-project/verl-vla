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

from verl_vla.teleop.config import KeyboardTeleopConfig
from verl_vla.teleop.devices import DeviceBase
from verl_vla.teleop.strategies.base import InterventionStrategyBase


class LiberoKeyboardStrategy(InterventionStrategyBase):
    env_type = "libero"
    device_type = "keyboard"
    _GRIPPER_TERM = True

    def __init__(self, cfg: KeyboardTeleopConfig | None = None):
        keyboard_cfg = cfg or KeyboardTeleopConfig()
        super().__init__(keyboard_cfg)
        self.cfg = keyboard_cfg
        self._active = False
        self._close_gripper = False
        self._gripper_active = False
        self._key_mapping = self._create_key_mapping()

    @override
    def reset(self) -> None:
        self._active = False
        self._close_gripper = False
        self._gripper_active = False

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
    def get_action(self, device: DeviceBase) -> Any:
        return self._command_from_device(device)

    @override
    def snapshot(self, device: DeviceBase) -> dict[str, Any]:
        command = self._command_from_device(device)
        return {
            "strategy": f"{self.env_type}:{self.device_type}",
            "is_intervening": self.is_intervening(device),
            "active": self._active,
            "close_gripper": self._close_gripper,
            "gripper_active": self._gripper_active,
            "command": command.astype(float).tolist(),
        }

    def _command_from_device(self, device: DeviceBase) -> np.ndarray:
        self._process_events(device)
        snapshot = device.snapshot()
        pressed_keys = set(snapshot.get("pressed_keys", []))
        delta_pos = np.zeros(3, dtype=np.float32)
        delta_rot = np.zeros(3, dtype=np.float32)
        for key_name in pressed_keys:
            if key_name in {"W", "S", "A", "D", "Q", "E", "PAGEUP", "PAGEDOWN"}:
                delta_pos += self._key_mapping[key_name]
            elif key_name in {"Z", "X", "T", "G", "C", "V", "ARROWUP", "ARROWDOWN", "ARROWLEFT", "ARROWRIGHT"}:
                delta_rot += self._key_mapping[key_name]
        rot_vec = Rotation.from_euler("XYZ", delta_rot).as_rotvec()
        command = np.concatenate([delta_pos, rot_vec]).astype(np.float32)
        if self._GRIPPER_TERM:
            gripper = 0.0
            if self._gripper_active:
                gripper = 1.0 if self._close_gripper else -1.0
            command = np.append(command, gripper).astype(np.float32)
        return command

    def _state_to_action(self, action: Any, relative_command: np.ndarray) -> np.ndarray:
        # LIBERO expects a relative 7D action, so the incoming state is only context.
        return relative_command.astype(np.float32)

    def _process_events(self, device: DeviceBase) -> None:
        for event in device.drain_events():
            key_name = str(event.get("code") or event.get("key") or "")
            event_type = str(event.get("event_type", "")).lower()
            repeat = bool(event.get("repeat", False))
            if event_type == "keydown" and key_name == "SPACE" and not repeat:
                self._active = not self._active
            elif event_type == "keydown" and key_name == "K" and not repeat:
                self._gripper_active = True
                self._close_gripper = not self._close_gripper
            elif event_type == "keydown" and key_name == "L":
                self.reset()
                device.reset()

    def _has_motion_key(self, device: DeviceBase) -> bool:
        pressed_keys = set(device.snapshot().get("pressed_keys", []))
        return bool(pressed_keys.intersection(self._key_mapping.keys()))

    def _default_command(self) -> np.ndarray:
        command = np.zeros(6, dtype=np.float32)
        if self._GRIPPER_TERM:
            command = np.append(command, 0.0).astype(np.float32)
        return command

    def _create_key_mapping(self) -> dict[str, np.ndarray]:
        return {
            "W": np.asarray([-1.0, 0.0, 0.0], dtype=np.float32) * self.cfg.pos_sensitivity,
            "S": np.asarray([1.0, 0.0, 0.0], dtype=np.float32) * self.cfg.pos_sensitivity,
            "A": np.asarray([0.0, 1.0, 0.0], dtype=np.float32) * self.cfg.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0], dtype=np.float32) * self.cfg.pos_sensitivity,
            "Q": np.asarray([0.0, 0.0, 1.0], dtype=np.float32) * self.cfg.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0], dtype=np.float32) * self.cfg.pos_sensitivity,
            "PAGEUP": np.asarray([0.0, 0.0, 1.0], dtype=np.float32) * self.cfg.pos_sensitivity,
            "PAGEDOWN": np.asarray([0.0, 0.0, -1.0], dtype=np.float32) * self.cfg.pos_sensitivity,
            "Z": np.asarray([1.0, 0.0, 0.0], dtype=np.float32) * self.cfg.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0], dtype=np.float32) * self.cfg.rot_sensitivity,
            "T": np.asarray([0.0, 1.0, 0.0], dtype=np.float32) * self.cfg.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0], dtype=np.float32) * self.cfg.rot_sensitivity,
            "C": np.asarray([0.0, 0.0, 1.0], dtype=np.float32) * self.cfg.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0], dtype=np.float32) * self.cfg.rot_sensitivity,
            "ARROWUP": np.asarray([0.0, 1.0, 0.0], dtype=np.float32) * self.cfg.rot_sensitivity,
            "ARROWDOWN": np.asarray([0.0, -1.0, 0.0], dtype=np.float32) * self.cfg.rot_sensitivity,
            "ARROWLEFT": np.asarray([0.0, 0.0, 1.0], dtype=np.float32) * self.cfg.rot_sensitivity,
            "ARROWRIGHT": np.asarray([0.0, 0.0, -1.0], dtype=np.float32) * self.cfg.rot_sensitivity,
        }
