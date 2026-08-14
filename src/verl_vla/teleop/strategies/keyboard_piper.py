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

from collections.abc import Callable
from typing import Any

import numpy as np
from typing_extensions import override

from verl_vla.teleop.config import KeyboardTeleopConfig
from verl_vla.teleop.devices import DeviceBase
from verl_vla.teleop.strategies.base import InterventionStrategyBase

_KEY_TO_AXIS = {
    "W": (0, 1.0),
    "S": (0, -1.0),
    "A": (1, 1.0),
    "D": (1, -1.0),
    "Q": (2, 1.0),
    "E": (2, -1.0),
    "Z": (3, 1.0),
    "X": (3, -1.0),
    "T": (4, 1.0),
    "G": (4, -1.0),
    "C": (5, 1.0),
    "V": (5, -1.0),
}


class PiperKeyboardStrategy(InterventionStrategyBase):
    """Map keyboard input to the configured Piper arms."""

    env_type = "piper"
    device_type = "keyboard"

    def __init__(
        self,
        cfg: KeyboardTeleopConfig | None = None,
        *,
        simulator_cfg: Any,
        task_delta_to_action: Callable[[np.ndarray], np.ndarray],
        task_target_sync: Callable[[str], None],
    ):
        cfg = cfg or KeyboardTeleopConfig()
        super().__init__(cfg)
        self._arm_names = tuple(arm.name for arm in simulator_cfg.arms)
        self._action_dim = int(simulator_cfg.action_dim)
        self._position_step = float(cfg.pos_sensitivity)
        self._rotation_step = float(cfg.rot_sensitivity)
        self._gripper_step = float(simulator_cfg.gripper_width_step)
        self._task_delta_to_action = task_delta_to_action
        self._task_target_sync = task_target_sync
        self._active_arm = 0
        self._was_intervening = False

    @override
    def reset(self) -> None:
        self._active_arm = 0
        self._was_intervening = False

    @override
    def is_intervening(self, device: DeviceBase) -> bool:
        keys = self._keys(device)
        self._select_arm(keys)
        return self._update_intervention(keys)

    @override
    def apply_action(self, action: Any, device: DeviceBase) -> Any:
        action_array = np.asarray(action)
        if action_array.shape != (self._action_dim,):
            raise ValueError(f"Piper action must have shape [{self._action_dim}], got {action_array.shape}")
        return self.get_action(device).astype(action_array.dtype, copy=False)

    @override
    def get_action(self, device: DeviceBase) -> np.ndarray:
        keys = self._keys(device)
        self._select_arm(keys)
        self._update_intervention(keys)
        return self._task_delta_to_action(self._task_delta_from_keys(keys))

    def _task_delta(self, device: DeviceBase) -> np.ndarray:
        keys = self._keys(device)
        self._select_arm(keys)
        return self._task_delta_from_keys(keys)

    def _task_delta_from_keys(self, keys: set[str]) -> np.ndarray:
        command = np.zeros(self._action_dim, dtype=np.float32)
        offset = self._active_arm * 7
        for key in keys & _KEY_TO_AXIS.keys():
            axis, sign = _KEY_TO_AXIS[key]
            scale = self._position_step if axis < 3 else self._rotation_step
            command[offset + axis] += sign * scale
        command[offset + 6] = (float("O" in keys) - float("K" in keys)) * self._gripper_step
        return command

    def _update_intervention(self, keys: set[str]) -> bool:
        active = bool(keys & (_KEY_TO_AXIS.keys() | {"O", "K"}))
        if active and not self._was_intervening:
            self._task_target_sync(self._arm_names[self._active_arm])
        self._was_intervening = active
        return active

    @override
    def snapshot(self, device: DeviceBase) -> dict[str, Any]:
        command = self._task_delta(device)
        return {
            "strategy": "piper:keyboard",
            "active_arm": self._arm_names[self._active_arm],
            "command": command.astype(float).tolist(),
            "unit": "m/rad device delta; environment action is absolute joint position",
            "key_bindings": self.key_bindings(),
        }

    def key_bindings(self) -> dict[str, str]:
        return {
            "1 / 2": "select first / second configured arm",
            "W/S": "+x / -x",
            "A/D": "+y / -y",
            "Q/E": "+z / -z",
            "Z/X": "+roll / -roll",
            "T/G": "+pitch / -pitch",
            "C/V": "+yaw / -yaw",
            "O/K": "open / close gripper",
        }

    @staticmethod
    def _keys(device: DeviceBase) -> set[str]:
        return set(device.snapshot().get("pressed_keys", []))

    def _select_arm(self, keys: set[str]) -> None:
        if "1" in keys:
            self._active_arm = 0
        elif "2" in keys and len(self._arm_names) == 2:
            self._active_arm = 1
