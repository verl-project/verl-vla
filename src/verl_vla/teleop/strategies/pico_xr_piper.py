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

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import override

from verl_vla.teleop.config import XRControllerTeleopConfig
from verl_vla.teleop.devices import DeviceBase, XRControllerDevice
from verl_vla.teleop.strategies.base import InterventionStrategyBase


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    return matrix


def _webxr_pose_matrix(pose: dict[str, Any]) -> np.ndarray | None:
    try:
        position = np.asarray(pose["position"], dtype=float)
        quaternion = np.asarray(pose["orientation"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if position.shape != (3,) or quaternion.shape != (4,):
        return None
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    matrix[:3, 3] = position
    return matrix


@dataclass
class _HandState:
    active: bool = False
    intervention_down: bool = False
    start_pose: np.ndarray | None = None
    previous_relative_pose: np.ndarray | None = None
    previous_trigger: float | None = None
    robot_zero_rotation: np.ndarray | None = None


class PiperPicoXRStrategy(InterventionStrategyBase):
    """Convert native WebXR frames into Piper environment actions."""

    env_type = "piper"
    device_type = "xr_controller"

    def __init__(
        self,
        cfg: XRControllerTeleopConfig | None = None,
        *,
        simulator_cfg: Any,
        arm_rotation_reader: Callable[[], dict[str, np.ndarray]],
        task_delta_to_action: Callable[[np.ndarray], np.ndarray],
        task_target_sync: Callable[[str], None],
    ):
        super().__init__(cfg or XRControllerTeleopConfig())
        self._action_dim = int(simulator_cfg.action_dim)
        self._arm_names = tuple(arm.name for arm in simulator_cfg.arms)
        self._position_scale = float(self.cfg.pos_sensitivity)
        self._rotation_scale = float(self.cfg.rot_sensitivity)
        self._gripper_open_width = float(simulator_cfg.gripper_open_width)
        self._arm_rotation_reader = arm_rotation_reader
        self._task_delta_to_action = task_delta_to_action
        self._task_target_sync = task_target_sync
        self._intervention_button = str(self.cfg.intervention_button)
        self._button_threshold = float(self.cfg.button_threshold)
        self._webxr_to_robot = np.array(
            [[0.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        self._controller_alignment = _rotation_matrix(-math.pi, 0.0, -math.pi / 2.0)
        self._robot_alignment = _rotation_matrix(-1.5708, 0.0, -1.5708)
        self.reset()

    @override
    def reset(self) -> None:
        self._timestamp: float | None = None
        self._hands = {name: _HandState() for name in self._arm_names}

    @override
    def is_intervening(self, device: DeviceBase) -> bool:
        self._update_buttons(cast(XRControllerDevice, device).latest_frame())
        return any(state.active for state in self._hands.values())

    @override
    def apply_action(self, action: Any, device: DeviceBase) -> Any:
        action_array = np.asarray(action)
        if action_array.shape != (self._action_dim,):
            raise ValueError(f"Piper action must have shape [{self._action_dim}], got {action_array.shape}")
        if not self.is_intervening(device):
            return action
        return self.get_action(device).astype(action_array.dtype, copy=False)

    @override
    def get_action(self, device: DeviceBase) -> np.ndarray:
        return self._action_from_frame(cast(XRControllerDevice, device).latest_frame())

    @override
    def snapshot(self, device: DeviceBase) -> dict[str, Any]:
        del device
        active_hands = {name: state.active for name, state in self._hands.items()}
        return {
            "strategy": "piper:xr_controller",
            "active": any(active_hands.values()),
            "active_hands": active_hands,
            "backend": "Piper direct joint control",
            "key_bindings": self.key_bindings(),
        }

    def key_bindings(self) -> dict[str, str]:
        return {
            "Squeeze": "start / pause this arm",
            "A/X": "save and end episode",
            "B/Y": "discard and restart episode",
            "Trigger": "gripper",
        }

    def _action_from_frame(self, frame: dict[str, Any]) -> np.ndarray:
        command = np.zeros(self._action_dim, dtype=np.float32)
        if not frame:
            return self._task_delta_to_action(command)
        timestamp = float(frame.get("timestamp", 0.0))
        if timestamp == self._timestamp:
            return self._task_delta_to_action(command)
        self._timestamp = timestamp
        controllers = frame.get("controllers")
        if not isinstance(controllers, dict):
            return self._task_delta_to_action(command)
        arm_rotations = self._arm_rotation_reader()

        for arm_index, hand in enumerate(self._arm_names):
            state = self._hands[hand]
            controller = controllers.get(hand)
            if not isinstance(controller, dict):
                continue
            self._update_hand_buttons(hand, state, controller, arm_rotations[hand])
            pose = controller.get("grip_pose") or controller.get("target_ray_pose")
            pose_matrix = _webxr_pose_matrix(pose) if isinstance(pose, dict) else None
            if pose_matrix is None:
                continue
            controller_pose = self._webxr_to_robot @ pose_matrix @ self._controller_alignment @ self._robot_alignment
            offset = arm_index * 7
            if state.active:
                if state.start_pose is None:
                    state.start_pose = controller_pose.copy()
                    state.previous_relative_pose = np.eye(4)
                relative = np.linalg.inv(state.start_pose) @ controller_pose
                relative[:3, 3] *= self._position_scale
                rotation = Rotation.from_matrix(relative[:3, :3])
                relative[:3, :3] = Rotation.from_rotvec(rotation.as_rotvec() * self._rotation_scale).as_matrix()
                previous = state.previous_relative_pose
                if previous is not None:
                    zero_rotation = state.robot_zero_rotation
                    if zero_rotation is None:
                        raise RuntimeError(f"Missing robot zero rotation for {hand} Piper arm")
                    command[offset : offset + 3] = zero_rotation @ (relative[:3, 3] - previous[:3, 3])
                    relative_rotation_delta = relative[:3, :3] @ previous[:3, :3].T
                    command[offset + 3 : offset + 6] = Rotation.from_matrix(
                        zero_rotation @ relative_rotation_delta @ zero_rotation.T
                    ).as_rotvec()
                state.previous_relative_pose = relative

            trigger = np.clip(self._button_value(controller, "trigger"), 0.0, 1.0)
            if state.previous_trigger is not None:
                command[offset + 6] = (trigger - state.previous_trigger) * self._gripper_open_width
            state.previous_trigger = trigger
        return self._task_delta_to_action(command)

    def _update_buttons(self, frame: dict[str, Any]) -> None:
        controllers = frame.get("controllers", {}) if frame else {}
        if not isinstance(controllers, dict):
            return
        arm_rotations = self._arm_rotation_reader()
        for hand, state in self._hands.items():
            controller = controllers.get(hand)
            if isinstance(controller, dict):
                self._update_hand_buttons(hand, state, controller, arm_rotations[hand])

    def _update_hand_buttons(
        self,
        hand: str,
        state: _HandState,
        controller: dict[str, Any],
        robot_rotation: np.ndarray,
    ) -> None:
        intervention_down = self._pressed(controller, self._intervention_button)
        if intervention_down and not state.intervention_down:
            state.active = not state.active
            state.start_pose = None
            state.previous_relative_pose = None
            if state.active:
                self._task_target_sync(hand)
            state.robot_zero_rotation = robot_rotation.copy() if state.active else None
        state.intervention_down = intervention_down

    @staticmethod
    def _button_value(controller: dict[str, Any], name: str) -> float:
        buttons = controller.get("buttons", {})
        button = buttons.get(name) if isinstance(buttons, dict) else None
        if not isinstance(button, dict):
            return 0.0
        return float(button.get("value", button.get("pressed", False)))

    def _pressed(self, controller: dict[str, Any], name: str) -> bool:
        return self._button_value(controller, name) >= self._button_threshold
