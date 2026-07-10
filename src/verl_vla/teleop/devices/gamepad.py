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

from dataclasses import dataclass
from typing import Any

from typing_extensions import override

from verl_vla.teleop.devices.device_base import DeviceBase, DeviceEvent


@dataclass(frozen=True)
class GamepadDeviceCfg:
    max_events: int = 256


class GamepadDevice(DeviceBase):
    name = "gamepad"

    def __init__(self, cfg: GamepadDeviceCfg | None = None):
        self.cfg = cfg or GamepadDeviceCfg()
        super().__init__(max_events=self.cfg.max_events)
        self._latest_state: dict[str, Any] = {}
        self._button_states: dict[str, bool] = {}
        self._axis_values: dict[str, float] = {}
        self._connected = False
        self._device_id = ""

    @override
    def reset(self) -> None:
        with self._lock:
            self._latest_state.clear()
            self._button_states.clear()
            self._axis_values.clear()
            self._events.clear()
            self._clear_record_control()
            self._connected = False
            self._device_id = ""

    @override
    def handle_event(self, event: DeviceEvent) -> None:
        with self._lock:
            event_type = event.event_type.lower()
            if event_type == "gamepad_update":
                self._latest_state = dict(event.raw)
                buttons_raw = event.raw.get("buttons")
                if not isinstance(buttons_raw, dict):
                    buttons_raw = {}
                axes_raw = event.raw.get("axes")
                if not isinstance(axes_raw, dict):
                    axes_raw = {}

                pressed_edges = set()
                for key, value in buttons_raw.items():
                    if isinstance(value, dict):
                        pressed = bool(value.get("pressed", False))
                    else:
                        pressed = bool(value)
                    if pressed and not self._button_states.get(key, False):
                        pressed_edges.add(key)
                    self._button_states[key] = pressed

                if "RB" in pressed_edges:
                    self._record_control["manual_reward"] = True
                if "LT" in pressed_edges:
                    self._record_control["restart_episode"] = True
                if "LB" in pressed_edges:
                    self._record_control["stop_episode"] = True

                for key, value in axes_raw.items():
                    self._axis_values[key] = float(value) if isinstance(value, int | float) else 0.0

                self._connected = True
                self._device_id = str(event.raw.get("id", ""))

            elif event_type == "gamepad_disconnect":
                self._connected = False
                self._device_id = ""
                self._latest_state.clear()
                self._button_states.clear()
                self._axis_values.clear()

            self._record_event(event)

    @override
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            pressed_buttons = sorted([k for k, v in self._button_states.items() if v])
            active_axes = {k: round(v, 3) for k, v in self._axis_values.items() if abs(v) > 0.01}
            is_active = bool(pressed_buttons) or bool(active_axes)
            return {
                "device": self.name,
                "connected": self._connected,
                "device_id": self._device_id,
                "pressed_buttons": pressed_buttons,
                "active_axes": active_axes,
                "timestamp": self._latest_state.get("timestamp"),
                "active": is_active,
                "key_bindings": self.key_bindings(),
            }

    def key_bindings(self) -> dict[str, str]:
        return {
            "Left Stick Y": "+x / -x",
            "Left Stick X": "+y / -y",
            "Right Stick Y": "+z / -z",
            "Right Stick X": "+yaw / -yaw",
            "D-Pad Left/Right": "+roll / -roll",
            "D-Pad Up/Down": "+pitch / -pitch",
            "RT": "intervention (hold)",
            "X": "toggle gripper",
            "LT": "restart recording episode",
            "LB": "start/stop recording episode",
            "RB": "manual reward",
        }

    def is_active(self) -> bool:
        with self._lock:
            return any(self._button_states.values()) or any(abs(v) > 0.01 for v in self._axis_values.values())

    def get_button(self, button_name: str) -> bool:
        with self._lock:
            return self._button_states.get(button_name, False)

    def get_axis(self, axis_name: str) -> float:
        with self._lock:
            return self._axis_values.get(axis_name, 0.0)
