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
class XRControllerDeviceCfg:
    max_events: int = 256


class XRControllerDevice(DeviceBase):
    name = "xr_controller"

    def __init__(self, cfg: XRControllerDeviceCfg | None = None):
        self.cfg = cfg or XRControllerDeviceCfg()
        super().__init__(max_events=self.cfg.max_events)
        self._latest_frame: dict[str, Any] = {}
        self._record_button_states: dict[str, bool] = {}
        self._frame_count = 0

    @override
    def reset(self) -> None:
        with self._lock:
            self._latest_frame.clear()
            self._record_button_states.clear()
            self._events.clear()
            self._clear_record_control()
            self._frame_count = 0

    @override
    def handle_event(self, event: DeviceEvent) -> None:
        with self._lock:
            if event.event_type == "xr_frame":
                self._latest_frame = dict(event.raw)
                self._update_record_controls_from_frame(self._latest_frame)
                self._frame_count += 1
            self._record_event(event)

    @override
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "device": self.name,
                "frame_count": self._frame_count,
                "timestamp": self._latest_frame.get("timestamp"),
                "key_bindings": self.key_bindings(),
            }

    def latest_frame(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest_frame)

    def key_bindings(self) -> dict[str, str]:
        return {
            "A/X or Enter": "stop recording episode",
            "B/Y or Backspace": "restart recording episode",
            "R": "manual reward",
        }

    def _update_record_controls_from_frame(self, frame: dict[str, Any]) -> None:
        controllers = frame.get("controllers", {})
        if not isinstance(controllers, dict):
            return

        for hand, controller in controllers.items():
            if not isinstance(controller, dict):
                continue
            primary_pressed = self._button_pressed(controller, "primary", raw_button_index=4)
            secondary_pressed = self._button_pressed(controller, "secondary", raw_button_index=5)
            self._set_record_control_on_press(f"{hand}:primary", primary_pressed, "stop_episode")
            self._set_record_control_on_press(f"{hand}:secondary", secondary_pressed, "restart_episode")

    def _set_record_control_on_press(self, key: str, pressed: bool, control_name: str) -> None:
        was_pressed = self._record_button_states.get(key, False)
        if pressed and not was_pressed:
            self._record_control[control_name] = True
        self._record_button_states[key] = pressed

    @staticmethod
    def _button_pressed(controller: dict[str, Any], button_name: str, *, raw_button_index: int) -> bool:
        buttons = controller.get("buttons", {})
        button = buttons.get(button_name, {}) if isinstance(buttons, dict) else {}
        if isinstance(button, dict) and button:
            return bool(button.get("pressed", False) or float(button.get("value", 0.0)) > 0.5)

        raw_buttons = controller.get("raw_buttons", [])
        if isinstance(raw_buttons, list) and len(raw_buttons) > raw_button_index:
            raw_button = raw_buttons[raw_button_index]
            if isinstance(raw_button, dict):
                return bool(raw_button.get("pressed", False) or float(raw_button.get("value", 0.0)) > 0.5)
        return False
