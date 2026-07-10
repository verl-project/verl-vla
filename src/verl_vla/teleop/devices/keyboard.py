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
class KeyboardDeviceCfg:
    max_events: int = 256


class KeyboardDevice(DeviceBase):
    name = "keyboard"

    def __init__(self, cfg: KeyboardDeviceCfg | None = None):
        self.cfg = cfg or KeyboardDeviceCfg()
        super().__init__(max_events=self.cfg.max_events)
        self._pressed_keys: set[str] = set()

    @override
    def reset(self) -> None:
        with self._lock:
            self._pressed_keys.clear()
            self._events.clear()
            self._clear_record_control()

    @override
    def handle_event(self, event: DeviceEvent) -> None:
        key_name = self.normalize_key_name(event.code or event.key)
        event_type = event.event_type.lower()
        normalized_event = DeviceEvent(
            event_type=event.event_type,
            key=key_name,
            code=key_name,
            timestamp=event.timestamp,
            repeat=event.repeat,
            raw=event.raw,
        )
        with self._lock:
            if event_type == "keydown":
                self._pressed_keys.add(key_name)
            elif event_type == "keyup":
                self._pressed_keys.discard(key_name)
            self._record_event(normalized_event)

    @override
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "device": self.name,
                "pressed_keys": sorted(self._pressed_keys),
                "key_bindings": self.key_bindings(),
            }

    def key_bindings(self) -> dict[str, str]:
        return {
            "W/S": "-x / +x",
            "A/D": "+y / -y",
            "Q/E, PgUp/PgDown": "+z / -z",
            "Z/X": "+roll / -roll",
            "T/G, Up/Down": "+pitch / -pitch",
            "C/V, Left/Right": "+yaw / -yaw",
            "Space": "toggle intervention",
            "K": "toggle gripper",
            "L": "reset device",
            "R": "manual reward",
            "Backspace": "restart recording episode",
            "Enter": "stop recording episode",
        }

    @staticmethod
    def normalize_key_name(value: str | None) -> str:
        if not value:
            return ""
        value = str(value)
        if value.startswith("Key") and len(value) == 4:
            return value[-1].upper()
        if value == "Space":
            return "SPACE"
        return value.upper()
