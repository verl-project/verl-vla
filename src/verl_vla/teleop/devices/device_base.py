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

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class DeviceEvent:
    event_type: str
    key: str | None = None
    code: str | None = None
    timestamp: float = field(default_factory=time.time)
    repeat: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DeviceEvent":
        return cls(
            event_type=str(payload.get("event_type") or payload.get("type") or ""),
            key=payload.get("key"),
            code=payload.get("code"),
            timestamp=float(payload.get("timestamp") or time.time()),
            repeat=bool(payload.get("repeat", False)),
            raw=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "key": self.key,
            "code": self.code,
            "timestamp": self.timestamp,
            "repeat": self.repeat,
            "raw": self.raw,
        }


class DeviceBase(ABC):
    name: str = "base"
    _MANUAL_REWARD_KEYS = {"R"}
    _RESTART_EPISODE_KEYS = {"BACKSPACE"}
    _STOP_EPISODE_KEYS = {"ENTER"}

    def __init__(self, max_events: int = 256):
        self._lock = Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._record_control = {
            "manual_reward": False,
            "restart_episode": False,
            "stop_episode": False,
        }

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def handle_event(self, event: DeviceEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        raise NotImplementedError

    def drain_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def pop_record_control(self) -> dict[str, bool]:
        with self._lock:
            control = dict(self._record_control)
            self._clear_record_control()
        return control

    def _record_event(self, event: DeviceEvent) -> None:
        self._events.append(event.to_dict())

        event_type = event.event_type.lower()
        if event_type == "keyboard_event":
            event_type = str(event.raw.get("event_type", "")).lower()
        if event_type != "keydown" or event.repeat:
            return

        key_name = self._normalize_key_name(event.code or event.key)
        if key_name in self._MANUAL_REWARD_KEYS:
            self._record_control["manual_reward"] = True
        elif key_name in self._RESTART_EPISODE_KEYS:
            self._record_control["restart_episode"] = True
        elif key_name in self._STOP_EPISODE_KEYS:
            self._record_control["stop_episode"] = True

    def _clear_record_control(self) -> None:
        for key in self._record_control:
            self._record_control[key] = False

    @staticmethod
    def _normalize_key_name(value: str | None) -> str:
        if not value:
            return ""
        value = str(value)
        if value.startswith("Key") and len(value) == 4:
            return value[-1].upper()
        if value == "Space":
            return "SPACE"
        return value.upper()
