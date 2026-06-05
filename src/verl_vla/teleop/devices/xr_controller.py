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
        self._frame_count = 0

    @override
    def reset(self) -> None:
        with self._lock:
            self._latest_frame.clear()
            self._events.clear()
            self._frame_count = 0

    @override
    def handle_event(self, event: DeviceEvent) -> None:
        with self._lock:
            if event.event_type == "xr_frame":
                self._latest_frame = dict(event.raw)
                self._frame_count += 1
            self._record_event(event)

    @override
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            controllers = self._latest_frame.get("controllers", {})
            return {
                "device": self.name,
                "frame_count": self._frame_count,
                "timestamp": self._latest_frame.get("timestamp"),
                "reference_space": self._latest_frame.get("reference_space"),
                "controllers": controllers,
                "latest_frame": self._latest_frame,
            }

    def latest_frame(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest_frame)
