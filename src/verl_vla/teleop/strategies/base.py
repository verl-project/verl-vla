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

from abc import ABC, abstractmethod
from typing import Any

from verl_vla.teleop.devices import DeviceBase


class InterventionStrategyBase(ABC):
    env_type: str = "base"
    device_type: str = "base"

    def __init__(self, cfg: Any):
        self.cfg = cfg

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_intervening(self, device: DeviceBase) -> bool:
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, action: Any, device: DeviceBase) -> Any:
        raise NotImplementedError

    def get_action(self, device: DeviceBase) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not support get_action.")

    @abstractmethod
    def snapshot(self, device: DeviceBase) -> dict[str, Any]:
        raise NotImplementedError
