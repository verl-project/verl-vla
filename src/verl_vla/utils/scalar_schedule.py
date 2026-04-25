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

import math
from dataclasses import dataclass


@dataclass
class ScheduledScalar:
    base_value: float
    enabled: bool = False
    initial_value: float | None = None
    final_value: float | None = None
    method: str = "square"
    clamp_min: float | None = 0.0
    clamp_max: float | None = 0.9999

    def __post_init__(self):
        self.base_value = self._clamp(self.base_value)
        self.initial_value = self._clamp(self.base_value if self.initial_value is None else self.initial_value)
        self.final_value = self._clamp(self.base_value if self.final_value is None else self.final_value)
        self.control_value = 0.0
        self.current_value = self.base_value

    def _clamp(self, value: float) -> float:
        value = float(value)
        if self.clamp_min is not None:
            value = max(float(self.clamp_min), value)
        if self.clamp_max is not None:
            value = min(float(self.clamp_max), value)
        return value

    @staticmethod
    def _control_01(value: float) -> float:
        return min(max(float(value), 0.0), 1.0)

    def _progress(self, control_value: float) -> float:
        x = self._control_01(control_value)
        if self.method == "linear":
            return x
        if self.method == "square":
            return x * x
        if self.method == "cos":
            return 0.5 - 0.5 * math.cos(math.pi * x)
        raise ValueError(f"Unsupported scalar schedule method: {self.method}")

    def refresh(self, control_value: float) -> float:
        self.control_value = self._control_01(control_value)
        if self.enabled:
            progress = self._progress(self.control_value)
            self.current_value = self._clamp(self.initial_value + (self.final_value - self.initial_value) * progress)
        else:
            self.current_value = self.base_value
        return self.current_value
