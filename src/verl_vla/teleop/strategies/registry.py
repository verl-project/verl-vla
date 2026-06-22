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

from verl_vla.teleop.strategies.base import InterventionStrategyBase
from verl_vla.teleop.strategies.gamepad_libero import LiberoGamepadStrategy
from verl_vla.teleop.strategies.keyboard_libero import LiberoKeyboardStrategy
from verl_vla.teleop.strategies.xr_controller_arena import ArenaXRControllerStrategy
from verl_vla.teleop.strategies.xr_controller_libero import LiberoXRControllerStrategy


class InterventionStrategyRegistry:
    def __init__(self):
        self._strategies: dict[tuple[str, str], type[InterventionStrategyBase]] = {}

    def register(self, strategy_cls: type[InterventionStrategyBase]) -> None:
        key = (strategy_cls.env_type, strategy_cls.device_type)
        self._strategies[key] = strategy_cls

    def get(self, env_type: str, device_type: str, cfg: Any) -> InterventionStrategyBase:
        key = (env_type, device_type)
        if key not in self._strategies:
            raise NotImplementedError(
                f"No teleop intervention strategy registered for env={env_type} device={device_type}"
            )
        return self._strategies[key](cfg)


_REGISTRY = InterventionStrategyRegistry()
_REGISTRY.register(LiberoKeyboardStrategy)
_REGISTRY.register(LiberoXRControllerStrategy)
_REGISTRY.register(ArenaXRControllerStrategy)
_REGISTRY.register(LiberoGamepadStrategy)


def get_strategy(env_type: str, device_type: str, cfg: Any) -> InterventionStrategyBase:
    return _REGISTRY.get(env_type, device_type, cfg)
