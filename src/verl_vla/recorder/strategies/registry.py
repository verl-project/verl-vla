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

"""Registry for LeRobot recording strategies."""

from __future__ import annotations

from collections.abc import Callable

from .base import BaseLeRobotStrategy
from .libero import LiberoLeRobotStrategy

StrategyFactory = Callable[..., BaseLeRobotStrategy]

_REGISTRY: dict[str, StrategyFactory] = {
    "libero": LiberoLeRobotStrategy,
}


def register_lerobot_strategy(env_type: str, factory: StrategyFactory) -> None:
    key = _normalize_env_type(env_type)
    if key in _REGISTRY:
        raise ValueError(f"LeRobot strategy already registered for env_type={env_type!r}.")
    _REGISTRY[key] = factory


def get_lerobot_strategy(env_type: str, **kwargs) -> BaseLeRobotStrategy:
    key = _normalize_env_type(env_type)
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"No LeRobot strategy registered for env_type={env_type!r}. Available: {available}")
    return _REGISTRY[key](**kwargs)


def list_lerobot_strategy_env_types() -> list[str]:
    return sorted(_REGISTRY)


def _normalize_env_type(env_type: str) -> str:
    return env_type.strip().lower()
