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

"""LIBERO-to-LeRobot frame conversion helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import override

from verl_vla.envs.libero_env.utils import get_libero_image, get_libero_wrist_image
from verl_vla.recorder.strategies.base import BaseLeRobotStrategy

DEFAULT_LIBERO_FPS = 10
DEFAULT_LIBERO_ROBOT_TYPE = "panda"


class LiberoLeRobotStrategy(BaseLeRobotStrategy):
    """LeRobot recording strategy for LIBERO Franka/Panda environments."""

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int] = (256, 256, 3),
        fps: int = DEFAULT_LIBERO_FPS,
        robot_type: str = DEFAULT_LIBERO_ROBOT_TYPE,
    ) -> None:
        self.image_shape = image_shape
        self._fps = fps
        self._robot_type = robot_type

    @property
    @override
    def fps(self) -> int:
        return self._fps

    @property
    @override
    def robot_type(self) -> str:
        return self._robot_type

    @override
    def features(self) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {
            "observation.images.image": {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist_image": {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],
            },
            "reward": {"dtype": "float32", "shape": (1,), "names": None},
            "done": {"dtype": "bool", "shape": (1,), "names": None},
            "truncated": {"dtype": "bool", "shape": (1,), "names": None},
            "is_intervention": {"dtype": "bool", "shape": (1,), "names": None},
        }
        return features

    @override
    def make_frame(
        self,
        *,
        obs: dict[str, Any],
        state: Any,
        action: Any,
        task: str,
        reward: Any = 0.0,
        done: Any = False,
        truncated: Any = False,
        is_intervention: Any = False,
    ) -> dict[str, Any]:
        return {
            "observation.images.image": np.ascontiguousarray(get_libero_image(obs)),
            "observation.images.wrist_image": np.ascontiguousarray(get_libero_wrist_image(obs)),
            "observation.state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": np.asarray(reward, dtype=np.float32).reshape(1),
            "done": np.asarray(done, dtype=bool).reshape(1),
            "truncated": np.asarray(truncated, dtype=bool).reshape(1),
            "is_intervention": np.asarray(is_intervention, dtype=bool).reshape(1),
            "task": str(task),
        }
