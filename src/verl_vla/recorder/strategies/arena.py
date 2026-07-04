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

"""Arena-to-LeRobot frame conversion helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import override

from verl_vla.recorder.strategies.base import BaseLeRobotStrategy


class ArenaLeRobotStrategy(BaseLeRobotStrategy):
    """Generic recording strategy for Isaac Lab Arena environments."""

    def __init__(
        self,
        *,
        camera_names: tuple[str, ...] = ("robot_head_cam_rgb",),
        image_shape: tuple[int, int, int] = (480, 640, 3),
        state_dim: int = 50,
        action_dim: int = 50,
        fps: int = 50,
        robot_type: str | None = "arena",
    ) -> None:
        self.camera_names = tuple(camera_names)
        self.image_shape = tuple(image_shape)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self._fps = int(fps)
        self._robot_type = robot_type

    @property
    @override
    def fps(self) -> int:
        return self._fps

    @property
    @override
    def robot_type(self) -> str | None:
        return self._robot_type

    @override
    def features(self) -> dict[str, dict[str, Any]]:
        features = {
            f"observation.images.{name}": {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channel"],
            }
            for name in self.camera_names
        }
        features.update(
            {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (self.state_dim,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (self.action_dim,),
                    "names": ["action"],
                },
                "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
                "next.terminated": {"dtype": "bool", "shape": (1,), "names": None},
                "next.truncated": {"dtype": "bool", "shape": (1,), "names": None},
                "next.success": {"dtype": "bool", "shape": (1,), "names": None},
                "info.is_intervention": {"dtype": "bool", "shape": (1,), "names": None},
            }
        )
        return features

    @override
    def make_frame(
        self,
        *,
        observation: dict[str, Any],
        action: Any,
        task: str,
        next_reward: Any = 0.0,
        next_terminated: Any = False,
        next_truncated: Any = False,
        next_success: Any = False,
        is_intervention: Any = False,
    ) -> dict[str, Any]:
        frame = {
            "observation.state": np.asarray(observation["observation.state"], dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "next.reward": np.asarray(next_reward, dtype=np.float32).reshape(1),
            "next.terminated": np.asarray(next_terminated, dtype=bool).reshape(1),
            "next.truncated": np.asarray(next_truncated, dtype=bool).reshape(1),
            "next.success": np.asarray(next_success, dtype=bool).reshape(1),
            "info.is_intervention": np.asarray(is_intervention, dtype=bool).reshape(1),
            "task": str(task),
        }
        for name in self.camera_names:
            key = f"observation.images.{name}"
            if key in observation:
                frame[key] = np.ascontiguousarray(observation[key])
        return frame
