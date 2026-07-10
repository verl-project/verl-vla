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

from typing import Any

import numpy as np
import torch
from typing_extensions import override

from verl_vla.envs.base import BaseEnv
from verl_vla.teleop.devices.device_base import DeviceBase, DeviceEvent


class DummyDevice(DeviceBase):
    name = "dummy"

    @override
    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._clear_record_control()

    @override
    def handle_event(self, event: DeviceEvent) -> None:
        with self._lock:
            self._record_event(event)

    @override
    def snapshot(self) -> dict[str, Any]:
        return {"device": self.name}


def test_record_control_survives_event_drain() -> None:
    device = DummyDevice()

    device.handle_event(DeviceEvent(event_type="keydown", code="KeyR"))

    assert device.drain_events()
    assert device.pop_record_control() == {
        "manual_reward": True,
        "restart_episode": False,
        "stop_episode": False,
    }
    assert device.pop_record_control() == {
        "manual_reward": False,
        "restart_episode": False,
        "stop_episode": False,
    }


def test_record_control_reset_and_ignores_non_press_events() -> None:
    device = DummyDevice()

    device.handle_event(DeviceEvent(event_type="keyup", code="Enter"))
    device.handle_event(DeviceEvent(event_type="keydown", code="Enter", repeat=True))
    assert device.pop_record_control() == {
        "manual_reward": False,
        "restart_episode": False,
        "stop_episode": False,
    }

    device.handle_event(DeviceEvent(event_type="keydown", code="Backspace"))
    device.reset()
    assert device.pop_record_control() == {
        "manual_reward": False,
        "restart_episode": False,
        "stop_episode": False,
    }


def test_manual_step_overrides_support_global_env_masks() -> None:
    env = object.__new__(BaseEnv)
    result = {
        "next.reward": torch.as_tensor([0.0, 0.0], dtype=torch.float32),
        "next.success": torch.as_tensor([False, False], dtype=torch.bool),
        "next.terminated": torch.as_tensor([False, False], dtype=torch.bool),
        "next.truncated": torch.as_tensor([False, False], dtype=torch.bool),
    }

    updated = env._apply_manual_step_overrides(
        result,
        env_ids=np.asarray([1, 3], dtype=np.int64),
        manual_reward=np.asarray([False, True, False, False], dtype=bool),
        force_truncated=np.asarray([False, False, False, True], dtype=bool),
    )

    assert updated["next.reward"].tolist() == [1.0, 0.0]
    assert updated["next.success"].tolist() == [True, False]
    assert updated["next.terminated"].tolist() == [True, False]
    assert updated["next.truncated"].tolist() == [False, True]
