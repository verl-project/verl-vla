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

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from verl_vla.envs.piper.backend import PiperBackend, _ArmRuntime


class _IK:
    lower_limits = np.full(6, -2.0)
    upper_limits = np.full(6, 2.0)

    def sync_state(self, joints):
        self.seed = np.asarray(joints)

    def solve(self, pose):
        return np.concatenate([pose[:3, 3], np.zeros(3)])

    def forward_kinematics(self, joints):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(joints)[:3]
        return pose

    def clip_to_limits(self, joints):
        return np.clip(joints, self.lower_limits, self.upper_limits)


class _LossyIK(_IK):
    def solve(self, pose):
        return np.concatenate([pose[:3, 3] * 0.5, np.zeros(3)])


class _Driver:
    def __init__(self):
        self.commands = []

    def move_js(self, joints):
        self.command = np.asarray(joints)
        self.commands.append(self.command)

    def get_tcp_pose(self):
        return SimpleNamespace(msg=[0.0] * 6)


class _Gripper:
    def __init__(self):
        self.commands = []

    def move_gripper_m(self, *, value, force):
        self.command = (value, force)
        self.commands.append(self.command)


def _backend() -> PiperBackend:
    backend = object.__new__(PiperBackend)
    backend.cfg = SimpleNamespace(
        gripper_close_width=0.0,
        gripper_open_width=0.1,
        gripper_force=1.0,
    )
    backend._lock = threading.RLock()
    backend._arm_names = ("left",)
    backend._arms = {
        "left": _ArmRuntime(
            driver=_Driver(),
            gripper=_Gripper(),
            ik=_IK(),
            target_joints=np.zeros(6),
            target_pose=np.eye(4),
            target_gripper=0.04,
        )
    }
    backend._read_joint_actions = lambda: np.asarray([[0.0] * 6 + [0.04]])
    return backend


def test_task_delta_is_converted_before_the_environment_action_boundary() -> None:
    backend = _backend()

    action = backend.task_delta_to_action(np.asarray([0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 0.005]))

    np.testing.assert_allclose(action, [0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 0.045])


def test_zero_task_delta_holds_the_last_absolute_joint_target() -> None:
    backend = _backend()

    action = backend.task_delta_to_action(np.zeros(7))

    np.testing.assert_allclose(action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04])


def test_task_ik_uses_live_joint_feedback_as_its_seed() -> None:
    backend = _backend()
    backend._arms["left"].target_joints = np.ones(6)

    backend.task_delta_to_action(np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(backend._arms["left"].ik.seed, np.zeros(6))


def test_task_target_preserves_unsolved_ik_residual_across_joint_execution() -> None:
    backend = _backend()
    backend._arms["left"].ik = _LossyIK()

    first = backend.task_delta_to_action(np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    backend.apply_action(first)
    second = backend.task_delta_to_action(np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(second[0], 0.01)
    np.testing.assert_allclose(backend._arms["left"].target_pose[0, 3], 0.02)


def test_joint_action_is_sent_directly_with_hardware_limits() -> None:
    backend = _backend()

    executed = backend.apply_action(np.asarray([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.2]))

    np.testing.assert_allclose(executed, [1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.1])
    np.testing.assert_allclose(backend._arms["left"].driver.command, executed[:6])
    assert backend._arms["left"].gripper.command == (0.1, 1.0)


def test_unchanged_joint_action_is_not_repeated() -> None:
    backend = _backend()

    backend.apply_action(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04]))

    assert backend._arms["left"].driver.commands == []
    assert backend._arms["left"].gripper.commands == []
