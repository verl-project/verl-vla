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

import numpy as np

from verl_vla.envs.piper.ik import PiperIKSolver


def test_piper_ik_forward_kinematics_and_joint_limits(monkeypatch) -> None:
    class _Placement:
        homogeneous = np.eye(4)

    class _Data:
        oMf = [_Placement()]

    class _Model:
        nq = 6
        lowerPositionLimit = np.full(6, -1.0)
        upperPositionLimit = np.full(6, 1.0)

    solver = object.__new__(PiperIKSolver)
    solver.robot = type("Robot", (), {"model": _Model(), "data": _Data()})()
    solver.frame_id = 0
    solver.lower_limits = np.full(6, -1.0)
    solver.upper_limits = np.full(6, 1.0)

    monkeypatch.setattr("verl_vla.envs.piper.ik.pin.framesForwardKinematics", lambda *args: None)
    np.testing.assert_array_equal(solver.forward_kinematics(np.zeros(6)), np.eye(4))
    np.testing.assert_array_equal(solver.clip_to_limits(np.full(6, 2.0)), np.ones(6))
