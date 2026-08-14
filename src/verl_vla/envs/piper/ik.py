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

"""Pinocchio/CasADi inverse kinematics for Piper task-space teleoperation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from scipy.spatial.transform import Rotation


def _xyzrpy_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    matrix[:3, 3] = xyz
    return matrix


class PiperIKSolver:
    """Six-axis Piper FK/IK solver for incremental teleoperation."""

    def __init__(
        self,
        *,
        urdf_path: str | Path,
        locked_joints: Sequence[str],
        ee_parent_joint: str,
        tool_translation_xyz: Sequence[float],
        tool_pre_rotation_rpy: Sequence[float],
        position_weight: float,
        orientation_weight: float,
        regularization_weight: float,
        smooth_weight: float,
        max_iterations: int,
        tolerance: float,
    ) -> None:
        urdf_path = Path(urdf_path).expanduser().resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Piper URDF does not exist: {urdf_path}")

        robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), package_dirs=[str(urdf_path.parents[4])])
        locked = [name for name in dict.fromkeys(locked_joints) if robot.model.getJointId(name) > 0]
        self.robot = robot.buildReducedRobot(locked, np.zeros(robot.model.nq))
        if self.robot.model.nq != 6:
            raise ValueError(f"Piper IK expects six active joints, URDF produced {self.robot.model.nq}")

        tool = _xyzrpy_matrix((0.0, 0.0, 0.0), tool_pre_rotation_rpy) @ _xyzrpy_matrix(
            tool_translation_xyz, (0.0, 0.0, 0.0)
        )
        quaternion = Rotation.from_matrix(tool[:3, :3]).as_quat()
        frame_name = "verl_vla_piper_tcp"
        self.robot.model.addFrame(
            pin.Frame(
                frame_name,
                self.robot.model.getJointId(ee_parent_joint),
                pin.SE3(
                    pin.Quaternion(quaternion[3], quaternion[0], quaternion[1], quaternion[2]),
                    tool[:3, 3],
                ),
                pin.FrameType.OP_FRAME,
            )
        )
        self.robot.data = self.robot.model.createData()
        self.frame_id = self.robot.model.getFrameId(frame_name)
        self.lower_limits = np.asarray(self.robot.model.lowerPositionLimit, dtype=float)
        self.upper_limits = np.asarray(self.robot.model.upperPositionLimit, dtype=float)

        model = cpin.Model(self.robot.model)
        data = model.createData()
        q = casadi.SX.sym("q", model.nq, 1)
        target = casadi.SX.sym("target", 4, 4)
        cpin.framesForwardKinematics(model, data, q)
        error = casadi.Function(
            "piper_pose_error",
            [q, target],
            [casadi.vertcat(cpin.log6(data.oMf[self.frame_id].inverse() * cpin.SE3(target)).vector)],
        )

        self._opti = casadi.Opti()
        self._q = self._opti.variable(model.nq)
        self._previous_q = self._opti.parameter(model.nq)
        self._target = self._opti.parameter(4, 4)
        pose_error = error(self._q, self._target)
        cost = (
            casadi.sumsqr(position_weight * pose_error[:3])
            + casadi.sumsqr(orientation_weight * pose_error[3:])
            + regularization_weight * casadi.sumsqr(self._q)
            + smooth_weight * casadi.sumsqr(self._q - self._previous_q)
        )
        self._opti.minimize(cost)
        self._opti.subject_to(self._opti.bounded(self.lower_limits, self._q, self.upper_limits))
        self._opti.solver(
            "ipopt",
            {
                "ipopt": {"print_level": 0, "max_iter": max_iterations, "tol": tolerance},
                "print_time": False,
            },
        )
        self._seed = np.zeros(model.nq, dtype=float)

    def sync_state(self, joint_positions: np.ndarray) -> None:
        joints = np.asarray(joint_positions, dtype=float)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError(f"Piper IK state must contain six finite joint positions, got {joints}")
        self._seed = joints.copy()

    def solve(self, target_pose: np.ndarray) -> np.ndarray:
        target_pose = np.asarray(target_pose, dtype=float)
        if target_pose.shape != (4, 4) or not np.all(np.isfinite(target_pose)):
            raise ValueError("Piper IK target pose must be a finite 4x4 matrix")
        self._opti.set_initial(self._q, self._seed)
        self._opti.set_value(self._previous_q, self._seed)
        self._opti.set_value(self._target, target_pose)
        self._opti.solve_limited()
        solution = np.asarray(self._opti.value(self._q), dtype=float).reshape(6)
        self._seed = solution
        return solution

    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        joints = np.asarray(joint_positions, dtype=float)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError(f"Piper FK state must contain six finite joint positions, got {joints}")
        pin.framesForwardKinematics(self.robot.model, self.robot.data, joints)
        return np.asarray(self.robot.data.oMf[self.frame_id].homogeneous, dtype=float).copy()

    def clip_to_limits(self, joint_positions: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(joint_positions, dtype=float), self.lower_limits, self.upper_limits)


__all__ = ["PiperIKSolver"]
