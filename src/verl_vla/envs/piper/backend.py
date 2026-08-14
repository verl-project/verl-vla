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

"""Direct Piper hardware backend using pyAgxArm, Pinocchio IK, and LeRobot cameras."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import ColorMode, OpenCVCameraConfig
from pyAgxArm import AgxArmFactory, create_agx_arm_config
from scipy.spatial.transform import Rotation

from .ik import PiperIKSolver

logger = logging.getLogger(__name__)


@dataclass
class _ArmRuntime:
    driver: Any
    gripper: Any
    ik: PiperIKSolver
    target_joints: np.ndarray | None = None
    target_pose: np.ndarray | None = None
    target_gripper: float | None = None
    reset_joints: np.ndarray | None = None


class PiperBackend:
    """Own all physical Piper, IK, and camera resources for one environment."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._lock = threading.RLock()
        self._arm_names = tuple(arm.name for arm in cfg.arms)
        self._arms: dict[str, _ArmRuntime] = {}
        self._cameras = {
            camera.name: OpenCVCamera(
                OpenCVCameraConfig(
                    index_or_path=camera.device,
                    fps=camera.fps,
                    width=camera.width,
                    height=camera.height,
                    color_mode=ColorMode.RGB,
                    warmup_s=camera.warmup_s,
                    fourcc=camera.fourcc,
                )
            )
            for camera in cfg.cameras
        }

    def start(self) -> None:
        connected_cameras: list[OpenCVCamera] = []
        try:
            for arm_cfg in self.cfg.arms:
                self._arms[arm_cfg.name] = self._start_arm(arm_cfg)
            self._wait_for_feedback()
            self._sync_targets(capture_reset=True)
            for camera in self._cameras.values():
                camera.connect(warmup=True)
                connected_cameras.append(camera)
        except Exception:
            for camera in reversed(connected_cameras):
                camera.disconnect()
            self.close()
            raise

    def close(self) -> None:
        for camera in self._cameras.values():
            if camera.is_connected:
                camera.disconnect()
        for runtime in reversed(list(self._arms.values())):
            try:
                runtime.driver.disconnect()
            except Exception:
                logger.exception("Failed to disconnect a Piper arm")
        self._arms.clear()

    def read_state(self) -> np.ndarray:
        with self._lock:
            return self._read_joint_actions().reshape(-1).astype(np.float32)

    def read_images(self) -> dict[str, np.ndarray]:
        return {
            name: camera.read_latest(max_age_ms=int(self.cfg.camera_stale_timeout_s * 1000)).copy()
            for name, camera in self._cameras.items()
        }

    def read_arm_rotations(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {hand: self._read_tcp_pose(self._arms[hand])[:3, :3] for hand in self._arm_names}

    def sync_task_target(self, hand: str) -> None:
        """Anchor one task-space command target to current hardware feedback."""
        with self._lock:
            index = self._arm_names.index(hand)
            feedback = self._read_joint_actions()[index]
            runtime = self._arms[hand]
            self._sync_runtime_target(runtime, feedback, self._read_tcp_pose(runtime))

    def task_delta_to_action(self, action: np.ndarray) -> np.ndarray:
        """Convert per-arm TCP/gripper deltas into the canonical absolute joint action."""
        deltas = np.asarray(action, dtype=float).reshape(len(self._arm_names), 7)
        if not np.all(np.isfinite(deltas)):
            raise ValueError("Piper task-space delta contains NaN or Inf")
        with self._lock:
            feedback = self._read_joint_actions()
            output = np.empty_like(deltas)
            for index, (hand, delta) in enumerate(zip(self._arm_names, deltas, strict=True)):
                runtime = self._arms[hand]
                feedback_joints = feedback[index, :6]
                base_pose = (
                    runtime.target_pose.copy()
                    if runtime.target_pose is not None
                    else runtime.ik.forward_kinematics(feedback_joints)
                )
                candidate_pose = base_pose.copy()
                candidate_pose[:3, 3] += delta[:3]
                candidate_pose[:3, :3] = Rotation.from_rotvec(delta[3:6]).as_matrix() @ candidate_pose[:3, :3]
                if np.any(delta[:6] != 0.0):
                    runtime.ik.sync_state(feedback_joints)
                    target_joints = runtime.ik.solve(candidate_pose)
                else:
                    target_joints = (
                        runtime.target_joints.copy() if runtime.target_joints is not None else feedback_joints.copy()
                    )

                base_gripper = (
                    runtime.target_gripper if runtime.target_gripper is not None else float(feedback[index, 6])
                )
                target_gripper = float(
                    np.clip(
                        base_gripper + delta[6],
                        float(self.cfg.gripper_close_width),
                        float(self.cfg.gripper_open_width),
                    )
                )
                runtime.target_pose = candidate_pose
                output[index, :6] = target_joints
                output[index, 6] = target_gripper
            return output.reshape(-1).astype(np.float32)

    def apply_action(self, action: np.ndarray) -> np.ndarray:
        """Safety-filter and execute canonical absolute joint targets."""
        requested = np.asarray(action, dtype=float).reshape(len(self._arm_names), 7)
        if not np.all(np.isfinite(requested)):
            raise ValueError("Piper joint action contains NaN or Inf")
        with self._lock:
            executed = requested.copy()
            for index, hand in enumerate(self._arm_names):
                runtime = self._arms[hand]
                limited = runtime.ik.clip_to_limits(executed[index, :6])
                gripper = float(
                    np.clip(
                        executed[index, 6],
                        float(self.cfg.gripper_close_width),
                        float(self.cfg.gripper_open_width),
                    )
                )
                if runtime.target_joints is None or not np.array_equal(limited, runtime.target_joints):
                    runtime.driver.move_js(limited.tolist())
                if runtime.target_gripper is None or gripper != runtime.target_gripper:
                    runtime.gripper.move_gripper_m(value=gripper, force=float(self.cfg.gripper_force))
                runtime.target_joints = limited
                runtime.target_gripper = gripper
                executed[index, :6] = limited
                executed[index, 6] = gripper
            return executed.reshape(-1).astype(np.float32)

    def reset(self) -> None:
        with self._lock:
            feedback = self._read_joint_actions()
            targets = np.stack([self._arms[hand].reset_joints for hand in self._arm_names])
            start = time.monotonic()
            duration = float(self.cfg.reset_duration_s)
            while True:
                progress = min((time.monotonic() - start) / duration, 1.0)
                smooth = progress * progress * (3.0 - 2.0 * progress)
                positions = feedback[:, :6] + smooth * (targets - feedback[:, :6])
                for hand, joints in zip(self._arm_names, positions, strict=True):
                    self._arms[hand].driver.move_j(joints.tolist())
                if progress >= 1.0:
                    break
                time.sleep(1.0 / float(self.cfg.control_hz))

            deadline = time.monotonic() + float(self.cfg.reset_timeout_s) - duration
            while time.monotonic() < deadline:
                current = self._read_joint_actions()[:, :6]
                if np.allclose(current, targets, rtol=0.0, atol=float(self.cfg.reset_joint_tolerance)):
                    self._sync_targets(capture_reset=False)
                    return
                time.sleep(0.02)
            raise TimeoutError("Piper arms did not reach their configured reset joint targets")

    def _start_arm(self, arm_cfg: Any) -> _ArmRuntime:
        driver_config = create_agx_arm_config(
            robot=arm_cfg.model,
            comm="can",
            channel=arm_cfg.can_channel,
            firmeware_version=arm_cfg.firmware_version,
        )
        driver = AgxArmFactory.create_arm(driver_config)
        try:
            driver.connect()
            deadline = time.monotonic() + float(self.cfg.startup_timeout_s)
            while time.monotonic() < deadline:
                if driver.enable():
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError(f"Timed out enabling {arm_cfg.name} Piper on {arm_cfg.can_channel}")
            if self.cfg.speed_percent is not None:
                driver.set_speed_percent(int(self.cfg.speed_percent))
            driver.set_tcp_offset([0.0, 0.0, float(self.cfg.driver_tcp_length), 0.0, 0.0, 0.0])
            gripper = driver.init_effector(driver.OPTIONS.EFFECTOR.AGX_GRIPPER)
            return _ArmRuntime(driver=driver, gripper=gripper, ik=self._create_ik(arm_cfg))
        except Exception:
            driver.disconnect()
            raise

    def _create_ik(self, arm_cfg: Any) -> PiperIKSolver:
        urdf = (
            Path(self.cfg.urdf_root).expanduser()
            / arm_cfg.model
            / "urdf"
            / f"{arm_cfg.model}_with_gripper_description.urdf"
        )
        return PiperIKSolver(
            urdf_path=urdf,
            locked_joints=("gripper_base_joint", "gripper", "gripper_joint1", "gripper_joint2"),
            ee_parent_joint="joint6",
            tool_translation_xyz=(0.0, 0.0, float(self.cfg.tool_length)),
            tool_pre_rotation_rpy=(0.0, 0.0, 0.0),
            position_weight=float(self.cfg.ik_position_weight),
            orientation_weight=float(self.cfg.ik_orientation_weight),
            regularization_weight=float(self.cfg.ik_regularization_weight),
            smooth_weight=float(self.cfg.ik_smooth_weight),
            max_iterations=int(self.cfg.ik_max_iterations),
            tolerance=float(self.cfg.ik_tolerance),
        )

    def _wait_for_feedback(self) -> None:
        deadline = time.monotonic() + float(self.cfg.startup_timeout_s)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._read_joint_actions()
                return
            except RuntimeError as exc:
                last_error = exc
                time.sleep(0.02)
        raise TimeoutError(f"Timed out waiting for Piper feedback: {last_error}")

    def _read_joint_actions(self) -> np.ndarray:
        actions = []
        for hand in self._arm_names:
            runtime = self._arms[hand]
            joints = runtime.driver.get_joint_angles()
            gripper = runtime.gripper.get_gripper_status()
            if joints is None or gripper is None:
                raise RuntimeError(f"Feedback is not ready for {hand} Piper arm")
            actions.append(np.concatenate([np.asarray(joints.msg, dtype=float), [float(gripper.msg.value)]]))
        return np.stack(actions)

    def _sync_targets(self, *, capture_reset: bool) -> None:
        feedback = self._read_joint_actions()
        for index, hand in enumerate(self._arm_names):
            runtime = self._arms[hand]
            self._sync_runtime_target(runtime, feedback[index], self._read_tcp_pose(runtime))
            if capture_reset:
                configured = self.cfg.arms[index].initial_joint_angles
                runtime.reset_joints = (
                    feedback[index, :6].copy() if configured is None else np.asarray(configured, dtype=float)
                )

    @staticmethod
    def _sync_runtime_target(runtime: _ArmRuntime, feedback: np.ndarray, task_pose: np.ndarray) -> None:
        runtime.target_joints = feedback[:6].copy()
        runtime.target_pose = task_pose.copy()
        runtime.target_gripper = float(feedback[6])
        runtime.ik.sync_state(runtime.target_joints)

    @staticmethod
    def _read_tcp_pose(runtime: _ArmRuntime) -> np.ndarray:
        feedback = runtime.driver.get_tcp_pose()
        if feedback is None:
            raise RuntimeError("Piper TCP feedback is not ready")
        pose = np.asarray(feedback.msg, dtype=float)
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler("xyz", pose[3:6]).as_matrix()
        matrix[:3, 3] = pose[:3]
        return matrix


__all__ = ["PiperBackend"]
