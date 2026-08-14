#!/usr/bin/env python3
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

"""Print configured Piper joint and gripper feedback as Hydra initial actions."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml
from pyAgxArm import AgxArmFactory, create_agx_arm_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=("left", "right"), default=["left", "right"])
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for action feedback")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[3] / "src/verl_vla/workflows/config/env/simulator/piper.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    arms = {arm["name"]: arm for arm in config["arms"]}
    missing = set(args.arms) - set(arms)
    if missing:
        raise ValueError(f"Hands are not configured in {args.config}: {sorted(missing)}")

    drivers = []
    try:
        for hand in args.arms:
            arm = arms[hand]
            driver_config = create_agx_arm_config(
                robot=arm["model"],
                comm="can",
                channel=arm["can_channel"],
                firmeware_version=arm["firmware_version"],
            )
            driver = AgxArmFactory.create_arm(driver_config)
            driver.connect()
            gripper = driver.init_effector(driver.OPTIONS.EFFECTOR.AGX_GRIPPER)
            drivers.append((hand, driver, gripper))

        deadline = time.monotonic() + args.timeout
        feedback = {}
        while len(feedback) != len(drivers) and time.monotonic() < deadline:
            for hand, driver, gripper in drivers:
                joints = driver.get_joint_angles()
                gripper_status = gripper.get_gripper_status()
                if joints is not None and joints.hz > 0 and gripper_status is not None and gripper_status.hz > 0:
                    feedback[hand] = [*joints.msg, gripper_status.msg.value]
            time.sleep(0.01)
        if len(feedback) != len(drivers):
            missing_feedback = sorted(set(args.arms) - set(feedback))
            raise TimeoutError(f"Timed out waiting for action feedback from {missing_feedback}")

        for hand in args.arms:
            values = ", ".join(f"{angle:.8f}" for angle in feedback[hand])
            print(f"{hand}: [{values}]")
        return 0
    finally:
        for _, driver, _ in reversed(drivers):
            driver.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
