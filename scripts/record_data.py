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
"""Record demonstration data with gamepad controller."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from verl_vla.recorder.config import (
    LeRobotRecorderConfig,
    RecorderConfig,
    VideoRecorderConfig,
)
from verl_vla.teleop.config import (
    GamepadTeleopConfig,
    KeyboardTeleopConfig,
    TeleopConfig,
    TeleopServerConfig,
)
from verl_vla.utils.recorder import merge_lerobot_datasets


def create_libero_config(args):
    from verl_vla.envs.libero_env.config import LiberoSimulatorConfig

    return LiberoSimulatorConfig(
        simulator_type="libero",
        task_suite_name=args.task_suite,
        max_episode_steps=args.max_steps,
        seed=args.seed,
        reset_warmup_steps=10,
        camera_depths=False,
        camera_heights=args.image_height,
        camera_widths=args.image_width,
        camera_names=args.camera_names,
    )


def create_env_config(simulator_type, simulator_config, teleop_config, recorder_config):
    return OmegaConf.create(
        {
            "num_envs": 1,
            "async_reset": False,
            "simulator": {
                "simulator_type": simulator_type,
                simulator_type: simulator_config,
            },
            "recorder": recorder_config,
            "teleop": teleop_config,
        }
    )


def create_env(simulator_type, cfg, rank=0, world_size=1, stage_id=0, task_id=0):
    if simulator_type == "libero":
        from verl_vla.envs.libero_env.libero_env import LiberoEnv

        return LiberoEnv(cfg=cfg, rank=rank, world_size=world_size, stage_id=stage_id)
    elif simulator_type == "isaac":
        os.environ["LIBERO_TASK_SUITE"] = cfg.simulator.libero.task_suite_name
        os.environ["LIBERO_TASK_ID"] = str(task_id)
        os.environ["LIBERO_OSC_TYPE"] = "pose_rel"
        from verl_vla.envs.isaac_env.isaac_env import IsaacEnv

        return IsaacEnv(cfg=cfg, rank=rank, world_size=world_size)
    elif simulator_type == "lerobot":
        from verl_vla.envs.lerobot_env.lerobot_env import LeRobotEnv

        return LeRobotEnv(cfg=cfg, rank=rank, world_size=world_size, stage_id=stage_id)
    else:
        raise ValueError(f"Unsupported simulator type: {simulator_type}")


def custom_reset_libero(env, task_id, trial_id):
    reset_states_for_task = env.reset_planner.reset_states_by_task[task_id]
    reset_state_row = reset_states_for_task[trial_id]
    reset_state_id = int(reset_state_row[2])

    reset_states = np.array([[task_id, trial_id, reset_state_id, -1]], dtype=np.int64)
    env._reset_to_states(reset_states, np.array([0]))

    reset_warmup_steps = int(env.libero_cfg.reset_warmup_steps)
    zero_actions = np.zeros((1, 7))
    for _ in range(reset_warmup_steps):
        raw_obs, _reward, _terminations, _info_lists = env.env.step(zero_actions)

    obs = {
        "observation": env._make_observations(raw_obs),
        "task": [env.task_descriptions[0]],
        "task_id": env.task_ids[np.array([0])].astype(np.int64, copy=False),
        "eval_episode_id": env.eval_episode_ids[np.array([0])].astype(np.int64, copy=False),
    }
    env._reset_elapsed_steps(np.array([0]))
    return obs


def custom_reset_isaac(env, task_id, trial_id):
    env._reset_metrics(np.array([0]))
    raw_obs, infos = env.env.reset()
    obs = env._wrap_obs(raw_obs)
    return {
        "observation": obs,
        "task": [env.task_descriptions[0]],
        "task_id": np.array([task_id], dtype=np.int64),
        "eval_episode_id": np.array([-1], dtype=np.int64),
    }


def custom_reset_lerobot(env, task_id, trial_id):
    state_id = trial_id
    obs, _ = env.reset_envs_to_state_ids([state_id], [task_id])
    return {
        "observation": obs,
        "task": [env.task_descriptions[0]],
        "task_id": np.array([task_id], dtype=np.int64),
        "eval_episode_id": np.array([-1], dtype=np.int64),
    }


def main():
    parser = argparse.ArgumentParser(description="Record demonstration data with gamepad controller")
    parser.add_argument(
        "--simulator-type",
        type=str,
        default="libero",
        choices=["libero", "isaac", "lerobot"],
        help="Simulator type",
    )
    parser.add_argument("--task-suite", type=str, default="libero_spatial", help="LIBERO task suite name")
    parser.add_argument("--task-name", type=str, default=None, help="Specific task name")
    parser.add_argument("--task-id", type=int, default=0, help="Fixed task index within the suite")
    parser.add_argument("--trial-id", type=int, default=None, help="Fixed trial id for the task")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to record")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--output-dir", type=str, default="/tmp/verl_vla_lerobot_records", help="Output directory")
    parser.add_argument("--repo-id", type=str, default="local/verl_vla_libero", help="Dataset repo ID")
    parser.add_argument(
        "--camera-names",
        type=str,
        nargs="+",
        default=["agentview", "robot0_eye_in_hand"],
        help="Camera names",
    )
    parser.add_argument("--image-height", type=int, default=256, help="Image height")
    parser.add_argument("--image-width", type=int, default=256, help="Image width")
    parser.add_argument(
        "--device",
        type=str,
        default="gamepad",
        choices=["gamepad", "keyboard", "xr_controller"],
        help="Input device type",
    )
    parser.add_argument("--server-port", type=int, default=18000, help="Teleop server base port")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    teleop_server_config = TeleopServerConfig(
        host="0.0.0.0",
        base_port=args.server_port,
        rank_stride=10000,
        stage_stride=1000,
        jpeg_quality=100,
        log_level="warning",
    )

    gamepad_config = GamepadTeleopConfig(
        pos_sensitivity=0.5,
        rot_sensitivity=1.0,
        intervention_button="RT",
        gripper_button="X",
        button_threshold=0.5,
        left_stick_x_axis="axis_0",
        left_stick_y_axis="axis_1",
        right_stick_y_axis="axis_3",
        right_stick_x_axis="axis_2",
        dpad_up_button="DUp",
        dpad_down_button="DDown",
        dpad_left_button="DLeft",
        dpad_right_button="DRight",
    )

    keyboard_config = KeyboardTeleopConfig(
        pos_sensitivity=0.5,
        rot_sensitivity=0.12,
    )

    teleop_config = TeleopConfig(
        enable=True,
        devices=(args.device,),
        server=teleop_server_config,
        gamepad=gamepad_config,
        keyboard=keyboard_config,
    )

    lerobot_config = LeRobotRecorderConfig(
        enable=True,
        root=args.output_dir,
        repo_id=args.repo_id,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
        batch_encoding_size=1,
        vcodec="libsvtav1",
        video_files_size_in_mb=1e-6,
    )

    video_config = VideoRecorderConfig(
        enable=True,
        root="/tmp/verl_vla_recorder_videos",
        fps=30,
        font_size=14,
    )

    recorder_config = RecorderConfig(
        enable=True,
        async_enable=False,
        recorders=("lerobot", "video"),
        lerobot=lerobot_config,
        video=video_config,
    )

    if args.simulator_type == "libero":
        simulator_config = create_libero_config(args)
    elif args.simulator_type == "isaac":
        simulator_config = create_libero_config(args)
    elif args.simulator_type == "lerobot":
        simulator_config = OmegaConf.create(
            {
                "num_envs": 1,
                "max_episode_steps": args.max_steps,
                "action_dim": 7,
                "state_dim": 8,
                "lerobot_config_path": args.task_suite,
                "init_params": {
                    "camera_heights": args.image_height,
                    "camera_widths": args.image_width,
                },
            }
        )
    else:
        raise ValueError(f"Unsupported simulator type: {args.simulator_type}")

    env_cfg = create_env_config(args.simulator_type, simulator_config, teleop_config, recorder_config)

    env = create_env(args.simulator_type, env_cfg, rank=0, world_size=1, stage_id=0, task_id=args.task_id)
    time.sleep(2)

    if args.simulator_type == "libero":
        reset_states_for_task = env.reset_planner.reset_states_by_task[args.task_id]
        num_trials_task0 = len(reset_states_for_task)
    elif args.simulator_type == "isaac":
        num_trials_task0 = 100
    elif args.simulator_type == "lerobot":
        num_trials_task0 = 100
    else:
        num_trials_task0 = 10

    success_count = 0
    for ep_idx in range(args.num_episodes):
        print(f"\nEpisode {ep_idx + 1}/{args.num_episodes}")
        print("-" * 40)

        if args.trial_id is not None:
            trial_id = args.trial_id
        else:
            trial_id = np.random.randint(0, num_trials_task0)

        if args.simulator_type == "libero":
            obs = custom_reset_libero(env, args.task_id, trial_id)
        elif args.simulator_type == "isaac":
            obs = custom_reset_isaac(env, args.task_id, trial_id)
        elif args.simulator_type == "lerobot":
            obs = custom_reset_lerobot(env, args.task_id, trial_id)
        else:
            obs = env.env_reset(env_ids=np.array([0]))

        env._latest_obs = obs

        current_task_id = args.task_id
        current_trial_id = trial_id

        if isinstance(obs, dict) and "task" in obs:
            print(f"Task: {obs['task'][0]}")
            print(f"  task_id: {current_task_id}, trial_id: {current_trial_id}")

        done = False
        episode_start = time.time()

        while not done:
            zero_action = np.zeros((1, 1, 7), dtype=np.float32)

            obs, rewards, terminateds, truncateds, success = env.step(zero_action)

            done = bool(terminateds[0, 0]) or bool(truncateds[0, 0])

            if done:
                step_count = env._elapsed_steps[0]
                if bool(terminateds[0, 0]):
                    success_count += 1
                    print(f"  Success! (step {step_count})")
                else:
                    print(f"  Truncated at step {step_count}")

        episode_time = time.time() - episode_start
        print(f"  Steps: {step_count}")
        print(f"  Time: {episode_time:.1f}s")

    print()
    print("=" * 60)
    print("Recording completed!")
    print(f"Total episodes: {args.num_episodes}")
    print(f"Successful episodes: {success_count}")
    print(f"Success rate: {success_count / args.num_episodes * 100:.1f}%")
    print("=" * 60)

    env.finish_rollout()
    dataset_info = env.pop_completed_dataset()

    if dataset_info:
        print(f"\nRaw dataset saved to: {dataset_info['root']}")

        merged_output = Path(args.output_dir) / args.repo_id
        print(f"\nMerging dataset to: {merged_output}")

        merge_lerobot_datasets(
            roots=[dataset_info["root"]],
            output_root=merged_output,
            repo_id=args.repo_id,
            repo_ids=[dataset_info["repo_id"]],
            append=merged_output.exists(),
            cleanup_roots=False,
        )

        print(f"\nFinal dataset saved to: {merged_output}")
        print("Use this path as SFT_REPO_ID for training.")
    else:
        print("\nNo dataset was recorded.")

    env.close()


if __name__ == "__main__":
    main()
