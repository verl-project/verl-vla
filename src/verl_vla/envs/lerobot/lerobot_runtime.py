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

import argparse
import logging
import os
import signal
import threading
import time

import draccus
import torch
from lerobot.processor import TransitionKey
from lerobot.rl.gym_manipulator import (
    GymManipulatorConfig,
    _mirror_reset_state_to_teleop,
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.transition import Transition

from .ipc_channel import clear_ipc, recv_obj, reply_obj, setup_ipc

logger = logging.getLogger(__name__)
_STOP = False
_RUNTIME = None


def _handle_stop_signal(signum, frame):
    del frame
    global _STOP
    _STOP = True
    logger.info("Received signal %s, stopping lerobot runtime.", signum)


class LerobotRuntime:
    def __init__(self, config_path: str, rank: int, stage_id: int):
        self.rank = rank
        self.stage_id = stage_id

        self.lerobot_config = draccus.parse(
            config_class=GymManipulatorConfig,
            config_path=config_path,
            args=[],
        )
        self.interaction_step = 0
        self.online_env, self.teleop_device = make_robot_env(self.lerobot_config.env)
        self.env_processor, self.action_processor = make_processors(
            self.online_env,
            self.teleop_device,
            self.lerobot_config.env,
        )

        self.obs = None
        self.info = None
        self.transition = None
        self.sum_reward_episode = 0
        self.list_transition_to_send_to_learner = []
        self.episode_intervention = False
        self.episode_intervention_steps = 0
        self.episode_total_steps = 0

    def reset(self, task_ids=None, state_ids=None):
        self.obs, self.info = self.online_env.reset()
        self.env_processor.reset()
        self.action_processor.reset()

        self.transition = create_transition(observation=self.obs, info=self.info)
        self.transition = self.env_processor(self.transition)
        self.sum_reward_episode = 0
        self.list_transition_to_send_to_learner = []
        self.episode_intervention = False
        self.episode_intervention_steps = 0
        self.episode_total_steps = 0

        _mirror_reset_state_to_teleop(self.teleop_device, self.online_env)

        obs = self.transition[TransitionKey.OBSERVATION]
        return obs

    def step(self, actions):
        if self.transition is None:
            self.reset()

        actions = torch.as_tensor(actions, dtype=torch.float32)
        if actions.ndim > 1:
            actions = actions[0]

        new_transition = step_env_and_process_transition(
            env=self.online_env,
            transition=self.transition,
            action=actions,
            env_processor=self.env_processor,
            action_processor=self.action_processor,
            teleop_device=self.teleop_device,
            mirror_robot_action_to_teleop=True,
        )

        next_obs = new_transition[TransitionKey.OBSERVATION]
        executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]

        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)

        self.sum_reward_episode += float(reward)
        self.episode_total_steps += 1
        self.interaction_step += 1

        intervention_info = new_transition[TransitionKey.INFO]
        if intervention_info.get(TeleopEvents.IS_INTERVENTION, False):
            self.episode_intervention = True
            self.episode_intervention_steps += 1

        self.list_transition_to_send_to_learner.append(
            Transition(
                state=self.obs,
                action=executed_action,
                reward=reward,
                next_state=next_obs,
                done=done,
                truncated=truncated,
                complementary_info={},
            )
        )
        self.obs = next_obs
        self.transition = new_transition

        if done or truncated:
            intervention_rate = 0.0
            if self.episode_total_steps > 0:
                intervention_rate = self.episode_intervention_steps / self.episode_total_steps

            logger.info(
                "Global step %s: Episode reward: %s Episode steps: %s Episode intervention: %s, "
                "Intervention rate: %.2f",
                self.interaction_step,
                self.sum_reward_episode,
                self.episode_total_steps,
                self.episode_intervention,
                intervention_rate,
            )

        return {
            "obs": next_obs,
            "reward": float(reward),
            "terminated": bool(done),
            "truncated": bool(truncated),
            "success": bool(done),
            "extra_info": {
                "is_intervention": intervention_info.get(TeleopEvents.IS_INTERVENTION, False),
                "executed_action": executed_action,
            },
        }

    def close(self):
        if self.teleop_device is not None and hasattr(self.teleop_device, "disconnect"):
            self.teleop_device.disconnect()
        if self.online_env is not None and hasattr(self.online_env, "close"):
            self.online_env.close()


def _terminate_runtime_process(rank: int, stage_id: int) -> None:
    global _RUNTIME
    try:
        if _RUNTIME is not None:
            _RUNTIME.close()
    except Exception:
        logger.exception("Failed to close LeRobot runtime during forced termination")

    try:
        clear_ipc(rank=rank, stage_id=stage_id)
    except Exception:
        logger.exception("Failed to clear IPC during forced termination")

    os._exit(0)


def _watch_owner_process(owner_pid: int, rank: int, stage_id: int) -> None:
    while True:
        if _STOP:
            return

        try:
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            logger.warning(
                "Owner process %s is gone, shutting down LeRobot runtime for rank=%s stage=%s",
                owner_pid,
                rank,
                stage_id,
            )
            _terminate_runtime_process(rank=rank, stage_id=stage_id)
        except PermissionError:
            return

        time.sleep(1.0)


def start_lerobot_runtime(config_path: str, rank: int, stage_id: int, owner_pid: int | None = None) -> None:
    global _RUNTIME
    logger.info("LeRobot runtime started with config: %s", config_path)
    setup_ipc(rank=rank, stage_id=stage_id)
    runtime = LerobotRuntime(config_path=config_path, rank=rank, stage_id=stage_id)
    _RUNTIME = runtime

    if owner_pid is not None:
        threading.Thread(
            target=_watch_owner_process,
            args=(owner_pid, rank, stage_id),
            name=f"lerobot-runtime-owner-watchdog-{rank}-{stage_id}",
            daemon=True,
        ).start()

    while not _STOP:
        msg = recv_obj(rank=rank, stage_id=stage_id)
        reply = None
        if msg.get("type") == "reset":
            reply = runtime.reset(
                task_ids=msg.get("content", {}).get("task_ids"),
                state_ids=msg.get("content", {}).get("state_ids"),
            )
        elif msg.get("type") == "step":
            reply = runtime.step(actions=msg.get("content", {}).get("actions"))
        else:
            logger.warning("Received unknown message type: %s", msg.get("type"))
            reply = {"status": "error", "message": f"Unknown message type: {msg.get('type')}"}

        reply_obj(reply, rank=rank, stage_id=stage_id)

    runtime.close()
    clear_ipc(rank=rank, stage_id=stage_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--stage_id", required=True, type=int)
    parser.add_argument("--owner_pid", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)
    start_lerobot_runtime(args.config_path, args.rank, args.stage_id, owner_pid=args.owner_pid)


if __name__ == "__main__":
    main()
