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


import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from verl_vla.utils.envs.action import put_info_on_image, save_rollout_video, to_tensor
from verl_vla.utils.image import preprocess_image_batch_to_uint8

from .ipc_channel import _ipc_paths, clear_ipc, send_obj

logger = logging.getLogger(__name__)
_LEROBOT_RUNTIME_SESSION_PREFIX = "lerobot_runtime"
_LEROBOT_IMAGE_CROP_SIZE = 480
_LEROBOT_IMAGE_RESIZE_SIZE = (224, 224)


class LeRobotEnv(gym.Env):
    def __init__(self, cfg, rank, world_size, stage_id: int = 0):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.stage_id = stage_id
        self.num_envs = cfg.num_envs
        assert self.num_envs == 1, f"LeRobotEnv only supports a single real-world environment, got {self.num_envs}"
        self.max_episode_steps = int(cfg.max_episode_steps)
        self.action_dim = int(getattr(cfg, "action_dim", 7))
        self.state_dim = int(getattr(cfg, "state_dim", 8))
        init_params = getattr(cfg, "init_params", None)
        self.image_height = int(getattr(init_params, "camera_heights", 256) if init_params is not None else 256)
        self.image_width = int(getattr(init_params, "camera_widths", 256) if init_params is not None else 256)
        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = []
        self._runtime_session_name = (
            f"{_LEROBOT_RUNTIME_SESSION_PREFIX}_{socket.gethostname()}_rank{self.rank}_stage{self.stage_id}"
        )

        self._episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._state_ids = np.zeros(self.num_envs, dtype=np.int64)
        self._task_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.task_descriptions = [self._task_description(0) for _ in range(self.num_envs)]
        self.total_num_group_envs = max(self.num_envs * self.world_size * 32, 1024)

        lerobot_config_path = getattr(cfg, "lerobot_config_path", None)
        if not lerobot_config_path:
            raise ValueError("`cfg.lerobot_config_path` is required for LeRobotEnv.")
        if not os.path.exists(lerobot_config_path):
            raise FileNotFoundError(f"LeRobot config not found: {lerobot_config_path}")
        self._lerobot_config_path = str(lerobot_config_path)
        self._runtime_restart_retry_sec = float(getattr(cfg, "runtime_restart_retry_sec", 1.0))
        self._runtime_ipc_retry_sec = float(getattr(cfg, "runtime_ipc_retry_sec", 0.5))
        self._runtime_boot_timeout_sec = float(getattr(cfg, "runtime_boot_timeout_sec", 10.0))
        self._runtime_default_ipc_timeout_sec = float(getattr(cfg, "runtime_default_ipc_timeout_sec", 10.0))
        self._runtime_log_path = self._resolve_runtime_log_path()
        self._ensure_tmux_lerobot_runtime(lerobot_config_path)

    def _ensure_tmux_lerobot_runtime(self, lerobot_config_path: str) -> None:
        if shutil.which("tmux") is None:
            raise RuntimeError("`tmux` is required for LeRobotEnv runtime process.")

        runtime_cmd = (
            f"exec {sys.executable} -u -m verl_vla.envs.lerobot.lerobot_runtime "
            f"--config_path {shlex.quote(str(lerobot_config_path))} "
            f"--rank {self.rank} --stage_id {self.stage_id} "
            f"--owner_pid {os.getpid()}"
        )

        while True:
            has_session = self._tmux_has_session()
            if has_session.returncode != 0:
                clear_ipc(rank=self.rank, stage_id=self.stage_id)
                subprocess.run(
                    ["tmux", "new-session", "-d", "-s", self._runtime_session_name],
                    check=True,
                )
                self._enable_tmux_log_forwarding()
                subprocess.run(
                    ["tmux", "send-keys", "-t", self._runtime_session_name, runtime_cmd, "C-m"],
                    check=True,
                )
                logger.info(
                    "Started LeRobot runtime in tmux session: %s (runtime log file: %s)",
                    self._runtime_session_name,
                    self._runtime_log_path,
                )
            else:
                self._enable_tmux_log_forwarding()

            if self._wait_runtime_ready(timeout_s=self._runtime_boot_timeout_sec):
                return

            logger.warning(
                "LeRobot runtime failed to become ready in %.1fs, restarting session: %s",
                self._runtime_boot_timeout_sec,
                self._runtime_session_name,
            )
            if self._tmux_has_session().returncode == 0:
                subprocess.run(
                    ["tmux", "kill-session", "-t", self._runtime_session_name],
                    check=False,
                )
            clear_ipc(rank=self.rank, stage_id=self.stage_id)
            time.sleep(self._runtime_restart_retry_sec)

    def _wait_runtime_ready(self, timeout_s: float) -> bool:
        req_path, resp_path = _ipc_paths(rank=self.rank, stage_id=self.stage_id)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._tmux_has_session().returncode != 0:
                return False
            time.sleep(0.1)
        return os.path.exists(req_path) and os.path.exists(resp_path)

    def _tmux_has_session(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["tmux", "has-session", "-t", self._runtime_session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _restart_tmux_lerobot_runtime(self) -> None:
        logger.warning("Restarting LeRobot runtime tmux session: %s", self._runtime_session_name)

        if shutil.which("tmux") is None:
            raise RuntimeError("`tmux` is required for LeRobotEnv runtime process.")

        if self._tmux_has_session().returncode == 0:
            subprocess.run(
                ["tmux", "kill-session", "-t", self._runtime_session_name],
                check=False,
            )
        self._ensure_tmux_lerobot_runtime(self._lerobot_config_path)

        logger.info("Restarted LeRobot runtime tmux session: %s", self._runtime_session_name)

    def _send_obj_with_runtime_recovery(
        self,
        msg_type: str,
        content: dict,
        timeout_s: Optional[float] = None,
    ):
        while True:
            kwargs = {
                "type": msg_type,
                "content": content,
                "rank": self.rank,
                "stage_id": self.stage_id,
                "timeout_s": timeout_s if timeout_s is not None else self._runtime_default_ipc_timeout_sec,
            }
            try:
                reply = send_obj(**kwargs)
                return reply
            except TimeoutError:
                logger.exception(
                    "send_obj timeout for runtime rank=%s stage=%s type=%s, restarting and retrying",
                    self.rank,
                    self.stage_id,
                    msg_type,
                )
            except Exception:
                logger.exception(
                    "send_obj failed for runtime rank=%s stage=%s type=%s, restarting and retrying",
                    self.rank,
                    self.stage_id,
                    msg_type,
                )

            self._restart_tmux_lerobot_runtime()
            time.sleep(self._runtime_ipc_retry_sec)

    def _enable_tmux_log_forwarding(self) -> None:
        Path(self._runtime_log_path).parent.mkdir(parents=True, exist_ok=True)
        prefix = f"[LeRobot runtime rank={self.rank} stage={self.stage_id}] "
        pipe_cmd = (
            f"awk -v p={shlex.quote(prefix)} '{{print p $0; fflush();}}' >> {shlex.quote(self._runtime_log_path)}"
        )
        subprocess.run(["tmux", "pipe-pane", "-t", self._runtime_session_name], check=True)
        subprocess.run(
            ["tmux", "pipe-pane", "-t", self._runtime_session_name, pipe_cmd],
            check=True,
        )

    def _resolve_runtime_log_path(self) -> str:
        cwd_dir = Path.cwd()
        cwd_dir.mkdir(parents=True, exist_ok=True)
        return str(cwd_dir / f"lerobot_runtime_rank{self.rank}_stage{self.stage_id}.log")

    def _task_description(self, task_id: int) -> str:
        del task_id
        return "Grab the white objects and put them in the plate."

    def _wrap_runtime_obs(self, runtime_obs: dict) -> dict:
        top_images = preprocess_image_batch_to_uint8(
            runtime_obs["observation.images.top"],
            crop_size=_LEROBOT_IMAGE_CROP_SIZE,
            resize_size=_LEROBOT_IMAGE_RESIZE_SIZE,
        )
        wrist_images = preprocess_image_batch_to_uint8(
            runtime_obs["observation.images.wrist"],
            crop_size=_LEROBOT_IMAGE_CROP_SIZE,
            resize_size=_LEROBOT_IMAGE_RESIZE_SIZE,
        )
        return {
            "images_and_states": to_tensor(
                {
                    "observation.images.top": top_images,
                    "observation.images.wrist": wrist_images,
                    "observation.state": runtime_obs["observation.state"],
                }
            ),
            "task_descriptions": list(self.task_descriptions),
            "task_id": self._task_ids.astype(np.int64, copy=False),
        }

    def add_new_frames(self, obs, plot_infos):
        info_item = {k: v if np.size(v) == 1 else v[0] for k, v in plot_infos.items()}
        top_tensor = obs["images_and_states"]["observation.images.top"][0].detach().cpu()
        wrist_tensor = obs["images_and_states"]["observation.images.wrist"][0].detach().cpu()
        if top_tensor.dtype in (torch.uint8, torch.int8):
            top_image = top_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        else:
            top_image = (top_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        if wrist_tensor.dtype in (torch.uint8, torch.int8):
            wrist_image = wrist_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        else:
            wrist_image = (wrist_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        top_image = put_info_on_image(top_image, info_item)
        wrist_image = put_info_on_image(wrist_image, info_item)
        self.render_images.append(np.concatenate([top_image, wrist_image], axis=1))

    def env_benchmark_size(self):
        return int(self.total_num_group_envs)

    def reset_envs_to_state_ids(self, state_ids_list, task_ids_list):
        self._state_ids = np.asarray(state_ids_list, dtype=np.int64)
        self._task_ids = np.asarray(task_ids_list, dtype=np.int64)
        self._episode_steps[:] = 0
        self._episode_returns[:] = 0.0
        self.task_descriptions = [self._task_description(task_id) for task_id in self._task_ids]
        obs = self._send_obj_with_runtime_recovery(
            msg_type="reset",
            content={
                "state_ids": self._state_ids.tolist(),
                "task_ids": self._task_ids.tolist(),
            },
        )
        obs = self._wrap_runtime_obs(obs)
        return obs, {}

    def step(self, action, critic_values=None):
        reply = self._send_obj_with_runtime_recovery(
            msg_type="step",
            content={
                "actions": action.tolist(),
            },
            timeout_s=10.0,
        )

        if not isinstance(reply, dict):
            raise RuntimeError(f"Invalid runtime reply: {reply}")
        obs = self._wrap_runtime_obs(reply["obs"])
        reward = to_tensor(np.asarray([reply["reward"]], dtype=np.float32))
        is_intervention = reply["extra_info"]["is_intervention"]
        executed_action = reply["extra_info"]["executed_action"]

        self._episode_steps += 1
        self._episode_returns += reward.numpy()

        terminations = to_tensor(np.asarray([reply["terminated"]], dtype=bool))
        raw_truncations = np.asarray([reply["truncated"]], dtype=bool)
        raw_truncations = np.logical_or(raw_truncations, self._episode_steps >= self.max_episode_steps)
        truncations = to_tensor(raw_truncations)
        successes = to_tensor(np.asarray([reply["success"]], dtype=bool))

        if self.video_cfg.save_video:
            plot_infos = {
                "rewards": reward.numpy(),
                "terminations": terminations.numpy(),
                "truncations": truncations.numpy(),
                "is_intervention": is_intervention,
                "task": self.task_descriptions,
            }
            if critic_values is not None:
                plot_infos["critic_value"] = np.asarray(critic_values, dtype=np.float32)
            self.add_new_frames(obs, plot_infos)

        infos = {}
        infos["is_intervention"] = is_intervention
        infos["executed_action"] = executed_action
        done_mask = torch.logical_or(terminations, truncations)
        if done_mask.any():
            infos["final_info"] = {
                "episode": {
                    "return": torch.as_tensor(self._episode_returns, dtype=torch.float32),
                    "episode_len": torch.as_tensor(self._episode_steps, dtype=torch.int32),
                    "success_once": terminations.clone(),
                }
            }
            self._episode_steps[done_mask.numpy()] = 0
            self._episode_returns[done_mask.numpy()] = 0.0

        return obs, reward, terminations, truncations, successes, infos

    def chunk_step(self, chunk_actions, chunk_values=None):
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        raw_chunk_successes = []
        intervention_info = {"obs": [], "action": [], "is_intervention": []}
        last_step_is_intervention = False
        last_executed_action = None

        step_idx = 0
        while step_idx < chunk_size or last_step_is_intervention:
            extracted_obs, step_reward, terminations, truncations, successes, infos = self.step(
                chunk_actions[:, step_idx, :] if step_idx < chunk_size else last_executed_action,
                critic_values=None if last_step_is_intervention else chunk_values,
            )

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)
            raw_chunk_successes.append(successes)
            executed_action = to_tensor(infos["executed_action"]).unsqueeze(0)
            intervention_info["action"].append(executed_action)
            intervention_info["is_intervention"].append(to_tensor(infos["is_intervention"]).unsqueeze(0))
            last_executed_action = executed_action
            if (step_idx + 1) % chunk_size == 0:
                intervention_info["obs"].append(extracted_obs)

            last_step_is_intervention = infos["is_intervention"]
            step_idx += 1

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)
        chunk_successes = torch.stack(raw_chunk_successes, dim=1)

        infos = {}
        intervention_obs = {
            f"obs.{key}": torch.stack(
                [step_obs["images_and_states"][key] for step_obs in intervention_info["obs"]], dim=1
            )
            for key in intervention_info["obs"][0]["images_and_states"]
        }

        is_intervention_tensor = torch.stack(intervention_info["is_intervention"], dim=1)
        if is_intervention_tensor.any():
            infos["intervention_info"] = {
                **intervention_obs,
                "obs.task_descriptions": np.array(
                    [step_obs["task_descriptions"] for step_obs in intervention_info["obs"]],
                    dtype=object,
                ).transpose(1, 0),
                "action": torch.stack(intervention_info["action"], dim=1),
                "is_intervention": is_intervention_tensor,
            }
        return extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, chunk_successes, infos

    def flush_video(self, video_sub_dir: Optional[str] = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"rank_{self.rank}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        save_rollout_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
        )
        self.video_cnt += 1
        self.render_images = []

    def load_state(self, state_buffer: bytes):
        del state_buffer
        logger.debug("LeRobotEnv.load_state is a no-op scaffold")

    def close(self):
        if shutil.which("tmux") is not None and self._tmux_has_session().returncode == 0:
            subprocess.run(
                ["tmux", "kill-session", "-t", self._runtime_session_name],
                check=False,
            )
            logger.info("Stopped LeRobot runtime tmux session: %s", self._runtime_session_name)
        clear_ipc(rank=self.rank, stage_id=self.stage_id)
        return None
