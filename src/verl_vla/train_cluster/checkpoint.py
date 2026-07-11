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

import logging
import re
from pathlib import Path
from typing import Any, Callable

from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

from verl_vla.train_cluster.config import TrainClusterCheckpointConfig

__all__ = ["CheckpointHelper"]

logger = logging.getLogger(__name__)


class CheckpointHelper:
    """Checkpoint helper for actor-only TrainCluster runs."""

    def __init__(
        self,
        config: TrainClusterCheckpointConfig,
        actor_config: Any,
        actor_worker_group: Any,
    ):
        self.config = config
        self.actor_config = actor_config
        self.actor_worker_group = actor_worker_group

    def step_dir(self, global_step: int) -> str:
        return str(Path(self.config.default_local_dir) / f"global_step_{global_step}")

    def load(self) -> tuple[int, str] | None:
        checkpoint_dir = self._find_resume_checkpoint_dir()
        if checkpoint_dir is None:
            logger.info("No checkpoint found; starting training from scratch.")
            return None

        global_step = self._parse_global_step(checkpoint_dir)
        logger.info(
            "Resuming checkpoint: step=%s, path=%s",
            global_step,
            checkpoint_dir,
        )

        self.actor_worker_group.load_checkpoint(
            str(Path(checkpoint_dir) / "actor"),
            del_local_after_load=self.config.del_local_ckpt_after_load,
        )
        return global_step, checkpoint_dir

    def save(
        self,
        global_step: int,
        save_extra_state: Callable[[str], None] | None = None,
    ) -> None:
        local_step_dir = self.step_dir(global_step)
        remote_path = (
            None
            if self.config.default_hdfs_dir is None
            else str(Path(self.config.default_hdfs_dir) / f"global_step_{global_step}" / "actor")
        )
        self.actor_worker_group.save_checkpoint(
            str(Path(local_step_dir) / "actor"),
            remote_path,
            global_step,
            max_ckpt_to_keep=self.config.max_actor_ckpt_to_keep,
        )
        if save_extra_state is not None:
            save_extra_state(local_step_dir)

        if self._async_save_enabled():
            return
        self._write_latest_marker(global_step)

    def _find_resume_checkpoint_dir(self) -> str | None:
        if self.config.resume_mode == "disable":
            return None
        if self.config.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")

        if self.config.resume_mode == "resume_path":
            if not isinstance(self.config.resume_from_path, str):
                raise TypeError("resume ckpt must be str type")
            if "global_step_" not in self.config.resume_from_path:
                raise ValueError("resume ckpt must specify the global_steps")
            checkpoint_dir = self.config.resume_from_path
        else:
            checkpoint_dir = find_latest_ckpt_path(str(Path(self.config.default_local_dir).absolute()))
            if checkpoint_dir is None:
                return None

        return checkpoint_dir if Path(checkpoint_dir).is_absolute() else str(Path.cwd() / checkpoint_dir)

    def _parse_global_step(self, checkpoint_dir: str) -> int:
        match = re.search(r"global_step_(\d+)", checkpoint_dir)
        if match is None:
            raise ValueError(f"checkpoint dir must contain global_step_<N>, got {checkpoint_dir}")
        return int(match.group(1))

    def _latest_marker_path(self) -> Path:
        return Path(self.config.default_local_dir) / "latest_checkpointed_iteration.txt"

    def _write_latest_marker(self, global_step: int) -> None:
        marker_path = self._latest_marker_path()
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(str(global_step))

    def _async_save_enabled(self) -> bool:
        checkpoint_config = self.actor_config.checkpoint
        return bool(getattr(checkpoint_config, "async_save", False) or checkpoint_config.get("async_save", False))
