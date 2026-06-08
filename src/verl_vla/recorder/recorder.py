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

"""Small wrapper around LeRobotDataset for per-env episode recording."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from verl_vla.recorder.strategies import get_lerobot_strategy
from verl_vla.utils.recorder import get_lerobot_dataset_cls


class LeRobotRecorder:
    """Record one environment stream into a temporary LeRobot dataset.

    Frames are first kept in per-env in-memory pending buffers. ``record_once()`` only
    normalizes the environment data and appends it to that buffer; the LeRobot
    dataset is written only when ``save_episode()`` is called. This keeps an
    unfinished episode available across chunk boundaries without creating a
    partial episode on disk.

    ``pop_completed()`` finalizes the dataset currently on disk and returns its
    root for the caller to pack, transfer, or merge. After that call the
    recorder has no active LeRobot dataset object. The returned root must be
    consumed before appending more frames.

    ``finalize()`` is a cleanup method, not a save method. It discards pending
    frames, removes the temporary dataset root, and clears local references.
    """

    def __init__(
        self,
        *,
        env_type: str,
        root: str | Path,
        repo_id: str,
        num_envs: int = 1,
        strategy_kwargs: dict[str, Any] | None = None,
        use_videos: bool = True,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        batch_encoding_size: int = 1,
        vcodec: str = "libsvtav1",
        video_files_size_in_mb: float = 1e-6,
    ) -> None:
        self.root = Path(root)
        self.repo_id = repo_id
        self.dataset_root = self.root / self.repo_id
        self.num_envs = num_envs
        self.env_type = env_type
        self.video_files_size_in_mb = video_files_size_in_mb
        self.strategy = get_lerobot_strategy(env_type, **(strategy_kwargs or {}))
        self._dataset_kwargs = {
            "use_videos": use_videos,
            "image_writer_processes": image_writer_processes,
            "image_writer_threads": image_writer_threads,
            "batch_encoding_size": batch_encoding_size,
            "vcodec": vcodec,
        }
        self._pending_frames: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
        self._has_completed_episode = False

        self.dataset = self._create_dataset()

    def record_once(
        self,
        *,
        env_id: int = 0,
        obs: dict[str, Any],
        state: Any,
        action: Any,
        task: str,
        reward: Any = 0.0,
        done: Any = False,
        truncated: Any = False,
        is_intervention: Any = False,
    ) -> None:
        """Record one environment step into the pending episode buffer."""
        if self.dataset is None:
            self.dataset = self._create_dataset()
        self._pending_frames[env_id].append(
            self.strategy.make_frame(
                obs=obs,
                state=state,
                action=action,
                task=task,
                reward=reward,
                done=done,
                truncated=truncated,
                is_intervention=is_intervention,
            )
        )

    def save_episode(self, env_id: int = 0) -> None:
        """Flush the current episode if it contains frames."""
        pending_frames = self._pending_frames[env_id]
        if not pending_frames:
            return
        for pending_frame in pending_frames:
            self.dataset.add_frame(dict(pending_frame))
        self.dataset.save_episode()
        pending_frames.clear()
        self._has_completed_episode = True

    def clear_episode(self, env_id: int = 0) -> None:
        """Discard pending frames for one environment."""
        self._pending_frames[env_id].clear()

    def pop_completed(self) -> Path | None:
        """Return a finalized dataset root with completed episodes."""
        if not self._has_completed_episode:
            return None

        self.dataset.finalize()
        completed_root = self.dataset_root

        self.dataset = None
        self._has_completed_episode = False

        return completed_root

    def finalize(self) -> None:
        """Discard local recorder cache and remove temporary dataset roots."""
        shutil.rmtree(self.dataset_root, ignore_errors=True)

        for pending_frames in self._pending_frames:
            pending_frames.clear()
        self._has_completed_episode = False
        self.dataset = None

    def _create_dataset(self):
        if self.dataset_root.exists():
            if self.dataset_root.is_dir():
                shutil.rmtree(self.dataset_root)
            else:
                self.dataset_root.unlink()
        lerobot_dataset_cls = get_lerobot_dataset_cls()
        dataset = lerobot_dataset_cls.create(
            repo_id=self.repo_id,
            root=self.dataset_root,
            fps=self.strategy.fps,
            robot_type=self.strategy.robot_type,
            features=self.strategy.features(),
            **self._dataset_kwargs,
        )
        dataset.meta.update_chunk_settings(video_files_size_in_mb=self.video_files_size_in_mb)
        return dataset
