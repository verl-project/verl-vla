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

"""Shared LeRobot output handling for recording entrypoints."""

from __future__ import annotations

import shutil
from pathlib import Path

from verl_vla.recorder.dataset import count_lerobot_episodes
from verl_vla.recorder.lerobot import REQUIRED_LEROBOT_META_FILES


def is_lerobot_dataset(root: str | Path) -> bool:
    root = Path(root)
    return all((root / path).exists() for path in REQUIRED_LEROBOT_META_FILES)


def prepare_lerobot_output_root(root: str | Path, *, resume: bool) -> int:
    root = Path(root)
    if is_lerobot_dataset(root):
        episode_count = count_lerobot_episodes(root)
        if not resume:
            raise FileExistsError(
                f"LeRobot dataset already exists with {episode_count} episodes: {root}. Set resume=true to append."
            )
        return episode_count
    if root.exists():
        raise FileExistsError(f"Output path exists but is not a complete LeRobot dataset: {root}")
    return 0


def move_lerobot_dataset_to_output(collected_root: str | Path, output_root: str | Path) -> Path:
    collected_root = Path(collected_root)
    output_root = Path(output_root)
    if not is_lerobot_dataset(collected_root):
        raise FileNotFoundError(f"Collected LeRobot dataset does not exist: {collected_root}")
    if output_root.exists():
        raise FileExistsError(f"Output path already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(collected_root), str(output_root))
    return output_root
