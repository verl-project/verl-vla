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

"""LeRobot dataset utility helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .tar import pack_directory_to_tar_bytes, unpack_tar_bytes_to_directory

REQUIRED_LEROBOT_META_FILES = ("meta/info.json", "meta/tasks.parquet", "meta/stats.json")


def get_lerobot_dataset_cls():
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def get_merge_datasets_fn():
    try:
        from lerobot.datasets.dataset_tools import merge_datasets
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import merge_datasets
    return merge_datasets


def merge_lerobot_datasets(
    roots: list[str | Path],
    *,
    output_root: str | Path,
    repo_id: str,
    repo_ids: list[str] | None = None,
    overwrite: bool = False,
    append: bool = False,
    cleanup_roots: bool = True,
    video_files_size_in_mb: float = 1e-6,
) -> dict[str, Any]:
    """Merge multiple LeRobot dataset roots and return the merged root metadata."""
    # 1. Normalize and validate source datasets.
    root_paths = [Path(root) for root in roots]
    if not root_paths:
        raise ValueError("No LeRobot dataset roots were provided for merge.")
    if repo_ids is None:
        repo_ids = [f"local/{root.name}" for root in root_paths]
    if len(repo_ids) != len(root_paths):
        raise ValueError(f"repo_ids length {len(repo_ids)} must match roots length {len(root_paths)}.")

    # 2. Prepare the output path. In append mode, a complete previous output is
    # treated as an additional input dataset. A partial output from a failed
    # merge is deleted so LeRobot does not try to resolve repo_id from the Hub.
    output_path = Path(output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_is_complete = all((output_path / path).exists() for path in REQUIRED_LEROBOT_META_FILES)

    if output_path.exists() and not append:
        if not overwrite:
            raise FileExistsError(f"Output dataset root already exists: {output_path}")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
        output_is_complete = False

    if output_path.exists() and append and not output_is_complete:
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
        output_is_complete = False

    # 3. Build aggregate input/output. Appending writes to a temporary dataset
    # first, then atomically replaces the old output.
    merge_output_path = output_path
    merge_root_paths = root_paths
    merge_repo_ids = repo_ids
    if append and output_is_complete:
        merge_output_path = output_path.parent / f".{output_path.name}.append_tmp"
        shutil.rmtree(merge_output_path, ignore_errors=True)
        merge_root_paths = [output_path, *root_paths]
        merge_repo_ids = [repo_id, *repo_ids]

    from lerobot.datasets.aggregate import aggregate_datasets

    # 4. Aggregate datasets. We call aggregate_datasets directly so we can force
    # video file rotation and avoid PyAV concat time_base failures.
    aggregate_datasets(
        repo_ids=merge_repo_ids,
        aggr_repo_id=repo_id,
        roots=merge_root_paths,
        aggr_root=merge_output_path,
        video_files_size_in_mb=video_files_size_in_mb,
    )

    # 5. Finalize the merged dataset and move the append temp root into place.
    merged_dataset = get_lerobot_dataset_cls()(repo_id=repo_id, root=merge_output_path)
    merged_dataset.finalize()

    if merge_output_path != output_path:
        shutil.rmtree(output_path)
        shutil.move(str(merge_output_path), str(output_path))

    if cleanup_roots:
        for root in root_paths:
            if root.resolve() != output_path.resolve():
                shutil.rmtree(root, ignore_errors=True)
    return {
        "root": output_path,
        "repo_id": repo_id,
    }


def pack_lerobot_dataset(
    root: str | Path,
    *,
    repo_id: str,
    max_payload_mb: int | None = None,
) -> dict[str, Any]:
    """Pack a LeRobot dataset root as a Ray-friendly payload dict."""
    root_path = Path(root)
    tar_bytes = pack_directory_to_tar_bytes(root_path)

    if max_payload_mb is not None and len(tar_bytes) > max_payload_mb * 1024 * 1024:
        raise ValueError(
            f"Packed dataset {repo_id} is {len(tar_bytes) / 1024 / 1024:.1f} MiB, "
            f"larger than max_payload_mb={max_payload_mb}."
        )
    return {
        "repo_id": repo_id,
        "tar_bytes": tar_bytes,
    }


def unpack_lerobot_dataset(payload: dict[str, Any], *, output_root: str | Path, overwrite: bool = False) -> Path:
    """Unpack a payload produced by pack_lerobot_dataset into a dataset root."""
    dataset_root = Path(output_root) / str(payload["repo_id"]).split("/")[-1]
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset root already exists: {dataset_root}")
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True)
    unpack_tar_bytes_to_directory(payload["tar_bytes"], dataset_root)
    return dataset_root
