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

"""Upload a local LeRobot dataset with the official LeRobot Hub metadata."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from verl_vla.utils.recorder import get_lerobot_dataset_cls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a local LeRobot dataset to Hugging Face. "
            "This uses LeRobotDataset.push_to_hub(), so README.md, LeRobot tags, "
            "the robotics task category, data_files config, and the v3.0 tag are generated."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Local LeRobot dataset root, e.g. /tmp/verl_vla_lerobot_records/local/verl_vla_libero.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face dataset repo id, e.g. Miical/verl_vla_libero.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Optional target branch/revision to upload to.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Extra dataset tags. The LeRobot tag is always added by LeRobot.",
    )
    parser.add_argument(
        "--license",
        default="apache-2.0",
        help="Dataset card license identifier.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the dataset repo as private.",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Do not upload videos/ files.",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional Hugging Face upload allow patterns.",
    )
    parser.add_argument(
        "--upload-large-folder",
        action="store_true",
        help="Use huggingface_hub upload_large_folder instead of upload_folder.",
    )
    return parser.parse_args()


def validate_local_dataset(root: Path) -> None:
    required_paths = (
        "meta/info.json",
        "meta/tasks.parquet",
        "meta/stats.json",
    )
    missing_paths = [path for path in required_paths if not (root / path).exists()]
    if missing_paths:
        raise FileNotFoundError(f"{root} is not a complete LeRobot dataset. Missing: {missing_paths}")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    validate_local_dataset(root)

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise OSError("Please set HF_TOKEN before uploading, e.g. export HF_TOKEN='hf_...'")

    dataset_cls = get_lerobot_dataset_cls()
    dataset = dataset_cls(repo_id=args.repo_id, root=root)
    dataset.push_to_hub(
        branch=args.branch,
        tags=args.tags,
        license=args.license,
        private=args.private,
        push_videos=not args.no_videos,
        allow_patterns=args.allow_patterns,
        upload_large_folder=args.upload_large_folder,
    )
    print(f"Uploaded LeRobot dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
