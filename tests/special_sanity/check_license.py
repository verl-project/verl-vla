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
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

license_headers = [
    "Copyright 2024 Bytedance Ltd. and/or its affiliates",
    "Copyright 2025 Bytedance Ltd. and/or its affiliates",
    "Copyright 2026 Bytedance Ltd. and/or its affiliates",
]


def _git_tracked_py_files() -> set[Path]:
    """Return git-tracked Python files while respecting .gitignore."""
    result = subprocess.run(["git", "ls-files", "*.py", "**/*.py"], capture_output=True, text=True, check=True)
    return {Path(line) for line in result.stdout.splitlines() if line and Path(line).exists()}


def get_py_files(path_arg: Path, tracked: set[Path]) -> Iterable[Path]:
    """Return Python files under a directory or the file itself if it is Python."""
    if path_arg.is_dir():
        return (path for path in tracked if path == path_arg or path_arg in path.parents)
    if path_arg.is_file() and path_arg.suffix == ".py":
        return [path_arg]
    return []


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--directories",
        "-d",
        required=True,
        type=Path,
        nargs="+",
        help="List of directories to check for license headers",
    )
    args = parser.parse_args()

    tracked = _git_tracked_py_files()
    pathlist = set(path for path_arg in args.directories for path in get_py_files(path_arg, tracked))

    for path in pathlist:
        with open(path, encoding="utf-8") as file_handle:
            file_content = file_handle.read()

        has_license = any(license_header in file_content for license_header in license_headers)
        assert has_license, f"file {path} does not contain license"
