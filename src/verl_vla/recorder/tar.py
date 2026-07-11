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

"""Safe tar serialization helpers for recorder payloads."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path


def pack_directory_to_tar_bytes(root: str | Path) -> bytes:
    root_path = Path(root)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for child in root_path.iterdir():
            tar.add(child, arcname=child.name)
    return buffer.getvalue()


def unpack_tar_bytes_to_directory(tar_bytes: bytes, output_root: str | Path) -> None:
    output_path = Path(output_root)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        base = output_path.resolve()
        for member in tar.getmembers():
            target = (output_path / member.name).resolve()
            if base != target and base not in target.parents:
                raise ValueError(f"Unsafe path in tar archive: {member.name}")
        tar.extractall(output_path)
