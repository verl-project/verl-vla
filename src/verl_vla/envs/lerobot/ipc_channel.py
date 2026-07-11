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

import errno
import os
import pickle
import select
import struct
import time
from pathlib import Path
from typing import Any, Optional


def _ipc_paths(rank: int, stage_id: int, host: str | None = None) -> tuple[str, str]:
    if host is None:
        host = os.uname().nodename
    base = f"/tmp/lerobot_ipc_{host}_rank{rank}_stage{stage_id}"
    return f"{base}.req.fifo", f"{base}.resp.fifo"


def _ensure_fifo(path: str) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()
    os.mkfifo(p)


def _remove_fifo(path: str) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()


def _send_raw_obj(path: str, obj: Any, timeout_s: Optional[float] = None) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    data = struct.pack("!I", len(payload)) + payload

    deadline = None if timeout_s is None else time.time() + max(timeout_s, 0.0)

    fd = None
    while True:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            break
        except OSError as e:
            if e.errno not in (errno.ENXIO, errno.ENOENT):
                raise
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(f"Timed out opening FIFO for write: {path}") from e
            time.sleep(0.05)

    try:
        offset = 0
        total = len(data)
        while offset < total:
            if deadline is None:
                _, writable, _ = select.select([], [fd], [], 1.0)
            else:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out writing FIFO: {path}")
                _, writable, _ = select.select([], [fd], [], remaining)

            if not writable:
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError(f"Timed out waiting FIFO writable: {path}")
                continue

            written = os.write(fd, data[offset:])
            if written <= 0:
                raise RuntimeError(f"Failed to write to FIFO: {path}")
            offset += written
    finally:
        os.close(fd)


def _recv_raw_obj(path: str, timeout_s: Optional[float] = None) -> Any:
    deadline = None if timeout_s is None else time.time() + max(timeout_s, 0.0)

    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)

    def _read_exact(size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            if deadline is None:
                readable, _, _ = select.select([fd], [], [], 1.0)
            else:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out reading {size} bytes from FIFO: {path}")
                readable, _, _ = select.select([fd], [], [], remaining)

            if not readable:
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError(f"Timed out waiting FIFO readable: {path}")
                continue

            chunk = os.read(fd, size - len(data))
            if not chunk:
                raise RuntimeError(f"FIFO closed before completing read: {path}")
            data.extend(chunk)
        return bytes(data)

    try:
        header = _read_exact(4)
        size = struct.unpack("!I", header)[0]
        payload = _read_exact(size)
        return pickle.loads(payload)
    finally:
        os.close(fd)


def setup_ipc(rank: int, stage_id: int) -> None:
    req_path, resp_path = _ipc_paths(rank=rank, stage_id=stage_id)
    _ensure_fifo(req_path)
    _ensure_fifo(resp_path)


def clear_ipc(rank: int, stage_id: int) -> None:
    req_path, resp_path = _ipc_paths(rank=rank, stage_id=stage_id)
    _remove_fifo(req_path)
    _remove_fifo(resp_path)


def send_obj(type: str, content: Any, rank: int, stage_id: int, timeout_s: float = 10.0) -> Any:
    req_path, resp_path = _ipc_paths(rank=rank, stage_id=stage_id)
    deadline = time.time() + timeout_s
    while not (Path(req_path).exists() and Path(resp_path).exists()):
        if time.time() >= deadline:
            raise TimeoutError(f"IPC FIFO is not ready: req={req_path}, resp={resp_path}")
        time.sleep(0.05)

    write_timeout = max(0.0, deadline - time.time())
    _send_raw_obj(req_path, {"type": type, "content": content}, timeout_s=write_timeout)

    read_timeout = max(0.0, deadline - time.time())
    return _recv_raw_obj(resp_path, timeout_s=read_timeout)


def recv_obj(rank: int, stage_id: int) -> Any:
    req_path, _ = _ipc_paths(rank=rank, stage_id=stage_id)
    return _recv_raw_obj(req_path, timeout_s=None)


def reply_obj(content: Any, rank: int, stage_id: int) -> None:
    _, resp_path = _ipc_paths(rank=rank, stage_id=stage_id)
    _send_raw_obj(resp_path, content, timeout_s=None)
