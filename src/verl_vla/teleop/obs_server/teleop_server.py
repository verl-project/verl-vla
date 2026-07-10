# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import logging
import queue
import time
from threading import Lock
from typing import Any, Callable

import cv2
import numpy as np
import torch

from verl_vla.teleop.config import TeleopServerConfig
from verl_vla.teleop.devices import DeviceBase
from verl_vla.teleop.obs_server.server import TeleopObsServer

logger = logging.getLogger(__name__)


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_as_jsonable(v) for v in value]
    return value


def encode_jpeg_base64(image: np.ndarray, quality: int = 80) -> str:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {image.shape}")

    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    success, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not success:
        raise RuntimeError("Failed to encode observation image as JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class ObsStore:
    def __init__(self, env_id: int, port: int, jpeg_quality: int = 80):
        self.env_id = env_id
        self.port = port
        self.jpeg_quality = jpeg_quality
        self._lock = Lock()
        self._latest: dict[str, Any] | None = None
        self._subscribers: set[queue.Queue] = set()

    def update(
        self,
        *,
        step: int,
        images: dict[str, np.ndarray],
        state: Any | None = None,
        extra: dict[str, Any] | None = None,
        task_description: str | None = None,
    ) -> None:
        encoded_images = {
            str(name): encode_jpeg_base64(image, quality=self.jpeg_quality) for name, image in images.items()
        }
        payload = {
            "env_id": self.env_id,
            "port": self.port,
            "step": int(step),
            "timestamp": time.time(),
            "task_description": task_description,
            "images": encoded_images,
            "state": _as_jsonable(state) if state is not None else None,
            "extra": _as_jsonable(extra or {}),
        }
        with self._lock:
            self._latest = payload
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            self._put_latest(subscriber, payload)

    def latest(self) -> dict[str, Any]:
        with self._lock:
            if self._latest is None:
                return {
                    "env_id": self.env_id,
                    "port": self.port,
                    "step": None,
                    "timestamp": None,
                    "task_description": None,
                    "images": {},
                    "state": None,
                    "extra": {},
                }
            return dict(self._latest)

    def subscribe(self) -> queue.Queue:
        subscriber = queue.Queue(maxsize=1)
        with self._lock:
            self._subscribers.add(subscriber)
            latest = self._latest
        if latest is not None:
            self._put_latest(subscriber, latest)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    @staticmethod
    def _put_latest(subscriber: queue.Queue, payload: dict[str, Any]) -> None:
        try:
            subscriber.put_nowait(payload)
            return
        except queue.Full:
            pass

        try:
            subscriber.get_nowait()
        except queue.Empty:
            pass
        try:
            subscriber.put_nowait(payload)
        except queue.Full:
            pass


class TeleopServer:
    def __init__(
        self,
        cfg: TeleopServerConfig,
        *,
        rank: int,
        stage_id: int,
        env_id: int,
        input_devices: dict[str, DeviceBase],
        latest_input_fn: Callable[[], dict[str, Any]],
    ):
        self.cfg = cfg
        self.rank = rank
        self.stage_id = stage_id
        self.env_id = env_id
        self.input_devices = input_devices
        self.latest_input_fn = latest_input_fn
        self._step = 0
        self._store: ObsStore | None = None
        self._server: TeleopObsServer | None = None

    @classmethod
    def from_cfg(
        cls,
        cfg: TeleopServerConfig,
        *,
        rank: int,
        stage_id: int,
        env_id: int,
        input_devices: dict[str, DeviceBase],
        latest_input_fn: Callable[[], dict[str, Any]],
    ) -> "TeleopServer":
        server = cls(
            cfg,
            rank=rank,
            stage_id=stage_id,
            env_id=env_id,
            input_devices=input_devices,
            latest_input_fn=latest_input_fn,
        )
        server.start()
        return server

    def port(self) -> int:
        return (
            self.cfg.base_port + self.rank * self.cfg.rank_stride + self.stage_id * self.cfg.stage_stride + self.env_id
        )

    def start(self) -> None:
        if self._server is not None:
            return
        port = self.port()
        store = ObsStore(env_id=self.env_id, port=port, jpeg_quality=self.cfg.jpeg_quality)
        server = TeleopObsServer(
            store=store,
            host=self.cfg.host,
            port=port,
            input_devices=self.input_devices,
            log_level=self.cfg.log_level,
            latest_input_fn=self.latest_input_fn,
            ssl_certfile=self.cfg.ssl_certfile,
            ssl_keyfile=self.cfg.ssl_keyfile,
        )
        server.start()
        self._store = store
        self._server = server
        print(
            f"[teleop] rank={self.rank} stage={self.stage_id} env={self.env_id} obs_url={server.url}",
            flush=True,
        )
        logger.info(
            "Teleop obs server started for rank=%s stage=%s env=%s: %s",
            self.rank,
            self.stage_id,
            self.env_id,
            server.url,
        )

    def publish_obs(
        self,
        *,
        images: dict[str, Any],
        state: Any | None = None,
        extra: dict[str, Any] | None = None,
        task_description: str | None = None,
    ) -> None:
        if self._store is None:
            return
        self._step += 1
        self._store.update(
            step=self._step,
            images=images,
            state=state,
            extra=extra,
            task_description=task_description,
        )

    def write_console(self, text: str) -> None:
        if self._server is not None:
            self._server.console.write_backend(text)

    def close(self) -> None:
        if self._server is not None:
            self._server.stop()
        self._server = None
        self._store = None
