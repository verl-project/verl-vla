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

import asyncio
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from verl_vla.teleop.devices import DeviceBase, DeviceEvent, KeyboardDevice

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from verl_vla.teleop.obs_server.teleop_server import ObsStore


_HTML_DIR = Path(__file__).with_name("html")
_INDEX_HTML_PATH = _HTML_DIR / "index.html"


def _load_index_html() -> str:
    return _INDEX_HTML_PATH.read_text(encoding="utf-8")


class WebConsoleSession:
    def __init__(self):
        self._lock = threading.Lock()
        self._events: list[dict] = []
        self._next_id = 1
        self._input_queue: queue.Queue[str] = queue.Queue()
        self._append_event("backend", "Console ready.")

    def snapshot(self) -> dict:
        with self._lock:
            return {"events": list(self._events)}

    def handle_message(self, message: dict) -> dict:
        payload = message.get("payload") or {}
        text = str(payload.get("text", ""))
        self.write_frontend(text)
        self._input_queue.put(text)
        return self.snapshot()

    def input(self, prompt: str) -> str:
        self.write_backend(prompt)
        return self._input_queue.get()

    def clear_inputs(self) -> None:
        while True:
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                return

    def write_backend(self, text: str) -> None:
        if text:
            self._append_event("backend", text)

    def write_frontend(self, text: str) -> None:
        self._append_event("frontend", text)

    def _append_event(self, event_type: str, text: str) -> int:
        with self._lock:
            event_id = self._next_id
            self._next_id += 1
            self._events.append(
                {
                    "id": event_id,
                    "type": event_type,
                    "text": text,
                    "timestamp": time.time(),
                }
            )
            self._events = self._events[-200:]
            return event_id


def create_app(
    store: "ObsStore",
    input_devices: dict[str, DeviceBase],
    latest_input_fn=None,
    console: WebConsoleSession | None = None,
) -> FastAPI:
    app = FastAPI(title=f"VERL-VLA Teleop Obs env {store.env_id}")
    app.mount("/static", StaticFiles(directory=_HTML_DIR), name="teleop-static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        device_types = [device.name for device in input_devices.values()]
        device_configs = {
            device.name: device.browser_config()
            for device in input_devices.values()
            if hasattr(device, "browser_config")
        }
        return (
            _load_index_html()
            .replace("__TELEOP_DEVICE_TYPES__", json.dumps(device_types))
            .replace("__TELEOP_DEVICE_CONFIGS__", json.dumps(device_configs))
        )

    @app.get("/api/obs/latest")
    def latest_obs():
        payload = store.latest()
        if latest_input_fn is not None:
            payload["teleop"] = latest_input_fn()
        return payload

    @app.get("/api/health")
    def health():
        return {"status": "ok", "env_id": store.env_id, "port": store.port}

    @app.get("/api/input/latest")
    def latest_input():
        if latest_input_fn is not None:
            payload = latest_input_fn()
        else:
            payload = {device_type: device.snapshot() for device_type, device in input_devices.items()}
        if console is not None:
            payload = dict(payload)
            payload["console"] = console.snapshot()
        return payload

    @app.get("/api/input/drain")
    def drain_input():
        return {
            "latest": {device_type: device.snapshot() for device_type, device in input_devices.items()},
            "events": {device_type: device.drain_events() for device_type, device in input_devices.items()},
        }

    @app.post("/api/input/reset")
    def reset_input():
        for input_device in input_devices.values():
            input_device.reset()
        return {device_type: device.snapshot() for device_type, device in input_devices.items()}

    @app.websocket("/ws/obs")
    async def obs_stream(websocket: WebSocket):
        await websocket.accept()
        subscriber = store.subscribe()
        try:
            while True:
                payload = await asyncio.to_thread(subscriber.get)
                if latest_input_fn is not None:
                    payload = dict(payload)
                    payload["teleop"] = latest_input_fn()
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            store.unsubscribe(subscriber)

    @app.websocket("/ws/input")
    async def input_stream(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_json()
                message_type = message.get("type")
                if message_type == "console_text":
                    if console is not None:
                        await websocket.send_json({"console": console.handle_message(message)})
                    continue
                device_type = str(message.get("device") or "")
                if device_type not in input_devices:
                    continue
                if message_type != "keyboard_event":
                    if device_type == "keyboard":
                        continue
                    if device_type == "xr_controller" and message_type != "xr_frame":
                        continue
                    if device_type == "gamepad" and message_type not in {"gamepad_update", "gamepad_disconnect"}:
                        continue
                    if device_type == "lerobot" and message_type not in {
                        "lerobot_open",
                        "lerobot_rx",
                        "lerobot_poll",
                        "lerobot_close",
                        "lerobot_error",
                    }:
                        continue
                payload = message.get("payload", {})
                if message_type != "lerobot_poll":
                    input_devices[device_type].handle_event(DeviceEvent.from_payload(payload))
                if device_type == "lerobot" and message_type in {"lerobot_poll", "lerobot_rx"}:
                    bridge_payload = {}
                    if hasattr(input_devices[device_type], "drain_tx_packets"):
                        bridge_payload["lerobot_tx"] = input_devices[device_type].drain_tx_packets()
                    await websocket.send_json(bridge_payload)
                    continue
                if latest_input_fn is not None:
                    payload = latest_input_fn()
                else:
                    payload = input_devices[device_type].snapshot()
                if console is not None:
                    payload = dict(payload)
                    payload["console"] = console.snapshot()
                if device_type == "lerobot" and hasattr(input_devices[device_type], "drain_tx_packets"):
                    payload = dict(payload)
                    payload["lerobot_tx"] = input_devices[device_type].drain_tx_packets()
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass

    return app


@dataclass
class TeleopObsServer:
    store: "ObsStore"
    host: str
    port: int
    input_devices: dict[str, DeviceBase] | None = None
    log_level: str = "warning"
    latest_input_fn: Callable[[], dict] | None = None
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None

    def __post_init__(self):
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.console = WebConsoleSession()
        if self.input_devices is None:
            self.input_devices = {"keyboard": KeyboardDevice()}
        self.input_devices = {device.name: device for device in self.input_devices.values()}
        for device in self.input_devices.values():
            if hasattr(device, "log_fn"):
                device.log_fn = self.console.write_backend
            if hasattr(device, "input_fn"):
                device.input_fn = self.console.input
            if hasattr(device, "clear_input_fn"):
                device.clear_input_fn = self.console.clear_inputs

    @property
    def url(self) -> str:
        scheme = "https" if self.ssl_certfile and self.ssl_keyfile else "http"
        return f"{scheme}://{self.host}:{self.port}"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        config = uvicorn.Config(
            create_app(
                self.store,
                self.input_devices,
                latest_input_fn=self.latest_input,
                console=self.console,
            ),
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            access_log=False,
            ssl_certfile=self.ssl_certfile,
            ssl_keyfile=self.ssl_keyfile,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name=f"teleop-obs-{self.port}", daemon=True)
        self._thread.start()
        logger.info("Started teleop obs server for env %s at %s", self.store.env_id, self.url)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

    def latest_input(self) -> dict:
        if self.latest_input_fn is not None:
            payload = self.latest_input_fn()
        else:
            payload = {device_type: device.snapshot() for device_type, device in self.input_devices.items()}
        payload = dict(payload)
        payload["console"] = self.console.snapshot()
        return payload
