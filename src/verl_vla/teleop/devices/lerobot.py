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

import queue
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from lerobot.motors import MotorCalibration
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import OperatingMode, SOLeader
from scipy.spatial.transform import Rotation
from typing_extensions import override

from verl_vla.teleop.devices.device_base import DeviceBase, DeviceEvent


@dataclass(frozen=True)
class LerobotDeviceCfg:
    teleop_type: str = "so101_leader"
    port_name: str = "tty.usbmodem5AB01836791"
    baud_rate: int = 1_000_000
    min_packet_timeout_ms: float = 1000.0
    urdf_path: str | None = None
    target_frame_name: str = "gripper_frame_link"


@dataclass
class LerobotLeaderConfig(SOLeaderTeleopConfig):
    teleop_type: str = "so101_leader"
    id: str | None = "so101_leader"
    bridge: "WebSerialBridge | None" = None
    min_packet_timeout_ms: float = 1000.0


class WebSerialBridge:
    def __init__(self):
        self._rx_queue: queue.Queue[int] = queue.Queue()
        self._tx_lock = threading.Lock()
        self._tx_packets: list[list[int]] = []
        self._connected = False
        self._baud_rate = 1_000_000

    def mark_connected(self, baud_rate: int) -> None:
        self._connected = True
        self._baud_rate = int(baud_rate)

    def mark_disconnected(self) -> None:
        self._connected = False
        self.clear()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def baud_rate(self) -> int:
        return self._baud_rate

    def push_rx(self, data: list[int]) -> None:
        for byte in data:
            self._rx_queue.put(int(byte) & 0xFF)

    def write_tx(self, data: list[int]) -> int:
        if not self._connected:
            return 0
        packet = [int(byte) & 0xFF for byte in data]
        if not packet:
            return 0
        with self._tx_lock:
            self._tx_packets.append(packet)
        return len(packet)

    def drain_tx_packets(self) -> list[list[int]]:
        with self._tx_lock:
            packets = self._tx_packets
            self._tx_packets = []
        return packets

    def read(self, length: int, timeout_s: float) -> list[int]:
        deadline = time.monotonic() + max(0.0, timeout_s)
        data = []
        while len(data) < length:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                data.append(self._rx_queue.get(timeout=remaining))
            except queue.Empty:
                break
        return data

    def clear(self) -> None:
        while True:
            try:
                self._rx_queue.get_nowait()
            except queue.Empty:
                break
        with self._tx_lock:
            self._tx_packets = []


class RemoteSerialPortHandler:
    def __init__(self, bridge: WebSerialBridge, port_name: str, min_packet_timeout_ms: float):
        self.bridge = bridge
        self.port_name = port_name
        self.min_packet_timeout_ms = float(min_packet_timeout_ms)
        self.is_open = False
        self.is_using = False
        self.baudrate = bridge.baud_rate
        self.packet_start_time = 0.0
        self.packet_timeout = self.min_packet_timeout_ms
        self.tx_time_per_byte = (1000.0 / float(self.baudrate)) * 10.0

    def openPort(self):  # noqa: N802
        self.is_open = self.bridge.connected
        return self.is_open

    def closePort(self):  # noqa: N802
        self.is_open = False

    def clearPort(self):  # noqa: N802
        self.bridge.clear()

    def setPortName(self, port_name):  # noqa: N802
        self.port_name = str(port_name)

    def getPortName(self):  # noqa: N802
        return self.port_name

    def setBaudRate(self, baudrate):  # noqa: N802
        self.baudrate = int(baudrate)
        self.tx_time_per_byte = (1000.0 / float(self.baudrate)) * 10.0
        return True

    def getBaudRate(self):  # noqa: N802
        return self.baudrate

    def getBytesAvailable(self):  # noqa: N802
        return 0

    def readPort(self, length):  # noqa: N802
        timeout_s = max(0.001, float(self.packet_timeout) / 1000.0)
        return self.bridge.read(int(length), timeout_s)

    def writePort(self, packet):  # noqa: N802
        return self.bridge.write_tx(list(packet))

    def setPacketTimeout(self, packet_length):  # noqa: N802
        self.packet_start_time = self.getCurrentTime()
        sdk_timeout = (self.tx_time_per_byte * int(packet_length)) + (self.tx_time_per_byte * 3.0) + 50
        self.packet_timeout = max(self.min_packet_timeout_ms, sdk_timeout)

    def setPacketTimeoutMillis(self, msec):  # noqa: N802
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = max(self.min_packet_timeout_ms, float(msec))

    def isPacketTimeout(self):  # noqa: N802
        return self.getTimeSinceStart() > self.packet_timeout

    def getCurrentTime(self):  # noqa: N802
        return time.monotonic() * 1000.0

    def getTimeSinceStart(self):  # noqa: N802
        return self.getCurrentTime() - self.packet_start_time

    def setupPort(self, cflag_baud):  # noqa: N802
        del cflag_baud
        return True

    def getCFlagBaud(self, baudrate):  # noqa: N802
        return int(baudrate)


class LerobotLeader(SOLeader):
    config_class = LerobotLeaderConfig
    name = "lerobot_leader"

    def __init__(self, config: LerobotLeaderConfig):
        if config.teleop_type not in {"so101_leader", "so100_leader"}:
            raise NotImplementedError(f"Unsupported LeRobot teleop_type: {config.teleop_type}")
        if config.bridge is None:
            raise ValueError("LerobotLeaderConfig.bridge must be set.")
        self._webserial_bridge = config.bridge
        super().__init__(config)
        self._patch_remote_port()

    def _patch_remote_port(self) -> None:
        import scservo_sdk as scs

        remote_port = RemoteSerialPortHandler(
            self._webserial_bridge,
            self.config.port,
            self.config.min_packet_timeout_ms,
        )
        self.bus.port_handler = remote_port
        self.bus.packet_handler = scs.PacketHandler(self.bus.protocol_version)
        self.bus.sync_reader = scs.GroupSyncRead(self.bus.port_handler, self.bus.packet_handler, 0, 0)
        self.bus.sync_writer = scs.GroupSyncWrite(self.bus.port_handler, self.bus.packet_handler, 0, 0)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback
        raise NotImplementedError


class LerobotDevice(DeviceBase):
    name = "lerobot"

    def __init__(self, cfg: LerobotDeviceCfg | None = None):
        self.cfg = cfg or LerobotDeviceCfg()
        super().__init__()
        self.bridge = WebSerialBridge()
        self.log_fn: Callable[[str], None] | None = None
        self.input_fn: Callable[[str], str] | None = None
        self.clear_input_fn: Callable[[], None] | None = None
        self._status = "idle"
        self._error = ""
        self._connect_attempt = 0
        self._connect_started_at: float | None = None
        self._connect_start_rx_bytes = 0
        self._connect_start_tx_bytes = 0
        self._watchdog_thread: threading.Thread | None = None
        self._connect_thread: threading.Thread | None = None
        self._teleop: LerobotLeader | None = None
        self._kinematics = None
        self._rx_bytes = 0
        self._tx_bytes = 0

    @override
    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._clear_record_control()

    @override
    def handle_event(self, event: DeviceEvent) -> None:
        with self._lock:
            self._record_event(event)
        event_type = event.event_type.lower()
        if event_type == "lerobot_open":
            baud_rate = int(event.raw.get("baud_rate") or self.cfg.baud_rate)
            self.bridge.mark_connected(baud_rate)
            self._status = "serial_open"
            self._error = ""
            self._log(
                "LeRobot browser serial opened: "
                f"type={self.cfg.teleop_type}, port={self.cfg.port_name}, baud={baud_rate}"
            )
            self.start_lerobot_connect()
        elif event_type == "lerobot_rx":
            data = event.raw.get("data") or []
            if isinstance(data, list):
                self.bridge.push_rx(data)
                self._rx_bytes += len(data)
        elif event_type == "lerobot_close":
            self.bridge.mark_disconnected()
            self._teleop = None
            self._kinematics = None
            self._status = "serial_closed"
            self._log("LeRobot browser serial closed.")
        elif event_type == "lerobot_error":
            self._error = str(event.raw.get("message") or "")
            self._status = "serial_error"
            self._log(f"LeRobot browser serial error: {self._error}")

    @override
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "device": self.name,
                "teleop_type": self.cfg.teleop_type,
                "configured_port": self.cfg.port_name,
                "baud_rate": self.cfg.baud_rate,
                "browser_connected": self.bridge.connected,
                "status": self._status,
                "error": self._error,
                "connect_attempt": self._connect_attempt,
                "connect_elapsed_s": None
                if self._connect_started_at is None
                else max(0.0, time.monotonic() - self._connect_started_at),
                "key_bindings": self.key_bindings(),
            }

    def key_bindings(self) -> dict[str, str]:
        return {
            "Space": "enter setup hold / exit intervention",
            "Tab": "start intervention / return to setup hold",
            "Move leader arm": "relative position control",
            "Rotate leader wrist": "relative rotation control when enabled",
            "Leader gripper": "open / close gripper",
            "R": "manual reward",
            "Backspace": "restart recording episode",
            "Enter": "stop recording episode",
        }

    def browser_config(self) -> dict[str, Any]:
        return {
            "teleop_type": self.cfg.teleop_type,
            "port_name": self.cfg.port_name,
            "baud_rate": self.cfg.baud_rate,
            "min_packet_timeout_ms": self.cfg.min_packet_timeout_ms,
            "urdf_path": self.cfg.urdf_path,
        }

    def drain_tx_packets(self) -> list[list[int]]:
        packets = self.bridge.drain_tx_packets()
        self._tx_bytes += sum(len(packet) for packet in packets)
        return packets

    def start_lerobot_connect(self) -> None:
        if self._connect_thread is not None and self._connect_thread.is_alive():
            elapsed = 0.0 if self._connect_started_at is None else time.monotonic() - self._connect_started_at
            self._log(
                "LeRobot connect already running: "
                f"elapsed={elapsed:.1f}s, status={self._status}, rx={self._rx_bytes}, tx={self._tx_bytes}, "
                f"error={self._error or '<none>'}."
            )
            return
        self._connect_started_at = time.monotonic()
        self._connect_start_rx_bytes = self._rx_bytes
        self._connect_start_tx_bytes = self._tx_bytes
        self._connect_thread = threading.Thread(target=self._connect_lerobot, name="lerobot-connect", daemon=True)
        self._connect_thread.start()
        self._start_connect_watchdog()

    def _connect_lerobot(self) -> None:
        while self.bridge.connected:
            self._status = "lerobot_connecting"
            teleop = None
            try:
                self._connect_attempt += 1
                self._log(f"LeRobot connect attempt {self._connect_attempt}: type={self.cfg.teleop_type}.")
                config = LerobotLeaderConfig(
                    teleop_type=self.cfg.teleop_type,
                    id=self.cfg.teleop_type,
                    port=self.cfg.port_name,
                    bridge=self.bridge,
                    min_packet_timeout_ms=self.cfg.min_packet_timeout_ms,
                )
                teleop = LerobotLeader(config)
                self._web_calibrate(teleop)
                self._teleop = teleop
                self._kinematics = self._create_kinematics()
                self._status = "lerobot_connected"
                self._error = ""
                self._connect_started_at = None
                self._log("LeRobot connect succeeded.")
                return
            except Exception as exc:
                if teleop is not None:
                    self._disconnect_bus_after_failed_probe(teleop)
                self._teleop = None
                self._kinematics = None
                self._status = "lerobot_retry_wait"
                self._log_exception("LeRobot connect failed, retrying", exc)
                time.sleep(1.0)
        self._connect_started_at = None

    def _web_calibrate(self, teleop: LerobotLeader) -> None:
        if self.input_fn is None:
            raise RuntimeError("LeRobot web calibration needs a console input_fn.")
        if self.clear_input_fn is not None:
            self.clear_input_fn()

        self._log("LeRobot calibration: connecting bus and checking motors.")
        self._connect_bus_with_retry(teleop)
        self._log("LeRobot calibration: bus probe succeeded.")

        if teleop.calibration:
            self._log(f"LeRobot calibration: using local calibration file {teleop.calibration_fpath}.")
            teleop.bus.write_calibration(teleop.calibration)
            teleop.configure()
            self._log("LeRobot calibration: local calibration written to motors.")
            return

        teleop.bus.disable_torque()
        for motor in teleop.bus.motors:
            teleop.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        self.input_fn(f"Move {teleop} to the middle of its range of motion, then press ENTER.")
        self._log("LeRobot calibration: capturing middle pose and homing offsets.")
        homing_offsets = teleop.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in teleop.bus.motors if motor != full_turn_motor]
        self.input_fn(
            f"Move all joints except '{full_turn_motor}' through their full range. Press ENTER to start recording."
        )
        self._log("LeRobot calibration: recording joint ranges. Press ENTER to stop.")
        range_mins, range_maxes = self._record_ranges_until_enter(teleop, unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        teleop.calibration = {}
        for motor, motor_cfg in teleop.bus.motors.items():
            teleop.calibration[motor] = MotorCalibration(
                id=motor_cfg.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        teleop.bus.write_calibration(teleop.calibration)
        teleop._save_calibration()
        teleop.configure()
        self._log(f"LeRobot calibration saved to {teleop.calibration_fpath}")

    def _connect_bus_with_retry(self, teleop: LerobotLeader) -> None:
        probe_attempt = 0
        while self.bridge.connected:
            probe_attempt += 1
            try:
                self._log(f"LeRobot bus probe attempt {probe_attempt}: connecting and handshaking.")
                teleop.bus.connect(handshake=True)
                self._log(f"LeRobot bus probe attempt {probe_attempt}: succeeded.")
                return
            except Exception as exc:
                self._status = "lerobot_retry_wait"
                self._log_exception(f"LeRobot bus probe attempt {probe_attempt} failed", exc)
                self._log_motor_config(teleop)
                self._disconnect_bus_after_failed_probe(teleop)
                time.sleep(1.0)
                self._status = "lerobot_connecting"
        raise RuntimeError("LeRobot browser serial closed while connecting.")

    def _start_connect_watchdog(self) -> None:
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_thread = threading.Thread(
            target=self._connect_watchdog,
            name="lerobot-connect-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _connect_watchdog(self) -> None:
        last_rx = self._connect_start_rx_bytes
        last_tx = self._connect_start_tx_bytes
        next_log_at = time.monotonic() + 5.0
        while self._connect_thread is not None and self._connect_thread.is_alive():
            now = time.monotonic()
            if now >= next_log_at:
                elapsed = 0.0 if self._connect_started_at is None else now - self._connect_started_at
                delta_rx = self._rx_bytes - last_rx
                delta_tx = self._tx_bytes - last_tx
                self._log(
                    "LeRobot connect still running: "
                    f"elapsed={elapsed:.1f}s, status={self._status}, "
                    f"rx={self._rx_bytes} (+{delta_rx}), tx={self._tx_bytes} (+{delta_tx}), "
                    f"browser_connected={self.bridge.connected}, error={self._error or '<none>'}."
                )
                last_rx = self._rx_bytes
                last_tx = self._tx_bytes
                next_log_at = now + 5.0
            time.sleep(0.2)

    def _disconnect_bus_after_failed_probe(self, teleop: LerobotLeader) -> None:
        try:
            if teleop.bus.is_connected:
                teleop.bus.disconnect(disable_torque=False)
        except Exception as exc:
            self._log(f"LeRobot bus cleanup after failed probe skipped: {exc!r}")

    def read_ee_pose(self) -> dict[str, Any] | None:
        teleop = self._teleop
        if not self.bridge.connected or teleop is None or not teleop.is_connected or self._kinematics is None:
            return None
        try:
            action = teleop.get_action()
            motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
            q = np.asarray([action[f"{motor}.pos"] for motor in motor_names], dtype=float)
            transform = self._kinematics.forward_kinematics(q)
            pos = transform[:3, 3].astype(float, copy=False)
            rotvec = Rotation.from_matrix(transform[:3, :3]).as_rotvec().astype(float, copy=False)
            pose = {
                "pos": pos.tolist(),
                "rotvec": rotvec.tolist(),
                "gripper": float(action["gripper.pos"]),
                "joints": {key: float(value) for key, value in action.items()},
            }
            return pose
        except Exception as exc:
            self._error = repr(exc)
            self._log(f"LeRobot ee pose read failed: {exc!r}")
            return None

    def _create_kinematics(self):
        if not self.cfg.urdf_path:
            self._log("LeRobot urdf_path is not set; ee pose is unavailable.")
            return None
        urdf_path = Path(self.cfg.urdf_path).expanduser()
        if not urdf_path.exists():
            self._log(f"LeRobot urdf_path does not exist: {urdf_path}; ee pose is unavailable.")
            return None
        try:
            from lerobot.model.kinematics import RobotKinematics

            return RobotKinematics(
                urdf_path=str(urdf_path),
                target_frame_name=self.cfg.target_frame_name,
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            )
        except Exception as exc:
            self._log(f"LeRobot failed to initialize kinematics: {exc!r}")
            return None

    def _record_ranges_until_enter(
        self, teleop: LerobotLeader, motor_names: list[str]
    ) -> tuple[dict[str, int], dict[str, int]]:
        stop_event = threading.Event()

        def wait_for_stop() -> None:
            if self.input_fn is not None:
                self.input_fn("Recording... press ENTER to stop recording ranges.")
            stop_event.set()

        stop_thread = threading.Thread(target=wait_for_stop, name="lerobot-calibration-stop", daemon=True)
        stop_thread.start()

        start_positions = teleop.bus.sync_read("Present_Position", motor_names, normalize=False)
        mins = {motor: int(pos) for motor, pos in start_positions.items()}
        maxes = dict(mins)
        last_log_at = 0.0

        while not stop_event.is_set():
            positions = teleop.bus.sync_read("Present_Position", motor_names, normalize=False)
            for motor, pos in positions.items():
                pos = int(pos)
                mins[motor] = min(mins[motor], pos)
                maxes[motor] = max(maxes[motor], pos)
            now = time.monotonic()
            if now - last_log_at > 1.0:
                summary = ", ".join(f"{motor}:{mins[motor]}..{maxes[motor]}" for motor in motor_names)
                self._log(f"LeRobot calibration ranges: {summary}")
                last_log_at = now
            time.sleep(0.05)

        same_min_max = [motor for motor in motor_names if mins[motor] == maxes[motor]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values: {same_min_max}")
        return mins, maxes

    def _log(self, text: str) -> None:
        if self.log_fn is not None:
            self.log_fn(text)

    def _log_exception(self, prefix: str, exc: Exception) -> None:
        summary = self._format_exception(exc)
        trace = traceback.format_exc()
        self._error = summary
        self._log(f"{prefix}: {summary}")
        self._log(trace.rstrip())

    def _format_exception(self, exc: Exception) -> str:
        parts = [f"{type(exc).__name__}: {exc}"]
        cause = getattr(exc, "__cause__", None)
        context = getattr(exc, "__context__", None)
        if cause is not None:
            parts.append(f"cause={type(cause).__name__}: {cause}")
        if context is not None and context is not cause:
            parts.append(f"context={type(context).__name__}: {context}")
        return " | ".join(parts)

    def _log_motor_config(self, teleop: LerobotLeader) -> None:
        try:
            motors = teleop.bus.motors
            summary = ", ".join(f"{name}:id={motor.id}" for name, motor in motors.items())
            self._log(f"LeRobot expected motors: {summary}")
        except Exception as exc:
            self._log(f"LeRobot expected motor config unavailable: {exc!r}")
