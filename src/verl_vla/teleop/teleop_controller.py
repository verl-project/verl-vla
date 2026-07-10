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

from typing import Any

import numpy as np

from verl_vla.teleop.config import TeleopConfig
from verl_vla.teleop.devices import (
    DeviceBase,
    GamepadDevice,
    GamepadDeviceCfg,
    KeyboardDevice,
    KeyboardDeviceCfg,
    LerobotDevice,
    LerobotDeviceCfg,
    XRControllerDevice,
    XRControllerDeviceCfg,
)
from verl_vla.teleop.obs_server.teleop_server import TeleopServer
from verl_vla.teleop.strategies import InterventionStrategyBase, get_strategy


class TeleopController:
    def __init__(
        self,
        teleop_cfg: TeleopConfig,
        *,
        rank: int,
        stage_id: int,
        env_id: int,
        env_type: str,
        device: str = "keyboard",
    ):
        self.teleop_cfg = teleop_cfg
        self.rank = rank
        self.stage_id = stage_id
        self.env_id = env_id
        self.env_type = env_type
        self.device = device
        self._teleop_server: TeleopServer | None = None
        self.input_devices: dict[str, DeviceBase] = {}
        self.strategies: dict[str, InterventionStrategyBase] = {}
        self._record_confirmation_required = False
        for device_type in self.teleop_cfg.devices:
            input_device = self._create_input_device(device_type)
            self.input_devices[input_device.name] = input_device
            self.strategies[input_device.name] = self._create_strategy(input_device.name)
        self._teleop_server = TeleopServer.from_cfg(
            self.teleop_cfg.server,
            rank=rank,
            stage_id=stage_id,
            env_id=env_id,
            input_devices=self.input_devices,
            latest_input_fn=self._get_teleop_info,
        )

    @classmethod
    def create(
        cls,
        teleop_cfg: TeleopConfig,
        *,
        rank: int,
        stage_id: int,
        env_id: int,
        env_type: str,
        device: str = "keyboard",
    ) -> "TeleopController | None":
        if not teleop_cfg.enable:
            return None
        return cls(teleop_cfg, rank=rank, stage_id=stage_id, env_id=env_id, env_type=env_type, device=device)

    def publish_obs(
        self,
        *,
        images: dict[str, np.ndarray],
        state: Any | None = None,
        extra: dict[str, Any] | None = None,
        task_description: str | None = None,
    ) -> None:
        if self._teleop_server is None:
            return
        obs_extra = {"env_type": self.env_type, **(extra or {})}
        self._teleop_server.publish_obs(
            images=images,
            state=state,
            extra=obs_extra,
            task_description=task_description,
        )

    def is_intervening(self) -> bool:
        return any(
            self.strategies[device_type].is_intervening(input_device)
            for device_type, input_device in self.input_devices.items()
        )

    def apply_action(self, action: Any) -> tuple[Any, bool, bool, bool]:
        overridden_action = action
        for device_type, input_device in self.input_devices.items():
            strategy = self.strategies[device_type]
            if strategy.is_intervening(input_device):
                overridden_action = strategy.apply_action(overridden_action, input_device)
        manual_reward, restart_episode, stop_episode = self._pop_record_control()
        return overridden_action, manual_reward, restart_episode, stop_episode

    def get_action(self, *, wait_for_confirm: bool = False) -> tuple[Any, bool, bool, bool]:
        if wait_for_confirm and not self._record_confirmation_required:
            self._record_confirmation_required = True
            self._write_console(
                f"[teleop] env {self.env_id}: press {self._record_start_key_label()} to start recording."
            )
        elif not wait_for_confirm:
            self._record_confirmation_required = False

        if not self.input_devices:
            raise RuntimeError("No teleop input devices are initialized.")
        device_type = self.device if self.device in self.input_devices else next(iter(self.input_devices))
        action = self.strategies[device_type].get_action(self.input_devices[device_type])
        manual_reward, restart_episode, stop_episode = self._pop_record_control()
        if wait_for_confirm and stop_episode:
            self._record_confirmation_required = False
            self._write_console(f"[teleop] env {self.env_id}: recording started.")
        return action, manual_reward, restart_episode, stop_episode

    def _write_console(self, text: str) -> None:
        if self._teleop_server is not None:
            self._teleop_server.write_console(text)

    def _pop_record_control(self) -> tuple[bool, bool, bool]:
        manual_reward = False
        restart_episode = False
        stop_episode = False
        for input_device in self.input_devices.values():
            control = input_device.pop_record_control()
            manual_reward = manual_reward or bool(control.get("manual_reward", False))
            restart_episode = restart_episode or bool(control.get("restart_episode", False))
            stop_episode = stop_episode or bool(control.get("stop_episode", False))
        return manual_reward, restart_episode, stop_episode

    def _get_teleop_info(self) -> dict[str, Any]:
        device_infos = []
        active_info = None
        is_intervening = False
        for device_type, input_device in self.input_devices.items():
            strategy = self.strategies[device_type]
            device_is_intervening = strategy.is_intervening(input_device)
            info = {
                "device_type": device_type,
                **input_device.snapshot(),
                **strategy.snapshot(input_device),
                "is_intervening": device_is_intervening,
                "active": device_is_intervening,
            }
            device_infos.append(info)
            is_intervening = is_intervening or device_is_intervening
            if not active_info and device_is_intervening:
                active_info = info
        return {
            "env_id": self.env_id,
            "port": self._teleop_server.port() if self._teleop_server is not None else None,
            "devices": device_infos,
            "device_types": list(self.input_devices),
            "active_device": active_info,
            "is_intervening": is_intervening,
            "active": is_intervening,
            "record_control": {
                "confirm_before_record": self._record_confirmation_required,
                "start_key": self._record_start_key_label(),
            },
        }

    def _record_start_key_label(self) -> str:
        if "xr_controller" in self.input_devices:
            return "Enter or A/X"
        return "Enter"

    def reset(self) -> None:
        for input_device in self.input_devices.values():
            input_device.reset()
        for strategy in self.strategies.values():
            strategy.reset()

    def close(self) -> None:
        if self._teleop_server is not None:
            self._teleop_server.close()
            self._teleop_server = None

    def _create_input_device(self, device_type: str) -> DeviceBase:
        if device_type == "keyboard":
            return KeyboardDevice(KeyboardDeviceCfg())
        if device_type == "xr_controller":
            return XRControllerDevice(XRControllerDeviceCfg(max_events=self.teleop_cfg.xr_controller.max_events))
        if device_type == "gamepad":
            return GamepadDevice(GamepadDeviceCfg(max_events=self.teleop_cfg.gamepad.max_events))
        if device_type == "lerobot":
            return LerobotDevice(
                LerobotDeviceCfg(
                    teleop_type=self.teleop_cfg.lerobot.teleop_type,
                    port_name=self.teleop_cfg.lerobot.port_name,
                    baud_rate=self.teleop_cfg.lerobot.baud_rate,
                    min_packet_timeout_ms=self.teleop_cfg.lerobot.min_packet_timeout_ms,
                    urdf_path=self.teleop_cfg.lerobot.urdf_path,
                    target_frame_name=self.teleop_cfg.lerobot.target_frame_name,
                )
            )
        raise NotImplementedError(f"Teleop device {device_type} is not implemented")

    def _create_strategy(self, device_type: str) -> InterventionStrategyBase:
        if device_type == "keyboard":
            return get_strategy(self.env_type, device_type, self.teleop_cfg.keyboard)
        if device_type == "xr_controller":
            return get_strategy(self.env_type, device_type, self.teleop_cfg.xr_controller)
        if device_type == "gamepad":
            return get_strategy(self.env_type, device_type, self.teleop_cfg.gamepad)
        if device_type == "lerobot":
            return get_strategy(self.env_type, device_type, self.teleop_cfg.lerobot)
        raise NotImplementedError(f"Teleop strategy for device {device_type} is not implemented")
