/*
Copyright 2026 Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

class TeleopGamepadDevice {
  constructor(socketProvider) {
    this.socketProvider = socketProvider;
    this.pollInterval = null;
    this.connected = false;
    this.lastButtons = {};
    this.lastAxes = {};
    this.deadzone = 0.15;
    this.threshold = 0.5;
    this.handleConnect = this.handleConnect.bind(this);
    this.handleDisconnect = this.handleDisconnect.bind(this);
    this.pollState = this.pollState.bind(this);

    this.buttonNames = {
      0: "A",
      1: "B",
      2: "X",
      3: "Y",
      4: "LB",
      5: "RB",
      6: "LT",
      7: "RT",
      8: "View",
      9: "Menu",
      10: "LS",
      11: "RS",
      12: "DUp",
      13: "DDown",
      14: "DLeft",
      15: "DRight"
    };
  }

  attach() {
    window.addEventListener("gamepadconnected", this.handleConnect);
    window.addEventListener("gamepaddisconnected", this.handleDisconnect);
    this.startPolling();
  }

  detach() {
    window.removeEventListener("gamepadconnected", this.handleConnect);
    window.removeEventListener("gamepaddisconnected", this.handleDisconnect);
    this.stopPolling();
  }

  handleConnect(event) {
    this.connected = true;
    console.log("Gamepad connected:", event.gamepad.id);
  }

  handleDisconnect(event) {
    this.connected = false;
    this.lastButtons = {};
    this.lastAxes = {};
    console.log("Gamepad disconnected:", event.gamepad.id);
  }

  startPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
    }
    this.pollInterval = setInterval(this.pollState, 16);
  }

  stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  applyDeadzone(value) {
    if (Math.abs(value) < this.deadzone) {
      return 0;
    }
    return value;
  }

  pollState() {
    const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
    let hasActiveGamepad = false;

    for (const gamepad of gamepads) {
      if (!gamepad || !gamepad.connected) {
        continue;
      }

      hasActiveGamepad = true;
      const buttons = {};
      const axes = {};
      let changed = false;

      for (let i = 0; i < gamepad.buttons.length; i++) {
        const button = gamepad.buttons[i];
        // 使用友好名称，如 "A", "B", "RT", "LT" 等
        const key = this.buttonNames[i] || `button_${i}`;
        buttons[key] = {
          pressed: button.pressed,
          touched: button.touched,
          value: button.value
        };
        if (this.lastButtons[key] === undefined ||
            this.lastButtons[key].pressed !== button.pressed ||
            Math.abs(this.lastButtons[key].value - button.value) > 0.01) {
          changed = true;
        }
      }

      for (let i = 0; i < gamepad.axes.length; i++) {
        const rawValue = gamepad.axes[i];
        const value = this.applyDeadzone(rawValue);
        const key = `axis_${i}`;
        axes[key] = value;
        if (this.lastAxes[key] === undefined ||
            Math.abs(this.lastAxes[key] - value) > 0.01) {
          changed = true;
        }
      }

      if (changed || Object.keys(this.lastButtons).length === 0) {
        this.sendState(gamepad, buttons, axes);
        this.lastButtons = buttons;
        this.lastAxes = axes;
      }
      break;
    }

    if (!hasActiveGamepad && Object.keys(this.lastButtons).length > 0) {
      this.sendDisconnect();
    }
  }

  sendState(gamepad, buttons, axes) {
    const socket = this.socketProvider();
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }

    const payload = {
      event_type: "gamepad_update",
      timestamp: Date.now() / 1000,
      id: gamepad.id,
      index: gamepad.index,
      mapping: gamepad.mapping,
      buttons,
      axes
    };

    socket.send(JSON.stringify({
      type: "gamepad_update",
      device: "gamepad",
      payload
    }));
  }

  sendDisconnect() {
    const socket = this.socketProvider();
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }

    socket.send(JSON.stringify({
      type: "gamepad_update",
      device: "gamepad",
      payload: {
        event_type: "gamepad_disconnect",
        timestamp: Date.now() / 1000
      }
    }));

    this.lastButtons = {};
    this.lastAxes = {};
  }
}