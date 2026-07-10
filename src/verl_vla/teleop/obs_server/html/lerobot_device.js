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

class TeleopLerobotDevice {
  constructor(socketProvider, configProvider) {
    this.socketProvider = socketProvider;
    this.configProvider = configProvider;
    this.port = null;
    this.reader = null;
    this.writer = null;
    this.reading = false;
    this.pollTimer = null;
    this.connecting = false;
  }

  attach() {
    if (!("serial" in navigator)) {
      this.appendConsole("WebSerial is unavailable. Use Chrome or Edge over HTTPS.");
      return;
    }
    this.startPolling();
    this.connectAuthorizedPort();
  }

  detach() {
    this.stopPolling();
  }

  keyboardEventCodes() {
    return ["Space", "Tab", "KeyR", "Backspace", "Enter"];
  }

  async requestConnect() {
    if (this.port || this.connecting || !("serial" in navigator)) {
      return;
    }
    try {
      this.connecting = true;
      const port = await navigator.serial.requestPort();
      await this.openPort(port);
    } catch (error) {
      this.sendEvent("lerobot_error", {message: String(error && error.message ? error.message : error)});
    } finally {
      this.connecting = false;
    }
  }

  async connectAuthorizedPort() {
    if (this.port || this.connecting || !("serial" in navigator)) {
      return;
    }
    const ports = await navigator.serial.getPorts();
    if (!ports.length) {
      const cfg = this.configProvider();
      this.appendConsole(`Lerobot serial permission required. Click this console and select ${cfg.port_name || "the SO101 port"}.`);
      return;
    }
    await this.openPort(ports[0]);
  }

  async openPort(port) {
    const cfg = this.configProvider();
    this.port = port;
    await this.port.open({baudRate: Number(cfg.baud_rate || 1000000)});
    this.writer = this.port.writable.getWriter();
    this.sendEvent("lerobot_open", {
      port_name: cfg.port_name || "",
      baud_rate: Number(cfg.baud_rate || 1000000),
      info: this.port.getInfo ? this.port.getInfo() : {}
    });
    this.appendConsole(`Lerobot WebSerial opened at ${cfg.baud_rate || 1000000}.`);
    this.readLoop();
  }

  async readLoop() {
    if (!this.port || !this.port.readable || this.reading) {
      return;
    }
    this.reading = true;
    try {
      while (this.port && this.port.readable) {
        this.reader = this.port.readable.getReader();
        try {
          while (true) {
            const {value, done} = await this.reader.read();
            if (done) {
              break;
            }
            if (value && value.length) {
              this.sendEvent("lerobot_rx", {data: Array.from(value)});
            }
          }
        } finally {
          this.reader.releaseLock();
          this.reader = null;
        }
      }
    } catch (error) {
      this.sendEvent("lerobot_error", {message: String(error && error.message ? error.message : error)});
    } finally {
      this.reading = false;
      this.sendEvent("lerobot_close", {});
      this.port = null;
      this.writer = null;
    }
  }

  async handleTxPackets(packets) {
    if (!this.writer || !Array.isArray(packets)) {
      return;
    }
    for (const packet of packets) {
      if (!Array.isArray(packet) || !packet.length) {
        continue;
      }
      try {
        await this.writer.write(new Uint8Array(packet));
      } catch (error) {
        this.sendEvent("lerobot_error", {message: `WebSerial write failed: ${String(error && error.message ? error.message : error)}`});
        throw error;
      }
    }
  }

  startPolling() {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
    }
    this.pollTimer = setInterval(() => {
      this.sendEvent("lerobot_poll", {});
    }, 10);
  }

  stopPolling() {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  sendEvent(eventType, payload) {
    const socket = this.socketProvider();
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    socket.send(JSON.stringify({
      type: eventType,
      device: "lerobot",
      payload: {
        event_type: eventType,
        timestamp: Date.now() / 1000,
        ...payload
      }
    }));
  }

  appendConsole(text) {
    if (typeof appendConsoleText === "function") {
      appendConsoleText(`${text}\n`);
      return;
    }
    const terminal = document.getElementById("console-terminal");
    if (!terminal) {
      return;
    }
    terminal.value += `${text}\n`;
    terminal.scrollTop = terminal.scrollHeight;
  }
}
