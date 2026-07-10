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

class TeleopKeyboardEventRouter {
  constructor(socketProvider) {
    this.socketProvider = socketProvider;
    this.routes = new Map();
    this.handleEvent = this.handleEvent.bind(this);
  }

  register(device, codes) {
    const filteredCodes = Array.from(new Set(codes || [])).filter(Boolean);
    if (!device || !filteredCodes.length) {
      return;
    }
    this.routes.set(device, new Set(filteredCodes));
  }

  attach() {
    window.addEventListener("keydown", this.handleEvent, true);
    window.addEventListener("keyup", this.handleEvent, true);
  }

  detach() {
    window.removeEventListener("keydown", this.handleEvent, true);
    window.removeEventListener("keyup", this.handleEvent, true);
  }

  handleEvent(event) {
    if (this.isTextInput(event.target)) {
      return;
    }
    const targetDevices = [];
    for (const [device, codes] of this.routes.entries()) {
      if (codes.has(event.code)) {
        targetDevices.push(device);
      }
    }
    if (!targetDevices.length) {
      return;
    }

    event.preventDefault();
    const socket = this.socketProvider();
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }

    const payload = {
      event_type: event.type,
      key: event.key,
      code: event.code,
      repeat: event.repeat,
      timestamp: Date.now() / 1000
    };
    for (const device of targetDevices) {
      socket.send(JSON.stringify({
        type: "keyboard_event",
        device,
        payload
      }));
    }
  }

  isTextInput(target) {
    if (!target) {
      return false;
    }
    const tagName = target.tagName ? target.tagName.toLowerCase() : "";
    return target.isContentEditable || tagName === "input" || tagName === "textarea" || tagName === "select";
  }
}
