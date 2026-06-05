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

class TeleopKeyboardDevice {
  constructor(socketProvider) {
    this.socketProvider = socketProvider;
    this.handleEvent = this.handleEvent.bind(this);
  }

  attach() {
    window.addEventListener("keydown", this.handleEvent);
    window.addEventListener("keyup", this.handleEvent);
  }

  detach() {
    window.removeEventListener("keydown", this.handleEvent);
    window.removeEventListener("keyup", this.handleEvent);
  }

  handleEvent(event) {
    const socket = this.socketProvider();
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    socket.send(JSON.stringify({
      type: "keyboard_event",
      device: "keyboard",
      payload: {
        event_type: event.type,
        key: event.key,
        code: event.code,
        repeat: event.repeat,
        timestamp: Date.now() / 1000
      }
    }));
  }
}
