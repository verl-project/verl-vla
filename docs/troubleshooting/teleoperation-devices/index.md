# Teleoperation devices

## USB XR for high-latency networks

When headset-to-computer Wi-Fi latency makes XR control intermittent, Android
Debug Bridge (ADB) reverse port forwarding can carry the existing WebXR
connection over USB:

```text
PICO browser https://localhost:18000
  -> ADB reverse over USB
  -> computer https://localhost:18000
  -> verl-vla WebSocket XR input
```

The browser still uses the existing HTTPS and WebSocket protocol, while its
traffic to the computer no longer depends on Wi-Fi. WebRTC, a VPN, and a remote
service are not required.

Enable USB debugging on the headset, connect it to the computer with a USB data
cable, keep it unlocked, and accept the debugging authorization prompt. Verify
the connection on the computer:

```bash
adb devices -l
```

The device state must be `device`, not `unauthorized` or `offline`. With the
local XR HTTPS service listening on port 18000, create the reverse mapping:

```bash
adb reverse tcp:18000 tcp:18000
```

The first port is the headset-side port and the second is the computer-side
port. Check the active mapping:

```bash
adb reverse --list
```

It should contain `tcp:18000 tcp:18000`. Open
`https://localhost:18000` in the headset browser and enter XR normally. In this
URL, `localhost` refers to the headset; ADB transparently forwards the
connection to port 18000 on the computer.

The reverse mapping is normally lost when the headset is unplugged or rebooted,
or when ADB reconnects. After reconnecting, verify `adb devices -l` and run the
`adb reverse` command again. For an `unauthorized` device, unlock the headset
and accept the debugging prompt. For an `offline` or missing device, reconnect
a USB data-capable cable, then restart ADB:

```bash
adb kill-server
adb start-server
```
