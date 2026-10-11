# drone_gui — control centre for the RB3 drone

A web panel that runs **on your machine** (WSL, or any Linux box), not on the drone. It shows
what the drone is doing, starts and stops the processes on the board, and keeps a log of what
happened while you were not watching.

```
 your machine (WSL)                                  drone (RB3 Gen 2)
 ┌──────────────────────────────┐                   ┌────────────────────────────────┐
 │ browser  ◄── SSE + MJPEG ──  │ ── ssh ─────────► │ agent.py (piped in, not        │
 │          http://localhost:8080│    telemetry      │   installed) ── ROS 2 topics   │
 │ drone_gui (stdlib only)      │ ── ssh ─────────► │ systemctl start/stop/restart   │
 └──────────────────────────────┘    control        └────────────────────────────────┘
```

* **No installation anywhere.** Python standard library only on this side; on the board,
  `agent.py` is piped into `python3 -u -` over SSH, so the copy that runs is always this repo's.
* **Nothing to keep alive.** Close the panel, reboot the board, restart the flight controller:
  the link reconnects and the page picks the data back up.
* **It never flies the drone.** No arm, no stick commands, no parameter writes. It starts and
  stops services and reads telemetry. The radio keeps arm/disarm exactly as before.

## Run it

```bash
cd drone_gui
./run.sh                 # then open http://localhost:8080
./run.sh --open          # ...and open a browser for you
```

`run.sh` sources ROS only for the optional DDS transport; the default SSH transport does not
need ROS on this machine at all.

First time, install the two units the panel controls (once, on the board):

```bash
./service/install_board_units.sh ubuntu@192.168.1.243
```

| Option | Default | |
| --- | --- | --- |
| `--host` | `192.168.1.243` | the board |
| `--port` / `--bind` | `8080` / `127.0.0.1` | loopback on purpose, see Security |
| `--domain-id` | `42` | must match the board |
| `--ssh-key` | `~/.ssh/rb3_claude` | needs to be readable with mode 600 |
| `--transport` | `ssh` | or `dds`, see below |

Environment variables work too (`DRONE_HOST`, `DRONE_GUI_PORT`, `DRONE_GUI_BIND`, …), see
`drone_gui/config.py`.

## What it shows

| Panel | |
| --- | --- |
| Flight controller | who has control (radio or RB3), armed state, arming flags, roll/pitch/yaw, both altitudes, battery, and the sticks the RB3 sent next to the ones the FC used |
| Stereo preview | the `/stereo/preview/compressed` JPEG stream as MJPEG |
| History | 2 minutes of altitude, attitude and battery |
| Processes | every unit's state, with start / stop / restart and a journal viewer |
| Board | uptime, load, temperature, memory, disk, whether `/dev/ttyACM0` exists and **which PID holds it**, and how many of the 4 sensor-core streams are alive |
| Event log | arm/disarm, node restarts, board reboots, unit state changes, every action taken from the panel — appended to `~/.local/share/drone_gui/events.jsonl`, so it survives restarts of the panel itself |

## Why the telemetry goes over SSH

The obvious design is to subscribe to the drone's ROS topics directly. That does not work from
WSL, and the failure is deceptive: discovery half-completes, so `/drone/...` topic names appear
in the graph and everything looks fine, but `count_publishers()` stays 0 and not one sample
arrives. The reason is that WSL's address is not reachable from the LAN — **the board cannot
even ping it** — so the drone has nowhere to send the data.

Fixing that means switching WSL to mirrored networking and opening Windows firewall rules.
Running the subscriber on the board instead avoids the whole problem, and the result comes back
over the SSH connection that already works. Telemetry is ~10 Hz JSON and the preview is
base64 JPEG, together well under a megabit.

If you run the panel somewhere the drone *can* reach — on the board, or a native Linux machine
on the same LAN — `--transport dds` subscribes directly instead.

## Security

The panel binds to **loopback** by default. It can start and stop processes and reboot the
board, and WSL2 forwards localhost to Windows, so `http://localhost:8080` works from the
Windows browser without exposing any of that. `--bind 0.0.0.0` makes it reachable from other
machines, with no authentication of any kind — only on a network you trust.

## Layout

```
run.sh                        start the panel
service/drone-control.service flight controller bridge, as a supervised unit
service/drone-stereo.service  stereo camera node
service/install_board_units.sh
drone_gui/agent.py            runs ON THE BOARD, piped in over SSH, emits JSON lines
drone_gui/ssh_link.py         default transport: spawns the agent, parses it, reconnects
drone_gui/ros_link.py         alternative transport: subscribe to DDS directly
drone_gui/store.py            latest values + plot history, shared by both transports
drone_gui/board.py            systemctl and health probes over SSH
drone_gui/events.py           the on-disk event log
drone_gui/server.py           HTTP: SSE for state, MJPEG for the camera, POST for actions
drone_gui/web/                the page (hand-written; no framework, no build step)
```

## Why the flight controller runs under systemd

Only one process may hold `/dev/ttyACM0`. A second opener gets corrupted MSP replies and both
die — an easy mistake to make when you can start the node from a panel as well as a terminal.
`drone-control.service` makes systemd the single owner, and the panel's buttons go through it.

The service does not take the drone over on a restart: the node still requires the MSP OVERRIDE
switch to go off and on again before it sends anything.

## Troubleshooting

| Symptom | |
| --- | --- |
| `telemetry no data`, board online | `drone-control` is not running — start it from the Processes table |
| `telemetry no data`, board unreachable | the board is off, rebooting, or on another address |
| link flaps | check the SSH key is mode 600 (a key on `/mnt/c` is 0777 and ssh refuses it) |
| `no frames` in the camera panel | `drone-stereo` is not running, or camss has no media device (see `rb3_stereo`) |
| port `8080` in use | an older panel is still running: `ss -ltnp \| grep 8080` |
| `ros2 topic list` empty in WSL | unrelated to this panel: the ros2 CLI daemon is broken there, use `--no-daemon` |
