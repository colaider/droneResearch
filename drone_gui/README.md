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

The default (SSH) transport needs **Python 3 and an ssh client. That is all** - no ROS, no WSL,
no pip install. Windows has both already, so either of these works:

```cmd
run.cmd                  :: Windows, then open http://localhost:8080
run.cmd --open           :: ...and open a browser for you
```

```bash
./run.sh                 # Linux / WSL
./run.sh --open
```

`run.sh` sources ROS, but only because the optional `--transport dds` needs it. Running from
Windows is the simpler choice: the panel talks to the board over SSH either way, and the board
reaches the Windows host directly (it cannot reach WSL at all - see below).

### First run on a new machine

```bat
run.cmd --login
```

```bash
./run.sh --login
```

It creates `~/.ssh/rb3_drone` if this machine has no usable key, installs the public half on
the board, and checks that key login and passwordless sudo work, then starts the panel.
**The board asks for its password and ssh reads it directly** - the script never sees it, so
it is not in Python memory, in argv or in any log. Running it again is harmless: it says the
key already works and changes nothing.

Without `--login` the panel picks up the first of `~/.ssh/rb3_drone`, `rb3_claude`,
`id_ed25519` or `id_rsa` that exists, or use `--ssh-key PATH`.

If the board does not allow password logins at all, add the key from a machine that already
has access:

```bash
cat ~/.ssh/rb3_drone.pub | ssh ubuntu@192.168.0.172 'cat >> ~/.ssh/authorized_keys'
```

First time, install the units the panel controls (once, on the board):

```bash
./service/install_board_units.sh ubuntu@192.168.0.172
```

| Option | Default | |
| --- | --- | --- |
| `--host` | `192.168.0.172` | the board |
| `--port` / `--bind` | `8080` / `127.0.0.1` | loopback on purpose, see Security |
| `--domain-id` | `42` | must match the board |
| `--ssh-key` | first key found | `rb3_drone`, `rb3_claude`, `id_ed25519`, `id_rsa`; mode 600 |
| `--login` | off | create and install a key on first run, see above |
| `--transport` | `ssh` | or `dds`, see below |

Environment variables work too (`DRONE_HOST`, `DRONE_GUI_PORT`, `DRONE_GUI_BIND`, …), see
`drone_gui/config.py`.

## What it shows

| Panel | |
| --- | --- |
| Flight controller | who has control (radio or RB3), armed state, arming flags, roll/pitch/yaw, both altitudes, battery, and the sticks the RB3 sent next to the ones the FC used |
| RB3 sensors | the board's own IMU, barometer and magnetometer: gyro, accel, mag with field magnitude, pressure and baro altitude, each with its stream rate, sample count and restart count. The sensors are on the sensor core, but `rb3_control` publishes them, so they read "not streaming" while `drone-control` is down |
| Position | x/y/z and velocity, each labelled with where it came from. The flight controller supplies height only - Betaflight has no X/Y without GPS and none is fitted - so x/y has to come from the stereo estimator |
| Path | the position over time, from above (x/y, coloured by height) and in 3D (drag to turn, scroll to zoom), with the drone drawn as a box turned by the FC's roll/pitch/yaw. One track per flight run, saved to `~/.local/share/drone_gui/tracks/`; the list replays earlier runs, `new track` starts a fresh one |
| Position and velocity from camera | the camera's own estimate over the same run as the path: x/y/z and vx/vy/vz (world frame) from `visual_position` / `visual_velocity` on `/vo/status`. A gap means the camera had no measurement on that axis (vz without depth velocity) |
| Cameras | up to 4 views, each showing any image topic on the board (`CompressedImage` as published, raw `Image` encoded to JPEG on the board), e.g. `/stereo/preview/compressed` next to `/stereoVisualReturn/processed/compressed`. Only topics a view shows are subscribed and sent over the link |
| History | 2 minutes of altitude, attitude and battery |
| Processes | every unit's state, with start / stop / restart and a journal viewer |
| Board | uptime, load, temperature, memory, disk, whether `/dev/ttyACM0` exists and **which PID holds it**, and how many of the 4 sensor-core streams are alive |
| Event log | arm/disarm, node restarts, board reboots, unit state changes, every action taken from the panel — appended to `~/.local/share/drone_gui/events.jsonl`, so it survives restarts of the panel itself |

## start_rb3 and the control loop

Two more units run the `GNNSDenaiedDrone` code from the board's checkout at `~/droneResearch`
(`git pull` there first; `install_board_units.sh` installs both):

| Unit | Runs | |
| --- | --- | --- |
| `drone-rb3` | `rb3/start_rb3.py` | IMU, baro, stereo cameras and the FC bridge on one node. Publishes the same `/drone/...` FC topics as `rb3_control` (attitude, altitude, altitude_stamped, battery, armed, status), so the Flight controller and RB3 sensors boxes work with either. It holds `/dev/ttyACM0` and the cameras, so it `Conflicts=` with `drone-control`, `drone-stereo` and `drone-vo`: starting it stops them. Restarts itself after a crash |
| `drone-gnss-control` | `controll/main.py` | The control loop. `BindsTo=drone-rb3`: starting it starts `drone-rb3` first, and `drone-rb3` stopping stops it. **It publishes `/drone/cmd_attitude`, so it can command the motors** once the radio has armed the drone and the MSP OVERRIDE switch is on. Never restarted automatically, because it calibrates assuming the drone is resting |

The page asks for confirmation before starting or restarting either one, and says why. Neither
is enabled at boot; `sudo systemctl enable drone-rb3` on the board changes that.

## Position, and what is missing for it

There is no X/Y from the flight controller: Betaflight only knows position with GPS, and none
is fitted (no UART carries the GPS function). What exists today:

| | source | |
| --- | --- | --- |
| height | flight controller | fused baro+IMU, published as `altitude`; the RB3's own barometer is a second, independent estimate |
| velocity x/y | stereo estimator | `/vo/status` carries `vx`, `vy` and `yaw_rate` from `VisulaAcEst` |
| position x/y | **nothing** | the estimator reports velocity, not position |

So the panel shows height and velocity, and says plainly that x/y has no source. Getting a
position out needs the velocity integrated - `swarm/filters/FKF.py` is the filter meant for
it - and published as `nav_msgs/Odometry` on `/drone/odom`. The panel already subscribes to
that topic and `rb3_control` already consumes it for the `cmd_vel` path, so the day something
publishes it, both light up with no further changes.

`drone-vo` and `drone-stereo` cannot run together: both open the cameras directly through
`rb3_stereo.stereo.StereoCamera`, so the units declare `Conflicts=` and systemd stops one when
you start the other. The camera panel shows whichever is publishing - the VO preview is the
one with the tracking overlay.

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

If you run the panel somewhere the drone *can* reach — on the board, on a native Linux machine
on the same LAN, or on Windows itself — `--transport dds` subscribes directly instead. On
Windows that also means installing ROS 2 Jazzy (a binary archive; Windows 10 + VS 2019 is a
Tier 1 platform) and letting its UDP through the firewall. For this panel there is no reason
to: the SSH transport needs neither.

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
drone_gui/tracks.py           position tracks, one file per flight run
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
| `no frames` in the camera panel | `drone-stereo` is not running (start it from Processes), or camss registered no media device - check `ls /dev/media*` for one whose driver is `qcom-camss`, see `drivers/devicetree/camss-disable-imx577.dtso` |
| port `8080` in use | an older panel is still running; the error message says how to find it |
| `ros2 topic list` empty in WSL | unrelated to this panel: the ros2 CLI daemon is broken there, use `--no-daemon` |
