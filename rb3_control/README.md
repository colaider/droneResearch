# rb3_control

Betaflight bridge for the Qualcomm RB3 Gen 2: MSP over the flight controller's **USB** port
(`/dev/ttyACM0`), exposed as ROS 2 topics. Replaces the UART/MAVLink test code in
`genesis_drones/utils/mavlink_rc.py`, which only read RC channels from a 2 Mbaud UART.

```
ROS 2 topics  <->  flight_controller node  <->  MSP over USB  <->  Betaflight  <->  motors
                                                      ^
                                           radio (arm/disarm + takeover)
```

## Safety model

The radio stays in charge. The node streams stick values **only** while all of these hold:

* the radio has **armed** the drone (`require_armed`),
* the **MSP OVERRIDE** switch is on (`require_override`), and it has been seen *off* since
  the node started, so a restart can never seize a flying drone,
* the radio link is up (no `FAILSAFE`/`RXLOSS` arming flag),
* a command arrived within `command_timeout` (0.5 s default).

When any of those stops being true the node simply stops sending, and Betaflight falls back
to the radio within about 100 ms. The node never arms, never disarms, and never writes to the
flight controller's configuration.

## Topics

Namespace `/drone` when started from the launch file.

| Topic | Type | Meaning |
| --- | --- | --- |
| `cmd_attitude` | `geometry_msgs/Twist` | `angular.x` roll [rad], `angular.y` pitch [rad], `angular.z` yaw rate [rad/s], `linear.z` throttle [0..1] |
| `cmd_vel` | `geometry_msgs/Twist` | body-frame velocity [m/s] + yaw rate; needs `odom`, otherwise ignored |
| `odom` | `nav_msgs/Odometry` | velocity feedback for `cmd_vel` (from the visual estimator) |
| `attitude` | `geometry_msgs/Vector3Stamped` | roll/pitch/yaw [rad] from Betaflight |
| `altitude` | `std_msgs/Float32` | baro+IMU altitude [m] |
| `battery` | `sensor_msgs/BatteryState` | voltage, current, mAh |
| `armed` | `std_msgs/Bool` | armed state |
| `status` | `std_msgs/String` | JSON: mode, switches, arming flags, sticks sent vs used |

`cmd_attitude` is the direct path. `cmd_vel` runs the ported velocity PID from
`swarm/droneClasses/dronePositionCTRL.py` (`control.VelocityController`) and only takes
effect while fresh odometry is arriving.

## Install on the RB3

```bash
ln -s ~/droneResearch/rb3_control ~/ros2_ws/src/rb3_control     # same pattern as rb3_stereo
cd ~/ros2_ws && colcon build --packages-select rb3_control --symlink-install
source install/setup.bash
```

The user needs access to the serial port once:

```bash
sudo usermod -aG dialout $USER     # log out and back in
```

If no `/dev/ttyACM*` appears at all, the Renesas USB host controller is missing its firmware.
Copy `renesas_usb_fw.mem` (Renesas uPD720201 firmware package) to `/lib/firmware/` and reboot;
see `docs/USB_SETUP.md`.

## Use

```bash
ros2 run rb3_control fc_check -- --watch        # read-only preflight + live view
ros2 launch rb3_control control.launch.py       # the bridge
ros2 run rb3_control teleop --ros-args -r __ns:=/drone   # keyboard sticks, props off
ros2 topic echo /drone/status
```

Send a command by hand (props off):

```bash
ros2 topic pub -r 20 /drone/cmd_attitude geometry_msgs/Twist \
  '{angular: {x: 0.0, y: 0.05, z: 0.0}, linear: {z: 0.1}}'
```

## Bring-up order

1. `fc_check` with props off: firmware, link latency, ARM and MSP OVERRIDE switches, arming flags.
2. `fc_check --watch` while flipping the switches: confirm the FC sees them where expected.
3. Props off, battery on, radio armed, override on: `teleop` and watch `sticks_fc` in `/drone/status`
   follow `sticks_sent`.
4. Only then fly, with a finger on the override switch.

## Parameters

See `config/control.yaml`. The ones that matter most: `max_tilt_deg` (how much tilt full stick
means), `hover_throttle` (measure it in a manual hover before using `cmd_vel`), and
`command_timeout` (how quickly a silent publisher hands control back to the radio).
