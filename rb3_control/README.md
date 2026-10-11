# rb3_control

Betaflight bridge for the Qualcomm RB3 Gen 2: MSP over the flight controller's **USB** port
(`/dev/ttyACM0`), exposed as ROS 2 topics. Replaces the UART/MAVLink test code in
`genesis_drones/utils/mavlink_rc.py`, which only read RC channels from a 2 Mbaud UART.

```
ROS 2 topics  <->  flight_controller node  <->  MSP over USB  <->  Betaflight  <->  motors
                            ^                         ^
         RB3 IMU/baro/mag --+              radio (arm/disarm + takeover)
         (sensor core, see.py)
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
| `status` | `std_msgs/String` | JSON: mode, switches, arming flags, sticks sent vs used, sensor stream health |
| `imu` | `sensor_msgs/Imu` | **RB3's own** IMU: gyro [rad/s] + accel [m/s²] |
| `mag` | `sensor_msgs/MagneticField` | RB3's own magnetometer [T] |
| `pressure` | `sensor_msgs/FluidPressure` | RB3's own barometer [Pa] |
| `baro_altitude` | `std_msgs/Float32` | altitude [m] above the start-up pressure level |

The last four come from the RB3's **own** sensors (ICM-42688-P, ICP-10111, AKM) through
Qualcomm's sensor core, not from the flight controller, so they keep publishing while the radio
has control. They are independent of MSP - see `docs/SENSORS.md`, and note the warning there
about *not* trying to reach them through the device tree.

`cmd_attitude` is the direct path. `cmd_vel` runs the ported velocity PID from
`swarm/droneClasses/dronePositionCTRL.py` (`control.VelocityController`) and only takes
effect while fresh odometry is arriving.

## Measured rates

With the flight controller on USB and all four RB3 sensors streaming (`control.launch.py`
defaults, props off):

| Topic | Rate | Set by |
| --- | --- | --- |
| `imu` | 198 Hz | `see_gyro_rate_hz` |
| `mag` | 50 Hz | `see_mag_rate_hz` |
| `pressure`, `baro_altitude` | 25 Hz | `see_pressure_rate_hz` (sensor maximum) |
| `status`, `armed` | 13 Hz | the control loop |
| `attitude`, `altitude` | 6.7 Hz | `telemetry_poll_hz`, decimated from the loop |
| `battery` | 1.3 Hz | `battery_poll_hz` |

The control loop does not reach its configured 20 Hz: an MSP request to Betaflight over USB
CDC costs about 28 ms round trip under load, and the loop makes two of them (arm state and RC
channels) before it can decide whether to send sticks. That puts the ceiling near 18 Hz, so
`telemetry_rate_hz: 20.0` self-limits to ~13 Hz once the decimated telemetry reads are mixed
in. Handover to the radio therefore takes about one 75 ms cycle, which is what the safety
model above assumes.

Two things follow from this, both already in the code:

* **The RB3 sensors get their own callback group** and the node runs on a
  `MultiThreadedExecutor`. Sharing a single-threaded executor with the blocking MSP reads cost
  the IMU 20 % of its samples (159 Hz instead of 200) and dragged `attitude` down to 7 Hz.
* **The MSP reads are decimated inside one timer** rather than split across several timers in
  one mutually-exclusive callback group. rclpy runs whichever timer is ready instead of
  round-robin, and the 20 Hz loop is always ready, which starved `attitude` to 1.3 Hz.

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

For the RB3's own sensors:

```bash
sudo apt install qcom-sensors-test-apps     # see_workhorse; see docs/SENSORS.md
```

## Use

```bash
ros2 run rb3_control fc_check -- --watch        # read-only preflight + live view
ros2 run rb3_control sensors                   # RB3 IMU/baro/mag only, no FC needed
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
`command_timeout` (how quickly a silent publisher hands control back to the radio). The
`see_*` parameters set the RB3 sensor rates; `see_enable: false` turns those streams off.
