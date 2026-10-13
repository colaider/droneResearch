"""ROS 2 node: command a Betaflight flight controller over USB (MSP) from the RB3.

Topics (relative to the node namespace, /drone with the launch file):
  SUBSCRIBE
    cmd_attitude   geometry_msgs/Twist   angular.x roll [rad], angular.y pitch [rad],
                                         angular.z yaw rate [rad/s], linear.z throttle [0..1]
    cmd_vel        geometry_msgs/Twist   body-frame velocity [m/s] + angular.z yaw rate;
                                         needs odom for feedback, ignored otherwise
    odom           nav_msgs/Odometry     velocity feedback for cmd_vel (from the visual estimator)
  PUBLISH
    attitude       geometry_msgs/Vector3Stamped   roll/pitch/yaw [rad], from the FC
    altitude       std_msgs/Float32               baro+IMU altitude [m], from the FC
    battery        sensor_msgs/BatteryState       voltage, current, mAh
    armed          std_msgs/Bool                  Betaflight armed state
    status         std_msgs/String (JSON)         full state: mode, switches, arming flags, rates
    imu            sensor_msgs/Imu                RB3 on-board IMU: gyro [rad/s] + accel [m/s^2]
    mag            sensor_msgs/MagneticField      RB3 on-board magnetometer [T]
    pressure       sensor_msgs/FluidPressure      RB3 on-board barometer [Pa]
    baro_altitude  std_msgs/Float32               altitude [m] from the RB3 barometer

The RB3's own sensors come from the sensor core, not from the flight controller - see see.py.
They are independent of the MSP link, so they keep publishing while the radio has control.

SAFETY MODEL (same as the bench-tested setup on the other airframe)
  - The radio arms and disarms. This node never arms.
  - Sticks are streamed only while: armed AND the MSP OVERRIDE switch is ON AND the radio
    link is up AND a command arrived within command_timeout.
  - Any of those failing stops the stream, and Betaflight is back on the radio in ~100 ms.
  - The override switch must go off->on after start-up before the node takes control, so a
    node restart can never grab the drone mid-flight.
"""
from __future__ import annotations

import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import BatteryState, FluidPressure, Imu, MagneticField
from std_msgs.msg import Bool, Float32, String

from . import msp as msp_proto
from . import see
from .control import VelocityController, sticks_from_command


class FlightControllerNode(Node):
    def __init__(self):
        super().__init__("flight_controller")
        p = self.declare_parameters("", [
            ("port", "/dev/ttyACM0"),
            ("baud", 115200),
            ("rc_rate_hz", 50.0),             # stick stream rate
            ("telemetry_rate_hz", 20.0),      # control loop rate (status + RC poll)
            ("telemetry_poll_hz", 10.0),      # attitude/altitude poll rate; see loop()
            ("battery_poll_hz", 2.0),         # battery changes slowly, and each MSP request
                                              # costs a ~19 ms round trip
            ("command_timeout", 0.5),         # stop sending if no command for this long [s]
            ("require_override", True),       # refuse to send unless MSP OVERRIDE is on
            ("require_armed", True),          # refuse to send unless the radio armed it
            ("max_tilt_deg", 15.0),
            ("max_yaw_rate_deg", 120.0),
            ("throttle_min", 1000),
            ("throttle_max", 2000),
            ("hover_throttle", 0.55),         # used by the cmd_vel controller
            ("odom_timeout", 0.5),
            # RB3 on-board sensors through the sensor core (see.py). 0 disables a stream;
            # rates are snapped down to one the sensor supports.
            ("see_enable", True),
            ("see_sudo", True),               # see_workhorse needs root to reach the sensor PD
            ("see_gyro_rate_hz", 200.0),      # drives the imu topic (max 500)
            ("see_accel_rate_hz", 200.0),     # merged into the imu topic (max 500)
            ("see_pressure_rate_hz", 25.0),   # max 25 on the ICP-10111
            ("see_mag_rate_hz", 50.0),        # max 100
            ("see_publish_rate_hz", 50.0),    # how often queued samples are drained
            ("imu_frame", "imu_link"),
            ("baro_reference_hpa", 0.0),      # 0: zero the barometer at start-up (relative
                                              # altitude); otherwise treat it as sea level
        ])
        self.cfg = {entry.name: entry.value for entry in p}
        self.max_tilt = math.radians(self.cfg["max_tilt_deg"])
        self.max_yaw_rate = math.radians(self.cfg["max_yaw_rate_deg"])

        self.msp = msp_proto.Msp(self.cfg["port"], self.cfg["baud"])
        identity = self.msp.identity()
        if identity is None:
            raise RuntimeError("no MSP answer on %s" % self.cfg["port"])
        self.get_logger().info(
            "connected to %(variant)s %(version)s (API %(api)s, board %(board)s) on " % identity
            + self.cfg["port"])

        self.arm = self.msp.mode_range(msp_proto.BOX_ARM)
        self.override = self.msp.mode_range(msp_proto.BOX_MSP_OVERRIDE)
        self.arm_bit = self.msp.box_bit(msp_proto.BOX_ARM)
        if self.arm is None or self.arm_bit is None:
            raise RuntimeError("ARM is not assigned to a radio switch - check the Modes tab")
        if self.override is None and self.cfg["require_override"]:
            raise RuntimeError(
                "MSP OVERRIDE is not assigned to a radio switch. Assign it in the Modes tab and set "
                "msp_override_channels_mask, or start with require_override:=false (UNSAFE)")
        if self.override:
            self.get_logger().info("ARM on AUX%d, MSP OVERRIDE on AUX%d"
                                   % (self.arm[0] - 3, self.override[0] - 3))

        self.rc = msp_proto.RcStream(self.msp, self.msp.rc_map(), self.cfg["rc_rate_hz"],
                                     idle_throttle=self.cfg["throttle_min"])
        self.rc.start()

        self.velocity = VelocityController(
            dt=1.0 / self.cfg["telemetry_rate_hz"],
            max_tilt=self.max_tilt,
            hover_throttle=self.cfg["hover_throttle"])

        self.command = None            # (roll, pitch, yaw_rate, throttle) in SI units
        self.command_stamp = 0.0
        self.cmd_vel = None
        self.cmd_vel_stamp = 0.0
        self.odom_vel = None
        self.odom_stamp = 0.0
        self.override_seen_off = not self.cfg["require_override"]
        self.in_control = False
        self.reason = "waiting for the radio"

        self.create_subscription(Twist, "cmd_attitude", self.on_cmd_attitude, 10)
        self.create_subscription(Twist, "cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(Odometry, "odom", self.on_odom, qos_profile_sensor_data)

        self.pub_attitude = self.create_publisher(Vector3Stamped, "attitude", 10)
        self.pub_altitude = self.create_publisher(Float32, "altitude", 10)
        self.pub_battery = self.create_publisher(BatteryState, "battery", 10)
        self.pub_armed = self.create_publisher(Bool, "armed", 10)
        self.pub_status = self.create_publisher(String, "status", 10)

        self.pub_imu = self.create_publisher(Imu, "imu", qos_profile_sensor_data)
        self.pub_mag = self.create_publisher(MagneticField, "mag", qos_profile_sensor_data)
        self.pub_pressure = self.create_publisher(FluidPressure, "pressure", qos_profile_sensor_data)
        self.pub_baro = self.create_publisher(Float32, "baro_altitude", 10)

        # Everything that talks MSP shares one serial port, so those callbacks must not
        # overlap; the RB3's own sensors are a separate process and get their own group, or
        # a slow MSP round trip would starve them (measured: the IMU lost 20% of its samples
        # and attitude dropped from 20 Hz to 7 Hz when they shared a single-threaded executor).
        self.serial_group = MutuallyExclusiveCallbackGroup()
        self.sensor_group = MutuallyExclusiveCallbackGroup()

        self.last_attitude = None
        self.last_altitude = None
        self.last_battery = None

        # One timer owns the serial port and decimates the slower reads itself. Separate
        # timers in one mutually-exclusive group would be simpler to read, but rclpy picks
        # whichever timer is ready rather than round-robin, and the 20 Hz one is always
        # ready: that starved attitude down to 1.3 Hz.
        self.cycle = 0
        self.telemetry_every = max(1, round(self.cfg["telemetry_rate_hz"]
                                            / max(self.cfg["telemetry_poll_hz"], 0.1)))
        self.battery_every = max(1, round(self.cfg["telemetry_rate_hz"]
                                          / max(self.cfg["battery_poll_hz"], 0.1)))
        self.timer = self.create_timer(1.0 / self.cfg["telemetry_rate_hz"], self.loop,
                                       callback_group=self.serial_group)
        self.sensors = self.start_sensors()

    # --- subscriptions ----------------------------------------------------
    def on_cmd_attitude(self, msg: Twist):
        self.command = (msg.angular.x, msg.angular.y, msg.angular.z, msg.linear.z)
        self.command_stamp = time.monotonic()

    def on_cmd_vel(self, msg: Twist):
        self.cmd_vel = ((msg.linear.x, msg.linear.y, msg.linear.z), msg.angular.z)
        self.cmd_vel_stamp = time.monotonic()

    def on_odom(self, msg: Odometry):
        twist = msg.twist.twist.linear
        self.odom_vel = (twist.x, twist.y, twist.z)
        self.odom_stamp = time.monotonic()

    # --- main loop --------------------------------------------------------
    def loop(self):
        """Safety path: arm state, arming flags and the override switch, then the sticks.

        Only the two MSP requests the decision needs (status + RC) run here; attitude,
        altitude and battery are polled on their own slower timers, because each request is
        a blocking ~19 ms round trip and five of them per cycle cannot fit in 50 ms.
        """
        now = time.monotonic()
        status = self.msp.status()
        rc_values = self.msp.rc()
        if status is None or not rc_values:
            self.release("no MSP telemetry")
            return

        modes, arming_flags, flag_count = status
        armed = bool(modes & self.arm_bit)
        rx_lost = bool(arming_flags & (msp_proto.ARMING_FAILSAFE | msp_proto.ARMING_RXLOSS))
        override_on = True
        if self.override and len(rc_values) > self.override[0]:
            value = rc_values[self.override[0]]
            override_on = self.override[1] <= value < self.override[2]
            if not override_on:
                self.override_seen_off = True

        command = self.resolve_command(now)
        fresh = command is not None and (now - self.command_stamp) < self.cfg["command_timeout"]

        allowed = (not self.cfg["require_armed"] or armed) and not rx_lost and fresh \
            and (override_on and self.override_seen_off)
        if allowed:
            roll, pitch, yaw_rate, throttle = command
            self.rc.set_sticks(*sticks_from_command(
                roll, pitch, yaw_rate, throttle, self.max_tilt, self.max_yaw_rate,
                throttle_min=self.cfg["throttle_min"], throttle_max=self.cfg["throttle_max"]))
            self.rc.paused = False
            self.in_control = True
            self.reason = "sending"
        else:
            self.release(
                "radio link lost" if rx_lost else
                "not armed" if self.cfg["require_armed"] and not armed else
                "override switch off" if not override_on else
                "override not re-armed (flip it off and on)" if not self.override_seen_off else
                "no recent command")

        self.publish_status(arming_flags, flag_count, armed, override_on, rc_values)

        # Telemetry is decimated: each MSP request is a blocking round trip, so polling all
        # of it every cycle would not fit in the loop period.
        self.cycle += 1
        if self.cycle % self.telemetry_every == 0:
            self.poll_telemetry()
        if self.cycle % self.battery_every == 0:
            self.poll_battery()

    def resolve_command(self, now):
        """cmd_attitude wins; cmd_vel is used when it is newer and odom feedback is fresh."""
        attitude_fresh = self.command is not None and (now - self.command_stamp) < self.cfg["command_timeout"]
        vel_fresh = self.cmd_vel is not None and (now - self.cmd_vel_stamp) < self.cfg["command_timeout"]
        odom_fresh = self.odom_vel is not None and (now - self.odom_stamp) < self.cfg["odom_timeout"]

        if vel_fresh and odom_fresh and (not attitude_fresh or self.cmd_vel_stamp > self.command_stamp):
            target, yaw_rate = self.cmd_vel
            out = self.velocity.step(target, self.odom_vel, yaw_rate)
            self.command_stamp = self.cmd_vel_stamp
            self.command = (out[0], out[1], out[2], out[3])
        elif vel_fresh and not odom_fresh:
            self.get_logger().warn("cmd_vel ignored: no odom feedback", throttle_duration_sec=5.0)
        return self.command

    def release(self, reason):
        if not self.rc.paused:
            self.get_logger().warn("control released: %s" % reason)
        self.rc.paused = True
        # centre the sticks we would send, so status does not keep showing a stale command
        self.rc.set_sticks(1500, 1500, 1500, self.cfg["throttle_min"])
        self.in_control = False
        self.reason = reason
        self.velocity.reset()

    # --- telemetry --------------------------------------------------------
    def poll_telemetry(self):
        """Attitude and altitude from the flight controller. Called from loop() only, so it
        never overlaps another MSP request."""
        stamp = self.get_clock().now().to_msg()
        attitude = self.msp.attitude()
        altitude = self.msp.altitude()

        if attitude:
            self.last_attitude = attitude
            msg = Vector3Stamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "base_link"
            msg.vector.x, msg.vector.y, msg.vector.z = attitude
            self.pub_attitude.publish(msg)
        if altitude:
            self.last_altitude = altitude
            self.pub_altitude.publish(Float32(data=float(altitude[0])))

    def poll_battery(self):
        battery = self.msp.battery()
        if not battery:
            return
        self.last_battery = battery
        volts, amps, mah, _rssi = battery
        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.voltage = float(volts)
        msg.current = float(-amps)          # ROS convention: negative = discharging
        msg.charge = float("nan")
        msg.capacity = float("nan")
        msg.design_capacity = float("nan")
        msg.percentage = float("nan")
        msg.present = volts > 1.0
        self.pub_battery.publish(msg)

    def publish_status(self, arming_flags, flag_count, armed, override_on, rc_values):
        """Armed state and the JSON status blob. Makes no MSP requests of its own: the
        telemetry values are whatever the slower timers last read."""
        attitude = self.last_attitude
        altitude = self.last_altitude
        battery = self.last_battery
        self.pub_armed.publish(Bool(data=armed))

        self.pub_status.publish(String(data=json.dumps({
            "armed": armed,
            "in_control": self.in_control,
            "reason": self.reason,
            "override_on": override_on,
            "arming_flags": msp_proto.arming_flag_names(arming_flags, flag_count),
            "attitude_rad": list(attitude) if attitude else None,
            "altitude_m": altitude[0] if altitude else None,
            "vario_ms": altitude[1] if altitude else None,
            "battery_v": battery[0] if battery else None,
            "sticks_sent": self.rc.snapshot()[:4],
            "sticks_fc": rc_values[:4],
            "frames_sent": self.rc.sent,
            "rb3_sensors": self.sensors.info() if self.sensors else None,
        })))

    # --- RB3 on-board sensors (sensor core) -------------------------------
    def start_sensors(self):
        """Start the SEE streams. Missing sensors are a warning, never fatal: the drone is
        still flyable on the radio and the flight controller's own telemetry."""
        if not self.cfg["see_enable"]:
            return None
        if not see.available():
            self.get_logger().warn(
                "see_workhorse not found - RB3 sensor topics disabled "
                "(sudo apt install qcom-sensors-test-apps)")
            return None

        # Host monotonic (what the streams timestamp with) -> ROS clock.
        self.time_offset_ns = self.get_clock().now().nanoseconds - int(time.monotonic() * 1e9)
        self.ssc_offset = None
        self.baro_reference = self.cfg["baro_reference_hpa"] or None
        self.baro_samples = []

        sensors = see.SensorSet({
            "gyro": self.cfg["see_gyro_rate_hz"],
            "accel": self.cfg["see_accel_rate_hz"],
            "pressure": self.cfg["see_pressure_rate_hz"],
            "mag": self.cfg["see_mag_rate_hz"],
        }, sudo=self.cfg["see_sudo"], logger=self.get_logger())
        sensors.start()
        self.create_timer(1.0 / self.cfg["see_publish_rate_hz"], self.pump_sensors,
                          callback_group=self.sensor_group)
        return sensors

    def stamp_for(self, host_time, ssc_time):
        """Timestamp a sample, preferring the sensor core's own clock.

        `host_time` is when the reader thread got the line, so it carries scheduling jitter;
        the SSC timestamp is taken in hardware. Both clocks run off the same SoC, so a single
        offset measured at the first sample keeps the two aligned.
        """
        if ssc_time is not None:
            if self.ssc_offset is None:
                self.ssc_offset = host_time - ssc_time
            host_time = ssc_time + self.ssc_offset
        return Time(nanoseconds=int(host_time * 1e9) + self.time_offset_ns).to_msg()

    def pump_sensors(self):
        """Publish everything the sensor core produced since the last call."""
        gyro = self.sensors.get("gyro")
        accel = self.sensors.get("accel")

        # The IMU topic is driven by the gyro, with the newest accel sample merged in - the
        # two streams are separate processes and never land on the same timestamp.
        if gyro is not None:
            latest_accel = accel.peek() if accel is not None else None
            if accel is not None:
                accel.drain()                  # keep the queue from going stale
            for host, ssc, values in gyro.drain():
                if len(values) < 3:
                    continue
                msg = Imu()
                msg.header.stamp = self.stamp_for(host, ssc)
                msg.header.frame_id = self.cfg["imu_frame"]
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = values[:3]
                if latest_accel is not None and len(latest_accel[2]) >= 3:
                    a = latest_accel[2]
                    msg.linear_acceleration.x = a[0]
                    msg.linear_acceleration.y = a[1]
                    msg.linear_acceleration.z = a[2]
                else:
                    msg.linear_acceleration_covariance[0] = -1.0
                msg.orientation.w = 1.0            # no orientation estimate here
                msg.orientation_covariance[0] = -1.0
                self.pub_imu.publish(msg)
        elif accel is not None:
            accel.drain()

        mag = self.sensors.get("mag")
        if mag is not None:
            for host, ssc, values in mag.drain():
                if len(values) < 3:
                    continue
                msg = MagneticField()
                msg.header.stamp = self.stamp_for(host, ssc)
                msg.header.frame_id = self.cfg["imu_frame"]
                # SEE reports microtesla, ROS wants tesla.
                msg.magnetic_field.x = values[0] * 1e-6
                msg.magnetic_field.y = values[1] * 1e-6
                msg.magnetic_field.z = values[2] * 1e-6
                self.pub_mag.publish(msg)

        pressure = self.sensors.get("pressure")
        if pressure is not None:
            for host, ssc, values in pressure.drain():
                hpa = values[0]
                msg = FluidPressure()
                msg.header.stamp = self.stamp_for(host, ssc)
                msg.header.frame_id = self.cfg["imu_frame"]
                msg.fluid_pressure = hpa * 100.0        # hPa -> Pa
                self.pub_pressure.publish(msg)
                if self.baro_reference is None:
                    # Average the first samples on the ground so baro_altitude starts at 0.
                    self.baro_samples.append(hpa)
                    if len(self.baro_samples) >= 20:
                        self.baro_reference = sum(self.baro_samples) / len(self.baro_samples)
                        self.get_logger().info("barometer zeroed at %.2f hPa" % self.baro_reference)
                    continue
                self.pub_baro.publish(Float32(
                    data=float(see.altitude_from_pressure(hpa, self.baro_reference))))

    def destroy_node(self):
        self.release("node shutting down")
        if self.sensors is not None:
            self.sensors.stop()
        self.rc.stop()
        self.msp.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FlightControllerNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
