"""Command topic: receive roll/pitch/yaw/throttle and command the motors via Betaflight.

Subscribes <ns>/cmd_attitude (geometry_msgs/Twist):
    angular.x roll [rad], angular.y pitch [rad], angular.z yaw rate [rad/s], linear.z throttle [0..1]
Converts those SI units to Betaflight stick values (sticks_from_command) and streams them over
MSP, which drives the motors -- the same conversion and the same safety gate as rb3_control.

SAFETY: sticks are sent ONLY while the radio has ARMED the drone AND the MSP OVERRIDE switch is
on (and has been seen off since start-up) AND the radio link is up AND a command is fresh. Any of
those failing stops the stream and Betaflight is back on the radio in ~100 ms. This never arms.

Also publishes the flight controller's telemetry, under the same names and types as rb3_control,
so drone_gui and the control loop see the same topics whichever of the two owns the FC:
    <ns>/armed             std_msgs/Bool
    <ns>/attitude          geometry_msgs/Vector3Stamped   roll/pitch/yaw [rad]
    <ns>/altitude          std_msgs/Float32               baro+IMU altitude [m]
    <ns>/altitude_stamped  geometry_msgs/PointStamped     same altitude in point.z, with ROS time
    <ns>/battery           sensor_msgs/BatteryState
    <ns>/status            std_msgs/String (JSON)         gate state, switches, arming flags, sticks
"""
import json
import math
import threading
import time

from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32, String

from rb3_control import msp as msp_proto
from rb3_control.control import sticks_from_command


class CommandTopic:
    def __init__(self, node, ns="/drone", port="/dev/ttyACM0", baud=115200,
                 rc_rate_hz=100.0, loop_hz=20.0, command_timeout=0.2,
                 max_tilt_deg=15.0, max_yaw_rate_deg=120.0,
                 throttle_min=1000, throttle_max=2000,
                 require_armed=True, require_override=True,
                 telemetry_poll_hz=10.0, battery_poll_hz=2.0, sensors_info=None):
        self.node = node
        self.command_timeout = command_timeout
        self.require_armed = require_armed
        self.throttle_min, self.throttle_max = throttle_min, throttle_max
        self.max_tilt = math.radians(max_tilt_deg)
        self.max_yaw_rate = math.radians(max_yaw_rate_deg)

        self.msp = msp_proto.Msp(port, baud)
        if self.msp.identity() is None:
            raise RuntimeError(f"no MSP answer on {port}")
        self.arm_bit = self.msp.box_bit(msp_proto.BOX_ARM)
        self.override = self.msp.mode_range(msp_proto.BOX_MSP_OVERRIDE)
        if self.arm_bit is None:
            raise RuntimeError("ARM is not assigned to a radio switch")
        self.require_override = require_override
        self.override_seen_off = not require_override
        # Silent-block guard: if OVERRIDE is required but not assigned in Betaflight, the gate can
        # never open -- say so loudly instead of blocking forever with no explanation.
        if require_override and self.override is None:
            node.get_logger().warn(
                "MSP OVERRIDE is required but NOT assigned to a radio switch in Betaflight -- "
                "commands will NEVER be sent. Assign it in the Modes tab, or pass "
                "require_override=False (bench, props off).")
        self._last_reason = None
        self._allowed = False            # set by the gate loop; lets _on_cmd push immediately

        self.rc = msp_proto.RcStream(self.msp, self.msp.rc_map(), rc_rate_hz,
                                     idle_throttle=throttle_min)
        self.rc.start()

        self.command = None
        self.command_stamp = 0.0
        self.pub_armed = node.create_publisher(Bool, f"{ns}/armed", 10)
        self.pub_attitude = node.create_publisher(Vector3Stamped, f"{ns}/attitude", 10)
        self.pub_altitude = node.create_publisher(Float32, f"{ns}/altitude", 10)
        self.pub_altitude_stamped = node.create_publisher(
            PointStamped, f"{ns}/altitude_stamped", qos_profile_sensor_data)
        self.pub_battery = node.create_publisher(BatteryState, f"{ns}/battery", 10)
        self.pub_status = node.create_publisher(String, f"{ns}/status", 10)

        # Telemetry is decimated like rb3_control's: each MSP request is a blocking ~19 ms round
        # trip, so attitude/altitude/battery cannot all be read on every gate cycle.
        self.telemetry_every = max(1, round(loop_hz / max(telemetry_poll_hz, 0.1)))
        self.battery_every = max(1, round(loop_hz / max(battery_poll_hz, 0.1)))
        self.cycle = 0
        self.last_attitude = self.last_altitude = self.last_battery = None
        # Callable returning the RB3 sensor-core stream info for the status blob (start_rb3 wires
        # it to ImuTopic/BaroTopic); the panel's "RB3 sensors" box reads it from there.
        self.sensors_info = sensors_info

        # Keep the depth-10 default QoS (matches the publisher; proven working). The gate thread
        # below -- not the QoS -- is what fixes the full/stop pulsing.
        node.create_subscription(Twist, f"{ns}/cmd_attitude", self._on_cmd, 10)

        # The gate loop makes two BLOCKING MSP round-trips (status + RC, ~28 ms each). It runs in
        # its own thread, like RcStream, so it never blocks the executor: if it starved _on_cmd,
        # commands would queue and go stale, the freshness check would trip, and the stream would
        # pause then resume -> motors pulse full/stop periodically. With nothing blocking in a
        # callback, start_rb3 can spin a single-threaded executor.
        self.error = None                # set if the gate loop died; start_rb3 re-raises it
        self._period = 1.0 / loop_hz
        self._running = True
        self._gate = threading.Thread(target=self._gate_loop, daemon=True, name="msp-gate")
        self._gate.start()

    def _on_cmd(self, m):
        self.command = (m.angular.x, m.angular.y, m.angular.z, m.linear.z)
        self.command_stamp = time.monotonic()
        # Low latency: push to the FC the instant a command arrives (100 Hz), instead of waiting
        # for the slow gate loop. The gate loop (safety reads, MSP-limited ~18 Hz) only decides
        # whether we are _allowed; it does not bottleneck the command itself.
        if self._allowed:
            self._apply_command()

    def _apply_command(self):
        roll, pitch, yaw_rate, throttle = self.command
        self.rc.set_sticks(*sticks_from_command(
            roll, pitch, yaw_rate, throttle, self.max_tilt, self.max_yaw_rate,
            throttle_min=self.throttle_min, throttle_max=self.throttle_max))
        self.rc.paused = False

    def _gate_loop(self):
        """Run _loop at loop_hz until stop() or shutdown. If it raises (e.g. the FC re-enumerated
        on USB), shut the node down so start_rb3 exits non-zero and drone-rb3.service restarts it,
        as it did when this was a timer callback. The sticks are released either way."""
        next_tick = time.monotonic()
        try:
            while self._running and self.node.context.ok():
                self._loop()
                next_tick = max(next_tick + self._period, time.monotonic())
                time.sleep(max(0.0, next_tick - time.monotonic()))
        except Exception as exc:                           # noqa: BLE001 - reported via self.error
            # A shutdown already under way also makes the publishers raise; that is not a failure.
            if self._running and self.node.context.ok():
                self.error = exc
                self.node.get_logger().error("command gate failed: %r" % (exc,))
                self.node.context.try_shutdown()
        finally:
            self._release()

    def _loop(self):
        now = time.monotonic()
        status = self.msp.status()
        rc_values = self.msp.rc()
        if status is None or not rc_values:
            self._release()
            return

        modes, arming_flags, flag_count = status
        armed = bool(modes & self.arm_bit)
        rx_lost = bool(arming_flags & (msp_proto.ARMING_FAILSAFE | msp_proto.ARMING_RXLOSS))
        override_on = True
        if self.override is not None and len(rc_values) > self.override[0]:
            value = rc_values[self.override[0]]
            override_on = self.override[1] <= value < self.override[2]
            if not override_on:
                self.override_seen_off = True
        self.pub_armed.publish(Bool(data=armed))

        fresh = self.command is not None and (now - self.command_stamp) < self.command_timeout
        allowed = ((not self.require_armed or armed) and not rx_lost and fresh
                   and override_on and self.override_seen_off)
        self._allowed = allowed
        if allowed:
            self._apply_command()              # keep sticks fresh / set them on first allow
            reason = "SENDING sticks to the FC"
        else:
            self._release()
            reason = ("radio link lost" if rx_lost else
                      "not armed (arm from the radio)" if self.require_armed and not armed else
                      "no recent command (is the control loop publishing cmd_attitude?)" if not fresh else
                      "OVERRIDE switch off" if self.override is not None and not override_on else
                      "OVERRIDE not re-armed since start (flip the switch off then on)"
                      if not self.override_seen_off else "blocked")

        # Log only when the reason changes -> shows the gate transitions without spamming.
        if reason != self._last_reason:
            self.node.get_logger().info("command gate: %s" % reason)
            self._last_reason = reason

        self._publish_status(armed, allowed, reason, override_on, arming_flags, flag_count,
                             rc_values)
        # Same thread as status/rc above, so these never overlap another MSP request.
        self.cycle += 1
        if self.cycle % self.telemetry_every == 0:
            self._poll_telemetry()
        if self.cycle % self.battery_every == 0:
            self._poll_battery()

    def _poll_telemetry(self):
        attitude = self.msp.attitude()
        if attitude:
            self.last_attitude = attitude
            msg = Vector3Stamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.vector.x, msg.vector.y, msg.vector.z = attitude
            self.pub_attitude.publish(msg)

        altitude = self.msp.altitude()
        if altitude:
            self.last_altitude = altitude
            self.pub_altitude.publish(Float32(data=float(altitude[0])))
            stamped = PointStamped()
            stamped.header.stamp = self.node.get_clock().now().to_msg()
            stamped.header.frame_id = "base_link"
            stamped.point.z = float(altitude[0])
            self.pub_altitude_stamped.publish(stamped)

    def _poll_battery(self):
        battery = self.msp.battery()
        if not battery:
            return
        self.last_battery = battery
        volts, amps, _mah, _rssi = battery
        msg = BatteryState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.voltage = float(volts)
        msg.current = float(-amps)          # ROS convention: negative = discharging
        msg.charge = msg.capacity = msg.design_capacity = msg.percentage = float("nan")
        msg.present = volts > 1.0
        self.pub_battery.publish(msg)

    def _publish_status(self, armed, allowed, reason, override_on, arming_flags, flag_count,
                        rc_values):
        """The JSON blob drone_gui shows, with the same keys as rb3_control's. No MSP requests
        of its own: telemetry values are whatever the slower polls last read."""
        attitude, altitude, battery = self.last_attitude, self.last_altitude, self.last_battery
        try:
            sensors = self.sensors_info() if self.sensors_info else None
        except Exception:                                  # noqa: BLE001 - status must not die
            sensors = None
        self.pub_status.publish(String(data=json.dumps({
            "armed": armed,
            "in_control": allowed,
            "reason": reason,
            "override_on": override_on,
            "arming_flags": msp_proto.arming_flag_names(arming_flags, flag_count),
            "attitude_rad": list(attitude) if attitude else None,
            "altitude_m": altitude[0] if altitude else None,
            "vario_ms": altitude[1] if altitude else None,
            "battery_v": battery[0] if battery else None,
            "sticks_sent": self.rc.snapshot()[:4],
            "sticks_fc": rc_values[:4],
            "frames_sent": self.rc.sent,
            "rb3_sensors": sensors,
        })))

    def _release(self):
        self._allowed = False
        self.rc.set_sticks(1500, 1500, 1500, self.throttle_min)
        self.rc.paused = True

    def stop(self):
        self._running = False
        self._gate.join(timeout=2.0)       # before closing the port it may be mid-request on
        try:
            self.rc.stop()
        finally:
            self.msp.close()
