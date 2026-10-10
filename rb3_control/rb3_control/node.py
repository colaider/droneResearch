"""ROS 2 node: command a Betaflight flight controller over USB (MSP) from the RB3.

Topics (relative to the node namespace, /drone with the launch file):
  SUBSCRIBE
    cmd_attitude   geometry_msgs/Twist   angular.x roll [rad], angular.y pitch [rad],
                                         angular.z yaw rate [rad/s], linear.z throttle [0..1]
    cmd_vel        geometry_msgs/Twist   body-frame velocity [m/s] + angular.z yaw rate;
                                         needs odom for feedback, ignored otherwise
    odom           nav_msgs/Odometry     velocity feedback for cmd_vel (from the visual estimator)
  PUBLISH
    attitude       geometry_msgs/Vector3Stamped   roll/pitch/yaw [rad]
    altitude       std_msgs/Float32               baro+IMU altitude [m]
    battery        sensor_msgs/BatteryState       voltage, current, mAh
    armed          std_msgs/Bool                  Betaflight armed state
    status         std_msgs/String (JSON)         full state: mode, switches, arming flags, rates

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
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32, String

from . import msp as msp_proto
from .control import VelocityController, sticks_from_command


class FlightControllerNode(Node):
    def __init__(self):
        super().__init__("flight_controller")
        p = self.declare_parameters("", [
            ("port", "/dev/ttyACM0"),
            ("baud", 115200),
            ("rc_rate_hz", 50.0),             # stick stream rate
            ("telemetry_rate_hz", 20.0),      # MSP polling rate
            ("command_timeout", 0.5),         # stop sending if no command for this long [s]
            ("require_override", True),       # refuse to send unless MSP OVERRIDE is on
            ("require_armed", True),          # refuse to send unless the radio armed it
            ("max_tilt_deg", 15.0),
            ("max_yaw_rate_deg", 120.0),
            ("throttle_min", 1000),
            ("throttle_max", 2000),
            ("hover_throttle", 0.55),         # used by the cmd_vel controller
            ("odom_timeout", 0.5),
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

        self.timer = self.create_timer(1.0 / self.cfg["telemetry_rate_hz"], self.loop)

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

        self.publish(modes, arming_flags, flag_count, armed, override_on, rc_values)

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
    def publish(self, modes, arming_flags, flag_count, armed, override_on, rc_values):
        stamp = self.get_clock().now().to_msg()
        attitude = self.msp.attitude()
        altitude = self.msp.altitude()
        battery = self.msp.battery()

        if attitude:
            msg = Vector3Stamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "base_link"
            msg.vector.x, msg.vector.y, msg.vector.z = attitude
            self.pub_attitude.publish(msg)
        if altitude:
            self.pub_altitude.publish(Float32(data=float(altitude[0])))
        if battery:
            volts, amps, mah, _rssi = battery
            msg = BatteryState()
            msg.header.stamp = stamp
            msg.voltage = float(volts)
            msg.current = float(-amps)          # ROS convention: negative = discharging
            msg.charge = float("nan")
            msg.capacity = float("nan")
            msg.design_capacity = float("nan")
            msg.percentage = float("nan")
            msg.present = volts > 1.0
            self.pub_battery.publish(msg)
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
        })))

    def destroy_node(self):
        self.release("node shutting down")
        self.rc.stop()
        self.msp.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FlightControllerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
