"""Command topic: receive roll/pitch/yaw/throttle and command the motors via Betaflight.

Subscribes <ns>/cmd_attitude (geometry_msgs/Twist):
    angular.x roll [rad], angular.y pitch [rad], angular.z yaw rate [rad/s], linear.z throttle [0..1]
Converts those SI units to Betaflight stick values (sticks_from_command) and streams them over
MSP, which drives the motors -- the same conversion and the same safety gate as rb3_control.

SAFETY: sticks are sent ONLY while the radio has ARMED the drone AND the MSP OVERRIDE switch is
on (and has been seen off since start-up) AND the radio link is up AND a command is fresh. Any of
those failing stops the stream and Betaflight is back on the radio in ~100 ms. This never arms.
"""
import math
import time

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from rb3_control import msp as msp_proto
from rb3_control.control import sticks_from_command


class CommandTopic:
    def __init__(self, node, ns="/drone", port="/dev/ttyACM0", baud=115200,
                 rc_rate_hz=50.0, loop_hz=20.0, command_timeout=0.5,
                 max_tilt_deg=15.0, max_yaw_rate_deg=120.0,
                 throttle_min=1000, throttle_max=2000,
                 require_armed=True, require_override=True):
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

        self.rc = msp_proto.RcStream(self.msp, self.msp.rc_map(), rc_rate_hz,
                                     idle_throttle=throttle_min)
        self.rc.start()

        self.command = None
        self.command_stamp = 0.0
        self.pub_armed = node.create_publisher(Bool, f"{ns}/armed", 10)
        node.create_subscription(Twist, f"{ns}/cmd_attitude", self._on_cmd, 10)
        self.timer = node.create_timer(1.0 / loop_hz, self._loop)

    def _on_cmd(self, m):
        self.command = (m.angular.x, m.angular.y, m.angular.z, m.linear.z)
        self.command_stamp = time.monotonic()

    def _loop(self):
        now = time.monotonic()
        status = self.msp.status()
        rc_values = self.msp.rc()
        if status is None or not rc_values:
            self._release()
            return

        modes, arming_flags, _ = status
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
        if allowed:
            roll, pitch, yaw_rate, throttle = self.command
            self.rc.set_sticks(*sticks_from_command(
                roll, pitch, yaw_rate, throttle, self.max_tilt, self.max_yaw_rate,
                throttle_min=self.throttle_min, throttle_max=self.throttle_max))
            self.rc.paused = False
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

    def _release(self):
        self.rc.set_sticks(1500, 1500, 1500, self.throttle_min)
        self.rc.paused = True

    def stop(self):
        try:
            self.rc.stop()
        finally:
            self.msp.close()
