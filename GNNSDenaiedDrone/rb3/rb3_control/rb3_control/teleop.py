"""Keyboard teleop: publishes geometry_msgs/Twist on cmd_attitude for bench testing.

    ros2 run rb3_control teleop --ros-args -r __ns:=/drone

Keys (hold-to-move; everything re-centres when you let go for command_timeout):
    w / s   pitch forward / back        i / k   throttle up / down
    a / d   roll left / right           space   throttle to zero, sticks centred
    q / e   yaw left / right            x       quit

PROPS OFF until you have watched the stick values arrive in `fc_check --watch`.
The flight controller only acts on these while the radio has armed it and the MSP OVERRIDE
switch is on, so this cannot start the motors by itself.
"""
from __future__ import annotations

import math
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

STEP_TILT = math.radians(2.0)
STEP_YAW = math.radians(15.0)
STEP_THROTTLE = 0.02
LIMIT_TILT = math.radians(15.0)
LIMIT_YAW = math.radians(120.0)


class Teleop(Node):
    def __init__(self):
        super().__init__("teleop")
        self.pub = self.create_publisher(Twist, "cmd_attitude", 10)
        self.roll = self.pitch = self.yaw_rate = 0.0
        self.throttle = 0.0
        self.create_timer(0.05, self.tick)          # 20 Hz, inside the node command_timeout

    def key(self):
        if select.select([sys.stdin], [], [], 0.0)[0]:
            return sys.stdin.read(1)
        return None

    def tick(self):
        key = self.key()
        if key == "w":
            self.pitch = min(self.pitch + STEP_TILT, LIMIT_TILT)
        elif key == "s":
            self.pitch = max(self.pitch - STEP_TILT, -LIMIT_TILT)
        elif key == "d":
            self.roll = min(self.roll + STEP_TILT, LIMIT_TILT)
        elif key == "a":
            self.roll = max(self.roll - STEP_TILT, -LIMIT_TILT)
        elif key == "e":
            self.yaw_rate = min(self.yaw_rate + STEP_YAW, LIMIT_YAW)
        elif key == "q":
            self.yaw_rate = max(self.yaw_rate - STEP_YAW, -LIMIT_YAW)
        elif key == "i":
            self.throttle = min(self.throttle + STEP_THROTTLE, 1.0)
        elif key == "k":
            self.throttle = max(self.throttle - STEP_THROTTLE, 0.0)
        elif key == " ":
            self.roll = self.pitch = self.yaw_rate = 0.0
            self.throttle = 0.0
        elif key == "x":
            raise KeyboardInterrupt

        msg = Twist()
        msg.angular.x = self.roll
        msg.angular.y = self.pitch
        msg.angular.z = self.yaw_rate
        msg.linear.z = self.throttle
        self.pub.publish(msg)
        print("\rroll %+5.1f  pitch %+5.1f  yaw %+6.1f deg/s  throttle %4.0f%%   "
              % (math.degrees(self.roll), math.degrees(self.pitch),
                 math.degrees(self.yaw_rate), self.throttle * 100),
              end="", flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = Teleop()
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        print(__doc__)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print()


if __name__ == "__main__":
    main()
