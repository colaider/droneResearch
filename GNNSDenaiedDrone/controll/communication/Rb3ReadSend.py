"""Reader: subscribe to the RB3 IMU + barometer, and publish motor commands.

In:  <ns>/imu (accel + gyro), <ns>/baro_altitude          -> read()/step() + getters
Out: <ns>/cmd_attitude (roll, pitch, yaw rate, throttle)  -> send()
"""
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32


class ReaderSender:
    def __init__(self, node=None, ns="/drone"):
        self._owns_node = node is None
        if self._owns_node and not rclpy.ok():
            rclpy.init()
        self.node = node if node is not None else rclpy.create_node("reader")
        ns = ns.rstrip("/")

        # latest received values, and the latched snapshot the getters return
        self._accel = np.zeros(3)
        self._gyro = np.zeros(3)
        self._baro = 0.0
        self._accel_s = np.zeros(3)
        self._gyro_s = np.zeros(3)
        self._baro_s = 0.0

        self.node.create_subscription(Imu, f"{ns}/imu", self._on_imu, qos_profile_sensor_data)
        self.node.create_subscription(Float32, f"{ns}/baro_altitude", self._on_baro, 10)
        self._cmd_pub = self.node.create_publisher(Twist, f"{ns}/cmd_attitude", 10)

    # --- listeners: keep the latest value ---
    def _on_imu(self, m):
        self._accel = np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        self._gyro = np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])

    def _on_baro(self, m):
        self._baro = float(m.data)

    # --- the one call main needs: spin + latch ---
    def step(self, timeout_sec=0.0):
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)
        self.read()

    def read(self):
        self._accel_s = self._accel.copy()
        self._gyro_s = self._gyro.copy()
        self._baro_s = self._baro

    # --- command out: roll, pitch [rad], yaw rate [rad/s], throttle [0..1] -> motors ---
    def send(self, cmd: np.ndarray):
        msg = Twist()
        msg.angular.x = float(cmd[0])
        msg.angular.y = float(cmd[1])
        msg.angular.z = float(cmd[2])
        msg.linear.z = float(cmd[3])
        self._cmd_pub.publish(msg)

    def get_accel(self):
        return self._accel_s.copy()      # [ax, ay, az] m/s^2

    def get_gyro(self):
        return self._gyro_s.copy()       # [wx, wy, wz] rad/s

    def get_baro(self):
        return self._baro_s              # altitude [m] above the start-up level

    def close(self):
        if self._owns_node:
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
