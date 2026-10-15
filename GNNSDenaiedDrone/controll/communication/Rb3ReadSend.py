import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


class ReaderSender:
    def __init__(self, node=None, ns="/drone"):
        self._owns_node = node is None
        if self._owns_node and not rclpy.ok():
            rclpy.init()
        self.node = node if node is not None else rclpy.create_node("reader")
        ns = ns.rstrip("/")

        # The RB3 IMU is z-DOWN (reads gravity on -z -> estimator saw roll = -pi and position
        # exploded on any tilt). Flip 180 deg about x (negate y and z) to the estimator's z-UP,
        # right-handed frame, so gravity reads +z and attitude is ~0 when level.
        # If horizontal x/y or yaw come out mirrored, adjust these signs (keep the product +1 to
        # stay right-handed).
        self._accel_sign = np.array([1.0, -1.0, -1.0])
        self._gyro_sign = np.array([1.0, -1.0, -1.0])

        # latest received values, and the latched snapshot the getters return
        self._accel = np.zeros(3)
        self._gyro = np.zeros(3)
        self._baro = 0.0
        self._accel_s = np.zeros(3)
        self._gyro_s = np.zeros(3)
        self._baro_s = 0.0
        self._imu_timestamp = None
        self._imu_timestamp_s = None
        self._baro_timestamp_ns = None
        self._baro_timestamp_ns_s = None
        self._fc_altitude = None
        self._fc_altitude_s = None
        self._fc_altitude_timestamp_ns = None
        self._fc_altitude_timestamp_ns_s = None

        self.node.create_subscription(Imu, f"{ns}/imu", self._on_imu, qos_profile_sensor_data)
        self.node.create_subscription(
            PointStamped,
            f"{ns}/baro_altitude_stamped",
            self._on_baro,
            qos_profile_sensor_data,
        )
        self.node.create_subscription(
            PointStamped,
            f"{ns}/altitude_stamped",
            self._on_fc_altitude,
            qos_profile_sensor_data,
        )
        self._cmd_pub = self.node.create_publisher(Twist, f"{ns}/cmd_attitude", 10)

    # --- listeners: keep the latest value ---
    def _on_imu(self, m):
        self._accel = np.array([m.linear_acceleration.x, m.linear_acceleration.y,
                                m.linear_acceleration.z]) * self._accel_sign
        self._gyro = np.array([m.angular_velocity.x, m.angular_velocity.y,
                               m.angular_velocity.z]) * self._gyro_sign
        stamp = m.header.stamp
        self._imu_timestamp = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _on_baro(self, m):
        self._baro = float(m.point.z)
        stamp = m.header.stamp
        self._baro_timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _on_fc_altitude(self, m):
        self._fc_altitude = float(m.point.z)
        stamp = m.header.stamp
        self._fc_altitude_timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    # --- the one call main needs: spin + latch ---
    def step(self, timeout_sec=0.0):
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)
        self.read()

    def read(self):
        self._accel_s = self._accel.copy()
        self._gyro_s = self._gyro.copy()
        self._baro_s = self._baro
        self._imu_timestamp_s = self._imu_timestamp
        self._baro_timestamp_ns_s = self._baro_timestamp_ns
        self._fc_altitude_s = self._fc_altitude
        self._fc_altitude_timestamp_ns_s = self._fc_altitude_timestamp_ns

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

    def get_imu_timestamp_ns(self):
        """Return the ROS timestamp in nanoseconds for the latched IMU reading."""
        return self._imu_timestamp_s

    def get_imu_timestamp(self):
        """Backward-compatible alias for :meth:`get_imu_timestamp_ns`."""
        timestamp_ns = self.get_imu_timestamp_ns()
        return None if timestamp_ns is None else timestamp_ns * 1e-9

    def get_baro_timestamp_ns(self):
        """Return the ROS timestamp in nanoseconds for the latched barometer reading."""
        return self._baro_timestamp_ns_s

    def get_fc_altitude(self):
        """Return the latched flight-controller altitude, if one has arrived."""
        return self._fc_altitude_s

    def get_fc_altitude_timestamp_ns(self):
        """Return the ROS timestamp in nanoseconds for the latched FC altitude."""
        return self._fc_altitude_timestamp_ns_s

    def close(self):
        if self._owns_node:
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
