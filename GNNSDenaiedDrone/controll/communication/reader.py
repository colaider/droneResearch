"""Reader: listen to the RB3 accelerometer, gyro and barometer, and hand out the latest values.

It subscribes to the topics the rb3 classes publish:
    <ns>/imu            sensor_msgs/Imu    -> accelerometer + gyro
    <ns>/baro_altitude  std_msgs/Float32   -> barometer altitude [m]

The callbacks keep the newest sample. Calling read() latches that newest sample into a snapshot;
the getters then return that snapshot, so one control tick sees one consistent set of values.

    reader = Reader(node)
    ...                      # node must be spinning (spin_once / an executor)
    reader.read()            # sample the latest values now
    acc = reader.get_accel()
    gyr = reader.get_gyro()
    alt = reader.get_baro()
"""
import numpy as np
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32


class Reader:
    def __init__(self, node, ns="/drone"):
        self.node = node
        # newest received values (written by the callbacks)
        self._accel = np.zeros(3)
        self._gyro = np.zeros(3)
        self._baro = 0.0
        # latched snapshot (written by read(), returned by the getters)
        self._accel_s = np.zeros(3)
        self._gyro_s = np.zeros(3)
        self._baro_s = 0.0

        ns = ns.rstrip("/")
        node.create_subscription(Imu, f"{ns}/imu", self._on_imu, qos_profile_sensor_data)
        node.create_subscription(Float32, f"{ns}/baro_altitude", self._on_baro, 10)

    # --- listeners: keep the latest value ---
    def _on_imu(self, m):
        self._accel = np.array([m.linear_acceleration.x,
                                m.linear_acceleration.y,
                                m.linear_acceleration.z])
        self._gyro = np.array([m.angular_velocity.x,
                               m.angular_velocity.y,
                               m.angular_velocity.z])

    def _on_baro(self, m):
        self._baro = float(m.data)

    # --- sample + getters ---
    def read(self):
        """Latch the latest accel, gyro and barometer values into the snapshot."""
        self._accel_s = self._accel.copy()
        self._gyro_s = self._gyro.copy()
        self._baro_s = self._baro

    def get_accel(self):
        return self._accel_s.copy()      # [ax, ay, az] m/s^2

    def get_gyro(self):
        return self._gyro_s.copy()       # [wx, wy, wz] rad/s

    def get_baro(self):
        return self._baro_s              # altitude [m] above the start-up level
