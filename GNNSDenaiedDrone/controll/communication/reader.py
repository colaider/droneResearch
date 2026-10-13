"""Reader: listen to the RB3 accelerometer, gyro and barometer, and hand out the latest values.

It subscribes to the topics the rb3 classes publish:
    <ns>/imu            sensor_msgs/Imu    -> accelerometer + gyro
    <ns>/baro_altitude  std_msgs/Float32   -> barometer altitude [m]

The callbacks keep the newest sample. Calling read() latches that newest sample into a snapshot;
the getters then return that snapshot, so one control tick sees one consistent set of values.

    reader = Reader()        # creates and owns its node
    while True:
        reader.step()        # spin + latch (all the plumbing in one call)
        gyr = reader.get_gyro()
        acc = reader.get_accel()
        alt = reader.get_baro()

If you instead run many components on one shared node that you spin yourself, pass that node in
(`Reader(node=my_node)`) and call read() (not step()) once per tick, spinning the node elsewhere.
"""
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32


class Reader:
    def __init__(self, node=None, ns="/drone"):
        # Own a node when none is given, so main does not have to create/spin one.
        self._owns_node = node is None
        if self._owns_node and not rclpy.ok():
            rclpy.init()
        self.node = node if node is not None else rclpy.create_node("reader")
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

    # --- the one call main needs: do all the utility steps ---
    def step(self, timeout_sec=0.0):
        """Process incoming messages, then latch them. Call once per loop; getters read after."""
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)
        self.read()

    def read(self):
        """Latch the latest accel, gyro and barometer values into the snapshot."""
        self._accel_s = self._accel.copy()
        self._gyro_s = self._gyro.copy()
        self._baro_s = self._baro

    def close(self):
        if self._owns_node:
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    def get_accel(self):
        return self._accel_s.copy()      # [ax, ay, az] m/s^2

    def get_gyro(self):
        return self._gyro_s.copy()       # [wx, wy, wz] rad/s

    def get_baro(self):
        return self._baro_s              # altitude [m] above the start-up level
