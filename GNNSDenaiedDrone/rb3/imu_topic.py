"""Publish the RB3 IMU with SSC capture time mapped through host monotonic to ROS time."""
import time

from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from rb3_control import see


class ImuTopic:
    def __init__(self, node, ns="/drone", gyro_hz=120.0, accel_hz=120.0,
                 publish_hz=120.0, frame="imu_link", sudo=True):
        self.node = node
        self.frame = frame
        self._pending_accel = []
        self._ssc_clock = see.SscClockMapper()
        self.pub = node.create_publisher(Imu, f"{ns}/imu", qos_profile_sensor_data)
        self.sensors = see.SensorSet({"gyro": gyro_hz, "accel": accel_hz},
                                     sudo=sudo, logger=node.get_logger())
        self.sensors.start()
        self.timer = node.create_timer(1.0 / publish_hz, self._publish)

    def _publish(self):
        gyro = self.sensors.get("gyro")
        accel = self.sensors.get("accel")
        samples = gyro.drain() if gyro is not None else ()
        if accel is not None:
            self._pending_accel.extend(accel.drain())
            self._pending_accel = self._pending_accel[-512:]
        for g in samples:
            if len(g[2]) < 3:
                continue
            a = None
            if self._pending_accel:
                nearest = min(
                    range(len(self._pending_accel)),
                    key=lambda i: abs(self._pending_accel[i][0] - g[0]),
                )
                if abs(self._pending_accel[nearest][0] - g[0]) <= 0.01:
                    a = self._pending_accel.pop(nearest)
            self._pending_accel = [
                sample for sample in self._pending_accel
                if sample[0] >= g[0] - 0.01
            ]

            msg = Imu()
            msg.header.frame_id = self.frame
            # Convert SSC capture time to host monotonic using its paired SEE event time,
            # then map that monotonic time into the ROS clock domain for the message header.
            ssc_seconds = g[1]
            if ssc_seconds is not None:
                self._ssc_clock.observe(g[0], ssc_seconds)
                monotonic_seconds = self._ssc_clock.to_monotonic(ssc_seconds)
                if monotonic_seconds is None:
                    msg.header.stamp = self.node.get_clock().now().to_msg()
                else:
                    mono_before_ns = time.monotonic_ns()
                    ros_now_ns = self.node.get_clock().now().nanoseconds
                    mono_after_ns = time.monotonic_ns()
                    mono_mid_ns = (mono_before_ns + mono_after_ns) // 2
                    ros_seconds = monotonic_seconds + (ros_now_ns - mono_mid_ns) * 1e-9
                    self._set_stamp(msg.header.stamp, ros_seconds)
            else:
                msg.header.stamp = self.node.get_clock().now().to_msg()

            gv = g[2]
            msg.angular_velocity.x = gv[0]
            msg.angular_velocity.y = gv[1] * -1
            msg.angular_velocity.z = gv[2] * -1

            if a is not None and len(a[2]) >= 3:
                av = a[2]
                msg.linear_acceleration.x = av[0] * -1
                msg.linear_acceleration.y = av[1]
                msg.linear_acceleration.z = av[2]
            else:
                msg.linear_acceleration_covariance[0] = -1.0               # no orientation estimate here
            self.pub.publish(msg)

    @staticmethod
    def _set_stamp(stamp, seconds):
        """Write float ROS-clock seconds into a builtin_interfaces/Time.

        Splits into whole seconds + nanoseconds, guarding the rounding case where the
        fractional part rounds up to a full second (nanosec must stay < 1e9).
        """
        sec = int(seconds)
        nanosec = int(round((seconds - sec) * 1_000_000_000))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        elif nanosec < 0:                       # defensive: seconds was negative/odd
            sec -= 1
            nanosec += 1_000_000_000
        stamp.sec = sec
        stamp.nanosec = nanosec

    def stop(self):
        self.sensors.stop()
