"""IMU topic: the RB3's OWN accelerometer + gyro, from the sensor core (see.py).

Only the RB3 IMU is used here (not Betaflight's), published as sensor_msgs/Imu on <ns>/imu.

header.stamp is the RB3 sensor-core INTERNAL clock (SSC/QTimer time, seconds) of the reading --
NOT ROS/system time and NOT publish time. This is the true capture time, so it carries no timer
jitter and lets consumers window IMU samples by the exact interval between readings. For visual
fusion the camera (and baro) must be stamped on this SAME sensor-core clock to be comparable.
"""
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from rb3_control import see


class ImuTopic:
    def __init__(self, node, ns="/drone", gyro_hz=200.0, accel_hz=200.0,
                 publish_hz=100.0, frame="imu_link", sudo=True):
        self.node = node
        self.frame = frame
        self._pending_accel = []
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
            # Stamp from the sensor-core internal clock of THIS gyro reading (g[1], seconds).
            # Fall back to ROS time only if the sample carried no sensor timestamp.
            ssc_seconds = g[1]
            if ssc_seconds is not None:
                self._set_stamp(msg.header.stamp, ssc_seconds)
            else:
                msg.header.stamp = self.node.get_clock().now().to_msg()

            gv = g[2]
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gv[0], gv[1], gv[2]
            if a is not None and len(a[2]) >= 3:
                av = a[2]
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = av[0], av[1], av[2]
            else:
                msg.linear_acceleration_covariance[0] = -1.0   # no accel this cycle
            msg.orientation.w = 1.0
            msg.orientation_covariance[0] = -1.0               # no orientation estimate here
            self.pub.publish(msg)

    @staticmethod
    def _set_stamp(stamp, seconds):
        """Write float seconds (sensor-core internal clock) into a builtin_interfaces/Time.

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
