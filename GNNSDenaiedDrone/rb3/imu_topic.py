"""Publish sample-averaged RB3 IMU readings with capture-time-derived ROS stamps."""
import time

from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from rb3_control import see

# Gyro and accel come from two separate see_workhorse streams, so a gyro reading often arrives
# before its accel reading. It is held back until the accel stream has caught up to its time,
# but for at most this long (host seconds), so a stalled accel stream cannot stop the topic.
_ACCEL_WAIT = 0.02


def _sample_time(sample):
    """Sensor-core time of a (host_time, ssc_time, values) sample, host time if it has none."""
    return sample[1] if sample[1] is not None else sample[0]


class _WindowAverage:
    """Mean of `size` consecutive gyro readings and their paired accel readings, stamped at
    the window centre. A gap or a backwards step in the stamps restarts the window, so each
    output covers one contiguous stretch of readings."""

    def __init__(self, size, max_gap):
        self.size, self.max_gap = size, max_gap
        self._stamps, self._gyro, self._accel = [], [], []

    def add(self, stamp, gyro, accel):
        """Add one reading (accel None if unpaired). Returns (stamp, gyro, accel) means when a
        window completes - accel None if no reading in it had one - else None."""
        if self._stamps and not 0 < stamp - self._stamps[-1] <= self.max_gap:
            self._stamps, self._gyro, self._accel = [], [], []
        self._stamps.append(stamp)
        self._gyro.append(gyro)
        if accel is not None:
            self._accel.append(accel)
        if len(self._stamps) < self.size:
            return None
        out = (sum(self._stamps) / self.size, self._mean(self._gyro),
               self._mean(self._accel) if self._accel else None)
        self._stamps, self._gyro, self._accel = [], [], []
        return out

    @staticmethod
    def _mean(vectors):
        return [sum(axis) / len(vectors) for axis in zip(*vectors)]


class ImuTopic:
    def __init__(self, node, ns="/drone", gyro_hz=500.0, accel_hz=500.0, average=4,
                 publish_hz=125.0, frame="imu_link", sudo=True):
        """average: consecutive readings averaged into each message, so <ns>/imu runs at
        gyro_hz / average (1 = every reading as it comes). publish_hz: how often the streams
        are drained and finished windows published; it bounds the added latency (up to
        1 / publish_hz, 8 ms at the default, which matches the 125 Hz output rate)."""
        self.node = node
        self.frame = frame
        self._pending_gyro = []
        self._pending_accel = []
        self._accel_until = float("-inf")
        self.pub = node.create_publisher(Imu, f"{ns}/imu", qos_profile_sensor_data)
        self.sensors = see.SensorSet({"gyro": gyro_hz, "accel": accel_hz},
                                     sudo=sudo, logger=node.get_logger())
        gyro_stream = self.sensors.get("gyro")
        accel_stream = self.sensors.get("accel")
        if gyro_stream is None or accel_stream is None:
            raise RuntimeError("gyro and accel streams are required for the IMU topic")
        gyro_rate = gyro_stream.rate                         # snapped to a supported rate
        # Accel readings pair with the gyro reading nearest in sensor time, within one period.
        self._pair_tolerance = 1.0 / accel_stream.rate
        if isinstance(average, bool) or int(average) < 1:
            raise ValueError("average must be a positive integer")
        self._window = _WindowAverage(int(average), max_gap=1.5 / gyro_rate)
        self._ssc_clock = see.SscClockMapper()
        self.sensors.start()
        self.timer = node.create_timer(1.0 / publish_hz, self._publish)

    def _pair_accel(self, t):
        """Pop the pending accel reading nearest to sensor time t (None if none within one
        period), and drop the ones too old to pair with any later gyro reading."""
        a = None
        if self._pending_accel:
            nearest = min(range(len(self._pending_accel)),
                          key=lambda i: abs(_sample_time(self._pending_accel[i]) - t))
            if abs(_sample_time(self._pending_accel[nearest]) - t) <= self._pair_tolerance:
                a = self._pending_accel.pop(nearest)
        self._pending_accel = [sample for sample in self._pending_accel
                               if _sample_time(sample) >= t - self._pair_tolerance]
        return a

    def _publish(self):
        gyro = self.sensors.get("gyro")
        accel = self.sensors.get("accel")
        if gyro is not None:
            self._pending_gyro.extend(gyro.drain())
        if accel is not None:
            drained = accel.drain()
            if drained:
                self._accel_until = max(self._accel_until, max(map(_sample_time, drained)))
            self._pending_accel.extend(drained)
            self._pending_accel = self._pending_accel[-512:]
        now = time.monotonic()
        while self._pending_gyro:
            g = self._pending_gyro[0]
            t = _sample_time(g)
            if g[1] is not None:
                self._ssc_clock.observe(g[0], g[1])
            # Wait until the accel reading for this instant (within half a period) is in.
            if (self._accel_until < t - self._pair_tolerance / 2
                    and now - g[0] < _ACCEL_WAIT):
                break
            self._pending_gyro.pop(0)
            if len(g[2]) < 3:
                continue
            a = self._pair_accel(t)
            averaged = self._window.add(t, g[2], a[2] if a is not None and len(a[2]) >= 3 else None)
            if averaged is None:
                continue

            sample_time, gyro_mean, accel_mean = averaged
            if g[1] is not None:
                monotonic_seconds = self._ssc_clock.to_monotonic(sample_time)
            else:
                monotonic_seconds = sample_time
            msg = Imu()
            msg.header.frame_id = self.frame
            if monotonic_seconds is None:
                msg.header.stamp = self.node.get_clock().now().to_msg()
            else:
                msg.header.stamp = self._monotonic_to_ros_stamp(monotonic_seconds)

            msg.angular_velocity.x = gyro_mean[0]
            msg.angular_velocity.y = -gyro_mean[1]
            msg.angular_velocity.z = -gyro_mean[2]

            if accel_mean is not None:
                msg.linear_acceleration.x = -accel_mean[0]
                msg.linear_acceleration.y = accel_mean[1]
                msg.linear_acceleration.z = accel_mean[2]
            else:
                msg.linear_acceleration_covariance[0] = -1.0
            self.pub.publish(msg)

    def _monotonic_to_ros_stamp(self, monotonic_seconds):
        mono_before_ns = time.monotonic_ns()
        ros_now_ns = self.node.get_clock().now().nanoseconds
        mono_after_ns = time.monotonic_ns()
        mono_mid_ns = (mono_before_ns + mono_after_ns) // 2
        ros_seconds = monotonic_seconds + (ros_now_ns - mono_mid_ns) * 1e-9
        stamp = self.node.get_clock().now().to_msg()
        self._set_stamp(stamp, ros_seconds)
        return stamp

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
