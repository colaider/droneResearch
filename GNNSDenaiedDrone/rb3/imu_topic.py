"""IMU topic: the RB3's OWN accelerometer + gyro, from the sensor core (see.py).

Only the RB3 IMU is used here (not Betaflight's), published as sensor_msgs/Imu on <ns>/imu.

header.stamp is the RB3 sensor-core INTERNAL clock (SSC/QTimer time, seconds) of the reading --
NOT ROS/system time and NOT publish time. This is the true capture time, so it carries no timer
jitter and lets consumers window IMU samples by the exact interval between readings. For visual
fusion the camera (and baro) must be stamped on this SAME sensor-core clock to be comparable.

Averaging: the sensor core is read at 500 Hz (the most see_workhorse offers) and every 4
consecutive readings are averaged into one message, so <ns>/imu runs at 125 Hz with lower noise
and with vibration above the output rate filtered out instead of aliased. Each message is
stamped at the centre of its 4-reading window (the mean of their sensor stamps), about 3 ms
before the newest reading in it.
"""
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
                 publish_hz=250.0, frame="imu_link", sudo=True):
        """average: consecutive readings averaged into each message, so <ns>/imu runs at
        gyro_hz / average (1 = every reading as it comes). publish_hz: how often the streams
        are drained and finished windows published; it bounds the added latency."""
        self.node = node
        self.frame = frame
        self._pending_gyro = []
        self._pending_accel = []
        self._accel_until = float("-inf")       # newest accel sensor time received so far
        self.pub = node.create_publisher(Imu, f"{ns}/imu", qos_profile_sensor_data)
        self.sensors = see.SensorSet({"gyro": gyro_hz, "accel": accel_hz},
                                     sudo=sudo, logger=node.get_logger())
        gyro_rate = self.sensors.get("gyro").rate            # snapped to a supported rate
        # Accel readings pair with the gyro reading nearest in sensor time, within one period.
        self._pair_tolerance = 1.0 / self.sensors.get("accel").rate
        self._window = _WindowAverage(int(average), max_gap=1.5 / gyro_rate)
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
            # Wait until the accel reading for this instant (within half a period) is in.
            if (self._accel_until < t - self._pair_tolerance / 2
                    and now - g[0] < _ACCEL_WAIT):
                break
            self._pending_gyro.pop(0)
            if len(g[2]) < 3:
                continue
            a = self._pair_accel(t)
            window = self._window.add(t, g[2][:3], a[2][:3] if a is not None and len(a[2]) >= 3 else None)
            if window is not None:
                self._publish_window(*window, from_sensor_clock=g[1] is not None)

    def _publish_window(self, stamp, gv, av, from_sensor_clock):
        msg = Imu()
        msg.header.frame_id = self.frame
        # Window centre on the sensor-core internal clock. Fall back to ROS time only if the
        # readings carried no sensor timestamp.
        if from_sensor_clock:
            self._set_stamp(msg.header.stamp, stamp)
        else:
            msg.header.stamp = self.node.get_clock().now().to_msg()

        msg.angular_velocity.x = gv[0]
        msg.angular_velocity.y = gv[1] * -1
        msg.angular_velocity.z = gv[2] * -1

        if av is not None:
            msg.linear_acceleration.x = av[0] * -1
            msg.linear_acceleration.y = av[1]
            msg.linear_acceleration.z = av[2]
        else:
            msg.linear_acceleration_covariance[0] = -1.0               # no orientation estimate here
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
