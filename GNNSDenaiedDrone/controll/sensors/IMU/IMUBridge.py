"""Capture timestamped IMU snapshots in a bounded recent-history buffer."""

from collections import deque

from sensors.IMU.imuSnapshot import IMUSnapshot


class IMUBridge:
    """Store up to ``capacity`` unique timestamped IMU snapshots."""

    def __init__(self, capacity=50):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._snapshots = deque(maxlen=capacity)
        self._last_timestamp_ns = None

    @property
    def snapshots(self):
        """Return an immutable view of the buffered snapshots, oldest first."""
        return tuple(self._snapshots)

    def capture(self, imu, timestamp_ns):
        """Capture one new ROS-stamped reading; ignore repeats of a latched stamp."""
        if timestamp_ns is None:
            return None
        snapshot = IMUSnapshot.from_imu(imu, timestamp_ns)
        if snapshot.timestamp_ns == self._last_timestamp_ns:
            return None
        self._snapshots.append(snapshot)
        self._last_timestamp_ns = snapshot.timestamp_ns
        return snapshot
