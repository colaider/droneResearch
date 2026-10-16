"""Capture timestamped IMU snapshots in a bounded recent-history buffer."""

from collections import deque
from threading import Lock

from sensors.IMU.imuSnapshot import IMUSnapshot


class IMUBridge:
    """Store up to ``capacity`` unique timestamped IMU snapshots."""

    def __init__(self, capacity=50):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._snapshots = deque(maxlen=capacity)
        self._last_timestamp_ns = None
        self._lock = Lock()

    @property
    def snapshots(self):
        """Return an immutable view of the buffered snapshots, oldest first."""
        with self._lock:
            return tuple(self._snapshots)

    def closest(self, timestamp_ns, max_delta_ns=None):
        """Return the nearest buffered sample to a timestamp, if within the optional tolerance."""
        with self._lock:
            if not self._snapshots:
                return None
            snapshot = min(self._snapshots, key=lambda item: abs(item.timestamp_ns - timestamp_ns))
        if max_delta_ns is not None and abs(snapshot.timestamp_ns - timestamp_ns) > max_delta_ns:
            return None
        return snapshot

    def capture(self, imu, timestamp_ns):
        """Capture one new monotonic-stamped reading; ignore repeated timestamps."""
        if timestamp_ns is None:
            return None
        snapshot = IMUSnapshot.from_imu(imu, timestamp_ns)
        with self._lock:
            if snapshot.timestamp_ns == self._last_timestamp_ns:
                return None
            self._snapshots.append(snapshot)
            self._last_timestamp_ns = snapshot.timestamp_ns
        return snapshot
