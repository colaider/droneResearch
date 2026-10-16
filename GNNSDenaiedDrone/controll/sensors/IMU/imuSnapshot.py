"""Immutable snapshots of IMU readings and estimated state."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class IMUSnapshot:
    """One IMU reading and state snapshot at a host-monotonic timestamp.

    Timestamp is integer nanoseconds in the host monotonic clock domain, used
    internally to match IMU readings with camera frames.
    Acceleration is the latest body-frame sensor reading; angular velocity is
    bias-corrected. Attitude is [roll, pitch, yaw] in radians; velocity and
    position are in the IMU's world frame.
    """

    timestamp_ns: int
    acceleration: Vector3
    angular_velocity: Vector3
    attitude: Vector3
    velocity: Vector3
    position: Vector3

    def __post_init__(self):
        if isinstance(self.timestamp_ns, bool) or not isinstance(self.timestamp_ns, Integral):
            raise ValueError("timestamp_ns must be an integer")
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        for name in ("acceleration", "angular_velocity", "attitude", "velocity", "position"):
            vector = tuple(float(value) for value in getattr(self, name))
            if len(vector) != 3 or not all(isfinite(value) for value in vector):
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, vector)

    @classmethod
    def from_imu(cls, imu, timestamp_ns: int) -> "IMUSnapshot":
        """Capture the latest public readings from an IMU instance."""
        return cls(
            timestamp_ns=timestamp_ns,
            acceleration=tuple(imu.get_acceleration()),
            angular_velocity=tuple(imu.get_angular_velocity()),
            attitude=tuple(imu.get_attitude()),
            velocity=tuple(imu.get_velocity()),
            position=tuple(imu.get_position()),
        )

    @property
    def timestamp(self) -> float:
        """Timestamp in seconds, for display and compatibility."""
        return self.timestamp_ns * 1e-9
