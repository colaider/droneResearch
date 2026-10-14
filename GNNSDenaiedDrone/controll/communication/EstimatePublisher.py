"""Publish the control loop's estimates as JSON on /vo/status, where drone_gui shows them.

Deliberately not /drone/odom: rb3_control feeds that topic into its velocity control, and an
estimate that is still drifting must not steer anything just by being displayed.
"""

import json
import math
import time
from collections import deque

import numpy as np


def _finite(values):
    """JSON-safe list: NaN/inf become null (json.dumps would write NaN, which browsers reject)."""
    if values is None:
        return None
    return [float(v) if math.isfinite(v) else None for v in np.asarray(values, float).ravel()]


class EstimatePublisher:
    def __init__(self, node, ns="/vo", rate_hz=10.0):
        from std_msgs.msg import String

        self._message_type = String
        self._publisher = node.create_publisher(String, "/" + ns.strip("/") + "/status", 10)
        self.node = node
        self.period = 1.0 / rate_hz
        self._next = 0.0
        self._frame_times = deque(maxlen=20)   # when each new visual frame was processed
        self._last_frame = None

    def publish(self, sensors, stage):
        """Call every control step; publishes at rate_hz."""
        now = time.monotonic()
        estimator = sensors.visual_estimator
        frame = getattr(estimator, "current_frame", None)
        if frame is not None and frame is not self._last_frame:
            self._last_frame = frame
            self._frame_times.append(now)
        if now < self._next:
            return
        self._next = now + self.period

        times = self._frame_times
        fps = 0.0
        if len(times) > 1 and now - times[-1] < 2.0 and times[-1] > times[0]:
            fps = (len(times) - 1) / (times[-1] - times[0])
        visual = _finite(sensors.get_visual_velocity())        # vx, vy, vz, yaw_rate
        depth = float(estimator.depth_median)
        status = {
            "mode": stage,
            "position": _finite(sensors.get_fused_position()),  # what the controller flies by
            "velocity": _finite(sensors.get_fused_velocity()),
            "visual_position": _finite(sensors.get_visual_position()),
            "vx": visual[0], "vy": visual[1], "vz": visual[2], "yaw_rate": visual[3],
            "tracked": int(np.count_nonzero(estimator.velocity_point_mask)),
            "depth_m": depth if math.isfinite(depth) else None,
            "depth_valid": float(estimator.depth_valid_frac),
            "fps": round(fps, 1),
        }
        message = self._message_type()
        message.data = json.dumps(status)
        self._publisher.publish(message)

    def close(self):
        self.node.destroy_publisher(self._publisher)
