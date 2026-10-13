"""Velocity -> stick conversion, ported from swarm/droneClasses/dronePositionCTRL.py.

The simulation version computed motor RPM itself. On real hardware Betaflight owns the
attitude and rate loops, so only the outer part survives: body-frame velocity error ->
target tilt angles + throttle. Gains are the ones from the simulation and are the first
thing to retune on the real airframe.
"""
from __future__ import annotations

import numpy as np


class VelocityController:
    """PID on body-frame velocity, output [roll, pitch, yaw_rate, throttle].

    roll/pitch in radians, yaw_rate passed through in rad/s, throttle normalised 0..1.
    """

    def __init__(self, dt=0.05, max_tilt=np.radians(15.0), hover_throttle=0.55,
                 kp=(2.0, 2.0, 1.0), ki=(0.4, 0.04, 0.4), kd=(0.01, 0.01, 0.01),
                 i_limit=0.25):
        self.dt = dt
        self.max_tilt = max_tilt
        self.hover_throttle = hover_throttle
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.i_limit = i_limit
        self.reset()

    def reset(self):
        self.last_err = np.zeros(3)
        self.err_sum = np.zeros(3)

    def step(self, target, measured, yaw_rate_target=0.0, dt=None):
        """target/measured: body-frame [vx, vy, vz] in m/s."""
        dt = self.dt if dt is None else max(1e-3, dt)
        err = np.asarray(target, dtype=float) - np.asarray(measured, dtype=float)

        p = self.kp * err
        d = self.kd * (err - self.last_err) / dt
        self.err_sum = np.clip(self.err_sum + err * dt, -self.i_limit / np.maximum(self.ki, 1e-6),
                               self.i_limit / np.maximum(self.ki, 1e-6))
        i = self.ki * self.err_sum
        self.last_err = err

        target_pitch = float(np.clip(p[0] + i[0] + d[0], -self.max_tilt, self.max_tilt))
        target_roll = float(np.clip(-(p[1] + i[1] + d[1]), -self.max_tilt, self.max_tilt))
        throttle = float(np.clip(self.hover_throttle + p[2] + i[2] + d[2], 0.0, 1.0))
        return np.array([target_roll, target_pitch, float(yaw_rate_target), throttle])


def sticks_from_command(roll, pitch, yaw_rate, throttle, max_tilt, max_yaw_rate,
                        centre=1500, span=500, throttle_min=1000, throttle_max=2000):
    """Convert a command in SI units to Betaflight stick values (1000..2000).

    roll/pitch [rad] map linearly onto +-max_tilt, yaw_rate [rad/s] onto +-max_yaw_rate,
    throttle [0..1] onto throttle_min..throttle_max.
    """
    def axis(value, limit):
        if limit <= 0.0:
            return centre
        return centre + span * float(np.clip(value / limit, -1.0, 1.0))

    throttle_value = throttle_min + (throttle_max - throttle_min) * float(np.clip(throttle, 0.0, 1.0))
    return (axis(roll, max_tilt), axis(pitch, max_tilt), axis(yaw_rate, max_yaw_rate), throttle_value)
