"""Cascade position controller -- blueprint: swarm DronePositionCTRL, rewritten to read from Sensors.

Outer loop : position error            -> target body velocity
Inner loop : velocity error (PID)      -> roll, pitch, throttle   (ported from velocity_to_cmd)
Yaw        : yaw error                 -> yaw rate
Output     : [roll, pitch, yaw_rate, throttle]  for ReaderSender.send()

No ROS, no DroneStruct -- it only reads a Sensors object, so it is pure and testable.
"""
import numpy as np


class Controller:
    def __init__(self, dt=0.01, hover_throttle=0.55, max_tilt_deg=15.0):
        self.dt = dt
        self.hover_throttle = hover_throttle
        self.max_tilt = np.radians(max_tilt_deg)

        # outer loop: position/yaw error -> target velocity / yaw rate
        self.kp_pos = np.array([0.3, 0.3, 0.75])
        self.kp_yaw = 0.3

        # inner loop: velocity error -> tilt/throttle (gains from the swarm blueprint)
        self.kp_vel = np.array([2.0, 2.0, 1.0])
        self.ki_vel = np.array([0.4, 0.04, 0.4])
        self.kd_vel = np.array([0.01, 0.01, 0.01])
        self._last_vel_err = np.zeros(3)
        self._vel_err_sum = np.zeros(3)

    def reset(self):
        self._last_vel_err = np.zeros(3)
        self._vel_err_sum = np.zeros(3)

    def update(self, sensors, setpoint):
        """setpoint = [x, y, z, yaw]. Returns [roll, pitch, yaw_rate, throttle]."""
        setpoint = np.asarray(setpoint, dtype=float)
        pos = sensors.get_fused_position()           # [x, y, z]
        vel = sensors.get_acc_velocity()             # [vx, vy, vz]
        yaw = sensors.get_attitude()[2]

        # outer loop -> targets for the inner loop
        target_vel = self.kp_pos * (setpoint[:3] - pos)
        yaw_rate_target = self.kp_yaw * self._wrap(setpoint[3] - yaw)

        return self._velocity_to_cmd(target_vel, vel, yaw_rate_target)

    def _velocity_to_cmd(self, target_vel, measured_vel, yaw_rate_target):
        err = np.asarray(target_vel, float) - np.asarray(measured_vel, float)
        P = self.kp_vel * err
        D = self.kd_vel * (err - self._last_vel_err) / self.dt
        self._vel_err_sum += err * self.dt
        I = self.ki_vel * self._vel_err_sum
        self._last_vel_err = err

        pitch = float(np.clip(P[0] + I[0] + D[0], -self.max_tilt, self.max_tilt))
        roll = float(np.clip(-(P[1] + I[1] + D[1]), -self.max_tilt, self.max_tilt))
        throttle = float(np.clip(self.hover_throttle + P[2] + I[2] + D[2], 0.0, 1.0))
        return np.array([roll, pitch, float(yaw_rate_target), throttle])

    @staticmethod
    def _wrap(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
