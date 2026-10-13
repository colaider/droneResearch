"""Sensors: the single interface the controller uses. It composes the sensor estimators and
forwards to them -- the IMU (attitude/velocity/position) plus the barometer. Later a visual
estimator joins here and get_fused_position() fuses them.
"""
import numpy as np

from sensors.IMU.IMU import IMU


class Sensors:
    def __init__(self, baro_vel_ema=0.8):
        self.imu = IMU()                  # accel + gyro estimator

        # barometer (kept here; it is not part of the IMU)
        self.baro_vel_ema = baro_vel_ema
        self._baro = 0.0
        self._baro_vel = 0.0
        self._prev_baro = None

    # ---------------- update ----------------
    def update(self, accel, gyro, baro, dt):
        self.imu.update(accel, gyro, dt)
        self._update_baro_velocity(float(baro), dt)

    def _update_baro_velocity(self, baro, dt):
        self._baro = baro
        if self._prev_baro is not None and dt > 0:
            raw = (baro - self._prev_baro) / dt
            b = self.baro_vel_ema
            self._baro_vel = b * self._baro_vel + (1 - b) * raw
        self._prev_baro = baro

    # ---------------- calibration (forwarded to the IMU) ----------------
    def start_calibration(self, n=200):
        self.imu.start_calibration(n)

    def is_calibrating(self):
        return self.imu.is_calibrating()

    # ---------------- forwarded IMU estimates ----------------
    def get_attitude(self):
        return self.imu.get_attitude()

    def get_acc_velocity(self):
        return self.imu.get_velocity()

    def get_acc_position(self):
        return self.imu.get_position()

    def get_acceleration(self):
        return self.imu.get_acceleration()

    def get_angular_velocity(self):
        return self.imu.get_angular_velocity()

    # ---------------- barometer ----------------
    def get_baro_position(self):
        return self._baro                 # barometer altitude [m]

    def get_baro_velocity(self):
        return self._baro_vel             # vertical velocity from the barometer

    # ---------------- fused ----------------
    def get_fused_position(self):
        # For now just the IMU position. Later: fuse the visual estimator / baro here.
        return self.imu.get_position()
