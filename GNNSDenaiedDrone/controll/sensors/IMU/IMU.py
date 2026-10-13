"""IMU estimator: accelerometer + gyroscope -> attitude, velocity, position, with bias calibration.

Pure numpy, no ROS, no barometer. Fed raw body accel + gyro + dt each tick. Owned and forwarded
by sensors.Sensors.
"""
import numpy as np

G = 9.81


class IMU:
    def __init__(self):
        # raw latest
        self._accel = np.zeros(3)        # body accel [ax, ay, az] m/s^2
        self._gyro = np.zeros(3)         # angular velocity [wx, wy, wz] rad/s
        # estimates
        self._att = np.zeros(3)          # [roll, pitch, yaw] rad
        self._vel = np.zeros(3)          # accel-integrated world velocity [vx, vy, vz]
        self._pos = np.zeros(3)          # accel-integrated world position [x, y, z]
        # biases measured at rest, subtracted from the RAW readings before anything else
        self.accel_bias = np.zeros(3)    # mean(accel) - [0,0,g]
        self.gyro_bias = np.zeros(3)     # mean(gyro)
        self._calib = None

    # ---------------- update ----------------
    def update(self, accel, gyro, dt):
        accel = np.asarray(accel, dtype=float)
        gyro = np.asarray(gyro, dtype=float)
        self._accel, self._gyro = accel, gyro

        if self._calib is not None:
            self._accumulate_calibration(accel, gyro)
            return                              # gathering bias: hold the estimates

        accel_c = accel - self.accel_bias
        gyro_c = gyro - self.gyro_bias
        self._update_attitude(accel_c, gyro_c, dt)
        self._update_vel_pos(accel_c, dt)

    # ---------------- calibration (accel + gyro) ----------------
    def start_calibration(self, n=200):
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self._calib = {"acc": np.zeros(3), "gyro": np.zeros(3), "count": 0, "n": int(n)}

    def _accumulate_calibration(self, accel, gyro):
        c = self._calib
        c["acc"] += accel
        c["gyro"] += gyro
        c["count"] += 1
        if c["count"] >= c["n"]:
            self.gyro_bias = c["gyro"] / c["count"]
            self.accel_bias = c["acc"] / c["count"] - np.array([0.0, 0.0, G])
            self._att[:] = 0.0              # start pose = world frame (level)
            self._vel[:] = 0.0
            self._pos[:] = 0.0
            self._calib = None

    def is_calibrating(self):
        return self._calib is not None

    # ---------------- estimators ----------------
    def _update_attitude(self, accel, gyro, dt):
        self._att += gyro * dt
        if abs(np.linalg.norm(accel) - G) < 1.0:
            roll_acc = np.arctan2(accel[1], accel[2])
            pitch_acc = np.arctan2(-accel[0], np.sqrt(accel[1] ** 2 + accel[2] ** 2))
            a = 0.98
            self._att[0] = a * self._att[0] + (1 - a) * roll_acc
            self._att[1] = a * self._att[1] + (1 - a) * pitch_acc

    def _update_vel_pos(self, accel, dt):
        roll, pitch, yaw = self._att
        ax, ay, az = accel
        # 1) body -> LEVELED frame (roll, pitch) and remove gravity on z
        a_level = np.array([
            ax * np.cos(pitch) + ay * np.sin(roll) * np.sin(pitch) + az * np.cos(roll) * np.sin(pitch),
            ay * np.cos(roll) - az * np.sin(roll),
            -ax * np.sin(pitch) + ay * np.sin(roll) * np.cos(pitch) + az * np.cos(roll) * np.cos(pitch) - G,
        ])
        # 2) LEVELED -> WORLD: rotate the horizontal components by yaw so x/y stay in a FIXED
        #    world frame (z is vertical, unaffected by yaw). This was missing -> x/y spun with heading.
        cy, sy = np.cos(yaw), np.sin(yaw)
        a_world = np.array([
            cy * a_level[0] - sy * a_level[1],
            sy * a_level[0] + cy * a_level[1],
            a_level[2],
        ])
        self._vel += a_world * dt          # integrate acceleration -> velocity (world frame)
        self._pos += self._vel * dt        # integrate velocity -> position (world frame)

    # ---------------- getters ----------------
    def get_attitude(self):
        return self._att.copy()              # [roll, pitch, yaw] rad

    def get_velocity(self):
        return self._vel.copy()              # world velocity from accel integration

    def get_position(self):
        return self._pos.copy()              # world position from accel integration

    def get_acceleration(self):
        return self._accel.copy()            # body accel [ax, ay, az] m/s^2

    def get_angular_velocity(self):
        return self._gyro - self.gyro_bias   # [wx, wy, wz] rad/s, bias removed
