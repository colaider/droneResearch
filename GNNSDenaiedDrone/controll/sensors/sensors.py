import numpy as np
G = 9.81


class Sensors:
    def __init__(self, baro_vel_ema=0.8):
        self.baro_vel_ema = baro_vel_ema

        # pass-through (raw latest)
        self._accel = np.zeros(3)        # body accel [ax, ay, az] m/s^2
        self._gyro = np.zeros(3)         # angular velocity [wx, wy, wz] rad/s
        self._baro = 0.0                 # barometer altitude [m]  (== baro position)

        # estimates
        self._att = np.zeros(3)          # [roll, pitch, yaw] rad
        self._vel = np.zeros(3)          # accel-integrated velocity (world) [vx, vy, vz]
        self._pos = np.zeros(3)          # accel-integrated position (world) [x, y, z]
        self._baro_vel = 0.0
        self._prev_baro = None

        # biases measured at rest, subtracted from the RAW body readings BEFORE anything else
        self.accel_bias = np.zeros(3)    # body accel offset: mean(accel) - [0,0,g]
        self.gyro_bias = np.zeros(3)     # gyro offset: mean(gyro)
        self._calib = None

        # low-pass on the acceleration (cuts noise; higher = smoother but laggier)
        self.accel_lp = 0.3
        self._accel_f = np.zeros(3)
        # z complementary filter: pull the drifty accel-z toward the drift-free barometer
        self.k_pos_z = 0.02              # correct position by this fraction of the baro error
        self.k_vel_z = 0.002             # and nudge velocity, which damps the drift itself

    # ---------------- update ----------------
    def update(self, accel, gyro, baro, dt):
        accel = np.asarray(accel, dtype=float)
        gyro = np.asarray(gyro, dtype=float)
        baro = float(baro)
        self._accel, self._gyro, self._baro = accel, gyro, baro
        self._update_baro_velocity(baro, dt)

        if self._calib is not None:
            self._accumulate_calibration(accel, gyro)
            return                              # gathering bias: hold the estimates

        # De-bias in the BODY frame first, so the attitude filter sees [0,0,g] at rest (stays
        # level) and a_world comes out clean on every axis -- including y.
        accel_c = accel - self.accel_bias
        gyro_c = gyro - self.gyro_bias
        # low-pass the de-biased acceleration before it feeds attitude + integration
        self._accel_f = self.accel_lp * self._accel_f + (1 - self.accel_lp) * accel_c
        self._update_attitude(self._accel_f, gyro_c, dt)
        self._update_accel_vel_pos(self._accel_f, dt)

    # ---------------- calibration (accel + gyro) ----------------
    def start_calibration(self, n=200):
        """Hold the drone STILL and LEVEL. Over the next n updates:
             gyro_bias  = mean(gyro)
             accel_bias = mean(accel) - [0, 0, g]
        The start pose becomes the world frame (attitude reset to level)."""
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

    def _update_accel_vel_pos(self, accel, dt):
        roll, pitch = self._att[0], self._att[1]
        ax, ay, az = accel
        a_world = np.array([
            ax * np.cos(pitch) + ay * np.sin(roll) * np.sin(pitch) + az * np.cos(roll) * np.sin(pitch),
            ay * np.cos(roll) - az * np.sin(roll),
            -ax * np.sin(pitch) + ay * np.sin(roll) * np.cos(pitch) + az * np.cos(roll) * np.cos(pitch) - G,
        ])
        self._vel += a_world * dt          # integrate acceleration -> velocity
        self._pos += self._vel * dt        # integrate velocity -> position

        # z complementary filter: the barometer is drift-free, so pull z (and vz) toward it.
        if self._prev_baro is not None:
            err = self._baro - self._pos[2]
            self._pos[2] += self.k_pos_z * err
            self._vel[2] += self.k_vel_z * err

    def _update_baro_velocity(self, baro, dt):
        if self._prev_baro is not None and dt > 0:
            raw = (baro - self._prev_baro) / dt
            b = self.baro_vel_ema
            self._baro_vel = b * self._baro_vel + (1 - b) * raw
        self._prev_baro = baro

    # ---------------- getters ----------------
    def get_attitude(self):
        return self._att.copy()            # [roll, pitch, yaw] rad

    def get_acc_velocity(self):
        return self._vel.copy()            # world velocity from accel integration

    def get_acc_position(self):
        return self._pos.copy()            # world position from accel integration

    def get_baro_velocity(self):
        return self._baro_vel              # vertical velocity from the barometer

    def get_fused_position(self):
        # For now just the IMU position. Later: fuse the visual estimator here.
        return self._pos.copy()

    def get_acceleration(self):
        return self._accel.copy()          # body accel [ax, ay, az] m/s^2

    def get_baro_position(self):
        return self._baro                  # barometer altitude [m]

    def get_angular_velocity(self):
        return self._gyro.copy()           # [wx, wy, wz] rad/s
