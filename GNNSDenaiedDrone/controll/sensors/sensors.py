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

        # bias: the residual world acceleration measured at rest, subtracted every tick
        self.accel_bias = np.zeros(3)
        self._calib = None               # {"sum", "count", "n"} while calibrating

    # ---------------- update ----------------
    def update(self, accel, gyro, baro, dt):
        accel = np.asarray(accel, dtype=float)
        gyro = np.asarray(gyro, dtype=float)
        baro = float(baro)
        self._accel, self._gyro, self._baro = accel, gyro, baro

        self._update_attitude(accel, gyro, dt)
        self._update_accel_vel_pos(accel, dt)
        self._update_baro_velocity(baro, dt)

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
        # rotate body acceleration to world and remove gravity
        a_world = np.array([
            ax * np.cos(pitch) + ay * np.sin(roll) * np.sin(pitch) + az * np.cos(roll) * np.sin(pitch),
            ay * np.cos(roll) - az * np.sin(roll),
            -ax * np.sin(pitch) + ay * np.sin(roll) * np.cos(pitch) + az * np.cos(roll) * np.cos(pitch) - G,
        ])
       

        # Calibrating: keep the drone still, average a_world, then lock it as the bias.
        if self._calib is not None:
            self._calib["sum"] += a_world
            self._calib["count"] += 1
            if self._calib["count"] >= self._calib["n"]:
                self.accel_bias = self._calib["sum"] / self._calib["count"]
                self._vel[:] = 0.0          # drop drift accumulated during calibration
                self._pos[:] = 0.0
                self._calib = None
            return                          # don't integrate while calibrating

        a_world = a_world - self.accel_bias  # remove the measured resting offset
        self._vel += a_world * dt            # integrate acceleration -> velocity
        self._pos += self._vel * dt          # integrate velocity -> position
        print(a_world)

    def _update_baro_velocity(self, baro, dt):
        if self._prev_baro is not None and dt > 0:
            raw = (baro - self._prev_baro) / dt
            b = self.baro_vel_ema
            self._baro_vel = b * self._baro_vel + (1 - b) * raw
        self._prev_baro = baro

    # ---------------- calibration ----------------
    def start_calibration(self, n=200):
        """Begin biasing: hold the drone still and level. Over the next n update() calls the
        resting a_world is averaged and stored as accel_bias; position/velocity stay frozen
        until it finishes. Call is_calibrating() to know when it is done."""
        self.accel_bias = np.zeros(3)       # measure the raw (uncorrected) residual
        self._calib = {"sum": np.zeros(3), "count": 0, "n": int(n)}

    def is_calibrating(self):
        return self._calib is not None

    # ---------------- estimates ----------------
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

    # ---------------- pass-through (raw latest) ----------------
    def get_acceleration(self):
        return self._accel.copy()          # body accel [ax, ay, az] m/s^2

    def get_baro_position(self):
        return self._baro                  # barometer altitude [m]

    def get_angular_velocity(self):
        return self._gyro.copy()           # [wx, wy, wz] rad/s
