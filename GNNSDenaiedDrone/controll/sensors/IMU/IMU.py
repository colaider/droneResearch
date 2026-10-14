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

        self.att_tau = 0.5

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
        if c["count"] < c["n"]:
            return

        mean_acc = c["acc"] / c["count"]
        norm = np.linalg.norm(mean_acc)
        self.gyro_bias = c["gyro"] / c["count"]

        if norm > 1e-6:
            roll0 = np.arctan2(mean_acc[1], mean_acc[2])
            pitch0 = np.arctan2(-mean_acc[0], np.sqrt(mean_acc[1] ** 2 + mean_acc[2] ** 2))
            self._att[:] = [roll0, pitch0, 0.0]            # yaw defines the world x axis = 0
            self.accel_bias = mean_acc - mean_acc * (G / norm)   # magnitude-only correction
        else:
            self._att[:] = 0.0
            self.accel_bias = np.zeros(3)

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
            a = self.att_tau / (self.att_tau + dt)   # dt-independent complementary weight
            self._att[0] = a * self._att[0] + (1 - a) * roll_acc
            self._att[1] = a * self._att[1] + (1 - a) * pitch_acc

    def _update_vel_pos(self, accel, dt):
        roll, pitch, yaw = self._att
        ax, ay, az = accel

        a_level = np.array([
            ax * np.cos(pitch) + ay * np.sin(roll) * np.sin(pitch) + az * np.cos(roll) * np.sin(pitch),
            ay * np.cos(roll) - az * np.sin(roll),
            -ax * np.sin(pitch) + ay * np.sin(roll) * np.cos(pitch) + az * np.cos(roll) * np.cos(pitch) - G,
        ])


        cy, sy = np.cos(yaw), np.sin(yaw)
        a_world = np.array([
            cy * a_level[0] - sy * a_level[1],
            sy * a_level[0] + cy * a_level[1],
            a_level[2],
        ])


        v_old = self._vel.copy()
        self._vel += a_world * dt                       # velocity (world frame)
        self._pos += v_old * dt + 0.5 * a_world * dt ** 2   # position (world frame)

    def fuse_vertical_velocity(self, ref_vz, w=0.9):
        """Anchor the drifty accel vz to an external vertical velocity (barometer)."""
        self._vel[2] = w * self._vel[2] + (1 - w) * ref_vz

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
