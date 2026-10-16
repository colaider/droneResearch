from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass
class ImuSample:
    """One timestamped sensor reading.

    Built by the IMU from accel + gyro + the topic timestamp (baro left empty);
    the Sensors facade fills in the barometer afterwards. This is the single timestamped
    record that later feeds time-windowed fusion.
    """
    t: float                       # sensor-core internal clock [s]
    accel: np.ndarray              # body accel [ax, ay, az] m/s^2
    gyro: np.ndarray               # body angular velocity [wx, wy, wz] rad/s
    baro: float = None             # altitude [m]; None until Sensors fills it


class IMU:
    @staticmethod
    def rotate_x_180(vector):
        vector = np.asarray(vector, dtype=float)
        return np.array([vector[0], -vector[1], -vector[2]])

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


        self.zupt_acc_tol = 0.3          # |‖accel_c‖ - g| below this = no net acceleration [m/s^2]
        self.zupt_gyro_tol = 0.05        # ‖gyro_c‖ below this = not rotating [rad/s] (~3 deg/s)
        self.zupt_hold = 10              # consecutive still samples required (~0.1 s at 100 Hz)
        self._zupt_count = 0
        self._stationary = False

        # IMU lever arm: offset of the IMU from the drone rotation center, BODY frame [m]
        # (same axis convention as the accel after remap). A displaced IMU senses rotation-induced
        # acceleration on top of the body's; we subtract it so attitude/vel/pos refer to the center.
        # Leave [0,0,0] to disable. alpha = dω/dt is noisy, so the tangential term is off by default.
        self.imu_offset = np.zeros(3)
        self.compensate_tangential = False
        self._prev_gyro_c = None
        self._accel_comp = np.zeros(3)   # last de-biased, lever-arm-compensated body accel [m/s^2]

    # ---------------- timestamped sample ----------------
    @staticmethod
    def build_sample(accel, gyro, t):
        """Construct a timestamped ImuSample from a reading (barometer left empty)."""
        return ImuSample(t=t, accel=accel, gyro=gyro)

    def update(self, accel, gyro, dt):
        if not np.isfinite(dt) or dt <= 0: return
        self._accel = np.asarray(accel, dtype=float).copy()
        self._gyro = np.asarray(gyro, dtype=float).copy()

        if self._calib is not None:
            self._accumulate_calibration(self._accel, self._gyro)
            return

        accel_c = self._accel - self.accel_bias
        gyro_c = self._gyro - self.gyro_bias
        accel_c = self._compensate_lever_arm(accel_c, gyro_c, dt)
        self._accel_comp = accel_c
        self._update_attitude(accel_c, gyro_c, dt)

        if self._detect_stationary(accel_c, gyro_c):
            self._vel[:] = 0.0
        else:
            self._update_vel_pos(accel_c, dt)


    def _compensate_lever_arm(self, accel, gyro, dt):
        r = self.imu_offset
        if not np.any(r):                                # no offset -> nothing to do
            self._prev_gyro_c = gyro
            return accel
        centripetal = gyro * np.dot(gyro, r) - np.dot(gyro, gyro) * r
        a = accel - centripetal
        if self.compensate_tangential and self._prev_gyro_c is not None and dt > 0:
            alpha = (gyro - self._prev_gyro_c) / dt
            a = a - np.cross(alpha, r)
        self._prev_gyro_c = gyro
        return a

    def _detect_stationary(self, accel_c, gyro_c):
        still = (abs(np.linalg.norm(accel_c) - G) < self.zupt_acc_tol and np.linalg.norm(gyro_c) < self.zupt_gyro_tol)
        
        self._zupt_count = self._zupt_count + 1 if still else 0   # reset the instant we move
        self._stationary = self._zupt_count >= self.zupt_hold
        return self._stationary

    def is_stationary(self):
        return self._stationary

    # ---------------- calibration (accel + gyro) ----------------
    def start_calibration(self, n=400):
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
        """Estimate roll, pitch and yaw using gyro integration and accel correction."""
        # 1. Predict attitude using bias-corrected gyro rates.
        self._att += gyro * dt

        # 2. Accelerometer gives a gravity reference when linear acceleration is small.
        accel_norm = np.linalg.norm(accel)
        if abs(accel_norm - G) > 1.0:
            return

        roll_acc = np.arctan2(accel[1], accel[2])
        pitch_acc = np.arctan2(
            -accel[0],
            np.sqrt(accel[1]**2 + accel[2]**2),
        )

        # 3. Complementary filter: trust gyro for fast changes, accel for long-term correction.
        tau = 0.5  # seconds
        alpha = tau / (tau + dt)

        self._att[0] = alpha * self._att[0] + (1 - alpha) * roll_acc
        self._att[1] = alpha * self._att[1] + (1 - alpha) * pitch_acc

    def _update_vel_pos(self, accel, dt):
        """Integrate body acceleration into world-frame velocity and position."""
        roll, pitch, yaw = self._att
        ax, ay, az = accel

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        a_level = np.array([
            ax * cp + ay * sr * sp + az * cr * sp,
            ay * cr - az * sr,
            -ax * sp + ay * sr * cp + az * cr * cp - G,
        ])

        a_world = np.array([
            cy * a_level[0] - sy * a_level[1],
            sy * a_level[0] + cy * a_level[1],
            a_level[2],
        ])

        self._pos += self._vel * dt + 0.5 * a_world * dt**2
        self._vel += a_world * dt

    # ---------------- getters ----------------
    def get_attitude(self):
        return self._att.copy()              # [roll, pitch, yaw] rad

    def get_velocity(self):
        return self._vel.copy()              # world velocity from accel integration

    def get_position(self):
        return self._pos.copy()              # world position from accel integration

    def get_acceleration(self):
        return self._accel.copy()            # raw body accel [ax, ay, az] m/s^2

    def get_body_accel(self):
        return self._accel_comp.copy()       # de-biased, lever-arm-compensated body accel [m/s^2]

    def get_angular_velocity(self):
        return self._gyro - self.gyro_bias   # [wx, wy, wz] rad/s, bias removed


    