import numpy as np
from collections import deque


class FusionKalmanFilter:
    """
    8D state: [px, py, pz, yaw, vx, vy, vz, yaw_rate]
      - position (px, py, pz)  — world frame
      - yaw                    — world frame
      - velocity (vx, vy, vz)  — world frame
      - yaw_rate               — body frame

    Position has NO direct measurement (no barometer, no GPS).
    It accumulates by integrating velocity through F (covariance coupling).
    Each camera / accel velocity update therefore also tightens position
    via the P[pos, vel] cross-covariance built up during predict().
    """

    def __init__(self,
                 # per-second process variance for each state axis
                 process_var=(0.05, 0.05, 0.03, 0.01,   # px, py, pz, yaw
                              0.6, 0.6, 0.4, 1.0),       # vx, vy, vz, yaw_rate
                 accel_var=(1.2, 1.2, 0.8),              # vx, vy, vz (accel vel drifts -> trust less)
                 camera_var=(0.4, 0.4, 1e6, 0.25),       # vx, vy, vz(unusable), yaw_rate (trust camera xy)
                 baro_var=(0.25, 0.4),                   # pz [m^2], vz [ (m/s)^2 ] (anchors vertical)
                 # extra P growth per second while camera is overdue
                 camera_staleness_rate=(0.5, 0.5, 0.0, 0.1,   # px, py, pz, yaw
                                        2.0, 2.0, 0.0, 0.1),  # vx, vy, vz, yaw_rate
                 expected_cam_interval=1.0 / 30.0,
                 history_seconds=2.0, dt_nominal=0.01):

        self.dim = 8
        self.x = np.zeros(self.dim)
        self.P = np.eye(self.dim)
        self.Q_rate = np.diag(process_var)
        self.Q_cam_stale = np.diag(camera_staleness_rate)
        self.R_accel = np.diag(accel_var)
        self.R_camera = np.diag(camera_var)
        self.R_baro = np.diag(baro_var)
        self.expected_cam_interval = expected_cam_interval
        self._last_cam_id = None
        self._time_since_cam = 0.0
        self.gravity = np.array([0.0, 0.0, 9.81])

        # Measurement selection matrices
        # Accel-derived velocity: 3-vector observes vx, vy, vz (indices 4,5,6)
        self.H_accel = np.zeros((3, self.dim))
        self.H_accel[0, 4] = 1.0
        self.H_accel[1, 5] = 1.0
        self.H_accel[2, 6] = 1.0

        # Camera: 4-vector observes vx, vy, vz, yaw_rate (indices 4,5,6,7)
        self.H_camera = np.zeros((4, self.dim))
        self.H_camera[0, 4] = 1.0
        self.H_camera[1, 5] = 1.0
        self.H_camera[2, 6] = 1.0
        self.H_camera[3, 7] = 1.0

        # Barometer: observes pz (index 2) and vz (index 6) -> anchors the vertical channel
        self.H_baro = np.zeros((2, self.dim))
        self.H_baro[0, 2] = 1.0
        self.H_baro[1, 6] = 1.0

        # OOSM state replay
        history_len = max(10, int(history_seconds / dt_nominal) + 10)
        self.history = deque(maxlen=history_len)
        self.now = 0.0

    # ---------------- core steps ----------------

    def predict(self, dt, imu_accel_body=None, attitude=None):
        # --- position & yaw: integrate from velocity & yaw_rate ---
        self.x[0] += self.x[4] * dt   # px += vx * dt
        self.x[1] += self.x[5] * dt   # py += vy * dt
        self.x[2] += self.x[6] * dt   # pz += vz * dt
        self.x[3] += self.x[7] * dt   # yaw += yaw_rate * dt

        # --- velocity: driven by IMU accel in world frame (no command prediction) ---
        # yaw_rate is left to drift under process noise and is corrected by the camera update.
        if imu_accel_body is not None and attitude is not None:
            R = self.rotation_matrix(attitude)
            accel_world = R @ np.asarray(imu_accel_body, float)[:3] - self.gravity
            self.x[4:7] += accel_world * dt

        # --- covariance propagation with full F (couples pos<-vel, yaw<-yaw_rate) ---
        F = np.eye(self.dim)
        F[0, 4] = dt   # dpx/dvx
        F[1, 5] = dt   # dpy/dvy
        F[2, 6] = dt   # dpz/dvz
        F[3, 7] = dt   # dyaw/dyaw_rate
        self.P = F @ self.P @ F.T + self.Q_rate * dt

        # --- camera staleness: inflate P only when overdue ---
        self._time_since_cam += dt
        overdue = self._time_since_cam - self.expected_cam_interval
        if overdue > 0:
            effective_dt = min(dt, overdue)
            self.P += self.Q_cam_stale * effective_dt

    def _update(self, x, P, z, H, R):
        """Kalman update with explicit measurement model H and noise R."""
        z = np.asarray(z, float)
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x_new = x + K @ y
        P_new = (np.eye(len(x)) - K @ H) @ P
        return x_new, P_new

    def update_accel(self, v):
        """Accel-derived velocity measurement (vx, vy, vz)."""
        v = np.asarray(v, float)[:3]
        self.x, self.P = self._update(self.x, self.P, v, self.H_accel, self.R_accel)

    def update_camera(self, v):
        """In-sequence camera update (vx, vy, vz, yaw_rate)."""
        v = np.asarray(v, float)[:4]
        self.x, self.P = self._update(self.x, self.P, v, self.H_camera, self.R_camera)
        self._time_since_cam = 0.0

    def update_baro(self, altitude, vertical_velocity):
        """Barometer update: altitude -> pz, vertical speed -> vz (anchors the vertical channel)."""
        z = np.array([float(altitude), float(vertical_velocity)])
        self.x, self.P = self._update(self.x, self.P, z, self.H_baro, self.R_baro)

    # ---------------- OOSM ----------------

    def _snapshot(self, dt, imu_accel_body, attitude, v_accel):
        """Record everything needed to replay this tick."""
        self.history.append({
            "t": self.now,
            "dt": dt,
            "imu_accel_body": None if imu_accel_body is None else np.asarray(imu_accel_body, float).copy(),
            "attitude":       None if attitude       is None else np.asarray(attitude, float).copy(),
            "v_accel":        None if v_accel        is None else np.asarray(v_accel, float).copy(),
            # pre-tick state (before this tick's predict + accel update ran)
            "x_pre": self.x.copy(),
            "P_pre": self.P.copy(),
        })

    def update_camera_oosm(self, v_camera, t_measurement):
        if not self.history or t_measurement >= self.now:
            self.update_camera(v_camera)
            return

        # Measurement older than anything we remember → apply with inflated R
        if t_measurement < self.history[0]["t"]:
            lag = self.now - t_measurement
            R_inflated = self.R_camera + np.eye(4) * lag ** 2
            v = np.asarray(v_camera, float)[:4]
            self.x, self.P = self._update(self.x, self.P, v, self.H_camera, R_inflated)
            self._time_since_cam = 0.0
            return

        # Find first tick AT or AFTER t_measurement
        split_idx = 0
        for i, snap in enumerate(self.history):
            if snap["t"] >= t_measurement:
                split_idx = i
                break

        # Rewind to that tick's pre-state (state at ~t_measurement)
        snap = self.history[split_idx]
        self.x = snap["x_pre"].copy()
        self.P = snap["P_pre"].copy()
        self.now = snap["t"]

        # Apply the (now in-sequence) camera update at that past time
        v = np.asarray(v_camera, float)[:4]
        self.x, self.P = self._update(self.x, self.P, v, self.H_camera, self.R_camera)

        # Replay all ticks from split_idx forward
        replay = list(self.history)[split_idx:]
        self.history.clear()
        self._time_since_cam = 0.0

        for snap in replay:
            self._snapshot(
                dt=snap["dt"],
                imu_accel_body=snap["imu_accel_body"],
                attitude=snap["attitude"],
                v_accel=snap["v_accel"],
            )
            self.predict(
                dt=snap["dt"],
                imu_accel_body=snap["imu_accel_body"],
                attitude=snap["attitude"],
            )
            if snap["v_accel"] is not None:
                self.update_accel(snap["v_accel"])
            self.now += snap["dt"]

    # ---------------- driver ----------------

    def step(self, accel_body, dt, attitude,
             t=None, v_camera=None, camera_id=None, camera_timestamp=None):
        """One fusion tick.

        accel_body : de-biased, lever-arm-compensated body acceleration [m/s^2] (drives velocity)
        attitude   : [roll, pitch, yaw] to rotate accel to world and remove gravity
        t          : absolute time at the START of this tick [s]; when given, the filter clock is
                     pinned to it so camera_timestamp (same clock) is directly comparable for OOSM.
        Velocity is driven ONLY by the IMU here (predict) and corrected by the camera update; there
        is no accel-velocity measurement and no command prediction.
        """
        if t is not None:
            self.now = float(t)

        self._snapshot(dt=dt, imu_accel_body=accel_body, attitude=attitude, v_accel=None)
        self.predict(dt, imu_accel_body=accel_body, attitude=attitude)
        self.now += dt

        if v_camera is not None and camera_id != self._last_cam_id:
            if camera_timestamp is None or camera_timestamp >= self.now:
                self.update_camera(v_camera)
            else:
                self.update_camera_oosm(v_camera, camera_timestamp)
            self._last_cam_id = camera_id

        return self.x.copy()

    # ---------------- getters & utilities ----------------

    def get(self):
        return self.x.copy()

    def get_position(self):
        return self.x[0:3].copy()

    def get_yaw(self):
        return float(self.x[3])

    def get_velocity(self):
        return self.x[4:7].copy()

    def get_yaw_rate(self):
        return float(self.x[7])

    def reset_position(self, p=None, yaw=None):
        """Clear accumulated position (e.g. at takeoff, when a new reference is set)."""
        if p is None:
            p = np.zeros(3)
        self.x[0:3] = np.asarray(p, float)
        if yaw is not None:
            self.x[3] = float(yaw)
        # tiny initial position variance, no cross-coupling to velocity
        self.P[0:4, :] = 0.0
        self.P[:, 0:4] = 0.0
        self.P[0, 0] = self.P[1, 1] = self.P[2, 2] = 0.01
        self.P[3, 3] = 0.01

    def camera_is_stale(self):
        return self._time_since_cam > 2.0 * self.expected_cam_interval


    @staticmethod
    def rotation_matrix(attitude):
        roll, pitch, yaw = attitude
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        return np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,    cp*sr,             cp*cr           ]
        ])






