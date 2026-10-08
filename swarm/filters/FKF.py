import numpy as np
from collections import deque


# temporary copy of the FusionKalmanFilter class from swarm/filters/FKF.py, so that the visualAccEst.py file can be run standalone for testing.

class FusionKalmanFilter:
    def __init__(self, dim=4, command_gain=0.4,
                 process_var=(0.5, 0.5, 0.3, 1.0),
                 accel_var=(1.0, 1.0, 1.0, 1e6),
                 camera_var=(1.3, 1.3, 1e6, 0.3),
                 camera_staleness_rate=(2.0, 2.0, 0.0, 5.0),
                 expected_cam_interval=1.0 / 30.0,
                 history_seconds=2.0, dt_nominal=0.01):

        self.dim = dim
        self.k = command_gain
        self.x = np.zeros(dim)
        self.P = np.eye(dim)
        self.Q_rate = np.diag(process_var)
        self.Q_cam_stale = np.diag(camera_staleness_rate)
        self.R_accel = np.diag(accel_var)
        self.R_camera = np.diag(camera_var)
        self.expected_cam_interval = expected_cam_interval
        self._last_cam_id = None
        self._time_since_cam = 0.0
        self.gravity = np.array([0.0, 0.0, 9.81])

        # OOSM state replay
        history_len = max(10, int(history_seconds / dt_nominal) + 10)
        self.history = deque(maxlen=history_len)
        self.now = 0.0   # current filter time (seconds)

    # ---------------- core steps ----------------

    def predict(self, dt, imu_accel_body=None, attitude=None, command=None):
        if imu_accel_body is not None and attitude is not None:
            R = self.rotation_matrix(attitude)
            accel_world = R @ np.asarray(imu_accel_body, float)[:3] - self.gravity
            self.x[:3] += accel_world * dt

        elif command is not None:
            u = np.asarray(command, float)[:self.dim]
            self.x[:3] += self.k * (u[:3] - self.x[:3]) * dt

        if command is not None: self.x[3] += self.k * (command[3] - self.x[3]) * dt

        self.P += self.Q_rate * dt

        self._time_since_cam += dt
        overdue = self._time_since_cam - self.expected_cam_interval
        if overdue > 0:
            effective_dt = min(dt, overdue)
            self.P += self.Q_cam_stale * effective_dt

    def _update(self, x, P, z, R):
        """Pure Kalman update: returns new (x, P) without touching self."""
        z = np.asarray(z, float)[:self.dim]
        K = P @ np.linalg.inv(P + R)
        x_new = x + K @ (z - x)
        P_new = (np.eye(self.dim) - K) @ P
        return x_new, P_new

    def update_accel(self, v):
        self.x, self.P = self._update(self.x, self.P, v, self.R_accel)

    def update_camera(self, v):
        """In-sequence camera update (measurement is current)."""
        self.x, self.P = self._update(self.x, self.P, v, self.R_camera)
        self._time_since_cam = 0.0

    # ---------------- OOSM ----------------

    def _snapshot(self, dt, imu_accel_body, attitude, command, v_accel):
        """Record everything needed to replay this tick."""
        self.history.append({
            "t": self.now,
            "dt": dt,
            "imu_accel_body": None if imu_accel_body is None else np.asarray(imu_accel_body, float).copy(),
            "attitude": None if attitude is None else np.asarray(attitude, float).copy(),
            "command": None if command is None else np.asarray(command, float).copy(),
            "v_accel": None if v_accel is None else np.asarray(v_accel, float).copy(),
            # Pre-tick state — the state BEFORE this tick's predict + accel update ran.
            "x_pre": self.x.copy(),
            "P_pre": self.P.copy(),
        })

    def update_camera_oosm(self, v_camera, t_measurement):
        if not self.history or t_measurement >= self.now:
            self.update_camera(v_camera)
            return

        # Oldest tick we remember
        if t_measurement < self.history[0]["t"]:
            lag = self.now - t_measurement
            R_inflated = self.R_camera + np.eye(self.dim) * lag ** 2
            self.x, self.P = self._update(self.x, self.P, v_camera, R_inflated)
            self._time_since_cam = 0.0
            return

        # Find the first tick AT or AFTER t_measurement
        split_idx = 0
        for i, snap in enumerate(self.history):
            if snap["t"] >= t_measurement:
                split_idx = i
                break

        # Rewind to pre-state of that tick (state as it was at t_measurement, approximately)
        snap = self.history[split_idx]
        self.x = snap["x_pre"].copy()
        self.P = snap["P_pre"].copy()
        self.now = snap["t"]

        # Apply camera update at this past time
        self.x, self.P = self._update(self.x, self.P, v_camera, self.R_camera)

        # Replay all ticks from split_idx forward (recompute each predict + accel update)
        replay = list(self.history)[split_idx:]
        self.history.clear()
        self._time_since_cam = 0.0   # the camera update we just applied is the fresh reference

        for snap in replay:
            # Re-snapshot with corrected pre-state (so future OOSM calls work)
            self._snapshot(
                dt=snap["dt"],
                imu_accel_body=snap["imu_accel_body"],
                attitude=snap["attitude"],
                command=snap["command"],
                v_accel=snap["v_accel"],
            )

            self.predict(
                dt=snap["dt"],
                imu_accel_body=snap["imu_accel_body"],
                attitude=snap["attitude"],
                command=snap["command"],
            )
            if snap["v_accel"] is not None:
                self.update_accel(snap["v_accel"])

            self.now += snap["dt"]

    # ---------------- driver ----------------

    def step(self, command, v_accel, dt, v_camera=None, camera_id=None, camera_timestamp=None, attitude=None):
        # Record what we're about to do (for OOSM replay)
        self._snapshot(
            dt=dt,
            imu_accel_body=v_accel,
            attitude=attitude,
            command=command,
            v_accel=v_accel,
        )

        # Normal predict + accel update
        self.predict(dt, imu_accel_body=v_accel, attitude=attitude, command=command)
        if v_accel is not None:
            self.update_accel(v_accel)

        self.now += dt

        # Camera: in-sequence if no timestamp, OOSM if timestamp given
        if v_camera is not None and camera_id != self._last_cam_id:
            if camera_timestamp is None or camera_timestamp >= self.now:
                self.update_camera(v_camera)
            else:
                self.update_camera_oosm(v_camera, camera_timestamp)
            self._last_cam_id = camera_id

        return self.x.copy()


    def get(self):
        return self.x.copy()


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






