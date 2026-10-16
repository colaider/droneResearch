import numpy as np


class FusionKalmanFilter:

    def __init__(self,
                 process_var=(10, 10, 10, 10),   # vx, vy, vz, yaw_rate  (per second)
                 camera_var=(0.4, 0.4, 100),        # vx, vy, yaw_rate
                 baro_vel_var=0.1,                    # vz measurement variance
                 baro_pos_gain=0.1):                  # complementary pull of pz toward baro altitude
        # --- two state vectors ---
        self.v = np.zeros(4)          # [vx, vy, vz, yaw_rate]
        self.p = np.zeros(4)          # [px, py, pz, yaw]
        self.Pv = np.eye(4)           # velocity covariance (position is not Kalman-tracked)

        self.Qv = np.diag(process_var)
        self.R_cam = np.diag(camera_var)
        self.R_baro_vz = float(baro_vel_var)
        self.baro_pos_gain = float(baro_pos_gain)
        self.gravity = np.array([0.0, 0.0, 9.81])
        self._last_cam_id = None

        # camera observes vx, vy, yaw_rate  -> velocity indices 0, 1, 3
        self.H_cam = np.array([[1.0, 0, 0, 0],
                               [0, 1.0, 0, 0],
                               [0, 0, 0, 1.0]])
        # baro observes vz -> velocity index 2
        self.H_vz = np.array([[0, 0, 1.0, 0]])

    # ---------------- predict ----------------
    def predict(self, accel_body, dt, attitude):
        if accel_body is not None and attitude is not None:
            a_world = self.rotation_matrix(attitude) @ np.asarray(accel_body, float)[:3] - self.gravity
            self.v[0:3] += a_world * dt           
        self.Pv = self.Pv + self.Qv * dt

        self.p[0:3] += self.v[0:3] * dt           # px, py, pz
        self.p[3] += self.v[3] * dt               # yaw += yaw_rate * dt

    # ---------------- updates ----------------
    def _update_v(self, z, H, R):
        y = np.asarray(z, float) - H @ self.v
        S = H @ self.Pv @ H.T + R
        K = self.Pv @ H.T @ np.linalg.inv(S)
        self.v = self.v + K @ y
        self.Pv = (np.eye(4) - K @ H) @ self.Pv

    def update_camera(self, v_camera):
        """v_camera = [vx, vy, vz, yaw_rate] (world); corrects vx, vy, yaw_rate (vz ignored here)."""
        v_camera = np.asarray(v_camera, float)
        self._update_v([v_camera[0], v_camera[1], v_camera[3]], self.H_cam, self.R_cam)

    def update_baro(self, altitude, vertical_velocity):
        """Corrects vz from the barometer's vertical speed, and pulls pz toward the altitude."""
        self._update_v([float(vertical_velocity)], self.H_vz, np.array([[self.R_baro_vz]]))
        self.p[2] += self.baro_pos_gain * (float(altitude) - self.p[2])

    # ---------------- driver ----------------
    def step(self, accel_body, dt, attitude, v_camera=None, camera_id=None):
        self.predict(accel_body, dt, attitude)
        if v_camera is not None and camera_id != self._last_cam_id:
            self.update_camera(v_camera)
            self._last_cam_id = camera_id
        return self.get()

    # ---------------- getters & utilities ----------------
    def get(self):
        return np.concatenate([self.p, self.v])      # [px,py,pz,yaw, vx,vy,vz,yaw_rate]

    def get_position(self):
        return self.p[0:3].copy()

    def get_yaw(self):
        return float(self.p[3])

    def get_velocity(self):
        return self.v[0:3].copy()

    def get_yaw_rate(self):
        return float(self.v[3])

    def reset_position(self, p=None, yaw=None):
        """Re-origin position (e.g. at takeoff). Velocity is left to the filter."""
        self.p[0:3] = np.zeros(3) if p is None else np.asarray(p, float)
        if yaw is not None:
            self.p[3] = float(yaw)

    @staticmethod
    def rotation_matrix(attitude):
        roll, pitch, yaw = attitude
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        return np.array([
            [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
            [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
            [-sp,      cp * sr,                 cp * cr],
        ])
