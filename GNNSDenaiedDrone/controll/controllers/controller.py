import numpy as np


class Controller:
    def __init__(self, dt=0.01, hover_throttle=0.55, max_tilt_deg=2.5):
        self.dt = dt
        self.hover_throttle = hover_throttle
        self.max_tilt = np.radians(max_tilt_deg)

        # outer loop: feedback-linearization model + position/yaw PD (4-DOF: x, y, z, yaw)
        self.k_f1 = np.array([0.5, 0.5, 0.5, 0.8])      # control effectiveness (inverted)
        self.k_f2 = np.array([0.15, 0.15, 0.2, 0.2])    # modelled drift / damping (compensated)
        self.kp_pos = np.diag([0.03, 0.03, 0.25, 0.06])
        self.kd_pos = np.diag([0.02, 0.02, 0.005, 0.1])

        # inner loop: velocity error -> tilt/throttle (gains from the swarm blueprint)
        self.kp_vel = np.array([2.0, 2.0, 1.3])
        self.ki_vel = np.array([0.4, 0.04, 0.8])
        self.kd_vel = np.array([0.05, 0.05, 0.05])
        self._last_vel_err = np.zeros(3)
        self._vel_err_sum = np.zeros(3)

        out_range = np.array([self.max_tilt, self.max_tilt, 1.0 - self.hover_throttle])
        self.i_limit = out_range / self.ki_vel

        self._pr_setpoint = None
        self._pr_dr_setpoint = np.zeros(4)

        self.posCMD = np.zeros(4)


    def reset(self):
        self._last_vel_err = np.zeros(3)
        self._vel_err_sum = np.zeros(3)
        self._pr_setpoint = None
        self._pr_dr_setpoint = np.zeros(4)


    def update(self, sensors, setpoint):
        setpoint = np.asarray(setpoint, dtype=float)
        pos = sensors.get_fused_position()               # world [x, y, z]
        vel_world = sensors.get_acc_velocity()           # world [vx, vy, vz]
        yaw = sensors.get_attitude()[2]
        yaw_rate = sensors.get_angular_velocity()[2]

        if self._pr_setpoint is None:                    # first call: no feedforward yet
            self._pr_setpoint = setpoint.copy()
            self._pr_dr_setpoint = np.zeros(4)

        R = self.rot(yaw)
        
        X_d = np.array([vel_world[0], vel_world[1], vel_world[2], yaw_rate]) 
        X_d_body = R.T @ X_d                             # rot(-yaw) @ X_d  (R orthonormal)
        X_body = np.array([pos[0], pos[1], pos[2], yaw])

        # setpoint feedforward: commanded velocity and acceleration
        S_d = (setpoint - self._pr_setpoint) / self.dt
        S_dd = (S_d - self._pr_dr_setpoint) / self.dt

        X_err = setpoint - X_body
        X_err[3] = self._wrap(X_err[3])                  # wrap yaw error (original did not)
        X_err_d = S_d - X_d

        v = S_dd + self.kp_pos @ X_err + self.kd_pos @ X_err_d    
        U = np.linalg.inv(self.f1(yaw)) @ (v + self.f2(yaw) @ X_d_body)  # target vel [vx,vy,vz,yaw_rate]
        # U[:3] = np.clip(U[:3], -0.05, 0.05)                        
        # U[3] = np.clip(U[3], 0,  0.5)
        self.posCMD = U
        self._pr_setpoint = setpoint.copy()
        self._pr_dr_setpoint = S_d
        return self._velocity_to_cmd(U, X_d_body[:3])    # inner loop on BODY velocity


    def _velocity_to_cmd(self, U, measured_vel):
        target_vel = np.asarray(U[:3], float)
        yaw_rate_target = float(U[3])
        err = target_vel - np.asarray(measured_vel, float)
        P = self.kp_vel * err
        D = self.kd_vel * (err - self._last_vel_err) / self.dt
        self._last_vel_err = err

        hi = np.array([self.max_tilt, self.max_tilt, 1.0 - self.hover_throttle])
        lo = -np.array([self.max_tilt, self.max_tilt, self.hover_throttle])

        # provisional output with the EXISTING integral -> see which axes are already railed
        out = P + self.ki_vel * self._vel_err_sum + D
        self._vel_err_sum = self._anti_windup(
            self._vel_err_sum, err, self.dt, out >= hi, out <= lo, self.i_limit)

        # recompute with the (possibly frozen / clamped) integral
        out = P + self.ki_vel * self._vel_err_sum + D
        pitch = float(np.clip(out[0], -self.max_tilt, self.max_tilt))
        roll = float(np.clip(-out[1], -self.max_tilt, self.max_tilt))
        throttle = float(np.clip(self.hover_throttle + out[2], 0.0, 0.63))
        return np.array([roll, pitch, yaw_rate_target, throttle])


    # ---------------- feedback-linearization model ----------------
    def f1(self, yaw):
        return self.rot(yaw) * self.k_f1                 # R @ diag(k_f1) via column broadcast

    def f2(self, yaw):
        return self.rot(yaw) * self.k_f2

    @staticmethod
    def rot(yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0, 0],
                         [s,  c, 0, 0],
                         [0,  0, 1, 0],
                         [0,  0, 0, 1]])

    @staticmethod
    def _anti_windup(i_sum, err, dt, at_hi, at_lo, i_limit):
        step = err * dt
        blocked = (at_hi & (err > 0)) | (at_lo & (err < 0))
        i_sum = i_sum + np.where(blocked, 0.0, step)
        return np.clip(i_sum, -i_limit, i_limit)

    @staticmethod
    def _wrap(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
