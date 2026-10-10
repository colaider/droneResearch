import numpy as np
from swarm.droneClasses.droneStruct import DroneStruct


class DroneCTRL:
    """
    Betaflight-style low-level controller.
    `lowLevelControl` expects U = [roll_target, pitch_target, yaw_rate_target, throttle]
      - roll_target, pitch_target : target tilt angles  [rad]
      - yaw_rate_target           : target yaw rate     [rad/s]
      - throttle                  : base thrust         [0..1]
    """

    def __init__(self, drone_entity: DroneStruct, dt=0.01):
        self.drone = drone_entity
        self.dt = dt
        self.__setup_low_level_control_variables()
        self.lin_pos = np.zeros(3)
        self.gyro_bias = None

    # ---------------- motors / helpers ----------------

    def set_propeller_rpm(self, rpms):
        rpms = np.clip(rpms * self.max_rpm, 0, self.max_rpm)
        self.drone.set_propellers_rpm(rpms)

    # ---------------- low level control ----------------

    def lowLevelControl(self, U: np.ndarray):
        roll_target, pitch_target, yaw_rate_target, throttle = U

        roll, pitch, _ = self.get_attitude()
        ang_vel = self.get_ang_vel()        # [roll_rate, pitch_rate, yaw_rate]

        # ---- Attitude PID (angle inner loop) ----
        att_err = np.array([roll_target - roll, pitch_target - pitch])

        P_att = self.kp_att * att_err
        D_att = -self.kd_att * np.array([ang_vel[0], ang_vel[1]])
        I_att = self.ki_att * self.last_att_err_sum
        self.last_att_err_sum += att_err * self.dt

        att_torque = P_att + I_att + D_att
        roll_cmd, pitch_cmd = att_torque[0], att_torque[1]

        # ---- Yaw-rate PID ----
        yaw_err = yaw_rate_target - ang_vel[2]
        self.yaw_err_sum += yaw_err * self.dt
        yaw_cmd = (self.kp_yaw * yaw_err
                   + self.ki_yaw * self.yaw_err_sum
                   - self.kd_yaw * (yaw_err - self.last_yaw_err) / self.dt)
        self.last_yaw_err = yaw_err

        # ---- Motor mixer ----
        # prop0 = front-right, prop1 = back-right, prop2 = back-left, prop3 = front-left
        # +pitch_cmd = nose down (forward),  +roll_cmd = roll right,  +yaw_cmd = yaw CCW
        throttle_motors = np.array([
            throttle - pitch_cmd - roll_cmd - yaw_cmd,
            throttle + pitch_cmd - roll_cmd + yaw_cmd,
            throttle + pitch_cmd + roll_cmd - yaw_cmd,
            throttle - pitch_cmd + roll_cmd + yaw_cmd,
        ])
        self.set_propeller_rpm(throttle_motors)

    # ---------------- gains / defaults ----------------

    def __setup_low_level_control_variables(self):
        # roll / pitch angle PID
        self.kp_att = np.array([2.0, 2.0])
        self.kd_att = np.array([0.08, 0.08])
        self.ki_att = np.array([0.1,  0.1])

        # yaw-rate PID
        self.kp_yaw = 0.5
        self.ki_yaw = 0.0
        self.kd_yaw = 0.0

        self.max_tilt = np.radians(15)

        self.max_rpm       = 93400
        self.hover_rpm     = 62300
        self.hover_throttle = self.hover_rpm / self.max_rpm

        self.last_att_err_sum = np.zeros(2)
        self.yaw_err_sum      = 0.0
        self.last_yaw_err     = 0.0

    # ---------------- sensors / state estimation ----------------

    def step(self, step=0):
        """Call this ONCE per scene.step(). Updates all cached estimates."""
        reading = self.drone.imu.read()
        acc  = reading.lin_acc.numpy()
        gyro = reading.ang_vel.numpy()

        if not hasattr(self, 'est_roll'):
            self.est_roll  = 0.0
            self.est_pitch = 0.0
            self.est_yaw   = 0.0
            self.v_x_sum   = 0.0
            self.v_y_sum   = 0.0
            self.v_z_sum   = 0.0
            self.v_roll    = 0.0
            self.v_pitch   = 0.0
            self.v_yaw     = 0.0

        self.est_roll  += gyro[0] * self.dt
        self.est_pitch += gyro[1] * self.dt
        self.est_yaw   += gyro[2] * self.dt
        self.v_roll, self.v_pitch, self.v_yaw = gyro[0], gyro[1], gyro[2]

        a = 0.99
        if abs(np.linalg.norm(acc) - 9.81) < 1.0:
            roll_acc  = np.arctan2(acc[1], acc[2])
            pitch_acc = np.arctan2(-acc[0], np.sqrt(acc[1] ** 2 + acc[2] ** 2))
            self.est_roll  = a * self.est_roll  + (1 - a) * roll_acc
            self.est_pitch = a * self.est_pitch + (1 - a) * pitch_acc

        roll, pitch = self.est_roll, self.est_pitch
        ax, ay, az = acc[0], acc[1], acc[2]
        a_x_world =  ax * np.cos(pitch) + ay * np.sin(roll) * np.sin(pitch) + az * np.cos(roll) * np.sin(pitch)
        a_y_world =  ay * np.cos(roll)  - az * np.sin(roll)
        a_z_world = -ax * np.sin(pitch) + ay * np.sin(roll) * np.cos(pitch) + az * np.cos(roll) * np.cos(pitch) - 9.81

        self.v_x_sum += a_x_world * self.dt
        self.v_y_sum += a_y_world * self.dt
        self.v_z_sum += a_z_world * self.dt

        gps_vel = self.drone.get_vel()
        vx_w, vy_w, vz_w = gps_vel[0], gps_vel[1], gps_vel[2]
        gps_vx =  vx_w * np.cos(self.est_yaw) + vy_w * np.sin(self.est_yaw)
        gps_vy = -vx_w * np.sin(self.est_yaw) + vy_w * np.cos(self.est_yaw)
        gps_vz = vz_w

        alpha = 0.95
        self.v_x_sum = alpha * self.v_x_sum + (1 - alpha) * gps_vx
        self.v_y_sum = alpha * self.v_y_sum + (1 - alpha) * gps_vy
        self.v_z_sum = alpha * self.v_z_sum + (1 - alpha) * gps_vz

        self.update_imu_pos()

    # ---------------- getters ----------------

    def get_position(self) -> np.ndarray:
        return self.drone.get_pos()

    def get_attitude(self):
        return self.est_roll, self.est_pitch, self.est_yaw

    def update_imu_pos(self):
        vel = self.get_lin_vel()
        self.lin_pos += vel * self.dt

    def get_imu_pos(self):
        return self.lin_pos.copy()

    def get_lin_vel(self):
        return np.array([self.v_x_sum, self.v_y_sum, self.v_z_sum])

    def get_ang_vel(self):
        raw = np.array([self.v_roll, self.v_pitch, self.v_yaw])
        if self.gyro_bias is not None:
            return raw - self.gyro_bias
        return raw

    def setAttitude(self, roll: float, pitch: float, yaw: float):
        self.est_roll  = roll
        self.est_pitch = pitch
        self.est_yaw   = yaw