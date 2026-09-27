import numpy as np
from swarm.utilities.geometry import as_numpy, rotation, integrate_attitude, quat_to_rpy
from swarm.droneClasses.droneStruct import DroneStruct


class DroneCTRL:
    def __init__(self, drone_entity: DroneStruct, dt=0.01, velocity_truth_weight=0.05):
        """Configure state estimation and low-level control for one DroneStruct.

        Args:
            drone_entity: Wrapper exposing a Genesis drone and its IMU sensor.
            dt: Physics/control interval in seconds.
            velocity_truth_weight: Simulator-velocity assistance fraction per
                physics step, clipped to [0, 1]. Zero disables this correction;
                the default 0.05 mixes in 5% ground-truth velocity each step.
        """
        self.drone = drone_entity
        self.dt = dt
        self.__setup_low_level_control_variables()
        self.lin_pos = np.zeros(3)
        # Explicit simulator assistance; set to zero for IMU-only dead reckoning.
        self.velocity_truth_weight = float(np.clip(velocity_truth_weight, 0, 1))

    def set_propeller_rpm(self, rpms):
        """Convert four normalized motor-speed commands to RPM and apply them.

        rpms is a (4,) NumPy array in propeller order 0..3. Despite its name,
        inputs are fractions of max_rpm, not RPM or newtons. Values are clipped
        after scaling. Thrust varies approximately with the square of rotor speed.
        """
        rpms = np.clip(rpms*self.max_rpm, 0, self.max_rpm)
        self.drone.set_propellers_rpm(rpms)


    def hover(self, thrust=0.5):
        """Apply thrust (a normalized motor-speed fraction, not newtons) equally to all four motors."""
        self.set_propeller_rpm(np.array([thrust]*4))


    def start(self, step: int):
        """Run the initial two-second altitude controller using physics step counter step.

        Returns 0 while commanding startup altitude (0.5 m), or 1 when startup
        has ended and the caller should supply normal flight commands.
        """
        elapsed_time = step * self.dt
        if elapsed_time < 2.0:
            if not hasattr(self, 'target_z'):
                self.target_z = 0.5
                self.z_error_prev = 0

            current_z = self.get_position()[2]
            error = self.target_z - current_z

            thrust_adjustment = 0.1 * error + 0.05 * (error - self.z_error_prev) / self.dt
            thrust = self.hover_throttle + thrust_adjustment
            self.hover(thrust=thrust)
            self.z_error_prev = error
            return 0
        return 1


    def compute_tilt(self, vx_target, vy_target):
        """Calculate roll/pitch mixer corrections for horizontal world velocity targets.

        vx_target and vy_target are desired world velocities in m/s. Measurements
        and targets are rotated into body axes, desired tilt is bounded in radians,
        and attitude feedback returns (2,) normalized [roll, pitch] mixer corrections.
        The returned values are motor-command corrections, not angles or torques.
        """
        if not hasattr(self, 'last_vel_xy_err'):
            self.last_vel_xy_err = np.zeros(2)
            self.last_att_err_sum = np.zeros(2)

        roll, pitch, yaw = self.get_attitude()
        vel = rotation(self.get_attitude()).T @ self.get_lin_vel()
        target = rotation(self.get_attitude()).T @ np.array([vx_target, vy_target, 0.0])
        vx_target, vy_target = target[:2]
        ang_vel = self.get_ang_vel()

        vx_err = vx_target - vel[0]
        vy_err = vy_target - vel[1]

        kp_vx = self.kp[0]*vx_err
        kp_vy = -self.kp[1]*vy_err
        kd_vx = self.kd[0]*(vx_err - self.last_vel_xy_err[0]) / self.dt
        kd_vy = -self.kd[1]*(vy_err - self.last_vel_xy_err[1]) / self.dt


        target_pitch = np.clip(kp_vx + kd_vx, -self.max_tilt, self.max_tilt)
        target_roll  = np.clip(kp_vy + kd_vy, -self.max_tilt, self.max_tilt)

        att_err = np.array([target_roll - roll, target_pitch - pitch])

        P = self.kp_att * att_err
        D = -self.kd_att * np.array([ang_vel[0], ang_vel[1]])
        I = self.ki_att * self.last_att_err_sum 
        self.last_vel_xy_err = np.array([vx_err, vy_err])
        self.last_att_err_sum = np.clip(self.last_att_err_sum + att_err * self.dt, -1, 1)
        return P + I + D


    def lowLevelControl(self, U: np.ndarray):
        """Apply motor commands for U=[vx, vy, vz, body_z_rate], shape (4,).

        Linear targets are in world m/s; the last target is body z angular rate
        in rad/s. Uses cached estimates from step(), updates PID history, and
        sends four rotor speeds. Returns None. Call at most once per physics step.
        """
        vel = self.get_lin_vel()
        ang_vel = self.get_ang_vel()

        X_dot = np.array([vel[0], vel[1], vel[2], ang_vel[2]])
        error = U - X_dot

        if not hasattr(self, 'pid_started'):
            self.last_error = error
            self.pid_started = True

        P = self.kp * error
        self.integral_error = np.clip(self.integral_error + error * self.dt, -1, 1)
        I = self.ki * self.integral_error
        D = self.kd * (error - self.last_error) / self.dt
        pid_output = P + I + D

        vz  = pid_output[2]
        yaw = pid_output[3]

        tilt_cmd = self.compute_tilt(U[0], U[1])
        roll_cmd  = tilt_cmd[0]
        pitch_cmd = tilt_cmd[1]

        # prop0 = front-right, prop1 = back-right, prop2 = back-left, prop3 = front-left
        # forward pitch = tilt nose down = back motors up, front motors down
        # roll right    = right motors down, left motors up
        # yaw CCW       = prop1 + prop3 up, prop0 + prop2 down  (using default spin -1,1,-1,1)
        
        throttle = np.array([
            self.hover_throttle + vz - pitch_cmd - roll_cmd - yaw,
            (self.hover_throttle + vz + pitch_cmd - roll_cmd + yaw),
            (self.hover_throttle + vz + pitch_cmd + roll_cmd - yaw),
            self.hover_throttle + vz - pitch_cmd + roll_cmd + yaw,
        ])

        self.set_propeller_rpm(throttle)
        self.last_error = error


    def get_position(self) -> np.ndarray:
        """Return simulator ground-truth world [x, y, z], shape (3,), in metres."""
        return as_numpy(self.drone.get_pos()).reshape(3)


    def __setup_low_level_control_variables(self):
        self.kp = np.array([2, 2, 1.0, 0.5])   # vx, vy, vz, yaw
        self.kd = np.array([0.01, 0.01, 0.01, 0.0])
        self.ki = np.array([0.4, 0.04, 0.4, 0])

        self.kp_att = np.array([2, 2])   # roll, pitch
        self.kd_att = np.array([0.08,0.08])
        self.ki_att = np.array([0.1, 0.1])
        self.lats_no_tilt_err = np.array([0, 0])
        self.max_tilt = np.radians(15)  # Maximum tilt angle in radians

        self.max_rpm = 93400
        self.hover_rpm = 62300
        self.hover_throttle = self.hover_rpm / self.max_rpm 

        self.last_error = np.zeros(4)
        self.integral_error = np.zeros(4)
        self.last_pos = np.zeros(3)
        self.last_ori = np.zeros(3)
        self.last_filtered_derivative = np.zeros(4)
        self.last_control = np.zeros(4)


    def step(self, step = 0):
        """Read sensors and update cached estimates once per physics step.

        step is the optional physics counter retained for caller compatibility;
        integration uses self.dt, not the counter. Returns None. The accelerometer
        supplies body-frame specific force in m/s^2; gyro supplies body [p,q,r]
        in rad/s. Rotating specific force to world and subtracting [0,0,9.81]
        gives world acceleration. The first call initializes pose from simulator
        truth; later calls integrate IMU readings and optional velocity assistance.
        """
        

        reading = self.drone.imu.read()
        acc = as_numpy(reading.lin_acc).reshape(3)
        gyro = as_numpy(reading.ang_vel).reshape(3)
        if not hasattr(self, 'est_roll'):
            self.est_roll, self.est_pitch, self.est_yaw = quat_to_rpy(
                as_numpy(self.drone.get_quat()).reshape(4))
            self.v_x_sum = self.v_y_sum = self.v_z_sum = 0.0
            self.lin_pos = self.get_position()

        attitude = integrate_attitude(self.get_attitude(), gyro, self.dt)
        if abs(np.linalg.norm(acc) - 9.81) < 1.0:
            tilt = np.array([np.arctan2(acc[1], acc[2]),
                             np.arctan2(-acc[0], np.hypot(acc[1], acc[2]))])
            delta = np.arctan2(np.sin(tilt-attitude[:2]), np.cos(tilt-attitude[:2]))
            attitude[:2] += 0.01 * delta
        self.est_roll, self.est_pitch, self.est_yaw = attitude
        self.body_rates = gyro
        self.ax, self.ay, self.az = acc
        world_acc = rotation(attitude) @ acc - np.array([0., 0., 9.81])
        vel = self.get_lin_vel() + world_acc * self.dt
        if self.velocity_truth_weight:
            truth = as_numpy(self.drone.get_vel()).reshape(3)
            vel += self.velocity_truth_weight * (truth - vel)
        self.v_x_sum, self.v_y_sum, self.v_z_sum = vel
        self.update_imu_pos()


    def get_attitude(self):
        """Return estimated (roll, pitch, yaw) in radians; call step() before first use."""
        return self.est_roll, self.est_pitch, self.est_yaw


    def update_imu_pos(self):
        """Integrate cached world velocity over self.dt seconds into world position."""
        vel = self.get_lin_vel()
        self.lin_pos += vel * self.dt


    def reset_position(self):
        """Set estimated position to world [0,0,0] metres; leaves velocity and attitude unchanged."""
        self.lin_pos = np.zeros(3)


    def get_imu_pos(self):
        """Return a copy of estimated world [x,y,z], shape (3,), in metres."""
        return self.lin_pos.copy()    


    def get_lin_vel(self):
        """Return estimated world [vx,vy,vz], shape (3,), in m/s after step() initialization."""
        return np.array([self.v_x_sum, self.v_y_sum, self.v_z_sum])


    def get_lin_acc(self):
        """Return the latest body-frame accelerometer specific force, shape (3,), in m/s^2; gravity is not removed."""
        return np.array([self.ax, self.ay, self.az])


    def get_ang_vel(self):
        """Return a copy of body gyro rates [p,q,r], shape (3,), in rad/s, not Euler-angle derivatives."""
        return self.body_rates.copy()

    def set_pose(self, pos):
        """Replace the estimated world position with a copy of pos, a (3,) vector in metres."""
        self.lin_pos = np.asarray(pos, dtype=float).copy()

           