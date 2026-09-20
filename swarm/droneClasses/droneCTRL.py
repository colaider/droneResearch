import genesis as gs
from genesis.engine.entities import DroneEntity
import numpy as np
from swarm.droneClasses.droneStruct import DroneStruct


class DroneCTRL:
    def __init__(self, drone_entity: DroneStruct, dt = 0.01):
        self.drone = drone_entity
        self.dt = dt
        self.__setup_low_level_control_variables()
        self.lin_pos = np.zeros(3)

    def set_propeller_rpm(self, rpms):
        rpms = np.clip(rpms*self.max_rpm, 0, self.max_rpm)
        self.drone.set_propellers_rpm(rpms)


    def hover(self, thrust=0.5):
        self.set_propeller_rpm(np.array([thrust]*4))


    def start(self, step: int):
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
        if not hasattr(self, 'last_vel_xy_err'):
            self.last_vel_xy_err = np.zeros(2)
            self.last_att_err_sum = np.zeros(2)

        roll, pitch, yaw = self.get_attitude()
        vel = self.get_lin_vel()
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
        self.last_att_err_sum += att_err * self.dt
        return P + I + D


    def lowLevelControl(self, U: np.ndarray):
        vel = self.get_lin_vel()
        ang_vel = self.get_ang_vel()

        X_dot = np.array([vel[0], vel[1], vel[2], ang_vel[2]])
        error = U - X_dot

        if not hasattr(self, 'pid_started'):
            self.last_error = error
            self.pid_started = True

        P = self.kp * error
        self.integral_error += error * self.dt
        I = self.ki * self.integral_error
        D = -self.kd * (error - self.last_error) / self.dt
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
        return self.drone.get_pos()


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
        """Call this ONCE per scene.step(). Updates all cached estimates."""
        

        reading = self.drone.imu.read()
        acc  = reading.lin_acc.numpy()   # array shape (3,)
        gyro = reading.ang_vel.numpy()   # array shape (3,)
        mag  = reading.mag.numpy() 

        if not hasattr(self, 'est_roll'):
            self.est_roll = 0.0
            self.est_pitch = 0.0
            self.est_yaw = 0.0
            self.v_x_sum = 0.0
            self.v_y_sum = 0.0
            self.v_z_sum = 0.0
            self.ax = 0.0
            self.ay = 0.0
            self.az = 0.0
            self.v_pitch = 0
            self.v_roll = 0
            self.v_yaw = 0
            
            

        

        self.est_roll  += gyro[0] * self.dt
        self.est_pitch += gyro[1] * self.dt
        self.est_yaw   += gyro[2] * self.dt
        self.v_pitch, self.v_roll, self.v_yaw = gyro[0], gyro[1], gyro[2]

        a = 0.99
        if abs(np.linalg.norm(acc) - 9.81) < 1.0:
            roll_acc  = np.arctan2(acc[1], acc[2])
            pitch_acc = np.arctan2(-acc[0], np.sqrt(acc[1]**2 + acc[2]**2))
            self.est_roll  = a * self.est_roll  + (1-a) * roll_acc
            self.est_pitch = a * self.est_pitch + (1-a) * pitch_acc

        roll, pitch = self.est_roll, self.est_pitch
        ax, ay, az = acc[0], acc[1], acc[2]
        self.ax, self.ay, self.az = ax, ay, az

        a_x_world = ax*np.cos(pitch) + ay*np.sin(roll)*np.sin(pitch) + az*np.cos(roll)*np.sin(pitch)
        a_y_world = ay*np.cos(roll) - az*np.sin(roll)
        a_z_world = -ax*np.sin(pitch) + ay*np.sin(roll)*np.cos(pitch) + (az)*np.cos(roll)*np.cos(pitch) - 9.81

        self.v_x_sum += a_x_world * self.dt
        self.v_y_sum += a_y_world * self.dt
        self.v_z_sum += a_z_world * self.dt

        gps_vel = self.drone.get_vel()
        vx_w, vy_w, vz_w = gps_vel[0], gps_vel[1], gps_vel[2]
        gps_vx = vx_w*np.cos(self.est_yaw) + vy_w*np.sin(self.est_yaw)
        gps_vy = -vx_w*np.sin(self.est_yaw) + vy_w*np.cos(self.est_yaw)
        gps_vz = vz_w

        alpha = 0.95
        self.v_x_sum = alpha * self.v_x_sum + (1-alpha) * gps_vx
        self.v_y_sum = alpha * self.v_y_sum + (1-alpha) * gps_vy
        self.v_z_sum = alpha * self.v_z_sum + (1-alpha) * gps_vz
        
        self.update_imu_pos()


    def get_attitude(self):
        return self.est_roll, self.est_pitch, self.est_yaw


    def update_imu_pos(self):
        vel = self.get_lin_vel()
        self.lin_pos += vel * self.dt


    def reset_position(self):
        self.lin_pos = np.zeros(3)


    def get_imu_pos(self):
        return self.lin_pos.copy()    


    def get_lin_vel(self):
        return np.array([self.v_x_sum, self.v_y_sum, self.v_z_sum])


    def get_lin_acc(self):
        return np.array([self.ax, self.ay, self.az])


    def get_ang_vel(self):
        return np.array([self.v_pitch, self.v_roll, self.v_yaw])

           