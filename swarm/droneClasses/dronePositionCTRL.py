from swarm.droneClasses.droneCTRL import DroneCTRL
import numpy as np

class DronePositionCTRL(DroneCTRL):

    def __init__(self, drone_entity, dt=0.01):
        super().__init__(drone_entity, dt)

        self.k_f1 = np.array([0.5, 0.5, 0.5, 0.8])  
        self.k_f2 = np.array([0.15, 0.15, 0.2, 0.2])
        self.kp_pos = np.diag([0.35, 0.35, 0.45, 0.06])
        self.kd_pos = np.diag([0.02, 0.02, 0.02, 0.1])
        self.prev_U = np.zeros(4)

        self.kp_vel = np.array([2.0, 2.0, 1.0])    
        self.kd_vel = np.array([0.01, 0.01, 0.01])
        self.ki_vel = np.array([0.4,  0.04, 0.4])

    def position_control(self, setpoint, X_body=None, X_d_body=None):
        
        if X_body is None: X_body = self.get_imu_pos()
        if X_d_body is None: X_d_body = self.get_lin_vel()

        if not hasattr(self, 'pr_setpoint'):
            self.pr_setpoint = setpoint.copy()
            self.pr_dr_setpoint = np.zeros(4)

        omega = self.get_attitude()[2]
        omega_d = self.get_ang_vel()[2]
        S_d = (setpoint - self.pr_setpoint)/self.dt
        S_dd = (S_d - self.pr_dr_setpoint)/self.dt

        X_d_body = np.append(X_d_body, omega_d)
        X_body   = np.append(X_body, omega)

        X_d = self.rot(omega) @ X_d_body
        
        X_err = setpoint - X_body
        X_err_d = X_d - S_d

        v = S_dd + self.kp_pos @ X_err + self.kd_pos @ X_err_d

        U = np.linalg.inv(self.f1(omega)) @ (v + self.f2(omega) @ X_d_body)
        self.prev_U = U
        np.clip(U, -2,2)
        self.pr_setpoint = setpoint
        self.pr_dr_setpoint = S_d
        return self.velocity_to_cmd(U)

       

    def velocity_to_cmd(self, U):
        vx_target, vy_target, vz_target, yaw_rate_target = U[0], U[1], U[2], U[3]

        if not hasattr(self, 'last_vel_err'):
            self.last_vel_err = np.zeros(3)
            self.vel_err_sum  = np.zeros(3)

        vel = self.get_lin_vel()
        vel_err = np.array([ vx_target - vel[0], vy_target - vel[1], vz_target - vel[2] ])

        P = self.kp_vel * vel_err
        D = self.kd_vel * (vel_err - self.last_vel_err) / self.dt
        I = self.ki_vel * self.vel_err_sum


        target_pitch = np.clip(   P[0] + I[0] + D[0],  -self.max_tilt, self.max_tilt)
        target_roll  = np.clip( -(P[1] + I[1] + D[1]), -self.max_tilt, self.max_tilt)
        throttle = self.hover_throttle + P[2] + I[2] + D[2]

        self.last_vel_err = vel_err
        self.vel_err_sum += vel_err * self.dt

        return np.array([target_roll, target_pitch, yaw_rate_target, throttle])
        

    def f1(self, omega:float):
        return self.rot(omega) * self.k_f1


    def f2(self, omega:float):
       return self.rot(omega) * self.k_f2


    @staticmethod
    def rot(omega: float) -> np.array:
        return np.array([[np.cos(omega), -1*np.sin(omega),0,0],[np.sin(omega),np.cos(omega),0,0],[0,0,1,0],[0,0,0,1]])


