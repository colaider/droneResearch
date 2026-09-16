from swarm.droneClasses.droneCTRL import DroneCTRL
import numpy as np

class DronePositionCTRL(DroneCTRL):

    def __init__(self, drone_entity, dt=0.01):
        super().__init__(drone_entity, dt)

        self.k_f1 = np.array([0.5, 0.5, 0.5, 0.8])  
        self.k_f2 = np.array([0.15, 0.15, 0.2, 0.2])
        self.kp_pos = np.diag([0.45, 0.45, 1, 0.06])
        self.kd_pos = np.diag([0.01, 0.01, 0.01, 0.1])

        self.PCS = False


    def position_control(self, setpoint: np.array) -> np.array:

        if not hasattr(self, 'pr_setpoint'):
            self.pr_setpoint = setpoint.copy()
            self.pr_dr_setpoint = np.zeros(4)

        omega = self.get_attitude()[2]
        omega_d = self.get_ang_vel()[2]
        S_d = (setpoint -  self.pr_setpoint)/self.dt
        S_dd = (S_d - self.pr_dr_setpoint)/self.dt

        X_body = self.get_imu_pos()
        X_d_body = self.get_lin_vel()

        X_d_body = np.append(X_d_body, omega_d)
        X_body   = np.append(X_body, omega)

        X_d = self.rot(omega) @ X_d_body
        
        X_err = setpoint - X_body
        X_err_d = X_d - S_d

        v = S_dd + self.kp_pos @ X_err + self.kd_pos @ X_err_d

        U = np.linalg.inv(self.f1(omega)) @ (v + self.f2(omega) @ X_d_body)
        print(U)
        np.clip(U, -2.5,2.5)
        self.lowLevelControl(U)
        self.pr_setpoint = setpoint
        self.pr_dr_setpoint = S_d


    def f1(self, omega:float):
        return self.rot(omega) * self.k_f1


    def f2(self, omega:float):
       return self.rot(omega) * self.k_f2


    @staticmethod
    def rot(omega: float) -> np.array:
        return np.array([[np.cos(omega), -1*np.sin(omega),0,0],[np.sin(omega),np.cos(omega),0,0],[0,0,1,0],[0,0,0,1]]) 