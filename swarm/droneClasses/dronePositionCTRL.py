from swarm.droneClasses.droneCTRL import DroneCTRL
import numpy as np

class DronePositionCTRL(DroneCTRL):

    def __init__(self, drone_entity, dt=0.01):
        """Initialize position-loop gains for drone_entity (DroneStruct).

        dt is the control timestep in seconds. Gains and four-element buffers use
        the component order [x, y, z, yaw]. The f1/f2 coefficients are an empirical
        velocity-response model, not a complete quadrotor dynamics model.
        """
        super().__init__(drone_entity, dt)

        self.k_f1 = np.array([0.5, 0.5, 0.5, 0.8])  
        self.k_f2 = np.array([0.15, 0.15, 0.2, 0.2])
        self.kp_pos = np.diag([0.45, 0.45, 1, 0.06])
        self.kd_pos = np.diag([0.01, 0.01, 0.01, 0.1])
        self.prev_U = np.zeros(4)

    def position_control(self, setpoint: np.array) -> np.array:

        """Calculate a bounded velocity command from a position/yaw target.

        Args:
            setpoint: NumPy (4,) [x, y, z, yaw]: world coordinates in metres,
                yaw in radians. Smooth references avoid large numerical derivatives.

        Returns:
            U: NumPy (4,) [vx, vy, vz, body_z_rate]; world linear velocities in
            m/s and angular rate in rad/s, each clipped to [-2, 2]. This method
            does not actuate motors; pass U to lowLevelControl().

        S_d and S_dd are numerical reference velocity and acceleration. X_err
        is target minus estimated state, and X_err_d is reference minus measured
        velocity. The fourth measured velocity uses body z gyro rate as an
        approximation to Euler yaw rate; these differ when significantly tilted.
        omega is the historical name for yaw ANGLE here, not angular velocity.
        """
        if not hasattr(self, 'pr_setpoint'):
            self.pr_setpoint = setpoint.copy()
            self.pr_dr_setpoint = np.zeros(4)

        omega = self.get_attitude()[2]
        omega_d = self.get_ang_vel()[2]
        S_d = (setpoint -  self.pr_setpoint)/self.dt
        S_dd = (S_d - self.pr_dr_setpoint)/self.dt

        # Historical name: X_body actually holds WORLD position, then yaw.
        X_body = self.get_imu_pos()
        X_d_world = self.get_lin_vel()
        

        X_d = np.append(X_d_world, omega_d)
        X_d_body = self.rot(omega).T @ X_d
        X_body   = np.append(X_body, omega)

        X_d = self.rot(omega) @ X_d_body
        
        X_err = setpoint - X_body
        X_err[3] = np.arctan2(np.sin(X_err[3]), np.cos(X_err[3]))
        X_err_d = S_d - X_d

        # Desired world acceleration (fourth component is angular acceleration).
        v = S_dd + self.kp_pos @ X_err + self.kd_pos @ X_err_d

        U = np.linalg.inv(self.f1(omega)) @ (v + self.f2(omega) @ X_d_body)
        U = np.clip(self.rot(omega) @ U, -2, 2)
        self.prev_U = U.copy()
        self.pr_setpoint = setpoint.copy()
        self.pr_dr_setpoint = S_d
        return U
       
        


    def f1(self, omega:float):
        """Return the (4, 4) input-gain map at yaw omega (radians).

        Each column of the yaw rotation is scaled by its corresponding k_f1
        coefficient, mapping the empirical body command into world acceleration.
        """
        return self.rot(omega) * self.k_f1


    def f2(self, omega:float):
       """Return the (4, 4) empirical velocity-response map at yaw omega (radians)."""
       return self.rot(omega) * self.k_f2


    @staticmethod
    def rot(omega: float) -> np.array:
        """Return a (4, 4) yaw-only body-to-world rotation for omega radians.

        It rotates the first two components of [x, y, z, yaw_component], preserving
        the last two. Its transpose maps world components back to the yaw-aligned
        body frame. Roll and pitch are intentionally absent from this outer-loop map.
        """
        return np.array([[np.cos(omega), -1*np.sin(omega),0,0],[np.sin(omega),np.cos(omega),0,0],[0,0,1,0],[0,0,0,1]])

