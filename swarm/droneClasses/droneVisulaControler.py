from swarm.droneClasses.dronePositionCTRL import DronePositionCTRL
import numpy as np

class DroneVisulaCTRL(DronePositionCTRL):

    def __init__(self, drone_entity, dt=0.01):
        super().__init__(drone_entity, dt)
        self.cam_vel = np.zeros(3)
        self.stp_cam_count = 0
        self.stp_count = 0
      

    def get_camera_lin_v(self):
        out = np.zeros(3)
        out[:2] = self.drone.frame_processor.camera_velocity[:2]
        out[2] = self.get_lin_vel()[2]
        return out



    def position_ctrl_fused(self, setpoint:np.ndarray):

        if not hasattr(self, 'cam_pos') or self.drone.frame_processor.expected_vel_err >= 20:
            self.cam_pos = self.get_imu_pos()
            self.drone.frame_processor.drone_vel = self.get_lin_vel() 

        v_c = self.drone.frame_processor.camera_velocity
        v_imu = self.get_lin_vel()
        p_imu = self.get_imu_pos() 
        p_mixed = np.zeros(4)

        a = 0.9
        p_mixed = a*p_imu + (1-a)*self.cam_pos
        # self.set_pose(p_mixed)
        self.cam_pos += v_c*self.dt
        self.cam_pos[2] = self.get_bar_z()

        U = self.position_control(setpoint)        
        self.lowLevelControl(U)
        return p_mixed



    def fuse_state(self):
        v_imu = self.get_lin_vel()
        v_cam = self.drone.frame_processor.camera_velocity
        
        # Trust weight based on flow quality
        err = self.drone.frame_processor.expected_vel_err
        trust_cam = np.clip(1.0 - err / 20.0, 0.0, 1.0)  # 0 when err>=20, 1 when err=0
        alpha_v = 0.7 * trust_cam  # camera influence on velocity
        
        v_fused = (1 - alpha_v) * v_imu + alpha_v * v_cam
        
        # Integrate fused velocity into position estimate
        if not hasattr(self, 'p_estimate'):
            self.p_estimate = self.get_imu_pos()
        self.p_estimate += v_fused * self.dt
        
        # Anchor altitude to barometer (drift correction)
        self.p_estimate[2] = 0.98 * self.p_estimate[2] + 0.02 * self.get_bar_z()
        
        # Slow correction toward IMU absolute position (prevents unbounded drift)
        p_imu = self.get_imu_pos()
        
        self.p_estimate = 0.995 * self.p_estimate + 0.005 * p_imu
        
        return self.p_estimate, v_fused


    def camera_update_step(self, step):
        fps = 30
        camer_st = max(1, round((1 / fps) / self.dt))   # whole sim steps per frame -> 3
        dt_camera = camer_st * self.dt                  # real frame interval -> 0.03 s

        self.step(step)

        self.drone.frame_processor.drone_pos = self.get_imu_pos()
        self.drone.frame_processor.imu_att = np.array(self.get_attitude())
        self.drone.frame_processor.drone_vel =  self.get_lin_vel()
        self.drone.frame_processor.drone_ang_vel = self.get_ang_vel()
        self.drone.frame_processor.previous_cmd_vel = self.prev_U
        self.drone.frame_processor.drone_ang = self.get_attitude()[:2]
        self.drone.frame_processor.push_sensors()

        if step < 1: 
            self.drone.set_camera_dt(dt_camera)

        self.stp_count += 1
        if step % camer_st == 0: 
            self.drone.camera_step()
            self.stp_cam_count += 1

        self.drone.camera_show()


    def get_bar_z(self):
        return self.get_position()[2]