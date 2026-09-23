from swarm.droneClasses.dronePositionCTRL import DronePositionCTRL
import numpy as np

class DroneVisulaCTRL(DronePositionCTRL):

    def __init__(self, drone_entity, dt=0.01):
        super().__init__(drone_entity, dt)
        self.cam_vel = np.zeros(3)


    def get_camera_lin_v(self):
        out = np.zeros(3)
        out[:2] = self.drone.frame_processor.camera_velocity[:2]
        out[2] = self.get_lin_vel()[2]
        return out



    def position_ctrl_fused(self,setpoint:np.ndarray):
        v_c = self.drone.frame_processor.camera_velocity
        v_imu = self.get_lin_vel()
        p_imu = self.get_imu_pos() 



        U = self.position_control(setpoint)
        




        self.lowLevelControl(U)


    def camera_update_step(self, step):
        fps = 30
        dt_camera = 1/fps
        camer_st = int(dt_camera/ self.dt)

        self.step(step)


        self.drone.frame_processor.drone_pos = self.get_imu_pos()
        self.drone.frame_processor.imu_att = np.array(self.get_attitude())
        self.drone.frame_processor.drone_vel =  self.get_lin_vel()
        self.drone.frame_processor.drone_ang_vel = self.get_ang_vel()
        self.drone.frame_processor.previous_cmd_vel = self.prev_U
        if step < 1: self.drone.set_camera_dt(dt_camera)
                
        if step % camer_st == 0: self.drone.camera_step()
        self.drone.camera_show()