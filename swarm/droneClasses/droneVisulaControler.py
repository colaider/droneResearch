from swarm.droneClasses.dronePositionCTRL import DronePositionCTRL
import numpy as np

class DroneVisulaCTRL(DronePositionCTRL):

    def __init__(self, drone_entity, dt=0.01, camera_fps=30):
        super().__init__(drone_entity, dt)
        self.cam_vel = np.zeros(3)
        self.stp_cam_count = 0
        self.stp_count = 0
        self.camera_fps = camera_fps
        self.camera_frame_interval = 1.0 / camera_fps
        self.last_frame_time = None
      

    def get_camera_lin_v(self):
        out = np.zeros(3)
        out[:2] = self.drone.frame_processor.camera_velocity[:2]
        out[2] = self.get_lin_vel()[2]
        return out

    def get_camera_vel_v(self):
        return np.append(self.get_camera_lin_v(), self.drone.frame_processor.camera_velocity[2])

    def position_ctrl_fused(self, setpoint: np.ndarray):
        fp = self.drone.frame_processor

        if not hasattr(self, 'cam_pos'):
            self.cam_pos = np.append(self.get_imu_pos(), self.get_attitude()[2]).astype(float)
            self.last_frame_idx = fp.current_frame.idx

        if fp.current_frame.idx != self.last_frame_idx:
            dt_frame = fp.dt
            vx, vy, yaw_rate = np.asarray(fp.camera_velocity, dtype=float)[:3]
            v_body = np.array([vx, vy, 0.0, yaw_rate])
            yaw_mid = self.cam_pos[3] + 0.5 * yaw_rate * dt_frame
            self.cam_pos += (self.rot(yaw_mid) @ v_body) * dt_frame
            self.last_frame_idx = fp.current_frame.idx
            
        self.cam_pos[2] = self.get_bar_z()
        U = self.position_control(setpoint)
        self.lowLevelControl(U)
        return self.cam_pos


    def camera_update_step(self, step, sim_time):
        camer_st = max(1, round(self.camera_frame_interval / self.dt))

        self.step(step)
        fp = self.drone.frame_processor
        fp.drone_pos = self.get_imu_pos()
        fp.imu_att = np.array(self.get_attitude())
        fp.drone_vel = self.get_lin_vel()
        fp.drone_ang_vel = self.get_ang_vel()
        fp.previous_cmd_vel = self.prev_U
        fp.drone_ang = self.get_attitude()[:2]
        fp.push_sensors()

        if step < 1:
            self.last_frame_time = sim_time
            fp.set_dt(self.camera_frame_interval)

        self.stp_count += 1
        if step % camer_st == 0:
            if self.last_frame_time is not None:
                actual_dt = sim_time - self.last_frame_time
                fp.set_dt(actual_dt)
            self.last_frame_time = sim_time

            self.drone.camera_step()
            self.stp_cam_count += 1

        self.drone.camera_show()
    def get_bar_z(self):
        return self.get_position()[2]