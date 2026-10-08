from swarm.droneClasses.dronePositionCTRL import DronePositionCTRL
from swarm.filters.FKF import FusionKalmanFilter
import numpy as np
from swarm.droneClasses.stateMachine import Step, StateMachine

class DroneVisulaCTRL(DronePositionCTRL):
    def __init__(self, drone_entity, dt=0.01, camera_fps=30):
        super().__init__(drone_entity, dt)
        self.camera_frame_interval = 1.0 / camera_fps
        self.camera_step_period = max(1, round(self.camera_frame_interval / dt))
        self.last_frame_time = None
        self.cam_pos = None                      # lazily initialized on first position_ctrl_fused call
        self.mission = self._build_mission()
        # 4D fusion: body [vx, vy, vz, yaw_rate]
        self.fkf = FusionKalmanFilter(dim=4)
        self.fused_vel = np.zeros(3)             # [vx, vy, vz] body, for logging
        self.fused_yaw_rate = 0.0

    # ---------------- camera velocity accessors ----------------

    def get_camera_lin_v(self):
        """Camera-estimated body velocity, with vz from accelerometer."""
        cv = self.drone.frame_processor.camera_velocity
        return np.array([cv[0], cv[1], self.get_lin_vel()[2]])

    def get_camera_vel_v(self):
        """[vx, vy, vz, yaw_rate]."""
        cv = self.drone.frame_processor.camera_velocity
        return np.array([cv[0], cv[1], self.get_lin_vel()[2], cv[3]])

    # ---------------- fused control ----------------
    def position_ctrl_fused(self, setpoint):
        fp = self.drone.frame_processor

        if self.cam_pos is None:
            self.cam_pos = np.append(self.get_imu_pos(), self.get_attitude()[2]).astype(float)

        # Raw sensor readings
        accel_v = self.get_lin_vel()                       # body [vx, vy, vz]
        cam_v = fp.camera_velocity                         # body [vx, vy, vz, yaw_rate]

        # Build 4D vectors — pad axes a sensor doesn't measure with 0 (filter ignores via large R)
        accel_full = np.array([accel_v[0], accel_v[1], accel_v[2], 0.0])
        cam_full   = np.array([cam_v[0],   cam_v[1],   0.0,         cam_v[3]])
        command    = np.append(self.prev_U[:3], 0.0)       # [vx, vy, vz, 0]

        fused = self.fkf.step(
            command=command,
            v_accel=accel_full,
            dt=self.dt,
            v_camera=cam_full,
            camera_id=fp.current_frame.idx,
            attitude=self.get_attitude()
        )

        self.fused_vel = fused[:3]
        self.fused_yaw_rate = fused[3]

        # Dead-reckon world pose using the fused state
        yaw_mid = self.cam_pos[3] + 0.5 * self.fused_yaw_rate * self.dt
        v_body = np.array([fused[0], fused[1], fused[2], self.fused_yaw_rate])
        self.cam_pos += (self.rot(yaw_mid) @ v_body) * self.dt

        # Closed loop control on fused state
        U = self.position_control(setpoint, X_body=self.cam_pos[:3], X_d_body=self.fused_vel)
        self.lowLevelControl(U)
        return self.cam_pos

    # ---------------- stepping ----------------

    def camera_update_step(self, step, sim_time):
        self.step(step)

        fp = self.drone.frame_processor
        fp.drone_pos = self.get_imu_pos()
        fp.imu_att = np.array(self.get_attitude())
        fp.drone_vel = self.get_lin_vel()
        fp.drone_ang_vel = self.get_ang_vel()
        fp.previous_cmd_vel = self.prev_U
        fp.drone_ang = self.get_attitude()[:2]
        fp.push_sensors()

        if step % self.camera_step_period == 0:
            if self.last_frame_time is not None:
                fp.set_dt(sim_time - self.last_frame_time)
            else:
                fp.set_dt(self.camera_frame_interval)
            self.last_frame_time = sim_time
            self.drone.camera_step()

        self.drone.camera_show()

    # Add this to the imports in your visual controller file:
    # from <your module> import Step, StateMachine
    def get_bar_z(self):
        return self.get_position()[2]