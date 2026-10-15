from swarm.droneClasses.dronePositionCTRL import DronePositionCTRL
from swarm.filters.FKF import FusionKalmanFilter
import numpy as np
import time

class DroneVisulaCTRL(DronePositionCTRL):
    def __init__(self, drone_entity, dt=0.01, camera_fps=30):
        super().__init__(drone_entity, dt)
        # self.camera_frame_interval = 1.0 / camera_fps
        self.camera_frame_interval  = 0.01
        self.camera_step_period = max(1, round(self.camera_frame_interval / dt))
        self.last_frame_time = None
        self.cam_pos = None                      # lazily initialized on first position_ctrl_fused call
        self.raw_cam_pos = None                  # [x, y, z, yaw] dead-reckoned from camera velocity only
        self._raw_cam_idx = None
        self._prev_frame_att = None              # attitude at the previous camera frame
       
        # 4D fusion: body [vx, vy, vz, yaw_rate]
        self.fkf = FusionKalmanFilter()
        self.fused_vel = np.zeros(3)             # [vx, vy, vz] body, for logging
        self.fused_yaw_rate = 0.0



        self.calibrated = False

    # ---------------- camera velocity accessors ----------------

    def get_camera_lin_v(self):
        """Camera-estimated body velocity, with vz from accelerometer."""
        cv = self.drone.frame_processor.camera_velocity
        return np.array([cv[0], cv[1], cv[2]])

    def get_camera_vel_v(self):
        """[vx, vy, vz, yaw_rate]."""
        cv = self.drone.frame_processor.camera_velocity
        return np.array([cv[0], cv[1], cv[2], cv[3]])

    def integrate_camera_position(self):
        """Add one camera-frame step of raw visual-odometry velocity to raw_cam_pos."""
        fp = self.drone.frame_processor
        if fp.current_frame.idx == self._raw_cam_idx:
            return
        self._raw_cam_idx = fp.current_frame.idx
        self.raw_cam_pos += fp.dt * self.get_camera_vel_v()

  # ---------------- fused control ----------------
    def position_ctrl_fused(self, setpoint):
        fp = self.drone.frame_processor
        self.integrate_camera_position()


        # Raw sensor readings
        accel_v = self.get_lin_vel()                 # [vx, vy, vz]  (3D now)
        cam_v   = fp.camera_velocity                 # [vx, vy, vz, yaw_rate]
        command = np.append(self.prev_U[:3], 0.0)    # [vx, vy, vz, 0]

        # Camera: vz is noise (R_camera[2]=1e6), so pass 0.0 and let R kill it
        cam_full = np.array([cam_v[0], cam_v[1], 0.0, cam_v[3]])

        fused = self.fkf.step(
            command=command,
            v_accel=accel_v,            # 3D now, not 4D
            dt=self.dt,
            v_camera=cam_full,
            camera_id=fp.current_frame.idx,
            attitude=self.get_attitude()
        )

        # Fused 8D state: [px, py, pz, yaw, vx, vy, vz, yaw_rate]
        self.cam_pos[0:3]   = fused[0:3]
        self.cam_pos[3]     = fused[3]
        self.fused_vel      = fused[4:7]
        self.fused_yaw_rate = fused[7]

        # Closed loop control on fused state
        U = self.position_control(setpoint, X_body=self.cam_pos[0:3], X_d_body=self.fused_vel)
        self.lowLevelControl(U)
        return self.cam_pos


    def camera_update_step(self, step, sim_time):
        self.step(step)

        fp = self.drone.frame_processor
        fp.drone_pos = self.get_imu_pos()
        fp.imu_att = np.array(self.get_attitude())
        fp.drone_vel = self.get_lin_vel()
        fp.drone_ang_vel = self.get_ang_vel()
        fp.previous_cmd_vel = self.prev_U
        fp.drone_ang = self.get_attitude()[:2]
        if step % self.camera_step_period == 0:
            if self.last_frame_time is not None:
                fp.set_dt(sim_time - self.last_frame_time)
            else:
                fp.set_dt(self.camera_frame_interval)
            # Image rotation between exposures is the attitude change, not the sampled gyro.
            if self._prev_frame_att is not None:
                delta = fp.imu_att - self._prev_frame_att
                fp.drone_ang_vel = ((delta + np.pi) % (2 * np.pi) - np.pi) / fp.dt
            self._prev_frame_att = fp.imu_att.copy()
            self.last_frame_time = sim_time
            self.drone.camera_step()

        self.drone.camera_show()


    def calibrate(self, duration=1.0, p0=None, yaw0=None):
        n_samples = max(1, int(duration / self.dt))
        gyro_samples  = []
        accel_samples = []

        for _ in range(n_samples):
            gyro_samples.append(self.get_ang_vel())
            time.sleep(self.dt)

        gyro_samples  = np.asarray(gyro_samples)

        self.gyro_bias = gyro_samples.mean(axis=0)

        if p0 is None:   p0 = np.zeros(3)
        if yaw0 is None: yaw0 = 0.0
        self.fkf.reset_position(p=np.asarray(p0, float), yaw=float(yaw0))
        
        self.fkf.x[4:8] = 0.0
        self.cam_pos = np.append(p0, yaw0).astype(float)
        self.raw_cam_pos = self.cam_pos.copy()

        self.calibrated = True
        return self.gyro_bias


    def get_bar_z(self):
        return self.get_position()[2]


    