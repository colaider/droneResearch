from swarm.droneClasses.dronePositionCTRL import DronePositionCTRL
import numpy as np


class DroneVisulaCTRL(DronePositionCTRL):
    """Fuse world-frame optical-flow velocity into IMU dead reckoning."""
    def __init__(self, drone_entity, dt=0.01, camera_weight=0.2, camera_fps=30):
        """Configure the image sampling schedule and visual velocity correction.

        Args:
            drone_entity: DroneStruct containing the simulated drone, IMU,
                left/right cameras, and its VisulaAcEst frame processor.
            dt: Physics/control timestep in seconds (default 0.01 = 100 Hz).
            camera_weight: Fraction of the horizontal velocity discrepancy
                corrected per valid image sample; 0 disables vision, 1 accepts
                the filtered visual velocity. Vertical velocity is not blended.
            camera_fps: Requested image samples per simulated second. Actual
                sample intervals are quantized to physics steps and measured.
        """
        super().__init__(drone_entity, dt)
        if camera_fps <= 0 or dt <= 0 or not 0 <= camera_weight <= 1:
            raise ValueError("Positive timing and camera_weight in [0, 1] required")
        self.camera_weight = camera_weight
        self.camera_period = 1.0 / camera_fps
        self.next_camera_time = 0.0
        self.last_camera_time = None
        self.camera_fresh = False
        self.stp_cam_count = 0
        self.stp_count = 0

    def get_camera_lin_v(self):
        """Return (3,) world camera velocity [vx, vy, vz] in m/s.

        This may be the IMU fallback. Check drone.frame_processor.valid when
        distinguishing a genuine visual measurement from a fallback.
        """
        return self.drone.frame_processor.camera_velocity.copy()

    def position_ctrl_fused(self, setpoint):
        """Apply control toward setpoint using the estimate updated by camera_update_step.

        Args:
            setpoint: (4,) NumPy [x, y, z, yaw]; world position in metres and
                yaw in radians. Call camera_update_step once before this method.

        Returns:
            (3,) current estimated world position in metres. Motor RPMs are
            updated as a side effect; fusion itself occurs in camera_update_step.
        """
        self.lowLevelControl(self.position_control(setpoint))
        return self.get_imu_pos()

    def camera_update_step(self, step):
        """Update the IMU estimate and process a new camera sample when due.

        Args:
            step: Zero-based physics step counter, not a camera frame counter.
                Simulated time is step*self.dt. Call once before each scene.step().

        Updates cached position/velocity and camera diagnostics; returns None.
        A fresh valid visual sample corrects horizontal velocity once. Between
        samples the IMU propagates the estimate. cam_pos is a diagnostic integrated
        camera trajectory, with simulator altitude used for its displayed z.
        """
        self.step(step)
        processor = self.drone.frame_processor
        processor.drone_pos = self.get_imu_pos()
        processor.imu_att = np.array(self.get_attitude())
        processor.drone_vel = self.get_lin_vel()
        processor.drone_ang_vel = self.get_ang_vel()
        now = step * self.dt
        self.camera_fresh = False
        if now + 1e-9 >= self.next_camera_time:
            interval = self.camera_period if self.last_camera_time is None else now-self.last_camera_time
            self.drone.set_camera_dt(interval)
            self.drone.camera_step()
            self.last_camera_time = now
            while self.next_camera_time <= now + 1e-9:
                self.next_camera_time += self.camera_period
            self.stp_cam_count += 1
            self.camera_fresh = True
            if processor.valid:
                # Correct velocity only once per new image; subsequent IMU steps
                # propagate it in the same world frame. No repeated position blending.
                vel = self.get_lin_vel()
                corrected = vel.copy()
                corrected[:2] += self.camera_weight * (processor.camera_velocity[:2]-vel[:2])
                self.lin_pos += (corrected-vel) * self.dt
                self.v_x_sum, self.v_y_sum, self.v_z_sum = corrected
        if not hasattr(self, 'cam_pos'):
            self.cam_pos = self.get_imu_pos()
        camera_vel = processor.camera_velocity if processor.valid else self.get_lin_vel()
        self.cam_pos += camera_vel * self.dt
        self.cam_pos[2] = self.get_simulated_altitude()
        self.stp_count += 1
        self.drone.camera_show()

    def get_simulated_altitude(self):
        """Return simulator ground-truth world z in metres for diagnostics; no barometer is simulated."""
        return self.get_position()[2]

    def get_bar_z(self):
        """Legacy alias for get_simulated_altitude(); returns world z in metres, not a sensor reading."""
        return self.get_simulated_altitude()
