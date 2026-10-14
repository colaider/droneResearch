import numpy as np

from sensors.IMU.IMU import IMU
from sensors.visual.config.cameraConfig import STEREO_CAM


class Sensors:
    def __init__(self, baro_vel_ema=0.8, *, node=None, camera_ns="/stereo",
                 camera_config=None, image_filters=("clahe",),
                 convert_to_grayscale=True, filters_enabled=True,
                 filter_kernel_size=3, pyramid_levels=2,
                 max_stereo_skew_s=0.005):
        from sensors.visual import VisulaEst

        self.camera_receiver = None
        if node is not None:
            from communication.CameraStreamReceiver import CameraStreamReceiver

            self.camera_receiver = CameraStreamReceiver(node, ns=camera_ns)

        self.imu = IMU()                  # accel + gyro estimator

        # barometer (kept here; it is not part of the IMU)
        self.baro_vel_ema = baro_vel_ema
        self._baro = 0.0
        self._baro_vel = 0.0
        self._prev_baro = None
        self.latest_camera_frame = None
        self._visual_position = None
        self._visual_velocity_world = None     # latest camera velocity in world frame (nan axis = no measurement)
        self._last_visual_frame_idx = None
        self.visual_estimator = VisulaEst(
            STEREO_CAM if camera_config is None else camera_config,
            image_filters=image_filters,
            convert_to_grayscale=convert_to_grayscale,
            filters_enabled=filters_enabled,
            filter_kernel_size=filter_kernel_size,
            pyramid_levels=pyramid_levels,
            max_stereo_skew_s=max_stereo_skew_s,
        )

    # ---------------- update ----------------
    def update(self, accel, gyro, baro, dt):
        was_calibrating = self.imu.is_calibrating()
        self.imu.update(accel, gyro, dt)
        if was_calibrating and not self.imu.is_calibrating():
            self._visual_position = None
            self._visual_velocity_world = None
            self._last_visual_frame_idx = None
        self._update_baro_velocity(float(baro), dt)
        if not self.imu.is_calibrating():
            self.imu.fuse_vertical_velocity(self._baro_vel, 0.9)   # vz = 0.9 accel + 0.1 baro
        self._push_visual_sensor_sample()
        if self.camera_receiver is not None:
            frame = self.camera_receiver.get_frame()
            if frame is not None:
                self.latest_camera_frame = frame
                self.process_visual_frame(frame)

    def _push_visual_sensor_sample(self):
        estimator = self.visual_estimator
        estimator.drone_pos = self.get_fused_position()
        estimator.drone_vel = self.get_acc_velocity()
        estimator.drone_ang_vel = self.get_angular_velocity()
        estimator.imu_att = self.get_attitude()
        estimator.push_sensors()

    def process_visual_frame(self, frame, idx=None):
        """Process a stereo capture and integrate valid visual velocity into position."""
        processed = self.visual_estimator.processing(frame, idx)
        frame_idx = processed.idx
        if frame_idx == self._last_visual_frame_idx:
            return processed
        self._last_visual_frame_idx = frame_idx

        estimator = self.visual_estimator
        if (np.count_nonzero(estimator.velocity_point_mask) < 2
                or not np.isfinite(estimator.camera_velocity[:3]).all()):
            return processed

        imu_position = self.get_acc_position()
        dt = float(estimator.dt)
        if not np.isfinite(dt) or dt <= 0:
            return processed

        if self._visual_position is None:
            self._visual_position = imu_position.copy()

        roll, pitch, yaw = self.get_attitude()
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        body_to_world = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
        has_visual_vertical_velocity = np.isfinite(estimator.depth_velocity_z)
        visual_velocity = estimator.camera_velocity[:3].copy()
        if not has_visual_vertical_velocity:
            visual_velocity[2] = 0.0
        visual_velocity_world = body_to_world @ visual_velocity
        if not has_visual_vertical_velocity:
            visual_velocity_world[2] = 0.0
        self._visual_position += visual_velocity_world * dt
        if not has_visual_vertical_velocity:
            self._visual_position[2] = imu_position[2]

        # keep the latest world-frame visual velocity for get_fused_velocity(); nan on z when
        # the camera gave no depth velocity, so the fuse leaves vz to the IMU/baro.
        held = visual_velocity_world.copy()
        if not has_visual_vertical_velocity:
            held[2] = np.nan
        self._visual_velocity_world = held
        return processed

    def get_visual_velocity(self):
        """Return latest camera velocity [vx, vy, vz, yaw_rate]."""
        return self.visual_estimator.camera_velocity.copy()

    def get_visual_position(self):
        """Return integrated visual position, or None until visual tracks are usable."""
        return None if self._visual_position is None else self._visual_position.copy()

    def close(self):
        """Release optional camera-topic subscriptions."""
        if self.camera_receiver is not None:
            self.camera_receiver.close()
            self.camera_receiver = None

    def _update_baro_velocity(self, baro, dt):
        self._baro = baro
        if self._prev_baro is not None and dt > 0:
            raw = (baro - self._prev_baro) / dt
            b = self.baro_vel_ema
            self._baro_vel = b * self._baro_vel + (1 - b) * raw
        self._prev_baro = baro

    # ---------------- calibration (forwarded to the IMU) ----------------
    def start_calibration(self, n=400):
        self._visual_position = None
        self._visual_velocity_world = None
        self._last_visual_frame_idx = None
        self.imu.start_calibration(n)

    def is_calibrating(self):
        return self.imu.is_calibrating()

    # ---------------- forwarded IMU estimates ----------------
    def get_attitude(self):
        return self.imu.get_attitude()

    def get_acc_velocity(self):
        return self.imu.get_velocity()

    def get_acc_position(self):
        return self.imu.get_position()

    def get_acceleration(self):
        return self.imu.get_acceleration()

    def get_angular_velocity(self):
        return self.imu.get_angular_velocity()

    # ---------------- barometer ----------------
    def get_baro_position(self):
        return self._baro                 # barometer altitude [m]

    def get_baro_velocity(self):
        return self._baro_vel             # vertical velocity from the barometer

    # ---------------- fused ----------------
    def get_fused_position(self):
        imu_position = self.get_acc_position()
        if self._visual_position is None:
            return imu_position
        return 0.9 * imu_position + 0.1 * self._visual_position

    def get_fused_velocity(self):
        imu_velocity = self.get_acc_velocity()
        visual = self._visual_velocity_world
        if visual is None:
            return imu_velocity
        fused = imu_velocity.copy()
        usable = np.isfinite(visual)
        fused[usable] = 0.9 * imu_velocity[usable] + 0.1 * visual[usable]
        return fused
