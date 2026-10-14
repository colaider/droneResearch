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
        self.imu.update(accel, gyro, dt)
        self._update_baro_velocity(float(baro), dt)
        if not self.imu.is_calibrating():
            self.imu.fuse_vertical_velocity(self._baro_vel, 0.9)   # vz = 0.9 accel + 0.1 baro
        self._push_visual_sensor_sample()
        if self.camera_receiver is not None:
            frame = self.camera_receiver.get_frame()
            if frame is not None:
                self.process_visual_frame(frame)

    def _push_visual_sensor_sample(self):
        estimator = self.visual_estimator
        estimator.drone_pos = self.get_fused_position()
        estimator.drone_vel = self.get_acc_velocity()
        estimator.drone_ang_vel = self.get_angular_velocity()
        estimator.imu_att = self.get_attitude()
        estimator.push_sensors()

    def process_visual_frame(self, frame, idx=None):
        """Process a stereo capture using sensor samples buffered since the prior pair."""
        return self.visual_estimator.processing(frame, idx)

    def get_visual_velocity(self):
        """Return latest camera velocity [vx, vy, vz, yaw_rate]."""
        return self.visual_estimator.camera_velocity.copy()

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
        # For now just the IMU position. Later: fuse the visual estimator / baro here.
        return self.imu.get_position()
