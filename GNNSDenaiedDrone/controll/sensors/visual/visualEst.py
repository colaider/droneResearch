"""Coordinate frame preparation, temporal tracking, and stereo depth estimation."""

import threading
from collections import deque

import numpy as np

from .config.cameraConfig import StereoCameraConfig
from .frame import FrameProcessor
from .estimatorTypes import Frame, StereoFrame
from .visualDepthEst import (
    SparseStereoDepth,
    estimate_vertical_velocity,
    estimate_frame_depth,
)
from .visualVelEst import VisualVelocityEstimator, VelocityKalmanFilter


# Preserve the original public frame name.
EnumFrame = Frame


class VisulaEst(VisualVelocityEstimator):
    """Visual-odometry estimator configured by a camera rig and frame options.

    After ``processing()``, read the latest ``camera_velocity``, ``depth_points``,
    ``depth_colors`` and ``depth_sigma`` from this object.
    """

    def __init__(self, camera_config: StereoCameraConfig, *, image_filters: tuple[str, ...],
                 convert_to_grayscale: bool, filters_enabled: bool,
                 filter_kernel_size: int, pyramid_levels: int,
                 max_stereo_skew_s: float):
        if not np.isfinite(max_stereo_skew_s) or max_stereo_skew_s < 0:
            raise ValueError('max_stereo_skew_s must be finite and nonnegative')

        self.camera_config = camera_config
        self.max_stereo_skew_s = float(max_stereo_skew_s)
        self.fov = float(camera_config.fov)
        self.foc_l = float(camera_config.focal_px)
        self.dt = 0.01
        self.camera_saperation = float(camera_config.baseline)
        self.camera_offset_from_yaw_center = np.asarray(
            camera_config.camera_offset_from_yaw_center, dtype=float
        )
        self.disparity_direction = np.asarray(
            camera_config.disparity_direction(), dtype=float
        )

        self.frame_processor = FrameProcessor(
            image_filters=image_filters,
            convert_to_grayscale=convert_to_grayscale,
            filters_enabled=filters_enabled,
            filter_kernel_size=filter_kernel_size,
            pyramid_levels=pyramid_levels,
        )
        self.resolution = tuple(camera_config.res)
        self.use_stereo_depth = True
        self.current_frame = EnumFrame()
        self.buffer = deque(maxlen=100)

        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_ang_vel = np.zeros(3)
        self.imu_att = np.zeros(3)
        self.drone_cmd_vel = np.zeros(4)
        self.previous_cmd_vel = np.zeros(4)

        self.buf_ang_vel = deque(maxlen=20)
        self.buf_att = deque(maxlen=20)
        self.buf_pos = deque(maxlen=20)
        self.buf_cmd_vel = deque(maxlen=20)
        self._sensor_buffer_lock = threading.Lock()
        self.avg_ang_vel = np.zeros(3)
        self.avg_att = np.zeros(3)
        self.avg_pos = np.zeros(3)

        self._init_velocity_estimator(self.frame_processor.pyramid_levels)
        self.depth_median = np.nan
        self.depth_valid_frac = 0.0
        self.depth_points = np.empty((0, 3))
        self.previous_depth = None
        self.current_depth = None
        self.depth_velocity_z = np.nan
        self.stereo = SparseStereoDepth()
        self.depth_by_layer = []
        self.depth_colors = np.empty((0, 3), np.uint8)
        self.depth_sigma = np.empty(0)
        self.depth_view = None
        self.depth_frame_idx = -1

    @property
    def image_filters(self):
        return self.frame_processor.image_filters

    @image_filters.setter
    def image_filters(self, value):
        self.frame_processor.image_filters = value

    @property
    def filter_kernel_size(self):
        return self.frame_processor.filter_kernel_size

    @filter_kernel_size.setter
    def filter_kernel_size(self, value):
        self.frame_processor.filter_kernel_size = value

    @property
    def clahe(self):
        return self.frame_processor.clahe

    @clahe.setter
    def clahe(self, value):
        self.frame_processor.clahe = value

    def preprocess_frame(self, frame):
        """Filter an already-normalized BGR image."""
        return self.frame_processor.preprocess(frame)

    def update_frame(self, frame, idx=None):
        next_dt = None
        if isinstance(frame, StereoFrame):
            if frame.stereo_skew > self.max_stereo_skew_s:
                raise ValueError('Stereo capture skew exceeds max_stereo_skew_s')
            if self.buffer:
                previous = self.buffer[-1]
                if frame.sequence_id <= previous.idx:
                    raise ValueError('Stereo sequence IDs must increase')
                if previous.left_timestamp is not None:
                    next_dt = frame.left_timestamp - previous.left_timestamp
                    if next_dt <= 0 or frame.right_timestamp <= previous.right_timestamp:
                        raise ValueError('Stereo capture timestamps must increase for both cameras')
        prepared = self.frame_processor.process(frame, idx)
        height, width = prepared.frames[0].shape[:2]
        if (width, height) != self.resolution:
            raise ValueError(
                "Frame resolution differs from calibration; create a new estimator "
                "for this output size"
            )
        if next_dt is not None:
            self.set_dt(float(next_dt))
        self.frame_size = prepared.frames[0].shape
        self.buffer.append(prepared)
        self.current_frame = prepared.display_copy()

    def processing(self, frame, idx=None, *, drone_pos=None, drone_vel=None,
                   drone_ang_vel=None, imu_att=None):
        """Process StereoFrame, or legacy [left,right] plus idx.

        Sensor values may be supplied together to update the estimator and
        buffer one sensor sample as part of this call. Existing callers may
        continue buffering faster sensor samples with ``push_sensors()``.
        Returns the display Frame. Read camera_velocity ([vx, vy, vz, yaw_rate]),
        depth_points ([u, v, optical_depth]), depth_colors, and depth_sigma on
        this estimator. Capture metadata is on current_frame.
        Typed inputs use left capture-time differences for dt. Legacy callers
        retain their explicit set_dt() convention and have no invented timestamp.
        """
        sensor_values = (drone_pos, drone_vel, drone_ang_vel, imu_att)
        if any(value is not None for value in sensor_values):
            if any(value is None for value in sensor_values):
                raise ValueError(
                    "Provide drone_pos, drone_vel, drone_ang_vel, and imu_att together"
                )
            self.drone_pos = np.asarray(drone_pos, dtype=float)
            self.drone_vel = np.asarray(drone_vel, dtype=float)
            self.drone_ang_vel = np.asarray(drone_ang_vel, dtype=float)
            self.imu_att = np.asarray(imu_att, dtype=float)
            self.push_sensors()

        self.update_frame(frame, idx)
        self.aply_flow()
        self.estimate_depth()
        return_velocity = self.estimate_velocities(
            self.velocity_flows[self.velocity_point_mask],
            self.velocity_points[self.velocity_point_mask],
            vertical_velocity=self.depth_velocity_z,
        )

        self.camera_velocity = return_velocity
        return self.current_frame

    def estimate_depth(self):
        """Estimate current-frame left/right depth after the temporal velocity fit."""
        self.depth_frame_idx = self.current_frame.idx
        result = estimate_frame_depth(
            self.current_frame,
            self.all_points,
            self.point_colors,
            self.layer_tracks,
            self.stereo,
            use_stereo_depth=self.use_stereo_depth,
            focal_px=self.foc_l,
            baseline=self.camera_saperation,
            direction=self.disparity_direction,
            cross_layer_min_distance=self.cross_layer_min_distance,
        )
        self.previous_depth = self.current_depth
        self.current_depth = result
        self.depth_velocity_z = estimate_vertical_velocity(
            self.previous_depth, self.current_depth, self.dt
        )
        self.depth_by_layer = result.by_layer
        self.depth_points = result.points
        self.depth_colors = result.colors
        self.depth_sigma = result.sigma
        self.depth_median = result.median
        self.depth_valid_frac = result.valid_fraction
        self.depth_view = result.view

    def push_sensors(self):
        with self._sensor_buffer_lock:
            self.buf_ang_vel.append(np.array(self.drone_ang_vel, dtype=float))
            self.buf_att.append(np.array(self.imu_att, dtype=float))
            self.buf_pos.append(np.array(self.drone_pos, dtype=float))
            self.buf_cmd_vel.append(np.array(self.drone_cmd_vel, dtype=float))

    def _take_sensor_window(self):
        with self._sensor_buffer_lock:
            if self.buf_ang_vel:
                self.avg_ang_vel = np.mean(np.array(self.buf_ang_vel), axis=0)
                self.avg_att = np.mean(np.array(self.buf_att), axis=0)
                self.avg_pos = np.mean(np.array(self.buf_pos), axis=0)
                self.previous_cmd_vel = np.mean(np.array(self.buf_cmd_vel), axis=0)
            else:
                self.avg_ang_vel = np.array(self.drone_ang_vel, dtype=float)
                self.avg_att = np.array(self.imu_att, dtype=float)
                self.avg_pos = np.array(self.drone_pos, dtype=float)
                self.previous_cmd_vel = np.array(self.drone_cmd_vel, dtype=float)
            self.buf_ang_vel.clear()
            self.buf_att.clear()
            self.buf_pos.clear()
            self.buf_cmd_vel.clear()

    def _current_height(self):
        return float(self.avg_pos[2])
