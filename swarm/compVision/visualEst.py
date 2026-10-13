"""Coordinate frame preparation, temporal tracking, and stereo depth estimation."""

from collections import deque

import numpy as np

import swarm.config.cameraConfig as camConfig
from swarm.compVision.frame import Frame, FrameProcessor, compose_layer_display
from swarm.compVision.smallFeatures import LAYER_COLORS
from swarm.compVision.visualDepthEst import (
    SparseStereoDepth,
    estimate_vertical_velocity,
    estimate_frame_depth,
)
from swarm.compVision.visualVelEst import VisualVelocityEstimator, VelocityKalmanFilter


# Preserve the original public frame name.
EnumFrame = Frame


class VisulaEst(VisualVelocityEstimator):
    """Coordinate prepared frames with temporal visual velocity and stereo depth."""

    def __init__(self, res, fov):
        self.fov = fov
        self.foc_l = (res[1] / 2) / np.tan(np.deg2rad(fov) / 2)
        self.dt = 0.01
        self.camera_saperation = camConfig.STEREO_CAM.baseline

        self.frame_processor = FrameProcessor(
            pyramid_levels=2, image_filters=("clahe",), filter_kernel_size=3
        )
        self.resolution = tuple(res)
        self.use_stereo_depth = True
        self.current_frame = EnumFrame()
        self.buffer = deque(maxlen=100)

        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_ang_vel = np.zeros(3)
        self.imu_att = np.zeros(3)
        self.previous_cmd_vel = np.zeros(4)

        self.buf_ang_vel = deque(maxlen=20)
        self.buf_att = deque(maxlen=20)
        self.buf_pos = deque(maxlen=20)
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

    def update_frame(self, frame, idx):
        prepared = self.frame_processor.process(frame, idx)
        height, width = prepared.frames[0].shape[:2]
        if (width, height) != self.resolution:
            raise ValueError(
                "Frame resolution differs from calibration; create a new estimator "
                "for this output size"
            )
        self.frame_size = prepared.frames[0].shape
        self.buffer.append(prepared)
        self.current_frame = prepared.display_copy()

    def processing(self, frame, idx):
        """Run frame preparation, left-camera tracking, velocity, then stereo depth."""
        self.update_frame(frame, idx)
        self.aply_flow()
        self.estimate_depth()
        return_velocity = self.estimate_velocities(
            self.velocity_flows[self.velocity_point_mask],
            self.velocity_points[self.velocity_point_mask],
            vertical_velocity=self.depth_velocity_z,
        )

        self.layer_views = [
            compose_layer_display(
                gray,
                self.all_points[layer] / (2 ** layer),
                self.layer_tracks[layer].links,
                layer,
                LAYER_COLORS[layer % len(LAYER_COLORS)],
            )
            for layer, gray in enumerate(self.current_frame.pyramids[0])
        ]
        self.camera_velocity = return_velocity
        return self.current_frame

    def estimate_depth(self):
        """Estimate current-frame left/right depth after the temporal velocity fit."""
        self.depth_frame_idx = self.current_frame.idx
        direction = getattr(
            self, "disparity_direction", camConfig.STEREO_CAM.disparity_direction()
        )
        result = estimate_frame_depth(
            self.current_frame,
            self.all_points,
            self.point_colors,
            self.layer_tracks,
            self.stereo,
            use_stereo_depth=self.use_stereo_depth,
            focal_px=self.foc_l,
            baseline=self.camera_saperation,
            direction=direction,
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
        self.buf_ang_vel.append(np.array(self.drone_ang_vel, dtype=float))
        self.buf_att.append(np.array(self.imu_att, dtype=float))
        self.buf_pos.append(np.array(self.drone_pos, dtype=float))

    def _take_sensor_window(self):
        if self.buf_ang_vel:
            self.avg_ang_vel = np.mean(np.array(self.buf_ang_vel), axis=0)
            self.avg_att = np.mean(np.array(self.buf_att), axis=0)
            self.avg_pos = np.mean(np.array(self.buf_pos), axis=0)
        else:
            self.avg_ang_vel = np.array(self.drone_ang_vel, dtype=float)
            self.avg_att = np.array(self.imu_att, dtype=float)
            self.avg_pos = np.array(self.drone_pos, dtype=float)
        self.buf_ang_vel.clear()
        self.buf_att.clear()
        self.buf_pos.clear()

    def _current_height(self):
        return float(self.avg_pos[2])
