"""Temporal feature tracking and visual velocity estimation."""

from collections import deque

import cv2
import numpy as np
from visualization.display import draw_tracks

from .estimatorTypes import Tracks
from .gpuVision import VisionBackend
from .smallFeatures import TRACK_COLOR, build_triangles, filter_by_neighbors, replenish


class VisualVelocityEstimator:
    """Single-resolution LK tracking and image-flow velocity estimation."""

    def _init_velocity_estimator(self, *, vision_backend=None):
        self.vision_backend = VisionBackend(vision_backend)
        self.tracked_points = np.empty((0, 2), np.float32)
        self.tracks = Tracks.empty(0)
        self._tracked_frame_idx = None
        self.max_feature_points = 1500
        self.feature_min_distance = 30.0
        self.velocity_flows = np.empty((0, 2), np.float32)
        self.velocity_points = np.empty((0, 3), np.float32)
        self.velocity_point_mask = np.empty(0, bool)
        self.estimated_height = np.nan
        self.velocity_log = []
        self.tracking_view = None
        self.vkfs = {}
        self.camera_velocity = np.zeros(4)
        self.camera_velocity_buffer = deque(maxlen=3)

    def aply_flow(self):
        """Track the left image, then publish observations for velocity fitting."""
        self._reset_tracking_frame()
        if len(self.buffer) < 2:
            return
        previous, current = self.buffer[-2], self.buffer[-1]
        if min(len(previous.frames), len(current.frames)) < 2:
            return
        tracked = self._track(previous, current)
        if tracked is not None:
            self._publish_velocity_tracks(*tracked)

    def _reset_tracking_frame(self):
        self.tracks = Tracks.empty(self.current_frame.idx)
        self.velocity_flows = np.empty((0, 2), np.float32)
        self.velocity_points = np.empty((0, 3), np.float32)
        self.velocity_point_mask = np.empty(0, bool)

    def _track(self, previous, current):
        tracked = self.lucas_kanade_flow(
            previous.gray_frames[0], current.gray_frames[0]
        )
        annotated = [image.copy() for image in current.frames]
        if tracked is not None:
            _, _, new = tracked
            annotated[0], annotated[1] = draw_tracks(
                annotated[0], annotated[1], new[:, :2],
                np.full((len(new), 2), np.nan), self.tracks.links,
                color=TRACK_COLOR,
            )
        for frame in annotated:
            cv2.putText(
                frame, f'{len(self.tracks.new_points)} points', (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .5, TRACK_COLOR, 1, cv2.LINE_AA,
            )
        self.current_frame.frames = annotated
        return tracked

    def _publish_velocity_tracks(self, flow, old, new):
        """Publish finite observations; command-motion filtering stays disabled."""
        self.velocity_flows = flow
        self.velocity_points = new
        self.velocity_point_mask = (
            np.isfinite(flow).all(axis=1)
            & np.isfinite(old).all(axis=1)
            & np.isfinite(new).all(axis=1)
        )

    def lucas_kanade_flow(self, gray_prev, gray_curr):
        """Track left-image features and return (flow, old, new) in full pixels."""
        height, width = gray_prev.shape
        previous_idx = (
            self.buffer[-2].idx if len(self.buffer) >= 2
            else self.current_frame.idx - 1
        )
        if (self._tracked_frame_idx is not None
                and self._tracked_frame_idx != previous_idx):
            self.tracked_points = np.empty((0, 2), np.float32)

        lk = dict(
            winSize=(21, 21),
            maxLevel=1,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
        )
        seeds = replenish(
            gray_prev, self.tracked_points, self.max_feature_points,
            min_distance=self.feature_min_distance, backend=self.vision_backend,
        )
        if len(seeds) < 3:
            return self._lose_tracks()
        tracks = self._track_forward(gray_prev, gray_curr, seeds, width, height, lk)
        if tracks is None:
            return self._lose_tracks()
        filtered = self._filter_tracks(*tracks)
        if filtered is None:
            return self._lose_tracks()
        return self._finish_tracking(*filtered)

    def _lose_tracks(self):
        self._reset_tracking_frame()
        self.tracked_points = np.empty((0, 2), np.float32)
        self._tracked_frame_idx = self.current_frame.idx

    def _track_forward(self, gray_prev, gray_curr, seeds, width, height, lk):
        old = seeds.reshape(-1, 1, 2).astype(np.float32)
        new, status, _ = self.vision_backend.calcOpticalFlowPyrLK(
            gray_prev, gray_curr, old, None, **lk
        )
        if new is None or status is None:
            return None
        old, new = old.reshape(-1, 2), new.reshape(-1, 2)
        good = (
            (status.ravel() == 1)
            & np.isfinite(new).all(axis=1)
            & (new[:, 0] >= 0) & (new[:, 0] < width)
            & (new[:, 1] >= 0) & (new[:, 1] < height)
        )
        if not good.any():
            return None
        return old[good], new[good]

    def _filter_tracks(self, good_old, good_new):
        if len(good_old) >= 6:
            _, inliers = cv2.estimateAffinePartial2D(
                good_old, good_new, method=cv2.RANSAC,
                ransacReprojThreshold=2.0, maxIters=2000, confidence=.99,
            )
            if inliers is not None:
                keep = inliers.ravel().astype(bool)
                good_old, good_new = good_old[keep], good_new[keep]
        for _ in range(2):
            if len(good_new) < 3:
                break
            _, _, links, _ = self._build_triangles(good_new)
            good_old, good_new = self.filter_by_neighbors(
                good_old, good_new, links
            )
        if len(good_old) < 3:
            return None
        triangles, _, links, _ = self._build_triangles(good_new)
        return good_old, good_new, triangles, links

    def _finish_tracking(self, good_old, good_new, triangles, links):
        self.tracked_points = good_new.astype(np.float32)
        self._tracked_frame_idx = self.current_frame.idx
        flow = good_new - good_old
        self.tracks = Tracks(
            self.current_frame.idx, good_old.copy(), good_new.copy(),
            flow.copy(), triangles.copy(), links.copy(),
        )
        height = self._current_height()
        old_samples = np.column_stack((good_old, np.full(len(good_old), height)))
        new_samples = np.column_stack((good_new, np.full(len(good_new), height)))
        return flow, old_samples, new_samples

    def _build_triangles(self, points, max_edge=200.0, min_area=20.0):
        return build_triangles(points, max_edge, min_area)

    def filter_by_neighbors(self, good_old, good_new, links, thresh=2.0,
                            min_links=1, neighbor_min_links=3):
        return filter_by_neighbors(
            good_old, good_new, links, thresh, min_links, neighbor_min_links
        )

    def velocity_inputs(self):
        """Return tracked flow/points having valid same-frame stereo depth."""
        points = self.velocity_points
        flow = self.velocity_flows
        depths = self.point_depth.depths
        if len(points) != len(depths):
            return np.empty((0, 2)), np.empty((0, 3))
        valid = self.velocity_point_mask & np.isfinite(depths) & (depths > 0)
        points = points[valid].copy()
        flow = flow[valid]
        points[:, 2] = depths[valid]
        return flow, points

    def _average_camera_velocity(self, velocity):
        self.camera_velocity_buffer.append(np.asarray(velocity, dtype=float).copy())
        return np.mean(self.camera_velocity_buffer, axis=0)

    def estimate_velocities(self, flow, points, vertical_velocity=np.nan):
        """Estimate [vx, vy, vz, yaw_rate] and average final velocity samples."""
        vz = (
            float(vertical_velocity) if np.isfinite(vertical_velocity)
            else float(self.drone_vel[2])
        )
        if len(points) < 2:
            self.estimated_height = np.nan
            velocity = self._average_camera_velocity([0.0, 0.0, vz, 0.0])
            self.camera_velocity = velocity
            return velocity.copy()

        angular_velocity = getattr(self, 'avg_ang_vel', self.drone_ang_vel)
        attitude = getattr(self, 'avg_att', self.imu_att)
        height, width = self.frame_size[:2]
        center = np.array([width / 2, height / 2])
        z = points[:, 2]
        k = max(1, int(len(z) * .1))
        estimated_height = np.median(np.partition(z, -k)[-k:])

        visual_flow = flow * (z / (self.foc_l * self.dt))[:, None]
        rotation = z[:, None] / np.cos(attitude[[1, 0]]) ** 2
        rotation *= angular_velocity[[1, 0]]
        target = visual_flow - rotation
        position = (
            (z[:, None] / self.foc_l) * (points[:, :2] - center)
            + self.camera_offset_from_yaw_center
        )
        matrix = np.vstack((
            np.column_stack((np.ones(len(z)), np.zeros(len(z)), -position[:, 1])),
            np.column_stack((np.zeros(len(z)), np.ones(len(z)), position[:, 0])),
        ))
        measured, *_ = np.linalg.lstsq(
            matrix, np.concatenate((target[:, 0], target[:, 1])), rcond=None
        )
        measured = -measured

        kalman = self.vkfs.setdefault(0, VelocityKalmanFilter())
        if not kalman.started:
            kalman.state = measured.copy()
            kalman.started = True
        kalman.predict(kalman.state)
        kalman.update(measured)

        self.estimated_height = float(estimated_height)
        #velocity = np.insert(kalman.get(), 2, vz)
        velocity = np.insert(measured, 2, vz)
        self.camera_velocity = self._average_camera_velocity(velocity)
        return self.camera_velocity.copy()

    def set_dt(self, dt):
        self.dt = dt


class VelocityKalmanFilter:
    """Three-state Kalman filter for planar velocity and yaw rate."""

    def __init__(self, process_var=.6, measurement_var=5,
                 yaw_process_var=5.0, yaw_measurement_var=1):
        self.state = np.zeros(3)
        self.P = np.eye(3)
        self.Q = np.diag([process_var, process_var, yaw_process_var])
        self.R = np.diag([measurement_var, measurement_var, yaw_measurement_var])
        self.started = False

    def predict(self, prior):
        self.state = np.asarray(prior, dtype=float)
        self.P = self.P + self.Q

    def update(self, measured):
        gain = self.P @ np.linalg.inv(self.P + self.R)
        self.state = self.state + gain @ (measured - self.state)
        self.P = (np.eye(3) - gain) @ self.P

    def get(self):
        return self.state.copy()