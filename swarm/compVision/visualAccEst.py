"""Visual velocity estimation and the backwards-compatible pipeline entry point.

Frame -> large features -> small features -> depth -> velocity.
Image preparation and feature/depth algorithms live in their own modules;
VisulaAcEst keeps the public interface used by the simulator and camera runner.
"""
from collections import deque

import cv2
import numpy as np

import swarm.config.cameraConfig as camConfig
from swarm.compVision.frame import Frame, FrameProcessor
from swarm.compVision.largeFeatures import LargeFeatures, LargeFeatureDetector
from swarm.compVision.smallFeatures import SmallFeatureTracker
from swarm.compVision.depth import DepthEstimator

# Keep the original public name available to callers.
EnumFrame = Frame


class VisulaAcEst:
    def __init__(self, res, fov):
        self.fov = fov
        self.foc_l = (res[1] / 2) / np.tan(np.deg2rad(fov) / 2)
        self.dt = 0.01
        self.camera_saperation = camConfig.STEREO_CAM.baseline
        self.disparity_direction = np.asarray(camConfig.STEREO_CAM.disparity_direction(), float)
        # The simulator's common-intrinsic camera model supports this geometry.
        # Turn off for raw real-camera pairs until they are stereo-rectified.
        self.use_stereo_depth = True
        self.frame_processor = FrameProcessor()
        self.large_feature_detector = LargeFeatureDetector()
        self.small_features = SmallFeatureTracker()
        self.depth_estimator = DepthEstimator()
        self.large_features = LargeFeatures()
        self.previous_large_features = LargeFeatures()
        self.current_frame = Frame()
        # Only the current and preceding stereo sample are needed by tracking.
        self.buffer = deque(maxlen=2)
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.drone_ang_vel = np.zeros(3)
        self.drone_ang = np.zeros(2)
        self.imu_att = np.zeros(3)
        self.previous_cmd_vel = np.zeros(4)
        self.buf_ang_vel = deque(maxlen=20)
        self.buf_att = deque(maxlen=20)
        self.buf_pos = deque(maxlen=20)
        self.avg_ang_vel = np.zeros(3)
        self.avg_att = np.zeros(3)
        self.avg_pos = np.zeros(3)
        self.vkfs = {}
        # Only vx, vy, yaw_rate are solved visually; vz is supplied sensor vz.
        self.camera_velocity = np.zeros(4)
        self.estimated_height = np.nan
        self.vertical_divergence = 0.0
        self.depth_median = np.nan
        self.depth_valid_frac = 0.0
        self.depth_points = np.empty((0, 3))
        self.depth_frame_idx = -1
        self.previous_depth = None
        self.current_depth = None

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

    @property
    def tracked_points(self):
        return self.small_features.tracked_points

    @tracked_points.setter
    def tracked_points(self, value):
        self.small_features.tracked_points = value

    def preprocess_frame(self, frame):
        return self.frame_processor.preprocess(frame)

    def update_frame(self, frames, idx):
        prepared = self.frame_processor.process(frames, idx)
        if hasattr(self, 'frame_size') and prepared.frames[0].shape != self.frame_size:
            raise ValueError('Frame resolution changed; create a new estimator with calibration for the new resolution')
        self.frame_size = prepared.frames[0].shape
        self.buffer.append(prepared)
        self.current_frame = prepared.display_copy()
        self.previous_large_features = self.large_features
        self.large_features = self.large_feature_detector.detect(prepared.pyramids[0])

    def processing(self, frame, idx):
        self.update_frame(frame, idx)
        self.aply_flow()
        return self.current_frame

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

    def _clear_depth(self):
        self.vertical_divergence = 0.0
        self.depth_median = np.nan
        self.depth_valid_frac = 0.0
        self.depth_points = np.empty((0, 3))
        self.depth_frame_idx = self.current_frame.idx
        self.previous_depth = self.current_depth = None

    def aply_flow(self):
        self._take_sensor_window()
        self._clear_depth()
        if len(self.buffer) < 2:
            self.camera_velocity = self._fallback_velocity()
            return
        previous, current = self.buffer[-2], self.buffer[-1]
        processed, flow, olds, news, _, _ = self.lucas_kanade_flow(
            previous.frames[0], current.frames[0], previous.frames[1], current.frames[1],
            tracking_gray=(previous.gray_frames[0], current.gray_frames[0],
                           previous.gray_frames[1], current.gray_frames[1]))
        self.current_frame.frames = processed
        if flow is None:
            self.camera_velocity = self._fallback_velocity()
            return
        keep = self.point_prediction_filtering(olds[0], news[0])
        self.estimate_velocities(flow[keep], news[0][keep])

    def _depth_for(self, left, right, points, idx):
        calibration = dict(focal_px=self.foc_l, baseline=self.camera_saperation,
                           disparity_direction=self.disparity_direction,
                           fallback_height=self._current_height(), frame_idx=idx)
        if self.use_stereo_depth:
            return self.depth_estimator.estimate(left, right, points, **calibration)
        return self.depth_estimator.triangulate(
            points, np.full_like(points, np.nan), np.zeros(len(points), bool), **calibration)

    def lucas_kanade_flow(self, frame1, frame2, frame_r, frame_r_curr, tracking_gray=None):
        """Compatibility adapter: delegate tracking/depth and retain the six outputs."""
        if tracking_gray is None:
            tracking_gray = tuple(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                  for frame in (frame1, frame2, frame_r, frame_r_curr))
        gray1, gray2, gray_r = tracking_gray[:3]
        gray_r_curr = (tracking_gray[3] if len(tracking_gray) == 4 else
                       cv2.cvtColor(frame_r_curr, cv2.COLOR_BGR2GRAY))
        annotated, annotated_r = frame2.copy(), frame_r_curr.copy()
        tracks = self.small_features.track(gray1, gray2, self.previous_large_features.regions)
        if len(tracks.new_points) < 3:
            self._clear_depth()
            return [annotated, annotated_r], None, None, None, None, None
        previous_idx = self.buffer[-2].idx if len(self.buffer) >= 2 else self.current_frame.idx - 1
        self.previous_depth = self._depth_for(gray1, gray_r, tracks.old_points, previous_idx)
        self.current_depth = self._depth_for(gray2, gray_r_curr, tracks.new_points, self.current_frame.idx)
        self.depth_median = self.current_depth.median
        self.depth_valid_frac = self.current_depth.valid_fraction
        self.depth_points = self.current_depth.points
        self.depth_frame_idx = self.current_depth.frame_idx
        old_depth, new_depth = self.previous_depth, self.current_depth
        olds = [np.column_stack((old_depth.left_points, old_depth.depths)),
                np.column_stack((old_depth.right_points, old_depth.depths))]
        news = [np.column_stack((new_depth.left_points, new_depth.depths)),
                np.column_stack((new_depth.right_points, new_depth.depths))]
        flow, _, self.vertical_divergence = self.compensate_vertical(
            tracks.old_points, tracks.new_points, tracks.triangles)
        annotated, annotated_r = self.drawing(
            annotated, annotated_r, tracks.new_points, new_depth.right_points, tracks.links)
        # These current-frame regions really belong to the displayed left frame.
        for x0, y0, x1, y1 in self.large_features.regions:
            cv2.rectangle(annotated, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 255), 1)
        return [annotated, annotated_r], flow, olds, news, tracks.triangles, tracks.links

    # Thin compatibility delegates; the implementations live in their stages.
    def _replenish(self, gray, pts, cam, max_points=300):
        return self.small_features.replenish(gray, pts, cam, max_points,
                                             regions=self.previous_large_features.regions)

    def _build_triangles(self, points, max_edge=200.0, min_area=20.0):
        return self.small_features.build_triangles(points, max_edge, min_area)

    def filter_by_neighbors(self, good_old, good_new, links, thresh=2.0, min_links=1, neighbor_min_links=3):
        return self.small_features.filter_by_neighbors(
            good_old, good_new, links, thresh, min_links, neighbor_min_links)

    def trinagulate_altitude(self, news, flow, olds, triangles, links):
        """Legacy spelling retained; supplied pairs must be real correspondences."""
        calibration = dict(focal_px=self.foc_l, baseline=self.camera_saperation,
                           disparity_direction=self.disparity_direction,
                           fallback_height=self._current_height())
        for pairs in (olds, news):
            result = self.depth_estimator.triangulate(
                pairs[0][:, :2], pairs[1][:, :2], np.ones(len(pairs[0]), bool),
                frame_idx=self.current_frame.idx, **calibration)
            for points in pairs:
                points[:, 2] = result.depths
        self.depth_median, self.depth_valid_frac = result.median, result.valid_fraction
        self.depth_points, self.depth_frame_idx = result.points, result.frame_idx
        return news

    # ---------------- velocity calculations ----------------
    def compensate_vertical(self, good_old, good_new, triangles, min_area=4.0):
        flow = good_new - good_old
        if len(triangles) == 0:
            return flow, 1.0, 0.0
        def area(points):
            a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
            return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        a_old, a_new = area(good_old[triangles]), area(good_new[triangles])
        ok = (a_old > min_area) & (a_new > 0)
        if not ok.any():
            return flow, 1.0, 0.0
        scale = np.sqrt(np.median(a_new[ok] / a_old[ok]))
        h, w = self.frame_size[:2]
        flow_comp = flow - (scale - 1) * (good_old - np.array([w / 2, h / 2]))
        return flow_comp, float(scale), float(np.log(scale) / self.dt)

    def estimate_velocities(self, flow, points):
        flow, points = np.asarray(flow), np.asarray(points)
        valid = (np.isfinite(flow).all(axis=1) & np.isfinite(points).all(axis=1)
                 & (points[:, 2] > 0))
        flow, points = flow[valid], points[valid]
        if len(points) < 2:
            self.camera_velocity = self._fallback_velocity()
            return
        ang_vel, att = self.avg_ang_vel, self.avg_att
        cosine = np.cos(att[[1, 0]]) ** 2
        if (cosine < 1e-6).any() or not np.isfinite(cosine).all() or not np.isfinite(ang_vel).all():
            self.camera_velocity = self._fallback_velocity()
            return
        h, w = self.frame_size[:2]
        center = np.array([w / 2, h / 2])
        n, z = len(points), points[:, 2]
        k = max(1, int(len(z) * 0.10))
        self.estimated_height = float(np.median(np.partition(z, -k)[-k:]))
        velocity = flow * (z / (self.foc_l * self.dt))[:, None]
        rotational = z[:, None] / cosine * ang_vel[[1, 0]]
        target = velocity - rotational
        r = (z[:, None] / self.foc_l) * (points[:, :2] - center) + self.camera_saperation / 2
        matrix = np.vstack([
            np.column_stack([np.ones(n), np.zeros(n), -r[:, 1]]),
            np.column_stack([np.zeros(n), np.ones(n), r[:, 0]])])
        estimate, *_ = np.linalg.lstsq(matrix, np.concatenate([target[:, 0], target[:, 1]]), rcond=None)
        estimate = -estimate
        kf = self.vkfs.setdefault(0, VelocityKalmanFilter())
        if not kf.started:
            kf.state, kf.started = estimate.copy(), True
        kf.predict(kf.state)
        kf.update(estimate)
        vx, vy, yaw_rate = kf.get()
        self.camera_velocity = np.array([vx, vy, self.drone_vel[2], yaw_rate])

    def point_prediction_filtering(self, old, new):
        keep = np.zeros(len(old), bool)
        valid = (np.isfinite(old).all(axis=1) & np.isfinite(new).all(axis=1)
                 & (old[:, 2] > 0) & (new[:, 2] > 0))
        if not valid.any():
            return keep
        previous, current = old[valid], new[valid]
        v, ang = self.previous_cmd_vel[:3], self.avg_ang_vel
        h, w = self.frame_size[:2]
        u, v_pix, z = previous[:, 0] - w / 2, previous[:, 1] - h / 2, previous[:, 2]
        du = (-self.foc_l * v[0] + u * v[2]) / z * self.dt - v_pix * ang[2] * self.dt
        dv = (-self.foc_l * v[1] + v_pix * v[2]) / z * self.dt + u * ang[2] * self.dt
        predicted = previous[:, :2] + np.column_stack((du, dv))
        residuals = np.clip(np.linalg.norm(current[:, :2] - predicted, axis=1), 0, 150)
        threshold = 1.5
        if residuals.sum() > 0:
            threshold = residuals.min() + 0.75 * (residuals.max() - residuals.min())
        keep[valid] = np.isfinite(residuals) & (residuals <= threshold)
        return keep

    @staticmethod
    def drawing(annotated_frame, annotated_r, good_new, new_r, links):
        for frame, points in ((annotated_frame, good_new), (annotated_r, new_r)):
            finite = np.isfinite(points).all(axis=1)
            for a, b in links:
                if finite[a] and finite[b]:
                    cv2.line(frame, tuple(points[a].astype(int)), tuple(points[b].astype(int)), (255, 0, 0), 2)
            for x, y in points[finite].astype(int):
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        return annotated_frame, annotated_r

    def set_dt(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        self.dt = float(dt)

    def _fallback_velocity(self):
        self.estimated_height = np.nan
        return np.array([self.drone_vel[0], self.drone_vel[1], self.drone_vel[2], self.drone_ang_vel[2]])


class VelocityKalmanFilter:
    def __init__(self, process_var=0.6, measurement_var=4.5, yaw_process_var=5.0, yaw_measurement_var=1):
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
