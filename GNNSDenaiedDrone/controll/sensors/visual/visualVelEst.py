"""Temporal feature tracking and visual velocity estimation."""

import cv2
import numpy as np
from visualization.display import draw_tracks

from .smallFeatures import (
    LAYER_COLORS,
    LayerTracks,
    build_triangles,
    filter_by_neighbors,
    replenish,
    unique_point_mask,
)


class VisualVelocityEstimator:
    """Mixin providing layered temporal LK tracking and image-flow velocity.

    The host estimator supplies prepared frames, camera calibration, sensor
    state, and ``_take_sensor_window`` / ``_current_height`` helpers.
    """

    def _init_velocity_estimator(self, pyramid_levels):
        """Initialize tracking histories, budgets, per-frame flow, and velocity state."""
        self.tracked_points_by_layer = {0: {}}
        self.tracked_points = self.tracked_points_by_layer[0]
        self.layer_tracks = {}
        self._layer_frame_idx = {}
        self.max_feature_points = 1500
        self.feature_min_distance = 30.0
        self.cross_layer_min_distance = 5.0
        self.all_points = [np.empty((0, 2), np.float32) for _ in range(pyramid_levels)]
        self.point_colors = [np.empty((0, 3), np.uint8) for _ in self.all_points]
        self.point_layers = np.empty(0, int)
        self.velocity_flows = np.empty((0, 2), np.float32)
        self.velocity_points = np.empty((0, 3), np.float32)
        self.velocity_point_mask = np.empty(0, bool)
        self.estimated_height = np.nan
        self.vkfs = {}
        self.camera_velocity = np.zeros(4)

    def aply_flow(self):
        """Track all layers, then publish aligned observations for velocity fitting."""
        self._take_sensor_window()
        num_layers = self._reset_tracking_frame()
        if len(self.buffer) < 2:
            return

        previous, current = self.buffer[-2], self.buffer[-1]
        if min(len(previous.frames), len(current.frames)) < 2:
            return

        layer_results = self._track_pyramid_layers(previous, current, num_layers)
        self._publish_velocity_tracks(layer_results)

    def _reset_tracking_frame(self):
        """Clear per-frame outputs and create empty per-layer track records."""
        num_layers = (
            len(self.current_frame.pyramids[0])
            if self.current_frame.pyramids else 0
        )
        self.layer_tracks = {
            layer: LayerTracks.empty(layer, self.current_frame.idx)
            for layer in range(num_layers)
        }
        self.all_points = [np.empty((0, 2), np.float32) for _ in range(num_layers)]
        self.point_colors = [np.empty((0, 3), np.uint8) for _ in range(num_layers)]
        self.point_layers = np.empty(0, int)
        self.velocity_flows = np.empty((0, 2), np.float32)
        self.velocity_points = np.empty((0, 3), np.float32)
        self.velocity_point_mask = np.empty(0, bool)
        return num_layers

    def _track_pyramid_layers(self, previous, current, num_layers):
        """Run left-image LK per layer and return successful aligned track groups."""
        if (
            not num_layers
            or not current.pyramids
            or not previous.pyramids
            or len(current.pyramids[0]) != num_layers
            or len(previous.pyramids[0]) != num_layers
        ):
            raise ValueError('Previous and current left-image pyramids must have the same layers')

        annotated = [image.copy() for image in current.frames]
        results = []
        budget, remainder = divmod(self.max_feature_points, num_layers)
        # Each layer has independent seeds; all returned coordinates are full-resolution.
        for layer, (gray_prev, gray_curr) in enumerate(zip(
            previous.pyramids[0], current.pyramids[0]
        )):
            annotated, flow, olds, news, _, _ = self.lucas_kanade_flow(
                previous.frames[0], annotated[0], previous.frames[1], annotated[1],
                tracking_gray=(gray_prev, gray_curr),
                layer=layer, max_points=budget + (layer < remainder),
            )
            if flow is not None:
                self.all_points[layer] = news[0][:, :2].astype(np.float32, copy=True)
                self.point_colors[layer] = np.tile(
                    np.asarray(LAYER_COLORS[layer % len(LAYER_COLORS)], np.uint8),
                    (len(flow), 1),
                )
                results.append((layer, flow, olds[0], news[0]))

        for layer, tracks in self.layer_tracks.items():
            colour = LAYER_COLORS[layer % len(LAYER_COLORS)]
            text = f'L{layer} 1/{2 ** layer} | {len(tracks.new_points)} points'
            for frame in annotated:
                cv2.putText(
                    frame, text, (8, 20 + 20 * layer), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, colour, 1, cv2.LINE_AA,
                )
        self.current_frame.frames = annotated
        return results

    def _publish_velocity_tracks(self, layer_results):
        """Combine layer observations and mark rows eligible for velocity fitting."""
        if not layer_results:
            return

        self.point_layers = np.concatenate([
            np.full(len(flow), layer, int)
            for layer, flow, _, _ in layer_results
        ])
        flow = np.vstack([result[1] for result in layer_results])
        old = np.vstack([result[2] for result in layer_results])
        new = np.vstack([result[3] for result in layer_results])
        self.velocity_flows = flow
        self.velocity_points = new
        keep = (
            np.isfinite(flow).all(axis=1)
            & np.isfinite(old).all(axis=1)
            & np.isfinite(new).all(axis=1)
            & (old[:, 2] > 0)
            & (new[:, 2] > 0)
        )
        ids = np.flatnonzero(keep)
        if len(ids):
            keep[ids] = self.point_prediction_filtering(old[ids], new[ids])
        # Duplicate observations remain visible per layer but count once in the fit.
        ids = np.flatnonzero(keep)
        keep[ids] = unique_point_mask(new[ids, :2], self.cross_layer_min_distance)
        self.velocity_point_mask = keep

    def lucas_kanade_flow(self, frame1, frame2, frame_r, frame_r_curr,
                         tracking_gray=None, *, layer, max_points):
        """Coordinate one layer's temporal LK tracking and update its track state."""
        if tracking_gray is None or len(tracking_gray) != 2:
            raise ValueError('Supply previous-left and current-left grayscale layers')
        gray1, gray2 = tracking_gray
        pixel_scale, height, width = self._validate_tracking_layer(
            frame1, tracking_gray, layer)
        annotated_frame, annotated_r = frame2.copy(), frame_r_curr.copy()
        lost = ([annotated_frame, annotated_r], None, None, None, None, None)
        empty = np.empty((0, 2), np.float32)
        cache = self._layer_cache(layer)

        lk = dict(
            winSize=(21, 21),
            maxLevel=1,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        pts = self._replenish(
            gray1, cache.get(0, empty), 0,
            max_points=max_points, pixel_scale=pixel_scale,
        )
        if len(pts) < 3:
            self._reset_layer_tracks(layer, cache)
            return lost

        tracks = self._track_forward_backward(
            gray1, gray2, pts, pixel_scale, width, height, lk)
        if tracks is None:
            self._reset_layer_tracks(layer, cache)
            return lost
        good_old, good_new = tracks
        filtered = self._filter_layer_tracks(good_old, good_new, pixel_scale)
        if filtered is None:
            self._reset_layer_tracks(layer, cache)
            return lost
        good_old, good_new, triangles, links = filtered

        return self._finish_layer_tracking(
            annotated_frame, annotated_r, good_old, good_new, triangles, links,
            cache, pixel_scale, layer,
        )

    def _validate_tracking_layer(self, full_image, gray_images, layer):
        """Check pyramid dimensions and return pixel scale plus full-image bounds."""
        pixel_scale = float(2 ** layer)
        height, width = full_image.shape[:2]
        expected = (
            (height + int(pixel_scale) - 1) // int(pixel_scale),
            (width + int(pixel_scale) - 1) // int(pixel_scale),
        )
        if any(gray.shape != expected for gray in gray_images):
            raise ValueError('Tracking images do not match the pyramid layer')
        return pixel_scale, height, width

    def _layer_cache(self, layer):
        """Return native-pixel seeds, clearing them if the previous frame was skipped."""
        cache = self.tracked_points_by_layer.setdefault(layer, {})
        previous_idx = self.buffer[-2].idx if len(self.buffer) >= 2 else self.current_frame.idx - 1
        if layer in self._layer_frame_idx and self._layer_frame_idx[layer] != previous_idx:
            cache.clear()
        return cache

    def _reset_layer_tracks(self, layer, cache):
        """Clear a failed layer's seeds and publish empty tracks for the current frame."""
        cache[0] = np.empty((0, 2), np.float32)
        self._layer_frame_idx[layer] = self.current_frame.idx
        self.layer_tracks[layer] = LayerTracks.empty(layer, self.current_frame.idx)

    def _track_forward_backward(self, gray_prev, gray_curr, seeds, pixel_scale,
                                width, height, lk):
        """Run forward/backward LK and return valid full-resolution pairs or None."""
        old = seeds.reshape(-1, 1, 2).astype(np.float32)
        new, status_forward, _ = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray_curr, old, None, **lk)
        if new is None or status_forward is None:
            return None
        valid = (
            (status_forward.ravel() == 1)
            & np.isfinite(new.reshape(-1, 2)).all(axis=1)
        )
        if not valid.any():
            return None
        old, new = old[valid], new[valid]
        back, status_backward, _ = cv2.calcOpticalFlowPyrLK(
            gray_curr, gray_prev, new, None, **lk)
        if back is None or status_backward is None:
            return None

        old, new, back = old.reshape(-1, 2), new.reshape(-1, 2), back.reshape(-1, 2)
        good = (
            (status_backward.ravel() == 1)
            & np.isfinite(back).all(axis=1)
            & (np.abs((old - back) * pixel_scale).max(axis=1) < 1.0)
            & (new[:, 0] >= 0) & (new[:, 0] * pixel_scale < width)
            & (new[:, 1] >= 0) & (new[:, 1] * pixel_scale < height)
        )
        return old[good] * pixel_scale, new[good] * pixel_scale

    def _filter_layer_tracks(self, good_old, good_new, pixel_scale):
        """Apply affine/neighbor filters and build the surviving layer mesh."""
        if len(good_old) >= 6:
            _, inliers = cv2.estimateAffinePartial2D(
                good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                maxIters=2000, confidence=0.99,
            )
            if inliers is not None:
                mask = inliers.ravel().astype(bool)
                good_old, good_new = good_old[mask], good_new[mask]
        mesh_args = dict(
            max_edge=200.0 * pixel_scale,
            min_area=20.0 * pixel_scale ** 2,
        )
        for _ in range(2):
            if len(good_new) < 3:
                break
            _, _, links, _ = self._build_triangles(good_new, **mesh_args)
            good_old, good_new = self.filter_by_neighbors(good_old, good_new, links)
        if len(good_old) < 3:
            return None
        triangles, _, links, _ = self._build_triangles(good_new, **mesh_args)
        return good_old, good_new, triangles, links

    def _finish_layer_tracking(self, annotated_frame, annotated_r, good_old, good_new,
                               triangles, links, cache, pixel_scale, layer):
        """Update tracking history and create this layer's flow, metadata, and views."""
        cache[0] = (good_new / pixel_scale).astype(np.float32)
        self._layer_frame_idx[layer] = self.current_frame.idx
        flow, _, _ = self.compensate_vertical(good_old, good_new, triangles)
        old_r = np.full_like(good_old, np.nan)
        new_r = np.full_like(good_new, np.nan)
        self.layer_tracks[layer] = LayerTracks(
            layer, self.current_frame.idx, good_old.copy(), good_new.copy(),
            flow.copy(), triangles.copy(), links.copy(),
        )
        z = self._current_height()
        olds = [np.column_stack((points, np.full(len(points), z))) for points in (good_old, old_r)]
        news = [np.column_stack((points, np.full(len(points), z))) for points in (good_new, new_r)]
        annotated_frame, annotated_r = draw_tracks(
            annotated_frame, annotated_r, good_new, new_r, links,
            color=LAYER_COLORS[layer % len(LAYER_COLORS)],
        )
        return [annotated_frame, annotated_r], flow, olds, news, triangles, links

    def _replenish(self, gray, pts, cam, max_points=300, *, pixel_scale=1.0):
        """Scale configured full-resolution feature spacing to the selected layer."""
        min_distance = max(1.0, self.feature_min_distance / pixel_scale)
        return replenish(gray, pts, cam, max_points, min_distance=min_distance)

    def compensate_vertical(self, good_old, good_new, triangles, min_area=4.0):
        """Estimate and remove radial expansion from the tracked mesh flow."""
        flow = good_new - good_old
        if len(triangles) == 0:
            return flow, 1.0, 0.0

        def area(points):
            a = points[:, 1] - points[:, 0]
            b = points[:, 2] - points[:, 0]
            return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

        a_old = area(good_old[triangles])
        a_new = area(good_new[triangles])
        ok = a_old > min_area
        if not ok.any():
            return flow, 1.0, 0.0

        scale = np.sqrt(np.median(a_new[ok] / a_old[ok]))
        height, width = self.frame_size[:2]
        center = np.array([width / 2, height / 2])
        flow_comp = flow - (scale - 1) * (good_old - center)
        divergence = np.log(scale) / self.dt
        return flow_comp, scale, divergence

    def _build_triangles(self, points, max_edge=200.0, min_area=20.0):
        """Build spatial links and triangles for track coherence checks."""
        return build_triangles(points, max_edge, min_area)

    def filter_by_neighbors(self, good_old, good_new, links, thresh=2.0,
                            min_links=1, neighbor_min_links=3):
        """Remove tracks whose motion disagrees with their mesh neighbors."""
        return filter_by_neighbors(
            good_old, good_new, links, thresh, min_links, neighbor_min_links)

    def estimate_velocities(self, flow, points, vertical_velocity=np.nan):
        """Estimate and return ``[vx, vy, vz, yaw_rate]`` from visual and depth data."""
        vz = (
            float(vertical_velocity)
            if np.isfinite(vertical_velocity)
            else float(self.drone_vel[2])
        )
        if len(points) < 2:
            self.estimated_height = np.nan
            self.camera_velocity = np.zeros(4)   # no usable tracks -> report zero velocity
            return self.camera_velocity.copy()

        ang_vel, att = self.avg_ang_vel, self.avg_att
        height, width = self.frame_size[:2]
        center = np.array([width / 2, height / 2])
        z = points[:, 2]
        k = max(1, int(len(z) * 0.10))
        z_estimated = np.median(np.partition(z, -k)[-k:])

        visual_flow = flow * (z / (self.foc_l * self.dt))[:, None]
        angular_velocity = ang_vel[[1, 0]] * np.array([1, 1])
        rotation_flow = z[:, None] / np.cos(att[[1, 0]]) ** 2 * angular_velocity
        target = visual_flow - rotation_flow
        position = (
            (z[:, None] / self.foc_l) * (points[:, :2] - center)
            + self.camera_offset_from_yaw_center
        )
        matrix = np.vstack([
            np.column_stack([np.ones(len(z)), np.zeros(len(z)), -position[:, 1]]),
            np.column_stack([np.zeros(len(z)), np.ones(len(z)), position[:, 0]]),
        ])
        measured, *_ = np.linalg.lstsq(
            matrix, np.concatenate([target[:, 0], target[:, 1]]), rcond=None)
        measured = -measured

        kalman = self.vkfs.setdefault(0, VelocityKalmanFilter())
        if not kalman.started:
            kalman.state = measured.copy()
            kalman.started = True
        kalman.predict(kalman.state)
        kalman.update(measured)

        self.estimated_height = float(z_estimated)
        self.camera_velocity = np.insert(kalman.get(), 2, vz)
        return self.camera_velocity.copy()

    def point_prediction_filtering(self, old, new):
        """Reject tracks inconsistent with commanded motion and IMU rotation."""
        velocity = self.previous_cmd_vel[:3]
        angular_velocity = self.avg_ang_vel
        height, width = self.frame_size[:2]
        center = np.array([width / 2, height / 2])
        u = old[:, 0] - center[0]
        v_pix = old[:, 1] - center[1]
        z = old[:, 2]

        du = (-self.foc_l * velocity[0] + u * velocity[2]) / z * self.dt - v_pix * angular_velocity[2] * self.dt
        dv = (-self.foc_l * velocity[1] + v_pix * velocity[2]) / z * self.dt + u * angular_velocity[2] * self.dt
        predicted = np.column_stack([old[:, 0] + du, old[:, 1] + dv])
        residuals = np.clip(np.linalg.norm(new[:, :2] - predicted, axis=1), 0, 150)
        threshold = 1.5
        if residuals.sum() > 0:
            threshold = np.min(residuals) + 0.75 * (np.max(residuals) - np.min(residuals))
        return residuals < threshold

    def set_dt(self, dt):
        """Set the elapsed time used to convert image displacement into velocity."""
        self.dt = dt


class VelocityKalmanFilter:
    """Three-state Kalman filter for planar velocity and yaw rate."""

    def __init__(self, process_var=0.6, measurement_var=4.5,
                 yaw_process_var=5.0, yaw_measurement_var=1):
        self.state = np.zeros(3)
        self.P = np.eye(3)
        self.Q = np.diag([process_var, process_var, yaw_process_var])
        self.R = np.diag([measurement_var, measurement_var, yaw_measurement_var])
        self.started = False

    def predict(self, prior):
        """Advance state covariance and set the supplied predicted state."""
        self.state = np.asarray(prior, dtype=float)
        self.P = self.P + self.Q

    def update(self, measured):
        """Fuse one visual motion measurement into the predicted state."""
        gain = self.P @ np.linalg.inv(self.P + self.R)
        self.state = self.state + gain @ (measured - self.state)
        self.P = (np.eye(3) - gain) @ self.P

    def get(self):
        """Return a copy of the current filtered velocity/yaw state."""
        return self.state.copy()