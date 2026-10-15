"""Temporal feature tracking and visual velocity estimation."""

import cv2
import numpy as np
from swarm.visualization.display import draw_tracks

from swarm.compVision.estimatorTypes import Tracks
from swarm.compVision.smallFeatures import (
    TRACK_COLOR,
    build_triangles,
    filter_by_neighbors,
    replenish,
)
from swarm.compVision.gpuVision import VisionBackend


class VisualVelocityEstimator:
    """Mixin providing temporal LK tracking and image-flow velocity.

    The host estimator supplies prepared frames, camera calibration, sensor
    state and ``_current_height`` helper.
    """

    def _init_velocity_estimator(self, *, vision_backend=None):
        """Initialize tracking history, budget, per-frame flow, and velocity state."""
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
        self.height_smoothing = 0.2  # EMA weight of the newest height, 1 disables smoothing
        self._smoothed_z = None
        self.velocity_log = []
        self.tracking_view = None
        self.vkfs = {}
        self.camera_velocity = np.zeros(4)

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
        """Clear per-frame outputs and create an empty track record."""
        self.tracks = Tracks.empty(self.current_frame.idx)
        self.velocity_flows = np.empty((0, 2), np.float32)
        self.velocity_points = np.empty((0, 3), np.float32)
        self.velocity_point_mask = np.empty(0, bool)

    def _track(self, previous, current):
        """Run left-image LK and draw the surviving tracks on the display frames."""
        tracked = self.lucas_kanade_flow(previous.gray_frames[0], current.gray_frames[0])
        annotated = [image.copy() for image in current.frames]
        if tracked is not None:
            new = tracked[2]
            annotated[0], annotated[1] = draw_tracks(
                annotated[0], annotated[1], new[:, :2],
                np.full((len(new), 2), np.nan), self.tracks.links, color=TRACK_COLOR,
            )
        for frame in annotated:
            cv2.putText(
                frame, f'{len(self.tracks.new_points)} points', (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .5, TRACK_COLOR, 1, cv2.LINE_AA,
            )
        self.current_frame.frames = annotated
        return tracked

    def _publish_velocity_tracks(self, flow, old, new):
        """Store observations and mark rows eligible for velocity fitting."""
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
        self.velocity_point_mask = keep

    def lucas_kanade_flow(self, gray_prev, gray_curr):
        """Track seeds from the previous to the current left image.

        Returns (flow, old, new) with (u, v, height) rows, or None when tracking is lost.
        """
        height, width = gray_prev.shape
        previous_idx = self.buffer[-2].idx if len(self.buffer) >= 2 else self.current_frame.idx - 1
        if self._tracked_frame_idx is not None and self._tracked_frame_idx != previous_idx:
            self.tracked_points = np.empty((0, 2), np.float32)

        lk = dict(
            winSize=(21, 21),
            maxLevel=1,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        seeds = replenish(
            gray_prev, self.tracked_points, 0, max_points=self.max_feature_points,
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
        """Clear the seeds and publish empty tracks for the current frame."""
        self.tracked_points = np.empty((0, 2), np.float32)
        self._tracked_frame_idx = self.current_frame.idx
        self.tracks = Tracks.empty(self.current_frame.idx)

    def _track_forward(self, gray_prev, gray_curr, seeds, width, height, lk):
        """Run forward LK and return valid (old, new) pairs or None."""
        old = seeds.reshape(-1, 1, 2).astype(np.float32)

        new, status, _ = self.vision_backend.calcOpticalFlowPyrLK(gray_prev, gray_curr, old, None, **lk)

        if new is None or status is None: return None

        old = old.reshape(-1, 2)
        new = new.reshape(-1, 2)

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
        """Apply affine/neighbor filters and build the surviving mesh."""
        if len(good_old) >= 6:
            _, inliers = cv2.estimateAffinePartial2D(
                good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                maxIters=2000, confidence=0.99,
            )
            if inliers is not None:
                mask = inliers.ravel().astype(bool)
                good_old, good_new = good_old[mask], good_new[mask]
        mesh_args = dict(max_edge=200.0, min_area=20.0)
        for _ in range(2):
            if len(good_new) < 3:
                break
            _, _, links, _ = self._build_triangles(good_new, **mesh_args)
            good_old, good_new = self.filter_by_neighbors(good_old, good_new, links)
        if len(good_old) < 3:
            return None
        triangles, _, links, _ = self._build_triangles(good_new, **mesh_args)
        return good_old, good_new, triangles, links
    

    def _finish_tracking(self, good_old, good_new, triangles, links):
        """Update tracking history and return the flow with (u, v, height) rows."""
        self.tracked_points = good_new.astype(np.float32)
        self._tracked_frame_idx = self.current_frame.idx
        flow, _, _ = self.compensate_vertical(good_old, good_new, triangles)
        self.tracks = Tracks(
            self.current_frame.idx, good_old.copy(), good_new.copy(),
            flow.copy(), triangles.copy(), links.copy(),
        )
        z = self._current_height()
        olds = np.column_stack((good_old, np.full(len(good_old), z)))
        news = np.column_stack((good_new, np.full(len(good_new), z)))
        return flow, olds, news
    

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

    def velocity_inputs(self):
        """Return masked (flow, points) with column 2 set to each point's stereo depth.

        Points without a valid stereo depth keep the drone altitude.
        """
        keep = self.velocity_point_mask
        points = self.velocity_points[keep]
        depths = self.point_depth.depths[keep]
        points[:, 2] = np.where(np.isfinite(depths) & (depths > 0), depths, points[:, 2])
        return self.velocity_flows[keep], points

    def estimate_velocities(self, flow, points):
        """Estimate [vx, vy, vz, yaw_rate]; vz is the sensor value."""
        vz = float(self.drone_vel[2])

        if len(points) < 2:
            self.estimated_height = np.nan
            self.camera_velocity = np.array([0.0, 0.0, vz, 0.0])
            self._log_velocity(np.full(3, np.nan), len(points), np.nan)
            return self.camera_velocity.copy()

        ang_vel, att = self.drone_ang_vel, self.imu_att
        h, w = self.frame_size[:2]
        c = np.array([w / 2, h / 2])

        n = len(points)
        z = points[:, 2]

        k = max(1, int(n * 0.10))
        z_estimated = np.median(np.partition(z, -k)[-k:])

        v = flow * (z / (self.foc_l * self.dt))[:, None]

        # Pitch affects image x; roll affects image y.
        roll, pitch = att[0], att[1]
        roll_rate, pitch_rate = ang_vel[0], ang_vel[1]

        v_rot = z[:, None] * np.column_stack([
            pitch_rate / np.cos(pitch) ** 2,
            roll_rate / np.cos(roll) ** 2,
        ])

        B = v - v_rot

        r = (
            (z[:, None] / self.foc_l) * (points[:, :2] - c)
            + self.camera_offset_from_yaw_center
        )

        A = np.vstack([
            np.column_stack([np.ones(n), np.zeros(n), -r[:, 1]]),
            np.column_stack([np.zeros(n), np.ones(n),  r[:, 0]]),
        ])

        measured, *_ = np.linalg.lstsq(
            A, np.concatenate([B[:, 0], B[:, 1]]), rcond=None
        )
        measured = -measured

        kf = self.vkfs.setdefault(0, VelocityKalmanFilter())

        if not kf.started:
            kf.state = measured.copy()
            kf.started = True

        kf.predict(kf.state)
        kf.update(measured)

        self.estimated_height = float(z_estimated)
        self.camera_velocity = np.insert(kf.get(), 2, vz)

        self._log_velocity(measured, n, z_estimated)

        return self.camera_velocity.copy()

    LOG_COLUMNS = (
        'frame', 'dt', 'meas_vx', 'meas_vy', 'meas_yaw_rate',
        'vx', 'vy', 'vz', 'yaw_rate', 'height', 'points',
        'gyro_x', 'gyro_y', 'gyro_z', 'roll', 'pitch', 'yaw',
        'drone_vx', 'drone_vy', 'drone_vz',
    )

    def _log_velocity(self, measured, point_count, height):
        """Append one row: raw least-squares fit, filtered output and the inputs behind it."""
        self.velocity_log.append([
            self.current_frame.idx, self.dt, *measured, *self.camera_velocity,
            height, point_count, *self.drone_ang_vel, *self.imu_att, *self.drone_vel,
        ])

    def save_velocity_log(self, path):
        """Write the recorded per-frame velocity rows to a CSV file."""
        np.savetxt(path, np.asarray(self.velocity_log, float), delimiter=',',
                   header=','.join(self.LOG_COLUMNS), comments='')

    def point_prediction_filtering(self, old, new):
        """Reject tracks inconsistent with commanded motion and IMU rotation."""
        velocity = self.previous_cmd_vel[:3]
        angular_velocity = self.drone_ang_vel
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

    def __init__(self, process_var=0.6, measurement_var=5,
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